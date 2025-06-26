package jcog.tensor.rl.pg2;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg2.configs.PPOAgentConfig;
import jcog.tensor.rl.pg2.memory.OnPolicyBuffer;
import jcog.tensor.rl.pg2.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg2.networks.ValueNet;
import jcog.tensor.rl.pg2.stats.MetricCollector;
import jcog.tensor.rl.pg2.util.AgentUtils;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class PPOAgent extends BasePolicyGradientAgent {

    public final PPOAgentConfig config;
    public final GaussianPolicyNet policy;
    public final ValueNet valueFunction;
    public final Tensor.Optimizer policyOptimizer;
    public final Tensor.Optimizer valueOptimizer;

    private final Tensor.GradQueue policyGradQueue;
    private final Tensor.GradQueue valueGradQueue;

    public PPOAgent(PPOAgentConfig config, int stateDim, int actionDim, @Nullable MetricCollector metricCollector) {
        super(stateDim, actionDim, new OnPolicyBuffer(config.memoryConfig().episodeLength().intValue()), metricCollector);
        Objects.requireNonNull(config, "Agent configuration cannot be null");
        this.config = config;

        this.policy = new GaussianPolicyNet(config.policyNetworkConfig(), stateDim, actionDim);
        this.valueFunction = new ValueNet(config.valueNetworkConfig(), stateDim);

        this.policyOptimizer = config.policyNetworkConfig().optimizer().build();
        this.valueOptimizer = config.valueNetworkConfig().optimizer().build();

        this.policyGradQueue = new Tensor.GradQueue();
        this.valueGradQueue = new Tensor.GradQueue();

        setTrainingMode(true); // Initialize training mode by default
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        // This method is required by the interface. For PPO, it's often better to also get the logProb.
        // Callers can use selectActionWithLogProb if they need it for Experience2.
        ActionWithLogProb result = selectActionWithLogProb(state, deterministic);
        return result.action();
    }

    public record ActionWithLogProb(double[] action, Tensor logProb) {}

    public ActionWithLogProb selectActionWithLogProb(Tensor state, boolean deterministic) {
        try (var noGrad = Tensor.noGrad()) {
            AgentUtils.GaussianDistribution dist = policy.getDistribution(
                state,
                config.actionConfig().sigmaMin().floatValue(),
                config.actionConfig().sigmaMax().floatValue()
            );
            Tensor actionTensor = dist.sample(deterministic);
            // Detach actionTensor before logProb if it's used in logProb calculation and has grad history.
            // sample() from GaussianDistribution might or might not attach grad based on mu/sigma.
            // dist.logProb internally handles this. Detaching logProb itself is good practice.
            Tensor logProb = dist.logProb(actionTensor.detach());
            return new ActionWithLogProb(actionTensor.clipUnitPolar().array(), logProb.detach());
        }
    }

    @Override
    public void update(long totalSteps) {
        if (!this.trainingMode || this.memory.size() == 0) {
            return;
        }

        List<Experience2> batch = this.memory.getAll();

        Tensor states = Tensor.concatRows(batch.stream().map(Experience2::state).collect(Collectors.toList()));
        Tensor actions = Tensor.concatRows(batch.stream().map(e -> Tensor.row(e.action())).collect(Collectors.toList()));
        Tensor oldLogProbs = Tensor.concatRows(batch.stream().map(Experience2::oldLogProb).collect(Collectors.toList()));

        Tensor advantages;
        Tensor valueTargets;

        // Record mean of GAE advantages before normalization
        // However, advantages are computed inside the noGrad block and potentially normalized later.
        // It's better to record the version of advantages actually used for policy updates.
        // Let's record raw GAE advantages (before normalization) and normalized GAE advantages if normalization is applied.

        try (var noGrad = Tensor.noGrad()) {
            Tensor values = this.valueFunction.apply(states);

            Tensor lastNextValue;
            Experience2 lastExp = batch.getLast();
            if (!lastExp.done() && lastExp.nextState() != null) {
                 lastNextValue = this.valueFunction.apply(lastExp.nextState());
            } else {
                lastNextValue = Tensor.zeros(1,1);
            }

            GAEResult gaeResult = computeGAE(batch, values, lastNextValue.scalar());
            advantages = gaeResult.advantages();
            valueTargets = gaeResult.valueTargets();
        }

        // Record raw GAE advantages (mean)
        recordMetric("gae_advantages_raw_mean", advantages.mean().scalar());


        if (config.hyperparams().normalizeAdvantages()) {
            double[] advantagesArray = advantages.array().clone(); // Operate on a copy
            AgentUtils.normalize(advantagesArray);
            advantages = Tensor.row(advantagesArray).transpose(); // advantages variable is now normalized
            recordMetric("gae_advantages_normalized_mean", advantages.mean().scalar());
        }


        // Temporary variables to accumulate losses over epochs for averaging, if desired.
        // Or, record loss of the last epoch, or average loss.
        // For now, record loss from each epoch, differentiated by an epoch tag if multi-epoch recording is too verbose.
        // Simpler: record the final losses after all epochs for this update step.

        Tensor finalPolicyLoss = null;
        Tensor finalValueLoss = null;
        Tensor finalEntropy = null; // Average entropy over states for the last epoch

        for (int epoch = 0; epoch < config.hyperparams().epochs().intValue(); ++epoch) {
            // Value Function Update
            Tensor currentPredictedValues = this.valueFunction.apply(states);
            Tensor valueLoss = currentPredictedValues.loss(valueTargets.detach(), Tensor.Loss.MeanSquared);

            valueGradQueue.clear();
            valueLoss.minimize(valueGradQueue);
            valueGradQueue.optimize(valueOptimizer);
            finalValueLoss = valueLoss; // Keep track of the last one

            // Policy Function Update
            AgentUtils.GaussianDistribution dist = policy.getDistribution(
                states,
                config.actionConfig().sigmaMin().floatValue(),
                config.actionConfig().sigmaMax().floatValue()
            );
            Tensor newLogProbs = dist.logProb(actions);
            Tensor ratio = newLogProbs.sub(oldLogProbs.detach()).exp();

            Tensor detachedAdvantages = advantages.detach();

            Tensor surrogate1 = ratio.mul(detachedAdvantages);
            Tensor surrogate2 = ratio.clip(
                1.0f - config.hyperparams().ppoClip().floatValue(),
                1.0f + config.hyperparams().ppoClip().floatValue()
            ).mul(detachedAdvantages);

            Tensor policyLoss = Tensor.min(surrogate1, surrogate2).mean().neg(); // Negative because we minimize, but loss is typically positive.
            finalEntropy = dist.entropy().mean(); // Calculate entropy for this batch

            if (config.hyperparams().entropyBonus().floatValue() > 0) {
                policyLoss = policyLoss.sub(finalEntropy.mul(config.hyperparams().entropyBonus().floatValue()));
            }
            finalPolicyLoss = policyLoss; // Keep track of the last one


            policyGradQueue.clear();
            finalPolicyLoss.minimize(policyGradQueue);
            policyGradQueue.optimize(policyOptimizer);

            // Record sigma stats from the *last epoch's* distribution
            if (epoch == config.hyperparams().epochs().intValue() - 1) {
                Tensor sigmas = dist.stddev(); // Assuming stddev() gives the sigma tensor
                recordMetric("policy_sigma_mean", sigmas.mean().scalar());
                recordMetric("policy_sigma_std", sigmas.std(false).scalar()); // Population stddev of the sigmas
            }
        }

        // Record final losses and entropy after all epochs for this update step
        if (finalPolicyLoss != null) {
            // The policy loss is negative of the objective function (we maximize objective = minimize -objective)
            // So, record -finalPolicyLoss.scalar() if it was negated for minimization.
            // Current PPO policyLoss is already `Tensor.min(surrogate1, surrogate2).mean().neg();`
            // If entropy bonus was subtracted, it's part of this loss.
            // It's conventional to report positive loss values that are minimized.
            // If policyLoss is `L = - (objective - entropy_bonus)`, then we record `L`.
            // If it's `L = objective - entropy_bonus` and we maximize, then we record `-L`.
            // Given standard PPO, policyLoss is -(ClippedSurrogateObjective - c2*Entropy). We minimize this.
            // So, the raw value of policyLoss is what we record.
            recordMetric("policy_loss", finalPolicyLoss.scalar());
        }
        if (finalValueLoss != null) {
            recordMetric("value_loss", finalValueLoss.scalar());
        }
        if (finalEntropy != null && config.hyperparams().entropyBonus().floatValue() > 0) {
            recordMetric("entropy", finalEntropy.scalar());
        }


        incrementUpdateCount(); // This should be called once per `update` call.
                                // Metrics are recorded with the `updateCount` *before* this increment,
                                // representing the state for which this update was performed.
                                // The `recordMetric` in BasePolicyGradientAgent uses `this.updateCount`.
                                // So, metrics for update N are logged with step N. Then updateCount becomes N+1. This is correct.
    }

    // Constructor that takes PPOAgentConfig, stateDim, actionDim, and defaults MetricCollector to null
    public PPOAgent(PPOAgentConfig config, int stateDim, int actionDim) {
        this(config, stateDim, actionDim, null);
    }

    private record GAEResult(Tensor advantages, Tensor valueTargets) {}

    private GAEResult computeGAE(List<Experience2> batch, Tensor valuesAtTimesteps, double valueOfLastNextState) {
        int n = batch.size();
        double[] advantages = new double[n];
        double[] valueFunctionTargets = new double[n];

        double gaeAccumulator = 0; // This is A_hat_GAE for t+1 when calculating for t
        float gamma = config.hyperparams().gamma().floatValue();
        float lambda = config.hyperparams().lambda().floatValue();

        for (int t = n - 1; t >= 0; t--) {
            Experience2 exp = batch.get(t);
            double currentTimestepValue = valuesAtTimesteps.data(t,0); // V(s_t)

            // V(s_{t+1})
            double nextTimestepValue;
            if (exp.done()) {
                nextTimestepValue = 0; // Terminal state, V(s_{t+1}) is 0
            } else if (t == n - 1) {
                nextTimestepValue = valueOfLastNextState; // V(s_N) for last element from batch
            } else {
                nextTimestepValue = valuesAtTimesteps.data(t + 1, 0); // V(s_{t+1}) for non-last elements
            }

            double tdError = exp.reward() + gamma * nextTimestepValue - currentTimestepValue;
            gaeAccumulator = tdError + gamma * lambda * gaeAccumulator * (exp.done() ? 0.0 : 1.0); // if done, next GAE is 0
            advantages[t] = gaeAccumulator;
            valueFunctionTargets[t] = advantages[t] + currentTimestepValue; // R_t for value net = A_t + V(s_t)
        }
        return new GAEResult(Tensor.row(advantages).transpose(), Tensor.row(valueFunctionTargets).transpose());
    }

    @Override public Object getPolicy() { return this.policy; }
    @Override public Object getValueFunction() { return this.valueFunction; }
    @Override public Object getConfig() { return this.config; }



    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training);
        this.policy.train(training);
        this.valueFunction.train(training);
        if (!training) { // If switching to eval mode, clear any partial trajectory
            this.memory.clear();
        }
    }

    @Override public void clearMemory() { this.memory.clear(); }
}
