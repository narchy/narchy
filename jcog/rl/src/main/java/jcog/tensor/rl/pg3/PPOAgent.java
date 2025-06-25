package jcog.tensor.rl.pg3;

import jcog.tensor.Tensor;
//import jcog.tensor.rl.pg.util.Experience2; // Old import
import jcog.tensor.rl.pg3.memory.Experience2; // New import
import jcog.tensor.rl.pg3.configs.PPOAgentConfig;
import jcog.tensor.rl.pg3.memory.AgentMemory;
import jcog.tensor.rl.pg3.memory.OnPolicyBuffer;
import jcog.tensor.rl.pg3.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg3.networks.ValueNet;
import jcog.tensor.rl.pg3.util.AgentUtils;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class PPOAgent extends BasePolicyGradientAgent {

    public final PPOAgentConfig config;
    public final GaussianPolicyNet policy;
    public final ValueNet valueFunction;
    public final Tensor.Optimizer policyOptimizer;
    public final Tensor.Optimizer valueOptimizer;
    public final AgentMemory memory;

    private final Tensor.GradQueue policyGradQueue;
    private final Tensor.GradQueue valueGradQueue;
    @Nullable private final RunningObservationNormalizer obsNormalizer;

    public PPOAgent(PPOAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim);
        Objects.requireNonNull(config, "Agent configuration cannot be null");
        this.config = config;

        this.policy = new GaussianPolicyNet(config.policyNetworkConfig(), stateDim, actionDim);
        this.valueFunction = new ValueNet(config.valueNetworkConfig(), stateDim);

        this.policyOptimizer = config.policyNetworkConfig().optimizer().build();
        this.valueOptimizer = config.valueNetworkConfig().optimizer().build();

        this.memory = new OnPolicyBuffer(config.memoryConfig().episodeLength().intValue());

        this.policyGradQueue = new Tensor.GradQueue();
        this.valueGradQueue = new Tensor.GradQueue();

        if (config.obsNormConfig() != null && config.obsNormConfig().enabled()) {
            this.obsNormalizer = new RunningObservationNormalizer(stateDim, config.obsNormConfig().history());
        } else {
            this.obsNormalizer = null;
        }

        setTrainingMode(true); // Initialize training mode by default
    }

    private Tensor normalizeState(Tensor state) {
        if (obsNormalizer != null && this.trainingMode) { // Only normalize during training updates for PPO
            return obsNormalizer.normalize(state);
        }
        // If not training or normalizer disabled, pass through.
        // For action selection during eval, typically use fixed mean/std or no normalization.
        // PPO updates use batches, so normalization happens on batch.
        // For selectAction, if needed, it implies stateful normalization which is tricky.
        // The current RunningObservationNormalizer updates stats.
        // A separate eval-mode normalization (using fixed stats) would be needed if applied at selectAction.
        // For PPO, it's common to normalize the batch of states before feeding to networks in update().
        return state;
    }


    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        // This method is required by the interface. For PPO, it's often better to also get the logProb.
        // Callers can use selectActionWithLogProb if they need it for Experience2.
        // Normalization for selectAction: if obsNormalizer is active, use it.
        // Note: this updates normalizer stats. For pure eval, might want fixed stats.
        Tensor processedState = (obsNormalizer != null) ? obsNormalizer.normalize(state.copy()) : state;
        ActionWithLogProb result = selectActionWithLogProbInternal(processedState, deterministic);
        return result.action();
    }

    public record ActionWithLogProb(double[] action, Tensor logProb) {}

    // Renamed to selectActionWithLogProbInternal to avoid clash if we make the public one normalize
    public ActionWithLogProb selectActionWithLogProbInternal(Tensor state, boolean deterministic) {
        try (var noGrad = Tensor.noGrad()) {
            AgentUtils.GaussianDistribution dist = policy.getDistribution(
                state, // Already processed state
                config.actionConfig().sigmaMin().floatValue(),
                config.actionConfig().sigmaMax().floatValue()
            );
            Tensor actionTensor = dist.sample(deterministic);
            Tensor logProb = dist.logProb(actionTensor.detach());
            return new ActionWithLogProb(actionTensor.clipUnitPolar().array(), logProb.detach());
        }
    }

    // Public method for environment interaction loop to get action + logProb
    public ActionWithLogProb selectActionAndLogProb(Tensor state, boolean deterministic) {
        Tensor processedState = (obsNormalizer != null) ? obsNormalizer.normalize(state.copy()) : state;
        return selectActionWithLogProbInternal(processedState, deterministic);
    }


    @Override
    public void recordExperience(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience object cannot be null.");
        Objects.requireNonNull(experience.oldLogProb(), "Experience for PPO agent must include oldLogProb.");
         if (!this.trainingMode) {
            return; // Do not record or update if not in training mode
        }
        // State in experience should be the raw state. Normalization happens during update or before selectAction.
        this.memory.add(experience);

        boolean bufferFull = this.memory.size() >= config.memoryConfig().episodeLength().intValue();

        if (bufferFull) {
            update(0);
            this.memory.clear();
        }
    }

    @Override
    public void update(long totalSteps) {
        if (!this.trainingMode || this.memory.size() == 0) {
            return;
        }

        List<Experience2> batchExperiences = this.memory.getAll();

        // Normalize states from the batch if normalizer is active
        Tensor states;
        if (obsNormalizer != null) {
            List<Tensor> normalizedStatesList = batchExperiences.stream()
                .map(exp -> obsNormalizer.normalize(exp.state().copy())) // Normalize each state
                .collect(Collectors.toList());
            states = Tensor.concatRows(normalizedStatesList);
        } else {
            states = Tensor.concatRows(batchExperiences.stream().map(Experience2::state).collect(Collectors.toList()));
        }

        Tensor actions = Tensor.concatRows(batchExperiences.stream().map(e -> Tensor.row(e.action())).collect(Collectors.toList()));
        Tensor oldLogProbs = Tensor.concatRows(batchExperiences.stream().map(Experience2::oldLogProb).collect(Collectors.toList()));

        Tensor advantages;
        Tensor valueTargets;

        // Value function and GAE calculations use potentially normalized states
        try (var noGrad = Tensor.noGrad()) {
            Tensor values = this.valueFunction.apply(states); // V(s_norm)

            Tensor lastNextValue;
            Experience2 lastExp = batchExperiences.get(batchExperiences.size() - 1);
            if (!lastExp.done() && lastExp.nextState() != null) {
                 // Normalize next_state for the last transition if normalizer is active
                 Tensor nextStateNormalized = (obsNormalizer != null) ? obsNormalizer.normalize(lastExp.nextState().copy()) : lastExp.nextState();
                 lastNextValue = this.valueFunction.apply(nextStateNormalized);
            } else {
                lastNextValue = Tensor.zeros(1,1);
            }

            // GAE computation needs raw rewards, but values are from normalized states.
            // This is standard: rewards are not normalized, values are based on normalized state representations.
            GAEResult gaeResult = computeGAE(batchExperiences, values, lastNextValue.scalar());
            advantages = gaeResult.advantages();
            valueTargets = gaeResult.valueTargets(); // These are targets for V(s_norm)
        }

        if (config.hyperparams().normalizeAdvantages()) {
            double[] advantagesArray = advantages.array().clone();
            AgentUtils.normalize(advantagesArray);
            advantages = Tensor.row(advantagesArray).transpose();
        }

        for (int i = 0; i < config.hyperparams().epochs().intValue(); ++i) {
            Tensor currentPredictedValues = this.valueFunction.apply(states); // Re-evaluate V(s) for current value net params
            Tensor valueLoss = currentPredictedValues.loss(valueTargets.detach(), Tensor.Loss.MeanSquared);

            valueGradQueue.clear();
            valueLoss.minimize(valueGradQueue);
            valueGradQueue.optimize(valueOptimizer);

            AgentUtils.GaussianDistribution dist = policy.getDistribution(
                states,
                config.actionConfig().sigmaMin().floatValue(),
                config.actionConfig().sigmaMax().floatValue()
            );
            Tensor newLogProbs = dist.logProb(actions); // log pi_new(a|s)
            Tensor ratio = newLogProbs.sub(oldLogProbs.detach()).exp(); // (pi_new / pi_old)

            Tensor detachedAdvantages = advantages.detach(); // Advantages don't depend on policy params being optimized here

            Tensor surrogate1 = ratio.mul(detachedAdvantages);
            Tensor surrogate2 = ratio.clip(
                1.0f - config.hyperparams().ppoClip().floatValue(),
                1.0f + config.hyperparams().ppoClip().floatValue()
            ).mul(detachedAdvantages);

            Tensor policyLoss = Tensor.min(surrogate1, surrogate2).mean().neg();

            if (config.hyperparams().entropyBonus().floatValue() > 0) {
                Tensor entropy = dist.entropy().mean();
                policyLoss = policyLoss.sub(entropy.mul(config.hyperparams().entropyBonus().floatValue()));
            }

            policyGradQueue.clear();
            policyLoss.minimize(policyGradQueue);
            policyGradQueue.optimize(policyOptimizer);
        }
        incrementUpdateCount();
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
