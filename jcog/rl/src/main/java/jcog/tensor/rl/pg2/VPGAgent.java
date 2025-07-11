package jcog.tensor.rl.pg2;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg2.configs.VPGAgentConfig;
import jcog.tensor.rl.pg2.memory.OnPolicyBuffer;
import jcog.tensor.rl.pg2.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg2.networks.ValueNet;
import jcog.tensor.rl.pg2.util.AgentUtils;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class VPGAgent extends AbstractPolicyGradientAgent {

    public final VPGAgentConfig config;
    public final GaussianPolicyNet policy;
    public final ValueNet valueFunction;
    public final Tensor.Optimizer policyOptimizer;
    public final Tensor.Optimizer valueOptimizer;

    private final Tensor.GradQueue policyGradQueue;
    private final Tensor.GradQueue valueGradQueue;

    public VPGAgent(VPGAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim, new OnPolicyBuffer(config.memoryConfig().episodeLength().intValue()), null);
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
        try (var noGrad = Tensor.noGrad()) {
            AgentUtils.GaussianDistribution dist = policy.getDistribution(
                state,
                config.actionConfig().sigmaMin().floatValue(),
                config.actionConfig().sigmaMax().floatValue()
            );
            Tensor action = dist.sample(deterministic);
            return action.clipUnitPolar().array();
        }
    }


    @Override
    public void update(long totalSteps) {
        if (!this.trainingMode || this.memory.size() == 0) {
            return;
        }

        List<Experience2> episode = this.memory.getAll(); // Get a copy

        Tensor returns = computeReturns(episode);

        List<Tensor> statesList = episode.stream().map(Experience2::state).filter(Objects::nonNull).toList();
        if (statesList.isEmpty())
            return;
        if (statesList.size()!=returns.volume())
            return; //?

        Tensor statesBatch = Tensor.concatRows(statesList);

        // 1. Update Value Function
        Tensor predictedValues = this.valueFunction.apply(statesBatch);
        Tensor valueLoss = predictedValues.loss(returns.detach(), Tensor.Loss.MeanSquared);

        valueGradQueue.clear();
        valueLoss.minimize(valueGradQueue);
        valueGradQueue.optimize(valueOptimizer);

        // 2. Calculate Advantages (A(s,a) = R_t - V(s_t))
        Tensor advantages = returns.sub(predictedValues.detach());
        if (config.hyperparams().normalizeAdvantages()) {
            double[] advantagesArray = advantages.array().clone(); // Clone to normalize a copy
            AgentUtils.normalize(advantagesArray);
            advantages = Tensor.row(advantagesArray).transpose();
        }

        // 3. Update Policy
        List<Tensor> actionsList = episode.stream().map(e -> Tensor.row(e.action())).collect(Collectors.toList());
        Tensor actionsBatch = Tensor.concatRows(actionsList);

        AgentUtils.GaussianDistribution dist = policy.getDistribution(
            statesBatch,
            config.actionConfig().sigmaMin().floatValue(),
            config.actionConfig().sigmaMax().floatValue()
        );
        Tensor logProbs = dist.logProb(actionsBatch);

        Tensor policyLoss = logProbs.mul(advantages).mean().neg();

        if (config.hyperparams().entropyBonus().floatValue() > 0) {
            Tensor entropy = dist.entropy().mean();
            policyLoss = policyLoss.sub(entropy.mul(config.hyperparams().entropyBonus().floatValue()));
        }

        policyGradQueue.clear();
        policyLoss.minimize(policyGradQueue);
        policyGradQueue.optimize(policyOptimizer);

        incrementUpdateCount();
        // Memory is cleared in recordExperience after 'done' for VPG.
    }

    private Tensor computeReturns(List<Experience2> episode) {
        int n = episode.size();
        double[] G = new double[n];
        double currentReturn = 0; // This is R_t or G_t in equations

        // Iterate backwards through the episode to calculate discounted returns
        for (int t = n - 1; t >= 0; t--) {
            currentReturn = episode.get(t).reward() + config.hyperparams().gamma().floatValue() * currentReturn;
            G[t] = currentReturn;
        }

        Tensor returnsTensor = Tensor.row(G).transpose(); // Shape [n, 1]

        // Optional normalization of returns (targets for value function)
        // This is less common for VPG targets than normalizing advantages, but supported by config.
        if (config.hyperparams().normalizeReturns()) {
             double[] returnsToNormalize = returnsTensor.array().clone();
             AgentUtils.normalize(returnsToNormalize);
             return Tensor.row(returnsToNormalize).transpose();
        } else {
            return returnsTensor;
        }
    }

    @Override
    public Object getPolicy() {
        return this.policy;
    }

    @Override
    public Object getValueFunction() {
        return this.valueFunction;
    }

    @Override
    public Object getConfig() {
        return this.config;
    }

    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training);
        this.policy.train(training);
        this.valueFunction.train(training);
        if (!training) { // If switching to eval mode, clear any partial trajectory
            this.memory.clear();
        }
    }

    @Override
    public void clearMemory() {
        this.memory.clear();
    }
}
