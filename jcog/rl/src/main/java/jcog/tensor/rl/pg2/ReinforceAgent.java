package jcog.tensor.rl.pg2;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg2.configs.ReinforceAgentConfig;
import jcog.tensor.rl.pg2.memory.OnPolicyBuffer;
import jcog.tensor.rl.pg2.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg2.util.AgentUtils;

import java.util.List;
import java.util.Objects;

public class ReinforceAgent extends AbstractPolicyGradientAgent {

    public final ReinforceAgentConfig config;
    public final GaussianPolicyNet policy;
    public final Tensor.Optimizer policyOptimizer;

    private final Tensor.GradQueue policyGradQueue;

    public ReinforceAgent(ReinforceAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim, new OnPolicyBuffer(config.memoryConfig().episodeLength().intValue()), null);
        Objects.requireNonNull(config, "Agent configuration cannot be null");
        this.config = config;

        this.policy = new GaussianPolicyNet(config.policyNetworkConfig(), stateDim, actionDim);
        this.policyOptimizer = config.policyNetworkConfig().optimizer().build();

        this.policyGradQueue = new Tensor.GradQueue();

        setTrainingMode(true); // Initialize training mode by default
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        // Policy is set to eval mode (train(false)) if agent is not in training mode by setTrainingMode method.
        // No need to toggle here unless specific per-action logic is needed outside agent's global training state.
        try (var noGrad = Tensor.noGrad()) { // Ensure no gradients are computed during action selection
            AgentUtils.GaussianDistribution dist = policy.getDistribution(
                state,
                config.actionConfig().sigmaMin().floatValue(),
                config.actionConfig().sigmaMax().floatValue()
            );
            Tensor action = dist.sample(deterministic);
            // Assuming actions should be clipped to a standard range like [-1, 1] for continuous control.
            return action.clipUnitPolar().array();
        }
    }

    @Override
    public void update(long totalSteps) {
        if (!this.trainingMode || this.memory.size() == 0) {
            return;
        }

        List<Experience2> episode = this.memory.getAll(); // Gets a copy

        Tensor returns = computeReturns(episode);

        List<Tensor> statesList = episode.stream().map(Experience2::state).toList();
        List<Tensor> actionsList = episode.stream().map(e -> Tensor.row(e.action())).toList();

        // It's generally more efficient to batch forward passes if possible.
        Tensor statesBatch = Tensor.concatRows(statesList);
        Tensor actionsBatch = Tensor.concatRows(actionsList);

        AgentUtils.GaussianDistribution dist = policy.getDistribution(
            statesBatch,
            config.actionConfig().sigmaMin().floatValue(),
            config.actionConfig().sigmaMax().floatValue()
        );
        Tensor logProbs = dist.logProb(actionsBatch);

        // Ensure returns tensor is compatible for element-wise multiplication (e.g., same shape or broadcastable)
        // logProbs: (batch_size, 1), returns: (batch_size, 1)
        Tensor policyLoss = logProbs.mul(returns).mean().neg();

        if (config.hyperparams().entropyBonus().floatValue() > 0) {
            Tensor entropy = dist.entropy().mean(); // entropy is per-timestep, so mean over batch
            policyLoss = policyLoss.sub(entropy.mul(config.hyperparams().entropyBonus().floatValue()));
        }

        policyGradQueue.clear(); // Clear any old gradients
        policyLoss.minimize(policyGradQueue);
        policyGradQueue.optimize(policyOptimizer);

        incrementUpdateCount();
        // Memory is cleared in recordExperience after 'done' for REINFORCE.
    }

    private Tensor computeReturns(List<Experience2> episode) {
        int n = episode.size();
        double[] G = new double[n];
        double currentReturn = 0;

        for (int t = n - 1; t >= 0; t--) {
            currentReturn = episode.get(t).reward() + config.hyperparams().gamma().floatValue() * currentReturn;
            G[t] = currentReturn;
        }

        Tensor returnsTensor = Tensor.row(G).transpose();

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
        return null;
    }

    @Override
    public Object getConfig() {
        return this.config;
    }

    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training);
        this.policy.train(training);
        if (!training) { // If switching to eval mode, good practice to clear any partial trajectory
            this.memory.clear();
        }
    }

    @Override
    public void clearMemory() {
        this.memory.clear();
    }
}
