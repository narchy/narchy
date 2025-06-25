package jcog.tensor.rl.pg3;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg3.configs.SACAgentConfig;
import jcog.tensor.rl.pg3.memory.ReplayBuffer;
import jcog.tensor.rl.pg3.networks.CriticNet;
import jcog.tensor.rl.pg3.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg3.util.AgentUtils;

import java.util.List;
import java.util.Objects;

public class SACAgent extends BasePolicyGradientAgent {

    public final SACAgentConfig config;

    public final GaussianPolicyNet policy; // Actor
    public final CriticNet qNet1;          // Critic 1
    public final CriticNet qNet2;          // Critic 2
    public final CriticNet targetQNet1;    // Target Critic 1
    public final CriticNet targetQNet2;    // Target Critic 2

    public final Tensor.Optimizer policyOptimizer;
    public final Tensor.Optimizer q1Optimizer;
    public final Tensor.Optimizer q2Optimizer;
    public final Tensor.Optimizer alphaOptimizer; // For temperature auto-tuning

    public final ReplayBuffer memory;

    private final Tensor logAlpha; // Learnable temperature parameter
    private final float targetEntropy;

    private final Tensor.GradQueue policyGradQueue;
    private final Tensor.GradQueue q1GradQueue;
    private final Tensor.GradQueue q2GradQueue;
    private final Tensor.GradQueue alphaGradQueue;
    private final int actionDim;

    public SACAgent(SACAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim);
        Objects.requireNonNull(config, "Agent configuration cannot be null");
        this.config = config;
        this.actionDim = actionDim;

        // Initialize networks
        this.policy = new GaussianPolicyNet(config.policyNetworkConfig(), stateDim, actionDim);
        this.qNet1 = new CriticNet(config.qNetworkConfig(), stateDim, actionDim);
        this.qNet2 = new CriticNet(config.qNetworkConfig(), stateDim, actionDim);
        this.targetQNet1 = new CriticNet(config.qNetworkConfig(), stateDim, actionDim);
        this.targetQNet2 = new CriticNet(config.qNetworkConfig(), stateDim, actionDim);

        // Initialize optimizers
        this.policyOptimizer = config.policyOptimizerConfig().build(this.policy.getWeights());
        this.q1Optimizer = config.qOptimizerConfig().build(this.qNet1.getWeights());
        this.q2Optimizer = config.qOptimizerConfig().build(this.qNet2.getWeights());


        // Temperature (alpha)
        if (config.sacHyperparams().useFixedAlpha()) {
            this.logAlpha = Tensor.scalar(Math.log(config.sacHyperparams().initialAlpha())).constant();
            this.alphaOptimizer = null; // No optimizer needed for fixed alpha
            this.alphaGradQueue = null;
        } else {
            this.logAlpha = Tensor.scalar(Math.log(config.sacHyperparams().initialAlpha())).requiresGrad();
            this.alphaOptimizer = config.alphaOptimizerConfig().build(List.of(this.logAlpha));
            this.alphaGradQueue = new Tensor.GradQueue(List.of(this.logAlpha));
        }

        this.targetEntropy = config.sacHyperparams().targetEntropy() != null ?
            config.sacHyperparams().targetEntropy() :
            (float) -actionDim; // Default target entropy: -dim(A)

        // Initialize memory
        if (config.memoryConfig().replayBufferConfig() == null) {
            throw new IllegalArgumentException("SAC requires ReplayBufferConfig in MemoryConfig");
        }
        this.memory = new ReplayBuffer(config.memoryConfig().replayBufferConfig().capacity());

        // Copy initial weights to target networks
        Util.softUpdate(this.targetQNet1.getWeights(), this.qNet1.getWeights(), 1.0f); // Hard copy
        Util.softUpdate(this.targetQNet2.getWeights(), this.qNet2.getWeights(), 1.0f); // Hard copy

        this.policyGradQueue = new Tensor.GradQueue(this.policy.getWeights());
        this.q1GradQueue = new Tensor.GradQueue(this.qNet1.getWeights());
        this.q2GradQueue = new Tensor.GradQueue(this.qNet2.getWeights());


        setTrainingMode(true);
    }

    /**
     * Selects an action based on the current policy.
     * For SAC, actions are sampled from a Gaussian distribution and then squashed by a tanh function.
     * The log probability also needs to account for this squashing.
     *
     * @param state The current state.
     * @param deterministic If true, use the mean of the policy distribution (for evaluation).
     *                      If false, sample from the distribution (for training).
     * @return double array representing the action.
     */
    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        try (var noGrad = Tensor.noGrad()) {
            AgentUtils.GaussianDistribution dist = this.policy.getDistribution(
                state,
                config.policyNetworkConfig().sigmaMin(), // Use sigmaMin from policy config
                config.policyNetworkConfig().sigmaMax()  // Use sigmaMax from policy config
            );
            Tensor rawAction = dist.sample(deterministic);
            Tensor squashedAction = rawAction.tanh(); // Squash action to [-1, 1]
            return squashedAction.array(); // Assumes (1, actionDim)
        }
    }

    /**
     * Samples an action and also returns its log probability, accounting for the tanh squashing.
     * This is needed for the policy update.
     * log_prob(squashed_action) = log_prob(raw_action) - sum(log(1 - tanh(raw_action)^2 + epsilon))
     */
    private ActionWithLogProb sampleActionWithLogProb(Tensor state) {
        AgentUtils.GaussianDistribution dist = this.policy.getDistribution(
            state,
            config.policyNetworkConfig().sigmaMin(),
            config.policyNetworkConfig().sigmaMax()
        );
        Tensor rawAction = dist.sample(false); // Always sample for log_prob calculation during training
        Tensor logProbRaw = dist.logProb(rawAction);

        Tensor squashedAction = rawAction.tanh();

        // Correction for tanh squashing: log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        // Simpler: sum across action dimensions: log(1 - action_tanh^2). Add epsilon for stability.
        // Each component of log_prob_squashed = log_prob_raw_component - log(1 - tanh(raw_action_component)^2)
        // Summing log_prob_raw means it's already joint. Sum the correction term.
        Tensor logProbSquashed = logProbRaw.sub(
            rawAction.tanh().sqr().mul(-1).add(1).log().sum(-1, true) // sum over action dim
        );
        // Ensure logProbSquashed has shape (batch_size, 1) if logProbRaw was already summed.
        // If dist.logProb returns (batch_size, action_dim), sum it first.
        // GaussianPolicyNet.getDistribution().logProb() returns per-sample log_prob, (batch_size, 1)

        return new ActionWithLogProb(squashedAction, logProbSquashed);
    }

    // Helper record for return value of sampleActionWithLogProb
    // private record ActionWithLogProb(Tensor action, Tensor logProb) {} // Now in BasePolicyGradientAgent


    @Override
    protected ActionWithLogProb selectActionWithLogProb(Tensor state, boolean deterministic) {
        if (deterministic) {
            // For deterministic SAC action (e.g., evaluation), typically use the mean of the Gaussian
            // and don't apply the tanh squashing's log_prob correction, or ensure log_prob isn't used.
            // The `selectAction` method already handles deterministic mean.
            // Log prob might not be meaningful or needed for deterministic eval path in `apply`.
            try (var noGrad = Tensor.noGrad()) {
                AgentUtils.GaussianDistribution dist = this.policy.getDistribution(
                    state,
                    config.policyNetworkConfig().sigmaMin(),
                    config.policyNetworkConfig().sigmaMax()
                );
                Tensor rawAction = dist.getMean(); // Use mean for deterministic
                Tensor squashedAction = rawAction.tanh();
                // LogProb is tricky for deterministic mean if squashed.
                // For 'apply' caching, if it's eval mode, logProb might not be stored/used anyway.
                // Returning null for logProb in deterministic case seems safest if it's not well-defined for this path.
                return new ActionWithLogProb(squashedAction.array(), null);
            }
        } else {
            // Stochastic action with log_prob for training
            AgentUtils.GaussianDistribution dist = this.policy.getDistribution(
                state,
                config.policyNetworkConfig().sigmaMin(),
                config.policyNetworkConfig().sigmaMax()
            );
            Tensor rawAction = dist.sample(false); // Sample for stochastic
            Tensor logProbRaw = dist.logProb(rawAction);

            Tensor squashedAction = rawAction.tanh();
            Tensor logProbSquashed = logProbRaw.sub(
                rawAction.tanh().sqr().mul(-1).add(1).log().sum(-1, true)
            );
            return new ActionWithLogProb(squashedAction.array(), logProbSquashed);
        }
    }


    @Override
    public void recordExperience(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null");
        this.memory.add(experience);
        // Updates are typically driven by having enough samples in the replay buffer.
    }

    @Override
    public void update(long totalSteps) {
        if (!this.trainingMode || this.memory.size() < config.memoryConfig().replayBufferConfig().batchSize()) {
            return;
        }

        List<Experience2> batch = this.memory.sample(config.memoryConfig().replayBufferConfig().batchSize());
        if (batch.isEmpty()) {
            return;
        }

        ReplayBuffer.BatchTuple tensors = ReplayBuffer.experiencesToBatchTuple(batch);
        Tensor states = tensors.states();
        Tensor actions = tensors.actions(); // These are actions taken in env, already squashed.
        Tensor rewards = tensors.rewards();
        Tensor nextStates = tensors.nextStates();
        Tensor dones = tensors.dones();     // 1.0 if done, 0.0 if not

        float alpha = this.logAlpha.exp().scalar(); // Current temperature

        // --- Update Q-functions (Critics) ---
        Tensor targetQValues;
        try (var noGrad = Tensor.noGrad()) {
            ActionWithLogProb nextActionWithLogProb = sampleActionWithLogProb(nextStates);
            Tensor nextActionsSquashed = nextActionWithLogProb.action();
            Tensor logProbNextActions = nextActionWithLogProb.logProb();

            // Target Q = min(Q_target1(s',a'), Q_target2(s',a')) - alpha * log_pi(a'|s')
            Tensor targetQ1 = this.targetQNet1.apply(new Tensor[]{nextStates, nextActionsSquashed});
            Tensor targetQ2 = this.targetQNet2.apply(new Tensor[]{nextStates, nextActionsSquashed});
            Tensor minTargetQ = Tensor.min(targetQ1, targetQ2);

            targetQValues = minTargetQ.sub(logProbNextActions.mul(alpha));

            Tensor y_i = rewards.add(
                dones.mul(-1).add(1) // (1 - dones)
                    .mul(config.hyperparams().gamma())
                    .mul(targetQValues)
            ); // y_i = r + gamma * (1-done) * (min_Q_target(s',a') - alpha * log_pi(a'|s'))

            // --- Q1 Loss ---
            q1GradQueue.clear();
            Tensor currentQ1 = this.qNet1.apply(new Tensor[]{states, actions});
            Tensor q1Loss = currentQ1.loss(y_i.detach(), Tensor.Loss.MeanSquared);
            q1Loss.minimize(q1GradQueue);
            q1GradQueue.optimize(q1Optimizer);

            // --- Q2 Loss ---
            q2GradQueue.clear();
            Tensor currentQ2 = this.qNet2.apply(new Tensor[]{states, actions});
            Tensor q2Loss = currentQ2.loss(y_i.detach(), Tensor.Loss.MeanSquared);
            q2Loss.minimize(q2GradQueue);
            q2GradQueue.optimize(q2Optimizer);
        }


        // --- Update Policy (Actor) ---
        policyGradQueue.clear();
        ActionWithLogProb currentActionWithLogProb = sampleActionWithLogProb(states);
        Tensor currentActionsSquashed = currentActionWithLogProb.action();
        Tensor logProbCurrentActions = currentActionWithLogProb.logProb();

        Tensor q1ForPolicyUpdate = this.qNet1.apply(new Tensor[]{states, currentActionsSquashed});
        Tensor q2ForPolicyUpdate = this.qNet2.apply(new Tensor[]{states, currentActionsSquashed});
        Tensor minQForPolicyUpdate = Tensor.min(q1ForPolicyUpdate, q2ForPolicyUpdate);

        // Policy loss = alpha * log_pi(a|s) - min_Q(s,a)
        Tensor policyLoss = logProbCurrentActions.mul(alpha).sub(minQForPolicyUpdate).mean();
        policyLoss.minimize(policyGradQueue);
        policyGradQueue.optimize(policyOptimizer);


        // --- Update Alpha (Temperature) ---
        if (config.sacHyperparams().automaticAlphaTuning() && !config.sacHyperparams().useFixedAlpha() && alphaOptimizer != null) {
            alphaGradQueue.clear();
            // Alpha loss = -log_alpha * (log_pi(a|s) + target_entropy).detach()
            // Or, if optimizing log_alpha directly: loss = log_alpha * (-log_pi(a|s) - target_entropy).detach()
            Tensor alphaLoss = this.logAlpha.mul(
                    logProbCurrentActions.add(this.targetEntropy).detach().neg()
            ).mean();
            alphaLoss.minimize(alphaGradQueue);
            alphaGradQueue.optimize(alphaOptimizer);
        }

        // --- Soft update target Q-networks ---
        if (getUpdateCount() % config.sacHyperparams().targetUpdateInterval() == 0) {
            Util.softUpdate(this.targetQNet1.getWeights(), this.qNet1.getWeights(), config.sacHyperparams().tau());
            Util.softUpdate(this.targetQNet2.getWeights(), this.qNet2.getWeights(), config.sacHyperparams().tau());
        }

        incrementUpdateCount();
    }


    @Override
    public Object getPolicy() {
        return this.policy;
    }

    @Override
    public Object getValueFunction() {
        // SAC has Q-functions, not a single state-value function like VPG/PPO.
        // Return one of them, or a structure containing both. For now, QNet1.
        return this.qNet1;
    }

    public CriticNet getQNet1() { return this.qNet1; }
    public CriticNet getQNet2() { return this.qNet2; }
    public Tensor getLogAlpha() { return this.logAlpha; }


    @Override
    public Object getConfig() {
        return this.config;
    }

    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training);
        this.policy.train(training);
        this.qNet1.train(training);
        this.qNet2.train(training);
        // Target networks are for stable targets, usually not in training mode for dropout/batchnorm effects
        this.targetQNet1.train(false);
        this.targetQNet2.train(false);
    }

    @Override
    public void clearMemory() {
        this.memory.clear();
    }
}
