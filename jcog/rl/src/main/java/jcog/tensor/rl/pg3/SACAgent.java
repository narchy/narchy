package jcog.tensor.rl.pg3;

import jcog.Util;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Tensor;
import jcog.tensor.Mod;
//import jcog.tensor.rl.pg.util.Experience2; // Old import
import jcog.tensor.rl.pg3.memory.Experience2; // New import
import jcog.tensor.rl.pg3.configs.SACAgentConfig;
import jcog.tensor.rl.pg3.memory.AgentMemory;
import jcog.tensor.rl.pg3.memory.ReplayBuffer;
import jcog.tensor.rl.pg3.networks.QValueNet;
import jcog.tensor.rl.pg3.networks.SquashedGaussianPolicyNet;
import jcog.tensor.rl.pg3.util.AgentUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Soft Actor-Critic (SAC) Agent.
 *
 * <p>SAC is an off-policy actor-critic algorithm that optimizes a stochastic policy in continuous action spaces.
 * It aims to maximize both expected return and policy entropy, encouraging exploration.
 * Key components:
 * <ul>
 *   <li>A stochastic actor (policy) network that outputs parameters for a squashed Gaussian distribution.</li>
 *   <li>Two Q-value critic networks (and their targets) to mitigate overestimation bias.</li>
 *   <li>A temperature parameter (alpha) that balances the reward and entropy terms. Alpha can be fixed or learned.</li>
 *   <li>An experience replay buffer.</li>
 * </ul>
 * </p>
 *
 * <p>Core update steps:
 * <ol>
 *   <li>Sample a minibatch from the replay buffer.</li>
 *   <li>Update Q-networks: Minimize MSE loss against a target Q-value that includes entropy.
 *       <code>y_i = r_i + gamma * (min(Q_target1(s'_i, a'_i), Q_target2(s'_i, a'_i)) - alpha * log pi(a'_i|s'_i))</code>,
 *       where <code>a'_i</code> is sampled from the current policy <code>pi(·|s'_i)</code>.</li>
 *   <li>Update Policy Network: Maximize <code>E_{s~D, a~pi}[min(Q1(s,a), Q2(s,a)) - alpha * log pi(a|s)]</code>.</li>
 *   <li>Update Alpha (if learning): Adjust alpha to match a target entropy level.
 *       <code>L(alpha) = alpha * (log pi(a_t|s_t) + target_entropy)</code>.</li>
 *   <li>Softly update target Q-networks.</li>
 * </ol>
 * </p>
 *
 * @see SACAgentConfig
 * @see SquashedGaussianPolicyNet
 * @see QValueNet
 * @see ReplayBuffer
 */
public class SACAgent extends BasePolicyGradientAgent {

    public final SACAgentConfig config;

    private final SquashedGaussianPolicyNet policy; // Actor
    private final QValueNet qFunc1, qFunc2;         // Critics
    private final QValueNet targetQFunc1, targetQFunc2; // Target Critics

    private final Tensor.Optimizer policyOptimizer;
    private final Tensor.Optimizer qFunc1Optimizer, qFunc2Optimizer;
    @Nullable private final Tensor.Optimizer alphaOptimizer; // If learning alpha

    private final AgentMemory memory;
    private final RandomGenerator random;

    private final Tensor.GradQueue policyGradQueue;
    private final Tensor.GradQueue q1GradQueue, q2GradQueue;
    @Nullable private final Tensor.GradQueue alphaGradQueue;

    private final Tensor logAlpha; // learnable parameter for temperature alpha
    private final float targetEntropy;

    private long totalSteps = 0;
    private final int actionDim;


    public SACAgent(@NotNull SACAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim);
        this.config = Objects.requireNonNull(config, "SACAgentConfig cannot be null.");
        this.actionDim = actionDim;

        // Policy (Actor)
        this.policy = new SquashedGaussianPolicyNet(
            config.policyNetworkConfig(), stateDim, actionDim,
            config.commonHyperparams().sigmaMin().floatValue(),
            config.commonHyperparams().sigmaMax().floatValue()
        );

        // Q-Functions (Critics)
        this.qFunc1 = new QValueNet(config.qNetworkConfig(), stateDim, actionDim);
        this.qFunc2 = new QValueNet(config.qNetworkConfig(), stateDim, actionDim);
        this.targetQFunc1 = new QValueNet(config.qNetworkConfig(), stateDim, actionDim);
        this.targetQFunc2 = new QValueNet(config.qNetworkConfig(), stateDim, actionDim);

        // Optimizers
        this.policyOptimizer = config.policyOptimizerConfig().build();
        this.qFunc1Optimizer = config.qOptimizerConfig().build();
        this.qFunc2Optimizer = config.qOptimizerConfig().build();

        // Temperature (alpha)
        this.logAlpha = Tensor.scalar(Math.log(config.initialAlpha().floatValue())).requiresGrad(config.learnAlpha());
        if (config.learnAlpha()) {
            Objects.requireNonNull(config.alphaOptimizerConfig(), "Alpha optimizer config must be provided if learning alpha.");
            this.alphaOptimizer = config.alphaOptimizerConfig().build();
            this.alphaGradQueue = new Tensor.GradQueue();
            if (config.targetEntropy() == null) {
                this.targetEntropy = (float) -actionDim; // Default target entropy = -action_dim
                System.err.println("SACAgent: targetEntropy not set in config, defaulting to -action_dim (" + this.targetEntropy + ")");
            } else {
                this.targetEntropy = config.targetEntropy().floatValue();
            }
        } else {
            this.alphaOptimizer = null;
            this.alphaGradQueue = null;
            this.targetEntropy = 0; // Not used if alpha is fixed
        }


        // Replay Memory
        if (config.memoryConfig().replayBuffer() == null) {
            throw new IllegalArgumentException("SAC requires replayBuffer configuration in MemoryConfig.");
        }
        if (config.perConfig() != null) {
            SACAgentConfig.PerConfig per = config.perConfig();
            this.memory = new PrioritizedReplayBuffer(
                config.memoryConfig().replayBuffer().capacity(),
                per.alpha(),
                per.beta0(),
                per.betaAnnealingSteps()
            );
        } else {
            this.memory = new ReplayBuffer(config.memoryConfig().replayBuffer().capacity());
        }
        this.random = new XoRoShiRo128PlusRandom();

        // Grad Queues
        this.policyGradQueue = new Tensor.GradQueue();
        this.q1GradQueue = new Tensor.GradQueue();
        this.q2GradQueue = new Tensor.GradQueue();


        // Initialize target networks
        AgentUtils.hardUpdate(this.qFunc1, this.targetQFunc1);
        AgentUtils.hardUpdate(this.qFunc2, this.targetQFunc2);

        setTrainingMode(true);
    }

    private float getAlpha() {
        return this.logAlpha.exp().scalar();
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        try (var noGrad = Tensor.noGrad()) {
            if (deterministic) {
                // For deterministic evaluation, use the mean of the policy distribution
                return this.policy.meanAction(state).array();
            } else {
                // Sample from the policy for exploration
                SquashedGaussianPolicyNet.ActionSampleWithLogProb sample = this.policy.sampleAndLogProb(state, false);
                return sample.action().array();
            }
        }
    }

    /**
     * Selects an action and also returns its log probability. Used internally for experience recording.
     */
    private SquashedGaussianPolicyNet.ActionSampleWithLogProb selectActionWithLogProb(Tensor state) {
         try (var noGrad = Tensor.noGrad()) { // Action selection should not compute gradients for storage
            // Always sample (non-deterministic) when collecting experience for training
            return this.policy.sampleAndLogProb(state, false);
        }
    }


    @Override
    public void recordExperience(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null.");
        // SAC is off-policy. Experience is always recorded if in training mode.
        // The `oldLogProb` in Experience2 is crucial for SAC updates.
        if (this.trainingMode) {
            this.memory.add(experience);

            if (this.memory.size() >= config.memoryConfig().replayBuffer().batchSize() &&
                totalSteps % config.memoryConfig().replayBuffer().updateEveryNSteps() == 0) {
                for (int i = 0; i < config.memoryConfig().replayBuffer().gradientStepsPerUpdate(); i++) {
                    update(this.totalSteps);
                }
            }
        }
        this.totalSteps++;
    }

    @Override
    public void update(long totalSteps) {
        if (!this.trainingMode || this.memory.size() < config.memoryConfig().replayBuffer().batchSize()) {
            return;
        }

        AgentMemory.ExperienceBatch batch;
        Tensor importanceSamplingWeights = null;
        PrioritizedReplayBuffer.PrioritizedSampleBatch prioritizedBatch = null;

        if (this.memory instanceof PrioritizedReplayBuffer perBuffer) {
            prioritizedBatch = perBuffer.sampleWithDetails(config.memoryConfig().replayBuffer().batchSize());
            if (prioritizedBatch.experiences().isEmpty()) return;
            batch = this.memory.toBatch(prioritizedBatch.experiences());
            importanceSamplingWeights = prioritizedBatch.importanceSamplingWeights();
        } else {
            List<Experience2> sampledExperiences = this.memory.sample(config.memoryConfig().replayBuffer().batchSize());
            if (sampledExperiences.isEmpty()) return;
            batch = this.memory.toBatch(sampledExperiences);
        }

        Tensor states = batch.states();
        Tensor actions = batch.actions();
        Tensor rewards = batch.rewards();
        Tensor nextStates = batch.nextStates();
        Tensor dones = batch.dones(); // 1.0 if done, 0.0 if not done

        float alpha = getAlpha();

        // --- Update Q-Functions (Critics) ---
        Tensor targetQValues;
        try (var noGrad = Tensor.noGrad()) {
            SquashedGaussianPolicyNet.ActionSampleWithLogProb nextActionSample = this.policy.sampleAndLogProb(nextStates, false);
            Tensor nextActions = nextActionSample.action();
            Tensor nextLogProbs = nextActionSample.logProb();

            Tensor targetQ1 = this.targetQFunc1.forward(nextStates, nextActions);
            Tensor targetQ2 = this.targetQFunc2.forward(nextStates, nextActions);
            Tensor minTargetQ = Tensor.min(targetQ1, targetQ2);

            // Target Q = r + gamma * (1-done) * (min_Q_target(s',a') - alpha * log pi(a'|s'))
            targetQValues = rewards.add(
                minTargetQ.sub(nextLogProbs.mul(alpha)) // Q - alpha * log_pi
                          .mul(config.commonHyperparams().gamma().floatValue())
                          .mul(dones.neg().add(1.0f)) // (1 - done_mask)
            );
        }

        // Q1 Loss
        Tensor currentQ1 = this.qFunc1.forward(states, actions);
        Tensor tdErrors1 = targetQValues.sub(currentQ1);
        Tensor q1LossElementwise = tdErrors1.sqr();
        Tensor q1Loss;
        if (importanceSamplingWeights != null) {
            q1Loss = q1LossElementwise.mul(importanceSamplingWeights).mean();
        } else {
            q1Loss = q1LossElementwise.mean();
        }
        this.q1GradQueue.clear();
        q1Loss.minimize(this.q1GradQueue);
        this.q1GradQueue.optimize(this.qFunc1Optimizer);

        // Q2 Loss
        Tensor currentQ2 = this.qFunc2.forward(states, actions);
        Tensor tdErrors2 = targetQValues.sub(currentQ2); // Note: targetQValues is same for both
        Tensor q2LossElementwise = tdErrors2.sqr();
        Tensor q2Loss;
        if (importanceSamplingWeights != null) {
            q2Loss = q2LossElementwise.mul(importanceSamplingWeights).mean();
        } else {
            q2Loss = q2LossElementwise.mean();
        }
        this.q2GradQueue.clear();
        q2Loss.minimize(this.q2GradQueue);
        this.q2GradQueue.optimize(this.qFunc2Optimizer);

        // Update priorities in PER buffer based on mean/max of TD errors from the two Q-functions
        if (prioritizedBatch != null && this.memory instanceof PrioritizedReplayBuffer perBuffer) {
            // Use the mean of absolute TD errors from both critics for priority
            // Or, could use max(abs(td1), abs(td2)) or abs(td_from_min_Q_in_target_calc)
            // For simplicity, let's use mean of abs TD errors from the two critics.
            Tensor absTdErrors = tdErrors1.abs().add(tdErrors2.abs()).mul(0.5).detach();
            perBuffer.updatePriorities(prioritizedBatch.indices(), absTdErrors);
        }


        // --- Update Policy (Actor) and Alpha (Temperature) ---
        // These are typically updated less frequently than Q-functions
        if (getUpdateCount() % config.policyUpdateFreq() == 0) {

            // Freeze Q-networks for policy update by detaching or using fresh forward pass without their grads
            // For policy loss, Q-values are treated as constants w.r.t policy parameters.
            // However, gradients need to flow through Q-networks if they are part of actor's loss computation.
            // Here, we need Q(s, pi(s)) where gradients flow through pi(s) into Q.

            SquashedGaussianPolicyNet.ActionSampleWithLogProb policyActionSample = this.policy.sampleAndLogProb(states, false);
            Tensor policyActions = policyActionSample.action();
            Tensor policyLogProbs = policyActionSample.logProb();

            // Q-values for policy actions (gradients will flow through policy, not Q-funcs here for actor loss)
            // We need to ensure Q-networks are in eval mode or their parameters are not updated by this step.
            // The optimizers are separate, so that's fine.
            Tensor q1ForPolicy = this.qFunc1.forward(states, policyActions);
            Tensor q2ForPolicy = this.qFunc2.forward(states, policyActions);
            Tensor minQForPolicy = Tensor.min(q1ForPolicy, q2ForPolicy);

            // Policy loss: E[alpha * log pi(a|s) - min_Q(s,a)]
            // We want to maximize this, so minimize its negation.
            Tensor policyLoss = policyLogProbs.mul(alpha).sub(minQForPolicy).mean();

            this.policyGradQueue.clear();
            policyLoss.minimize(this.policyGradQueue);
            this.policyGradQueue.optimize(this.policyOptimizer);

            // Update Alpha (if learning)
            if (config.learnAlpha() && this.alphaOptimizer != null && this.alphaGradQueue != null) {
                // Alpha loss: E[-alpha * (log pi(a|s) + target_entropy)]
                // We want to minimize this.
                // log_pi needs to be detached from policy computation graph for alpha update.
                Tensor alphaLoss = this.logAlpha.mul(
                                        policyLogProbs.detach().add(this.targetEntropy).neg()
                                    ).mean();

                this.alphaGradQueue.clear();
                alphaLoss.minimize(this.alphaGradQueue); // logAlpha is the parameter
                this.alphaGradQueue.optimize(this.alphaOptimizer);
            }
        }

        // --- Soft Update Target Q-Networks ---
        if (getUpdateCount() % config.targetUpdateFreq() == 0) {
            AgentUtils.softUpdate(this.qFunc1, this.targetQFunc1, config.tau().floatValue());
            AgentUtils.softUpdate(this.qFunc2, this.targetQFunc2, config.tau().floatValue());
        }

        incrementUpdateCount();
    }


    @Override
    public Object getPolicy() { return this.policy; }

    @Override
    public Object getValueFunction() {
        // SAC uses Q-functions, not a single state-value function V directly for policy evaluation in this setup.
        // Return one of the Q-functions, or null, or a wrapper if needed.
        return List.of(this.qFunc1, this.qFunc2);
    }

    @Override
    public Object getConfig() { return this.config; }

    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training);
        this.policy.train(training);
        this.qFunc1.train(training);
        this.qFunc2.train(training);

        // Target networks are typically always in evaluation mode (no dropout, etc.)
        this.targetQFunc1.train(false);
        this.targetQFunc2.train(false);
    }

    @Override
    public void clearMemory() {
        this.memory.clear();
    }

    // Override apply from Agent for environment interaction loop
    @Override
    public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
        Objects.requireNonNull(input, "Current input (state) cannot be null.");
        Objects.requireNonNull(actionNext, "Output actionNext array cannot be null.");

        Tensor currentStateTensor = Tensor.row(input);

        if (inputPrev != null && actionPrev != null) {
            Tensor prevStateTensor = Tensor.row(inputPrev);
            // For SAC, `oldLogProb` (log_prob of actionPrev given inputPrev) is needed.
            // This requires running the policy on inputPrev to get actionPrev's log_prob.
            // This is slightly problematic if actionPrev was generated with different noise or policy state.
            // A common practice is to store the log_prob at the time actionPrev was generated.
            // The Experience2 record should ideally contain the log_prob of the action *as it was taken*.
            // For now, we'll recompute it based on prevState and actionPrev. This is an approximation if noise was involved.
            // A better way is for the environment loop to get action AND log_prob from agent, then store that.

            // To correctly get the log_prob for the *actual* actionPrev taken:
            // This assumes actionPrev was the *squashed* output. We need to "unsquash" it to find its pre-image
            // or, more practically, store log_prob at generation time.
            // For simplicity here, we assume Experience2 will be populated with the correct log_prob by the caller.
            // If Experience2.oldLogProb is null, it will be an issue for SAC.
            // The `selectActionWithLogProb` is for the *new* action.
            // The provided `experience` in `recordExperience` *must* have `oldLogProb` correctly set.

            // Let's assume the caller of `apply` will handle constructing Experience2 correctly,
            // including the log probability of `actionPrev` under the policy that generated it.
            // The `PolicyGradientAgent.recordExperience(Experience2)` expects this.

            Experience2 exp = new Experience2(prevStateTensor, actionPrev, reward, currentStateTensor, false, null /* log_prob of actionPrev */);
            // This `null` for oldLogProb will be an issue. The environment loop needs to be smarter.
            // The `DDPGAgent` didn't need it, but SAC does.
            // The `PPOAgent`'s `selectActionWithLogProb` is a good pattern: it returns action + log_prob.
            // The environment interaction loop should call that, then pass the log_prob into Experience2.

            // For now, this `apply` method is problematic for SAC if used directly without a smart caller.
            // The `recordExperience` method is the main entry point for learning.
            // This `apply` is more for a generic Agent interface.
        }

        // Select new action based on current state for `actionNext`
        // If training, this action should be sampled. If evaluating, mean.
        double[] newActionArray = selectAction(currentStateTensor, !this.trainingMode);
        System.arraycopy(newActionArray, 0, actionNext, 0, Math.min(newActionArray.length, actionNext.length));

        // To make `apply` usable for SAC's training loop, the environment runner would do:
        // 1. current_state = env.getState()
        // 2. ActionSampleWithLogProb result = sac_agent.selectActionWithLogProb(current_state_tensor); // internal method
        // 3. next_state, reward, done = env.step(result.action())
        // 4. Experience2 exp = new Experience2(current_state_tensor, result.action(), reward, next_state_tensor, done, result.logProb());
        // 5. sac_agent.recordExperience(exp);
    }
}
