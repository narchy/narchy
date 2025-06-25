package jcog.tensor.rl.pg3;

import jcog.Util;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Tensor;
//import jcog.tensor.rl.pg.util.Experience2; // Old import
import jcog.tensor.rl.pg3.memory.Experience2; // New import
import jcog.tensor.rl.pg3.configs.DDPGAgentConfig;
import jcog.tensor.rl.pg3.memory.AgentMemory;
import jcog.tensor.rl.pg3.memory.PrioritizedReplayBuffer;
import jcog.tensor.rl.pg3.memory.ReplayBuffer;
import jcog.tensor.rl.pg3.networks.DeterministicPolicyNet;
import jcog.tensor.rl.pg3.networks.QValueNet;
import jcog.tensor.rl.pg3.util.AgentUtils;
import jcog.tensor.rl.pg3.util.Noise;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Deep Deterministic Policy Gradient (DDPG) Agent.
 *
 * <p>DDPG is an off-policy actor-critic algorithm designed for continuous action spaces.
 * It utilizes:
 * <ul>
 *   <li>An actor network that deterministically maps states to actions.</li>
 *   <li>A critic network that learns the Q-value (action-value function).</li>
 *   <li>Target networks for both actor and critic to stabilize learning.</li>
 *   <li>An experience replay buffer to store and sample transitions.</li>
 *   <li>Action noise (e.g., Ornstein-Uhlenbeck process or Gaussian noise) for exploration.</li>
 * </ul>
 * </p>
 *
 * <p>Key steps in DDPG:
 * <ol>
 *   <li>Initialize actor, critic, and their target networks.</li>
 *   <li>Initialize replay buffer.</li>
 *   <li>For each timestep:
 *     <ol>
 *       <li>Select action based on actor network plus exploration noise.</li>
 *       <li>Execute action, observe reward and next state.</li>
 *       <li>Store transition in replay buffer.</li>
 *       <li>Sample a minibatch of transitions from the buffer.</li>
 *       <li>Update critic by minimizing the MSE loss:
 *           <code>L = (Q(s,a) - (r + gamma * Q_target(s', mu_target(s'))))^2</code></li>
 *       <li>Update actor using the deterministic policy gradient:
 *           <code>nabla_theta mu_theta Q(s, mu_theta(s))</code></li>
 *       <li>Softly update target networks:
 *           <code>theta_target = tau * theta + (1-tau) * theta_target</code></li>
 *     </ol>
 *   </li>
 * </ol>
 * </p>
 *
 * @see DDPGAgentConfig
 * @see DeterministicPolicyNet
 * @see QValueNet
 * @see ReplayBuffer
 * @see Noise
 */
public class DDPGAgent extends BasePolicyGradientAgent {

    public final DDPGAgentConfig config;

    private final DeterministicPolicyNet actor;
    private final DeterministicPolicyNet targetActor;
    private final QValueNet critic;
    private final QValueNet targetCritic;

    private final Tensor.Optimizer actorOptimizer;
    private final Tensor.Optimizer criticOptimizer;

    private final AgentMemory memory;
    private final Noise.NoiseStrategy noise;
    private final RandomGenerator random;

    private final Tensor.GradQueue actorGradQueue;
    private final Tensor.GradQueue criticGradQueue;

    private long totalSteps = 0;


    public DDPGAgent(@NotNull DDPGAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim); // inputs, outputs for Agent superclass
        this.config = Objects.requireNonNull(config, "DDPGAgentConfig cannot be null.");

        // Initialize networks
        this.actor = new DeterministicPolicyNet(config.actorNetworkConfig(), stateDim, actionDim);
        this.targetActor = new DeterministicPolicyNet(config.actorNetworkConfig(), stateDim, actionDim);
        this.critic = new QValueNet(config.criticNetworkConfig(), stateDim, actionDim);
        this.targetCritic = new QValueNet(config.criticNetworkConfig(), stateDim, actionDim);

        // Initialize optimizers
        this.actorOptimizer = config.actorOptimizerConfig().build();
        this.criticOptimizer = config.criticOptimizerConfig().build();

        // Initialize replay memory
        if (config.memoryConfig().replayBuffer() == null) {
            throw new IllegalArgumentException("DDPG requires replayBuffer configuration in MemoryConfig.");
        }
        if (config.perConfig() != null) {
            DDPGAgentConfig.PerConfig per = config.perConfig();
            this.memory = new PrioritizedReplayBuffer(
                config.memoryConfig().replayBuffer().capacity(),
                per.alpha(),
                per.beta0(),
                per.betaAnnealingSteps()
            );
        } else {
            this.memory = new ReplayBuffer(config.memoryConfig().replayBuffer().capacity());
        }

        // Initialize noise process
        this.noise = Noise.NoiseStrategy.create(config.noiseConfig(), actionDim);
        this.random = new XoRoShiRo128PlusRandom(); // Or use a configurable RNG

        // Initialize grad queues
        this.actorGradQueue = new Tensor.GradQueue();
        this.criticGradQueue = new Tensor.GradQueue();

        // Initialize target networks to be identical to main networks
        AgentUtils.hardUpdate(this.actor, this.targetActor);
        AgentUtils.hardUpdate(this.critic, this.targetCritic);

        setTrainingMode(true); // Default to training mode
    }


    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        // DDPG selection: actor_output + noise (if not deterministic/evaluating)
        try (var noGrad = Tensor.noGrad()) { // Action selection should not compute gradients
            Tensor actionTensor = this.actor.forward(state);
            double[] action = actionTensor.array().clone(); // Get as double array

            if (this.trainingMode && !deterministic) {
                this.noise.apply(action, this.random);
            }

            // Clip actions to ensure they are within valid bounds (e.g., [-1, 1])
            // This might depend on environment specifics or be handled by actor's output activation
            for (int i = 0; i < action.length; i++) {
                action[i] = Util.clampSafe(action[i], -1.0, 1.0); // Assuming actions are in [-1, 1]
            }
            return action;
        }
    }

    @Override
    public void recordExperience(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null.");
        // DDPG is off-policy, so it always records experience to its replay buffer during training.
        // The decision to update is typically based on buffer size and step counts, not 'done' flags directly.
        if (this.trainingMode) {
            this.memory.add(experience);

            // Trigger update based on replay buffer size and update frequency
            // This is a common pattern: update after every N steps or if buffer has enough samples
            if (this.memory.size() >= config.memoryConfig().replayBuffer().batchSize() &&
                totalSteps % config.memoryConfig().replayBuffer().updateEveryNSteps() == 0) {
                 // Update multiple times if specified (typical for DDPG to do one update per step after warmup)
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
            return; // Not enough samples in memory to form a batch, or not in training mode
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


        // --- Update Critic ---
        Tensor targetQValues;
        try (var noGrad = Tensor.noGrad()) { // Target computations should not have gradients
            Tensor nextActionsFromTargetActor = this.targetActor.forward(batch.nextStates());
            Tensor qValuesOfNextStateAction = this.targetCritic.forward(batch.nextStates(), nextActionsFromTargetActor);

            // Compute y_i = r_i + gamma * Q_target(s'_i, mu_target(s'_i)) * (1 - done_i)
            Tensor nonTerminalMask = batch.dones().neg().add(1.0f); // 1.0 if not done, 0.0 if done
            targetQValues = batch.rewards().add(
                qValuesOfNextStateAction.mul(config.commonHyperparams().gamma().floatValue()).mul(nonTerminalMask)
            );
        }

        // Current Q-values from critic: Q(s_i, a_i)
        Tensor currentQValues = this.critic.forward(batch.states(), batch.actions());
        Tensor tdErrors = targetQValues.sub(currentQValues); // Used for PER priority update

        // Critic loss: MSE( Q(s_i, a_i) - y_i )
        // If PER, weight loss by IS weights: mean( IS_weight_i * (Q_i - y_i)^2 )
        Tensor criticLossElementwise = tdErrors.sqr();
        Tensor criticLoss;
        if (importanceSamplingWeights != null) {
            criticLoss = criticLossElementwise.mul(importanceSamplingWeights).mean();
        } else {
            criticLoss = criticLossElementwise.mean();
        }

        this.criticGradQueue.clear();
        criticLoss.minimize(this.criticGradQueue);
        this.criticGradQueue.optimize(this.criticOptimizer);

        // Update priorities in PER buffer
        if (prioritizedBatch != null && this.memory instanceof PrioritizedReplayBuffer perBuffer) {
            Tensor absTdErrors = tdErrors.abs().detach(); // Ensure detached for priority updates
            perBuffer.updatePriorities(prioritizedBatch.indices(), absTdErrors);
        }

        // --- Update Actor ---
        // Actor loss: -mean( Q(s_i, mu(s_i)) )
        // We want to maximize Q-values for actions chosen by the actor.
        // Gradients flow from critic through actor.
        Tensor actionsFromActor = this.actor.forward(batch.states());
        Tensor actorLoss = this.critic.forward(batch.states(), actionsFromActor).mean().neg();

        this.actorGradQueue.clear();
        actorLoss.minimize(this.actorGradQueue);
        this.actorGradQueue.optimize(this.actorOptimizer);

        // --- Soft Update Target Networks ---
        if (getUpdateCount() % config.policyUpdateFreq() == 0) { // Check update frequency
            AgentUtils.softUpdate(this.actor, this.targetActor, config.tau().floatValue());
            AgentUtils.softUpdate(this.critic, this.targetCritic, config.tau().floatValue());
        }
        incrementUpdateCount(); // Increment after a full update cycle (critic + actor)
    }


    @Override
    public Object getPolicy() {
        return this.actor;
    }

    @Override
    public Object getValueFunction() {
        return this.critic; // DDPG's "value function" is the Q-value critic
    }

    @Override
    public Object getConfig() {
        return this.config;
    }

    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training); // Sets the trainingMode flag in BasePolicyGradientAgent
        this.actor.train(training);
        this.targetActor.train(false); // Target networks are usually in eval mode
        this.critic.train(training);
        this.targetCritic.train(false); // Target networks are usually in eval mode

        if (training) {
            this.noise.reset(); // Reset noise process when starting training
        }
    }

    @Override
    public void clearMemory() {
        this.memory.clear();
        this.noise.reset(); // Also reset noise state if memory is cleared externally
    }

    // Override apply from Agent for environment interaction loop
    @Override
    public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
        Objects.requireNonNull(input, "Current input (state) cannot be null.");
        Objects.requireNonNull(actionNext, "Output actionNext array cannot be null.");

        Tensor currentStateTensor = Tensor.row(input);
        // Note: inputPrev and actionPrev are from the *previous* step's (s, a) leading to `reward` and `input` (s')
        // For DDPG, experience is (s, a, r, s', done). `oldLogProb` is not used by DDPG.

        if (inputPrev != null && actionPrev != null) { // If not the first step of an episode
            Tensor prevStateTensor = Tensor.row(inputPrev);
            // The `reward` and `currentStateTensor` (as nextState) are outcomes of `actionPrev` taken in `prevStateTensor`.
            // `done` flag would ideally be passed here too. Assuming false if not passed, or needs to be managed by caller.
            // For now, let's assume 'done' is false unless explicitly set by environment interaction logic.
            // A more robust `apply` might need a `boolean done` parameter.
            // For DDPG, experience doesn't need oldLogProb.
            Experience2 exp = new Experience2(prevStateTensor, actionPrev, reward, currentStateTensor, false, null);
            recordExperience(exp);
        }

        // Select new action based on current state
        double[] newAction = selectAction(currentStateTensor, !this.trainingMode); // deterministic if not training
        System.arraycopy(newAction, 0, actionNext, 0, Math.min(newAction.length, actionNext.length));
    }

}
