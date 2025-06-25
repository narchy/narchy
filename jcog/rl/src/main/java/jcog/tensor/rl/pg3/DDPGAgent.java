package jcog.tensor.rl.pg3;

import jcog.Util;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg3.configs.DDPGAgentConfig;
import jcog.tensor.rl.pg3.memory.ReplayBuffer;
import jcog.tensor.rl.pg3.networks.CriticNet;
import jcog.tensor.rl.pg3.networks.DeterministicPolicyNet;

import java.util.List;
import java.util.Objects;
import java.util.Random;

public class DDPGAgentImpl extends BasePolicyGradientAgent {

    public final DDPGAgentConfig config;

    public final DeterministicPolicyNet actor;
    public final CriticNet critic;
    public final DeterministicPolicyNet targetActor;
    public final CriticNet targetCritic;

    public final Tensor.Optimizer actorOptimizer;
    public final Tensor.Optimizer criticOptimizer;

    public final ReplayBuffer memory;
    private final Random randomForNoise; // For action noise

    private final Tensor.GradQueue actorGradQueue;
    private final Tensor.GradQueue criticGradQueue;

    public DDPGAgentImpl(DDPGAgentConfig config, int stateDim, int actionDim) {
        super(stateDim, actionDim);
        Objects.requireNonNull(config, "Agent configuration cannot be null");
        this.config = config;

        // Initialize networks
        this.actor = new DeterministicPolicyNet(config.actorNetworkConfig(), stateDim, actionDim);
        this.critic = new CriticNet(config.criticNetworkConfig(), stateDim, actionDim);
        this.targetActor = new DeterministicPolicyNet(config.actorNetworkConfig(), stateDim, actionDim);
        this.targetCritic = new CriticNet(config.criticNetworkConfig(), stateDim, actionDim);

        // Initialize optimizers
        this.actorOptimizer = config.actorOptimizerConfig().build(this.actor.getWeights());
        this.criticOptimizer = config.criticOptimizerConfig().build(this.critic.getWeights());

        // Initialize memory
        if (config.memoryConfig().replayBufferConfig() == null) {
            throw new IllegalArgumentException("DDPG requires ReplayBufferConfig in MemoryConfig");
        }
        this.memory = new ReplayBuffer(config.memoryConfig().replayBufferConfig().capacity());

        this.randomForNoise = new XoRoShiRo128PlusRandom(); // Or make injectable

        // Copy initial weights to target networks
        Util.softUpdate(this.targetActor.getWeights(), this.actor.getWeights(), 1.0f); // Hard copy
        Util.softUpdate(this.targetCritic.getWeights(), this.critic.getWeights(), 1.0f); // Hard copy

        this.actorGradQueue = new Tensor.GradQueue(this.actor.getWeights());
        this.criticGradQueue = new Tensor.GradQueue(this.critic.getWeights());

        setTrainingMode(true); // Initialize training mode by default
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        Tensor actionTensor;
        try (var noGrad = Tensor.noGrad()) { // Action selection should not compute gradients
            actionTensor = this.actor.apply(state);
        }

        double[] action = actionTensor.array(); // Assumes actionTensor is (1, actionDim)

        if (!deterministic && this.trainingMode) {
            // Add noise for exploration if in training mode and not requesting deterministic action
            float noiseStddev = config.ddpgHyperparams().actionNoiseStddev();
            if (noiseStddev > 0) {
                for (int i = 0; i < action.length; i++) {
                    action[i] += randomForNoise.nextGaussian() * noiseStddev;
                }
            }
        }
        // Clip action to be within valid range (e.g., if actor output is tanh, it's already [-1,1])
        Util.clampSafe(action, -1.0, 1.0); // Assuming actions are normalized to [-1, 1]
        return action;
    }

    @Override
    protected ActionWithLogProb selectActionWithLogProb(Tensor state, boolean deterministic) {
        // DDPG is deterministic and doesn't use log probabilities for its experience tuples
        // in the same way stochastic policy agents do for entropy or importance sampling.
        double[] actionArray = selectAction(state, deterministic);
        return new ActionWithLogProb(actionArray, null); // LogProb is null
    }


    @Override
    public void recordExperience(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null");
        this.memory.add(experience);

        if (this.trainingMode && this.memory.size() >= config.memoryConfig().replayBufferConfig().batchSize()) {
            // Update can be called here if memory has enough samples.
            // For DDPG, often updates are triggered after a certain number of steps or when enough data.
            // update(0); // Call if updating on every valid recordExperience
        }
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
        Tensor actions = tensors.actions();
        Tensor rewards = tensors.rewards();
        Tensor nextStates = tensors.nextStates();
        Tensor dones = tensors.dones(); // 1.0 if done, 0.0 if not

        // --- Update Critic ---
        Tensor targetActions;
        try (var noGrad = Tensor.noGrad()) {
            targetActions = this.targetActor.apply(nextStates);
            Tensor targetQValues = this.targetCritic.apply(new Tensor[]{nextStates, targetActions});
            Tensor y_i = rewards.add(
                dones.mul(-1).add(1) // (1 - dones)
                    .mul(config.hyperparams().gamma())
                    .mul(targetQValues)
            );

            criticGradQueue.clear();
            Tensor currentQValues = this.critic.apply(new Tensor[]{states, actions});
            Tensor criticLoss = currentQValues.loss(y_i.detach(), Tensor.Loss.MeanSquared);
            criticLoss.minimize(criticGradQueue);
        }
        criticGradQueue.optimize(criticOptimizer);

        // --- Update Actor ---
        if (getUpdateCount() % config.ddpgHyperparams().policyUpdateFreq() == 0) {
            actorGradQueue.clear();
            Tensor newActionsFromActor = this.actor.apply(states);
            Tensor actorLoss = this.critic.apply(new Tensor[]{states, newActionsFromActor}).mean().neg();
            actorLoss.minimize(actorGradQueue);
            actorGradQueue.optimize(actorOptimizer);

            Util.softUpdate(this.targetActor.getWeights(), this.actor.getWeights(), config.ddpgHyperparams().tau());
            Util.softUpdate(this.targetCritic.getWeights(), this.critic.getWeights(), config.ddpgHyperparams().tau());
        }

        incrementUpdateCount();
    }


    @Override
    public Object getPolicy() {
        return this.actor;
    }

    @Override
    public Object getValueFunction() {
        return this.critic;
    }

    @Override
    public Object getConfig() {
        return this.config;
    }

    @Override
    public void setTrainingMode(boolean training) {
        super.setTrainingMode(training);
        this.actor.train(training);
        this.critic.train(training);
        this.targetActor.train(false);
        this.targetCritic.train(false);
    }

    @Override
    public void clearMemory() {
        this.memory.clear();
    }
}
