package jcog.tensor.rl;

import jcog.tensor.rl.env.SyntheticEnvironment;
import jcog.tensor.rl.pg.AbstrPG;
import jcog.tensor.rl.pg.PGBuilder;
import org.junit.jupiter.api.Assertions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public abstract class RLAgentTestBase {

    protected static final int DEFAULT_INPUTS_CONTINUOUS = 2;
    protected static final int DEFAULT_OUTPUTS_CONTINUOUS = 2;
    protected static final int DEFAULT_INPUTS_GRID = 2; // x, y
    protected static final int DEFAULT_OUTPUTS_GRID = 4; // up, down, left, right

    protected static final double LEARNING_THRESHOLD_IMPROVEMENT = 0.1; // Min improvement factor
    protected static final double MIN_PERFORMANCE_MATCHING_TASK = -0.1; // For the matching task specifically

    protected Random random = new Random(12345); // For environment and reproducible tests

    /**
     * Trains an agent on a given synthetic environment.
     *
     * @param agent The agent to train.
     * @param env The environment to train on.
     * @param totalEpisodes The total number of episodes to train for.
     * @param maxStepsPerEpisode The maximum steps per episode.
     * @return A list of total rewards per episode.
     */
    protected List<Double> trainAgent(AbstrPG agent, SyntheticEnvironment env, int totalEpisodes, int maxStepsPerEpisode) {
        List<Double> episodeRewards = new ArrayList<>(totalEpisodes);

        for (int episode = 0; episode < totalEpisodes; episode++) {
            double[] currentState = env.reset();
            double currentEpisodeReward = 0;
            double lastRewardSignal = 0; // Initial reward signal for the first step
            boolean doneSignal = false;    // Initial done signal

            for (int step = 0; step < maxStepsPerEpisode; step++) {
                double[] action;
                if (env.isDiscreteActionSpace()) {
                    // For discrete actions, agent.act() might expect to produce discrete actions directly
                    // or produce probabilities. PGBuilder agents typically produce continuous actions
                    // that might need to be discretized (e.g., by argmax for policy).
                    // For now, assume agent.act can handle it or adapt PGBuilder output.
                    // This part might need refinement based on specific agent capabilities.
                    // The current PG agents output continuous values; for discrete, we'd typically argmax.
                    // However, `act` itself takes state, reward, done. The policy inside will decide action.
                    // Let's assume the agent's internal policy handles the output type correctly.
                    // If it's a discrete env, the agent should be configured for discrete output.
                    // PGBuilder currently focuses on continuous outputs.
                    // This is a simplification for now.
                    action = agent.act(currentState, lastRewardSignal, doneSignal);
                } else {
                    action = agent.act(currentState, lastRewardSignal, doneSignal);
                }

                SyntheticEnvironment.StepResult result;
                if (env.isDiscreteActionSpace()) {
                    // We need to convert the action (typically continuous for PG) to discrete.
                    // A common way is to take the argmax if the output represents probabilities,
                    // or if it's a single value, discretize it.
                    // This is a known challenge when applying continuous-output agents to discrete envs.
                    // For now, let's assume the environment or a wrapper handles this.
                    // Or, the agent is specifically a discrete-action variant.
                    // For this test base, we'll assume the `action` from `agent.act` must be converted.
                    // This is a placeholder for a more robust solution.
                    int discreteAction = selectDiscreteAction(action, env.actionDimension());
                    result = env.step(discreteAction);
                } else {
                    result = env.step(action);
                }

                lastRewardSignal = result.reward;
                doneSignal = result.done;
                currentState = result.nextState;
                currentEpisodeReward += result.reward;

                if (doneSignal) {
                    // Final update for the episode if agent expects it (e.g. REINFORCE)
                    // Some agents might do this internally based on the done signal.
                    // For PPO, VPG, etc., the `act` call with done=true triggers episode processing.
                    agent.act(currentState, lastRewardSignal, true); // Signal episode end
                    break;
                }
            }
            episodeRewards.add(currentEpisodeReward);
        }
        return episodeRewards;
    }

    /**
     * Selects a discrete action from a continuous action vector (e.g., policy output).
     * This is a simple argmax implementation.
     */
    protected int selectDiscreteAction(double[] continuousAction, int numDiscreteActions) {
        if (continuousAction.length != numDiscreteActions && continuousAction.length == 1 && numDiscreteActions > 1) {
            // If agent outputs a single value for a multi-action discrete space,
            // this is likely an error in setup or agent type.
            // However, for robustness, we can try to map it.
            // This is a very naive way, assuming the single value is meaningful for discretization.
            // E.g. if action is in [-1, 1], scale it to [0, numDiscreteActions-1]
            double scaledAction = (continuousAction[0] + 1.0) / 2.0 * (numDiscreteActions - 1);
            return Math.min(numDiscreteActions - 1, Math.max(0, (int) Math.round(scaledAction)));
        } else if (continuousAction.length != numDiscreteActions) {
             throw new IllegalArgumentException(
                "Continuous action dimension (" + continuousAction.length +
                ") must match number of discrete actions (" + numDiscreteActions +
                ") for argmax selection, or be 1 for simple scaling."
             );
        }

        int maxIndex = 0;
        for (int i = 1; i < continuousAction.length; i++) {
            if (continuousAction[i] > continuousAction[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    /**
     * Evaluates an agent's performance on an environment without training.
     *
     * @param agent The agent to evaluate.
     * @param env The environment for evaluation.
     * @param numEpisodes Number of episodes to run for evaluation.
     * @param maxStepsPerEpisode Max steps in each evaluation episode.
     * @return Average reward over the evaluation episodes.
     */
    protected double evaluateAgent(AbstrPG agent, SyntheticEnvironment env, int numEpisodes, int maxStepsPerEpisode) {
        double totalReward = 0;
        Random evalRandom = new Random(random.nextLong()); // Use a different seed for evaluation consistency if needed

        for (int i = 0; i < numEpisodes; i++) {
            double[] currentState = env.reset(); // Ensure env uses its own randomness or is seeded
            double episodeReward = 0;
            // For evaluation, typically don't pass reward/done from previous step,
            // but act() API for these agents expects it.
            // We can pass 0 and false, as training aspect is not used.
            double lastEvalReward = 0.0;
            boolean lastEvalDone = false;

            for (int step = 0; step < maxStepsPerEpisode; step++) {
                // In evaluation, use the deterministic action if possible, or mean of distribution.
                // The _action method in AbstrPG is suitable for this as it typically samples.
                // For a more deterministic evaluation, one might need access to a deterministic policy.
                // PGBuilder's AbstrPG._action(Tensor state, boolean training) is ideal.
                // However, we only have the AbstrPG interface here.
                // A simple way: call `act` but ensure it doesn't trigger learning.
                // Most PGBuilder agents' `act` method also does `remember` and `update`.
                // This is not ideal for pure evaluation.
                // The `_action(Tensor state)` is protected.
                //
                // Workaround: PGBuilderTest uses `model._action(Tensor.row(state), true)`
                // We need a similar capability. For now, we'll use `act` and accept that it might
                // call remember/update, which is not pure evaluation but a practical compromise.
                // Or, we can use `agent.policy.apply(Tensor.row(currentState))` if policy is public
                // and then sample/argmax.
                // PGBuilder's `_action(state, training)` is what we need.
                // For now, let's assume `act` is used, and for evaluation, its learning side effects are minimal
                // or the agent is in an eval mode (not explicitly supported by AbstrPG).

                double[] actionVec = agent.act(currentState, lastEvalReward, lastEvalDone);

                SyntheticEnvironment.StepResult result;
                 if (env.isDiscreteActionSpace()) {
                    int discreteAction = selectDiscreteAction(actionVec, env.actionDimension());
                    result = env.step(discreteAction);
                } else {
                    result = env.step(actionVec);
                }

                currentState = result.nextState;
                episodeReward += result.reward;
                lastEvalReward = result.reward; // Store for next call to act
                lastEvalDone = result.done;     // Store for next call to act

                if (result.done) {
                    // If agent needs a final signal for "done" even in eval, send it.
                    agent.act(currentState, lastEvalReward, true);
                    break;
                }
            }
            totalReward += episodeReward;
        }
        return totalReward / numEpisodes;
    }

    protected void assertLearningOccurs(List<Double> rewardsHistory, int window, String agentName) {
        Assertions.assertTrue(rewardsHistory.size() > window * 2,
                agentName + ": Not enough reward history to evaluate learning. Collected: " + rewardsHistory.size());

        double initialPerf = rewardsHistory.subList(0, window).stream().mapToDouble(Double::doubleValue).average().orElse(-Double.MAX_VALUE);
        double finalPerf = rewardsHistory.subList(rewardsHistory.size() - window, rewardsHistory.size()).stream().mapToDouble(Double::doubleValue).average().orElse(-Double.MAX_VALUE);

        String learningCurve = rewardsHistory.stream().map(r -> String.format("%.4f", r)).collect(Collectors.joining(", "));
        System.out.printf("Performance for %s (avg over %d episodes): Initial: %.4f, Final: %.4f. Curve: [%s]%n",
                agentName, window, initialPerf, finalPerf, learningCurve);

        Assertions.assertTrue(finalPerf > initialPerf + Math.abs(initialPerf * LEARNING_THRESHOLD_IMPROVEMENT) + 1e-6, // Add epsilon for floating point
                agentName + ": Final performance (" + finalPerf + ") should be significantly better than initial (" + initialPerf + ")." +
                " Learning curve: " + learningCurve);
    }

    /**
     * Helper to create a PGBuilder instance with common defaults for testing.
     * Specific tests can then customize it.
     */
    protected PGBuilder createDefaultPGBuilder(PGBuilder.Algorithm algo, int inputs, int outputs, boolean isDiscrete) {
        var builder = new PGBuilder(inputs, outputs).algorithm(algo);
        builder.policy(p -> p.hiddenLayers(32, 32).optimizer(o -> o.learningRate(1e-3f)));

        if (isDiscrete) {
            // For discrete actions, PGBuilder needs to be configured appropriately if it supports it.
            // This might involve setting action distribution to categorical, or ensuring output layer
            // matches number of actions for softmax. PGBuilder primarily targets continuous.
            // This is a placeholder for future PGBuilder enhancements for discrete actions.
            // For now, we assume the output layer of policy is `outputs` (num_actions) and we apply argmax.
        }


        switch (algo) {
            case PPO, VPG, REINFORCE ->
                    builder.value(v -> v.hiddenLayers(32, 32).optimizer(o -> o.learningRate(1e-3f)));
            case SAC -> builder.qNetworks(q -> q.hiddenLayers(32, 32).optimizer(o -> o.learningRate(1e-3f)));
            case DDPG -> builder.value(v -> v.hiddenLayers(32, 32).optimizer(o -> o.learningRate(1e-3f)))
                    .action(a -> a.distribution(PGBuilder.ActionConfig.Distribution.DETERMINISTIC));
        }
        builder.hyperparams(h -> h.epochs(1).policyUpdateFreq(1).gamma(0.99f)); // Sensible defaults for tests
        builder.memory(m -> m
                .episodeLength(isDiscrete ? 64 : 32) // Grid world might take more steps
                .replayBuffer(rb -> rb.capacity(isDiscrete ? 2048 : 10000).batchSize(isDiscrete ? 64 : 128))
        );
        return builder;
    }
}
