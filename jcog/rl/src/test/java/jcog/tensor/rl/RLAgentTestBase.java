package jcog.tensor.rl;

import jcog.agent.Agent;
import jcog.tensor.rl.env.SyntheticEnv;
import org.junit.jupiter.api.Assertions;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public abstract class RLAgentTestBase {


    protected static final double LEARNING_THRESHOLD_IMPROVEMENT = 0.1; // Min improvement factor
    protected static final double MIN_PERFORMANCE_MATCHING_TASK = -0.1; // For the matching task specifically

    protected Random random = new Random(12345); // For environment and reproducible tests



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
    protected double evaluateAgent(Agent agent, SyntheticEnv env, int numEpisodes, int maxStepsPerEpisode) {
        double totalReward = 0;

        for (int i = 0; i < numEpisodes; i++) {
            double[] currentState = env.reset(random); // Ensure env uses its own randomness or is seeded
            double episodeReward = 0;
            // For evaluation, typically don't pass reward/done from previous step,
            // but act() API for these agents expects it.
            // We can pass 0 and false, as training aspect is not used.
            double lastEvalReward = 0.0;
            boolean lastEvalDone = false;

            double[] actionVec = new double[agent.actions], actionPrev = new double[agent.actions];

            for (int step = 0; step < maxStepsPerEpisode; step++) {

                agent.act(actionPrev, (float)lastEvalReward, currentState, actionVec);

                SyntheticEnv.StepResult result;
                 if (env.isDiscreteActionSpace()) {
                    int discreteAction = selectDiscreteAction(actionVec, env.actionDimension());
                    result = env.step(discreteAction);
                } else {
                    result = env.step(actionVec);
                }

                currentState = result.nextState;
                episodeReward += result.reward;
                lastEvalReward = result.reward; // Store for next call to act
                //lastEvalDone = result.done;     // Store for next call to act
            }
            totalReward += episodeReward;
            System.arraycopy(actionVec, 0, actionPrev, 0, agent.actions);
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

}
