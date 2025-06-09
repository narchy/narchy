package jcog.tensor;

import jcog.tensor.Tensor.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static jcog.tensor.experimental.ModelsExperimental.DecisionTransformer;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DecisionTransformerTest {

    @Test
    void testDecisionTransformerEndToEnd() {
        // Parameters for a small model
        int stateSize = 2;
        int actionSize = 1;
        int returnSize = 1;
        int maxEpisodeLength = 5;
        int embeddingSize = 80;
        int numLayers = 2;
        int numHeads = 2;
        float dropoutRate = 0.1f;
        int numTrajectories = 200;
        int numEpochs = 100;

        var transformerOrULT =
            true;
            //false;

        DecisionTransformer dt = new DecisionTransformer(
            stateSize, actionSize, returnSize, maxEpisodeLength,
            embeddingSize, numLayers, numHeads, dropoutRate, transformerOrULT
        );

        // Generate synthetic data with a learnable pattern
        List<List<Tensor>> trajectories = trajectories(
                numTrajectories, maxEpisodeLength, stateSize, actionSize, returnSize
        );

        // Train the model
        Optimizer optimizer = new Optimizers.ADAM(()->0.001f).get();
        dt.train(
                batchIndex -> dt.training(trajectories),
                numEpochs,
                optimizer
        );

        // Test the model
        double correctActionPredictions = 0;
        int numTestCases = 100;

        for (int i = 0; i < numTestCases; i++) {
            List<Tensor> testTrajectory = trajectories(1, maxEpisodeLength, stateSize, actionSize, returnSize).get(0);
            DecisionTransformer.TrainingBatch testBatch = dt.training(List.of(testTrajectory));

            Tensor predictedAction = dt.getAction(
                    testBatch.states(), testBatch.actions(), testBatch.returns(),
                    testBatch.timesteps(), testBatch.attentionMask()
            );

            // Get the last state from the test trajectory
            Tensor lastState = testTrajectory.get(testTrajectory.size() - 1).slice(0, stateSize);
            double optimalAction = getOptimalAction(lastState.data(0), lastState.data(1));

            // Check if the predicted action is close to the optimal action
            if (Math.abs(predictedAction.scalar() - optimalAction) < 0.2) {
                correctActionPredictions++;
            }
        }

        double accuracyRate = correctActionPredictions / numTestCases;
        System.out.println("Accuracy rate: " + accuracyRate);

        // The model should predict the correct action at least 80% of the time
        assertTrue(accuracyRate > 0.8, "Model should predict correct actions with high accuracy");

        // Check if the model's loss decreases over time (keep this test from before)
        DecisionTransformer.TrainingBatch testBatch = dt.training(trajectories.subList(0, 10));
        double initialLoss = dt.computeLoss(testBatch).scalar();
        dt.train(
            batchIndex -> dt.training(trajectories),
            numEpochs / 2,
            optimizer
        );
        double finalLoss = dt.computeLoss(testBatch).scalar();

        assertTrue(finalLoss < initialLoss,
                   "Model's loss should decrease after additional training");
    }

    private List<List<Tensor>> trajectories(int numTrajectories, int maxLength,
                                            int stateSize, int actionSize, int returnSize) {
        List<List<Tensor>> trajectories = new ArrayList<>();
        Random random = new Random(42);  // Fixed seed for reproducibility

        for (int i = 0; i < numTrajectories; i++) {
            List<Tensor> trajectory = new ArrayList<>();
            double cumulativeReturn = 0;

            for (int j = 0; j < maxLength; j++) {
                double[] state = new double[stateSize];
                double[] action = new double[actionSize];
                double[] returns = new double[returnSize];

                // Generate synthetic state
                for (int k = 0; k < stateSize; k++) {
                    state[k] = random.nextDouble() * 2 - 1;  // Random value between -1 and 1
                }

                // Generate action based on the optimal policy
                action[0] = getOptimalAction(state[0], state[1]);

                // Calculate reward
                double reward = calculateReward(state[0], state[1], action[0]);
                cumulativeReturn += reward;
                returns[0] = cumulativeReturn;

                // Combine state, action, and return into a single tensor
                double[] combined = new double[stateSize + actionSize + returnSize];
                System.arraycopy(state, 0, combined, 0, stateSize);
                System.arraycopy(action, 0, combined, stateSize, actionSize);
                System.arraycopy(returns, 0, combined, stateSize + actionSize, returnSize);

                trajectory.add(Tensor.row(combined));
            }
            trajectories.add(trajectory);
        }

        return trajectories;
    }

    private double getOptimalAction(double state1, double state2) {
        // Optimal action: if state1 > state2, choose positive action; otherwise, choose negative action
        return state1 > state2 ? 0.8 : -0.8;
    }

    private double calculateReward(double state1, double state2, double action) {
        double optimalAction = getOptimalAction(state1, state2);
        // Reward is higher when the action is closer to the optimal action
        return 1.0 - Math.abs(action - optimalAction);
    }
}