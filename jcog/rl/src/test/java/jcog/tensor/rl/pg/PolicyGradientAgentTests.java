package jcog.tensor.rl.pg;

import jcog.tensor.rl.RLAgentTestBase;
import jcog.tensor.rl.env.SimpleGridWorld;
import jcog.tensor.rl.env.SyntheticEnvironment;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("RLAlgorithmTests")
@DisplayName("Policy Gradient Agent Tests")
public class PolicyGradientAgentTests extends RLAgentTestBase {

    // Helper to create agents using PGBuilder
    private AbstrPG createAgent(PGBuilder.Algorithm algo, int inputs, int outputs, boolean isDiscrete) {
        PGBuilder builder = createDefaultPGBuilder(algo, inputs, outputs, isDiscrete);

        // Specific configurations if needed, e.g., for discrete action spaces
        if (isDiscrete) {
            // PGBuilder's default action distribution is Gaussian, which is for continuous.
            // For discrete environments, the policy network's output (size `outputs`)
            // is typically fed into a softmax and then sampled (Categorical distribution).
            // RLAgentTestBase.selectDiscreteAction uses argmax on the raw output,
            // which is a common simplification if the network learns appropriate logits.
            // Ideally, PGBuilder would allow specifying a Categorical distribution.
            // builder.action(a -> a.distribution(PGBuilder.ActionConfig.Distribution.CATEGORICAL));
        }

        // Some algorithms might need longer episode lengths or different hyperparams by default
        if (algo == PGBuilder.Algorithm.REINFORCE && isDiscrete) {
            builder.memory(m -> m.episodeLength(128)); // REINFORCE might need more samples per update
        }
        if (algo == PGBuilder.Algorithm.SAC || algo == PGBuilder.Algorithm.DDPG) {
             builder.hyperparams(h -> h.gamma(0.95f)); // Common for off-policy
        }


        return builder.build();
    }

    @ParameterizedTest(name = "{0} on SimpleGridWorld")
    @EnumSource(value = PGBuilder.Algorithm.class,
                 names = {"REINFORCE", "VPG", "PPO"}) // SAC and DDPG are typically for continuous.
    @DisplayName("Discrete Action Test: SimpleGridWorld")
    @Timeout(value = 5, unit = TimeUnit.MINUTES) // Generous timeout for potentially slow learning
    void testAgentOnSimpleGridWorld(PGBuilder.Algorithm algo) {
        SimpleGridWorld env = new SimpleGridWorld(); // Default 5x5 grid
        AbstrPG agent = createAgent(algo, env.stateDimension(), env.actionDimension(), true);

        int totalEpisodes = 300;
        int maxStepsPerEpisode = env.maxSteps; // Use env's max steps
        int evaluationWindow = 30;

        if (algo == PGBuilder.Algorithm.REINFORCE) {
            totalEpisodes = 700; // REINFORCE is less sample efficient
            evaluationWindow = 70;
        }

        List<Double> rewardsHistory = trainAgent(agent, env, totalEpisodes, maxStepsPerEpisode);
        assertLearningOccurs(rewardsHistory, evaluationWindow, algo + " on SimpleGridWorld");

        double avgEvalReward = evaluateAgent(agent, env, 20, maxStepsPerEpisode);
        System.out.printf("%s SimpleGridWorld - Avg Eval Reward after training: %.4f%n", algo, avgEvalReward);

        // Performance expectation can vary. PPO/VPG should do better than REINFORCE.
        double expectedMinReward = -(maxStepsPerEpisode * env.stepPenalty) * 0.5; // Should at least do better than random walk
        if (algo == PGBuilder.Algorithm.PPO || algo == PGBuilder.Algorithm.VPG) {
            expectedMinReward = 0; // Expect PPO/VPG to solve it (positive reward)
        }

        assertTrue(avgEvalReward > expectedMinReward,
                algo + " should achieve a reasonable score on SimpleGridWorld. Got: " + avgEvalReward + ", Expected > " + expectedMinReward);
    }


    // MatchingEnv for continuous tests
    private static class MatchingEnv implements SyntheticEnvironment {
        private final int dim;
        private final Random randomInstance; // Use instance for better control if needed, or base class random
        private double[] currentState;

        public MatchingEnv(int dim, Random randomSeedGenerator) {
            this.dim = dim;
            this.randomInstance = new Random(randomSeedGenerator.nextLong()); // Seed per instance
        }

        @Override public int stateDimension() { return dim; }
        @Override public int actionDimension() { return dim; }
        @Override public boolean isDiscreteActionSpace() { return false; }

        @Override public double[] reset() {
            currentState = randomInstance.doubles(dim, -1.0, 1.0).toArray();
            return currentState;
        }

        @Override public StepResult step(double[] action) {
            double mse = 0;
            for (int i = 0; i < dim; i++) {
                mse += Math.pow(currentState[i] - action[i], 2);
            }
            double reward = -mse / dim;
            // For matching task, each step is an independent trial, so "done" is true.
            // The next state is a new random state.
            double[] nextState = randomInstance.doubles(dim, -1.0, 1.0).toArray();
            // currentState = nextState; // Not needed as it's reset effectively
            return new StepResult(nextState, reward, true);
        }
        @Override public StepResult step(int action) { throw new UnsupportedOperationException("MatchingEnv uses continuous actions."); }
    }

    @ParameterizedTest(name = "{0} on Matching Task")
    @EnumSource(PGBuilder.Algorithm.class) // Test all algorithms that support continuous actions
    @DisplayName("Continuous Action Test: Matching Task")
    @Timeout(value = 5, unit = TimeUnit.MINUTES)
    void testAgentOnMatchingTask(PGBuilder.Algorithm algo) {
        // DDPG and SAC are primarily for continuous action spaces.
        // REINFORCE, VPG, PPO can also handle continuous actions with appropriate policy distributions (e.g., Gaussian).
        MatchingEnv env = new MatchingEnv(DEFAULT_INPUTS_CONTINUOUS, this.random); // Use base class random for seeding
        AbstrPG agent = createAgent(algo, env.stateDimension(), env.actionDimension(), false);

        // Matching task uses 1-step episodes. totalEpisodes effectively means total training steps.
        int totalTrainingSteps = (algo == PGBuilder.Algorithm.REINFORCE) ? 12000 : 8000;
        if (algo == PGBuilder.Algorithm.SAC || algo == PGBuilder.Algorithm.DDPG) {
            totalTrainingSteps = 10000; // Off-policy might need a bit more interaction for this setup
        }

        int maxStepsPerEpisode = 1; // Env is always done in 1 step
        int evaluationWindow = totalTrainingSteps / 10; // Evaluate over last 10% of steps

        List<Double> rewardsHistory = trainAgent(agent, env, totalTrainingSteps, maxStepsPerEpisode);
        assertLearningOccurs(rewardsHistory, evaluationWindow, algo + " on MatchingTask");

        double avgEvalReward = evaluateAgent(agent, env, 200, maxStepsPerEpisode);
        System.out.printf("%s MatchingTask - Avg Eval Reward after training: %.4f%n", algo, avgEvalReward);

        double expectedMinPerf = MIN_PERFORMANCE_MATCHING_TASK;
        if (algo == PGBuilder.Algorithm.SAC || algo == PGBuilder.Algorithm.DDPG) {
            expectedMinPerf = -0.05; // Expect better performance from SAC/DDPG
        }

        assertTrue(avgEvalReward > expectedMinPerf,
             algo + " should achieve a good score on MatchingTask. Got: " + avgEvalReward + ", Expected > " + expectedMinPerf);
    }
}
