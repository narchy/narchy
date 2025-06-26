package jcog.tensor.rl.pg2;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg2.configs.VPGAgentConfig;
import jcog.tensor.rl.pg2.stats.DefaultMetricCollector;
import jcog.tensor.rl.pg2.stats.MetricCollector;
import jcog.tensor.rl.pg2.configs.PPOAgentConfig;
import jcog.tensor.rl.pg2.configs.ReinforceAgentConfig;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static java.lang.Math.round;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PolicyGradientTest {

    private static final int ENV_STATE_DIM = 1;
    private static final int ENV_ACTION_DIM = 1;
    private static final int MAX_STEPS_PER_EPISODE = 50;
    private static final int NUM_EPISODES = 200; // Reduced for faster test runs, can be increased for more thoroughness
    private static final int HIDDEN_LAYER_SIZE_FACTOR = 4;
    private static final int DEFAULT_EPISODE_LEN_FOR_UPDATE = 32; // Common default for on-policy updates


    // Simple environment for testing: PointNavigation1D
    static class PointNavigation1D {
        private double currentState;
        private final double targetState = 0.0;
        private final double stepSize = 0.1; // Max action magnitude
        private final double stateMin = -1.0;
        private final double stateMax = 1.0;
        //private final int maxStepsPerEpisode = 50; // Use constant
        private int currentStepInEpisode;
        private final Random random = new Random(123); // Seeded for reproducibility

        public PointNavigation1D() {
            reset();
        }

        public Tensor reset() {
            this.currentState = random.nextDouble() * (stateMax - stateMin) + stateMin; // Random start
            this.currentStepInEpisode = 0;
            return Tensor.scalar(this.currentState);
        }

        // Returns <NextState, Reward, Done>
        public Triplet<Tensor, Double, Boolean> step(double[] action) {
            var move = Util.clamp(action[0], -stepSize, stepSize);
            this.currentState = Util.clamp(this.currentState + move, stateMin, stateMax);
            this.currentStepInEpisode++;

            var reward = -Math.abs(this.currentState - this.targetState); // Negative distance to target

            var done = this.currentStepInEpisode >= MAX_STEPS_PER_EPISODE || Math.abs(this.currentState - targetState) < 0.01;

            if (Math.abs(this.currentState - targetState) < 0.01) {
                reward += 10.0; // Bonus for reaching target
            }

            // Small penalty for existing
            // reward -= 0.01;

            return new Triplet<>(Tensor.scalar(this.currentState), reward, done);
        }

        public int getStateDim() {
            return ENV_STATE_DIM;
        }

        public int getActionDim() {
            return ENV_ACTION_DIM;
        }
    }

    // Helper record for environment step results
    record Triplet<S, R, D>(S state, R reward, D done) {}

    static Stream<Supplier<AbstractPolicyGradientAgent>> agentProvider() {
        return Stream.of(
                PolicyGradientTest::createVPGAgent,
                PolicyGradientTest::createPPOAgent,
                PolicyGradientTest::createReinforceAgent
        );
    }

    private static AbstractPolicyGradientAgent createVPGAgent() {
        var h = round(ENV_STATE_DIM * HIDDEN_LAYER_SIZE_FACTOR);
        return new VPGAgent(new VPGAgentConfig(new int[]{h, h}, DEFAULT_EPISODE_LEN_FOR_UPDATE), ENV_STATE_DIM, ENV_ACTION_DIM);
    }

    private static AbstractPolicyGradientAgent createPPOAgent() {
        var h = round(ENV_STATE_DIM * HIDDEN_LAYER_SIZE_FACTOR);
        int[] hiddenLayers = {h, h};

        HyperparamConfig hyperparams = new HyperparamConfig(); // Default hyperparams
        NetworkConfig policyNetworkConfig = new NetworkConfig(
                OptimizerConfig.of(hyperparams.policyLR().floatValue()),
                hiddenLayers
        );
        NetworkConfig valueNetworkConfig = new NetworkConfig(
                OptimizerConfig.of(hyperparams.valueLR().floatValue()),
                hiddenLayers
        );
        ActionConfig actionConfig = new ActionConfig(); // Default action config
        MemoryConfig memoryConfig = new MemoryConfig(DEFAULT_EPISODE_LEN_FOR_UPDATE);

        PPOAgentConfig ppoConfig = new PPOAgentConfig(
                hyperparams,
                policyNetworkConfig,
                valueNetworkConfig,
                actionConfig,
                memoryConfig
        );
        return new PPOAgent(ppoConfig, ENV_STATE_DIM, ENV_ACTION_DIM);
    }

    private static AbstractPolicyGradientAgent createReinforceAgent() {
        var h = round(ENV_STATE_DIM * HIDDEN_LAYER_SIZE_FACTOR);
        int[] hiddenLayers = {h, h};

        HyperparamConfig hyperparams = new HyperparamConfig(); // Default hyperparams
        NetworkConfig policyNetworkConfig = new NetworkConfig(
                OptimizerConfig.of(hyperparams.policyLR().floatValue()),
                hiddenLayers
        );
        ActionConfig actionConfig = new ActionConfig(); // Default action config
        MemoryConfig memoryConfig = new MemoryConfig(DEFAULT_EPISODE_LEN_FOR_UPDATE);

        // ReinforceAgentConfig does not take a valueNetworkConfig in its primary constructor
        // It expects: HyperparamConfig, NetworkConfig (policy), ActionConfig, MemoryConfig
        ReinforceAgentConfig reinforceConfig = new ReinforceAgentConfig(
                hyperparams,
                policyNetworkConfig,
                actionConfig,
                memoryConfig
        );
        return new ReinforceAgent(reinforceConfig, ENV_STATE_DIM, ENV_ACTION_DIM);
    }


    @ParameterizedTest
    @MethodSource("agentProvider")
    void testAgentLearnsOnPointNavigation(Supplier<AbstractPolicyGradientAgent> agentSupplier) {
        var env = new PointNavigation1D();
        //var stateDim = env.getStateDim(); // Using constants
        //var actionDim = env.getActionDim(); // Using constants
        MetricCollector metrics = new DefaultMetricCollector(); // Each test run gets its own metrics

        AbstractPolicyGradientAgent agent = agentSupplier.get();
        agent.setMetricCollector(metrics); // Ensure agent uses the collector
        agent.setTrainingMode(true);

        List<Double> episodeRewards = new ArrayList<>();
        double totalRewardLast50Episodes = 0;

        // Store initial metric values
        // To get initial values, we might need a dummy update or rely on first recorded values.
        // For simplicity, we'll record values after a small number of initial updates / episodes.
        // Alternatively, capture them if present after the first few episodes where updates have occurred.
        final int initialCheckEpisode = 20; // Check initial metrics after these many episodes (allows some updates)
        final double[] initialPolicyLoss = {Double.NaN};
        final double[] initialValueLoss = {Double.NaN};
        final double[] initialEntropy = {Double.NaN};
        final boolean[] initialMetricsCaptured = {false};


        System.out.printf("Starting training with %s on PointNavigation1D...%n", agent.getClass().getSimpleName());

        for (var episode = 0; episode < NUM_EPISODES; episode++) {
            var state = env.reset();
            double currentEpisodeReward = 0;

            for (var t = 0; t < MAX_STEPS_PER_EPISODE; t++) {
                PolicyGradientAgent.ActionOutput actionOutput = agent.selectAction(state, false);
                var action = actionOutput.action();
                Tensor currentLogProb = actionOutput.logProb(); // Will be null if not PPO/not applicable

                var stepResult = env.step(action);
                var nextState = stepResult.state();
                double reward = stepResult.reward();
                boolean done = stepResult.done();

                // Store experience: state, action, reward, nextState, done, oldLogProb (currentLogProb is used as oldLogProb for PPO)
                agent.memory.add(new Experience2(state, action, reward, nextState, done, currentLogProb));

                state = nextState;
                currentEpisodeReward += reward;

                if (agent.memory.size() >= agent.config.memoryConfig().episodeLength().intValue()) {
                    // Global step can be estimated or tracked more accurately if needed
                    agent.update(episode * MAX_STEPS_PER_EPISODE + t);
                    // For VPG, PPO, REINFORCE, they are on-policy and memory should be cleared after updates.
                    // This can be inferred if replayBuffer is null in their MemoryConfig.
                    if (agent.config.memoryConfig().replayBuffer() == null && agent.config.memoryConfig().episodeLength() != null) {
                        agent.memory.clear();
                    }
                }

                if (done) break;
            }
            episodeRewards.add(currentEpisodeReward);
            if (episode >= NUM_EPISODES - 50) {
                totalRewardLast50Episodes += currentEpisodeReward;
            }

            if ((episode + 1) % 20 == 0) {
                System.out.printf("Agent: %s, Episode %d, Avg Reward (last 20): %.2f, Total Updates: %d%n",
                        agent.getClass().getSimpleName(),
                        episode + 1,
                        episodeRewards.subList(Math.max(0, episodeRewards.size()-20), episodeRewards.size())
                                .stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN),
                        agent.getUpdateCount());

                metrics.getSummary("policy_loss").ifPresent(s -> System.out.printf("  Policy Loss: %.4f%n", s.getMean()));
                metrics.getSummary("value_loss").ifPresent(s -> System.out.printf("  Value Loss: %.4f%n", s.getMean()));
                metrics.getSummary("entropy").ifPresent(s -> System.out.printf("  Entropy: %.4f%n", s.getMean()));

                // Capture initial metrics
                if (!initialMetricsCaptured[0] && episode >= initialCheckEpisode && agent.getUpdateCount() > 0) {
                    metrics.getSummary("policy_loss").ifPresent(s -> initialPolicyLoss[0] = s.getMean());
                    metrics.getSummary("value_loss").ifPresent(s -> initialValueLoss[0] = s.getMean());
                    metrics.getSummary("entropy").ifPresent(s -> initialEntropy[0] = s.getMean());
                    if (!Double.isNaN(initialPolicyLoss[0])) { // If policy loss is captured, assume others are too (if they exist)
                        initialMetricsCaptured[0] = true;
                        System.out.printf("[%s] Initial metrics captured at episode %d (Update count: %d): PL=%.4f, VL=%.4f, E=%.4f%n",
                                agent.getClass().getSimpleName(), episode + 1, agent.getUpdateCount(), initialPolicyLoss[0], initialValueLoss[0], initialEntropy[0]);
                    }
                }
            }
        }

        System.out.printf("Training finished for %s. Total updates: %d%n", agent.getClass().getSimpleName(), agent.getUpdateCount());

        // Assertions
        assertTrue(agent.getUpdateCount() > 0, agent.getClass().getSimpleName() + " should have performed updates.");

        // Reward Assertions
        double avgRewardFirst50 = episodeRewards.subList(0, Math.min(50, episodeRewards.size()))
                .stream().mapToDouble(r -> r).average().orElse(Double.NEGATIVE_INFINITY);
        double avgRewardLast50 = episodeRewards.size() > 50 ?
                totalRewardLast50Episodes / Math.min(50, episodeRewards.size() - (NUM_EPISODES - 50)) :
                avgRewardFirst50;

        System.out.printf("[%s] Avg Reward First 50 Episodes: %.2f%n", agent.getClass().getSimpleName(), avgRewardFirst50);
        System.out.printf("[%s] Avg Reward Last 50 Episodes: %.2f%n", agent.getClass().getSimpleName(), avgRewardLast50);

        double expectedMinRewardImprovement = 5.0;
        assertTrue(avgRewardLast50 > avgRewardFirst50 + expectedMinRewardImprovement || episodeRewards.size() <= 50,
                String.format("[%s] Average reward in later episodes should be significantly higher. Last50: %.2f, First50: %.2f",
                        agent.getClass().getSimpleName(), avgRewardLast50, avgRewardFirst50));

        // Loss and Entropy Assertions
        var finalPolicyLossSummaryOpt = metrics.getSummary("policy_loss");
        assertTrue(finalPolicyLossSummaryOpt.isPresent(), agent.getClass().getSimpleName() + ": Policy loss should have been recorded.");
        finalPolicyLossSummaryOpt.ifPresent(s -> {
            System.out.printf("[%s] Initial Policy Loss: %.4f, Final Policy Loss: %.4f%n", agent.getClass().getSimpleName(), initialPolicyLoss[0], s.getMean());
            assertFalse(Double.isNaN(s.getMean()), agent.getClass().getSimpleName() + ": Policy loss mean should not be NaN.");
            if (!Double.isNaN(initialPolicyLoss[0]) && s.getCount() > 1) { // Ensure multiple readings and initial was captured
                 assertTrue(s.getMean() < initialPolicyLoss[0] || s.getMean() == initialPolicyLoss[0] /*can plateau*/,
                        String.format("[%s] Final policy loss (%.4f) should be less than or equal to initial policy loss (%.4f)",
                                agent.getClass().getSimpleName(), s.getMean(), initialPolicyLoss[0]));
            }
        });

        var finalValueLossSummaryOpt = metrics.getSummary("value_loss");
        finalValueLossSummaryOpt.ifPresent(s -> { // Value loss is optional
            System.out.printf("[%s] Initial Value Loss: %.4f, Final Value Loss: %.4f%n", agent.getClass().getSimpleName(), initialValueLoss[0], s.getMean());
            assertFalse(Double.isNaN(s.getMean()), agent.getClass().getSimpleName() + ": Value loss mean should not be NaN.");
            if (!Double.isNaN(initialValueLoss[0]) && s.getCount() > 1) {
                 assertTrue(s.getMean() < initialValueLoss[0] || s.getMean() == initialValueLoss[0],
                        String.format("[%s] Final value loss (%.4f) should be less than or equal to initial value loss (%.4f)",
                                agent.getClass().getSimpleName(), s.getMean(), initialValueLoss[0]));
            }
        });

        var finalEntropySummaryOpt = metrics.getSummary("entropy");
        finalEntropySummaryOpt.ifPresent(s -> { // Entropy is optional but good to track
            System.out.printf("[%s] Initial Entropy: %.4f, Final Entropy: %.4f%n", agent.getClass().getSimpleName(), initialEntropy[0], s.getMean());
            assertFalse(Double.isNaN(s.getMean()), agent.getClass().getSimpleName() + ": Entropy mean should not be NaN.");
            if (!Double.isNaN(initialEntropy[0]) && s.getCount() > 1) {
                // Entropy can sometimes increase initially then decrease, or fluctuate.
                // A strict decrease is not always guaranteed throughout, but a general trend is expected for many agents.
                // For now, let's check it's not drastically increasing, or simply that it's recorded.
                // A more robust check might compare early vs late phase averages.
                // For this iteration, we'll check if it's lower or roughly similar, allowing for some noise.
                // Consider a more nuanced check if this fails often due to normal fluctuations.
                 assertTrue(s.getMean() < initialEntropy[0] + 0.5 || s.getMean() == initialEntropy[0], // Allow slight increase or plateau
                        String.format("[%s] Final entropy (%.4f) should ideally be lower or similar to initial entropy (%.4f)",
                                agent.getClass().getSimpleName(), s.getMean(), initialEntropy[0]));
            }
        });
    }

//    private static VPGAgent agent(int i, int o, MetricCollector metricCollector) {
//        float s = 4;
//        var h = round(i * s);
//        var episodeLen = 2;
//        return new VPGAgent(new VPGAgentConfig(new int[]{
//            h, h
//        }, episodeLen), i, o);
//    }
}
