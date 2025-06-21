package jcog.tensor.rl.pg;

import jcog.tensor.Models;
import jcog.tensor.Tensor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("PGBuilder Test Suite")
class PGBuilderTest {

    private static final int INPUTS = 4;
    private static final int OUTPUTS = 2;
    private static final double EPSILON = 1e-6;
    private static final Random random = new Random(12345);

    private static PGBuilder createTestBuilder(PGBuilder.Algorithm algo) {
        PGBuilder builder = new PGBuilder(INPUTS, OUTPUTS).algorithm(algo);
        builder.policy(p -> p.hiddenLayers(16, 16).optimizer(o -> o.learningRate(1e-3f)));
        switch (algo) {
            case PPO, VPG, REINFORCE ->
                    builder.value(v -> v.hiddenLayers(16, 16).optimizer(o -> o.learningRate(1e-3f)));
            case SAC -> builder.qNetworks(q -> q.hiddenLayers(16, 16).optimizer(o -> o.learningRate(1e-3f)));
            case DDPG -> builder.value(v -> v.hiddenLayers(16, 16).optimizer(o -> o.learningRate(1e-3f)))
                    .action(a -> a.distribution(PGBuilder.ActionConfig.Distribution.DETERMINISTIC));
        }
        builder.hyperparams(h -> h.epochs(1).policyUpdateFreq(1));
        builder.memory(m -> m
                .episodeLength(32)
                .replayBuffer(rb -> rb.capacity(128).batchSize(16))
        );
        return builder;
    }

    private static AlgorithmStrategy getStrategy(AbstrPG model) {
        return ((PolicyGradientModel) model).strategy;
    }

    private PGBuilder.MemoryConfig getMemoryConfig(AlgorithmStrategy strategy) {
        return strategy.getMemoryConfig();
    }

    private UnaryOperator<Tensor> getPolicyNetwork(AlgorithmStrategy strategy) {
        return strategy.getPolicy();
    }

    // ===================================================================================
    // End-to-End and Integration Tests
    // ===================================================================================
    @DisplayName("L4: End-to-End Model Build and Run")
    @ParameterizedTest(name = "Algorithm: {0}")
    @EnumSource(PGBuilder.Algorithm.class)
    void testBuildAndRunModel(PGBuilder.Algorithm algo) {
        assertDoesNotThrow(() -> {
            AbstrPG model = createTestBuilder(algo).build();
            assertNotNull(model, "Model should be built successfully for " + algo);

            AlgorithmStrategy strategy = getStrategy(model);
            var pn = getPolicyNetwork(strategy);
            Tensor initialParam = Tensor.parameters(pn).findFirst().orElseThrow();
            Tensor initialParamCopy = initialParam.detachCopy();

            int steps = 100;
            int doneInterval = 20;
            for (int i = 0; i < steps; i++) {
                double[] state = random.doubles(INPUTS, -1.0, 1.0).toArray();
                double reward = random.nextDouble(-1.0, 1.0);
                boolean done = (i > 0 && i % doneInterval == 0);

                double[] action = model.act(state, reward, done);

                assertEquals(OUTPUTS, action.length, "Action dimension mismatch");
                for (double v : action) {
                    assertTrue(v >= -1.0 && v <= 1.01, "Action out of range: " + v);
                }
            }

            if (strategy.getUpdateSteps() > 0) {
                Tensor finalParam = Tensor.parameters(pn).findFirst().orElseThrow();
                assertFalse(initialParamCopy.equals(finalParam), "Policy parameters did not change for " + algo + ", training likely failed.");
            } else if (algo.isOffPolicy()) {
                fail("Off-policy algorithm " + algo + " performed 0 updates, which is unexpected.");
            }

            if (algo.isOffPolicy()) {
                assertTrue(strategy.getUpdateSteps() > 0, "Off-policy model should have performed updates");
                assertTrue(strategy.getMemory().size() > 0, "Off-policy buffer should retain samples");
            } else {
                // --- FIX: Correct calculation for episodic updates ---
                // Updates occur when `done` is true, which happens every `doneInterval` steps.
                // The on-policy buffer is cleared after each update.
                int expectedUpdates = (steps - 1) / doneInterval;
                // The memory size is the number of steps accumulated since the last `done` flag.
                int expectedMemorySize = (steps - 1) % doneInterval;

                assertEquals(expectedUpdates, strategy.getUpdateSteps(), "On-policy model update count mismatch for " + algo);
                assertEquals(expectedMemorySize, strategy.getMemory().size(), "On-policy buffer size mismatch for " + algo);
            }
        });
    }

    // ===================================================================================
    // RL Performance Tests on Synthetic Tasks
    // ===================================================================================
    @Nested
    @DisplayName("L5: RL Performance on Synthetic Tasks")
    class RLPerformanceTests {

        private static final int PERF_INPUTS = 2;
        private static final int PERF_OUTPUTS = 2;

        private PGBuilder createPerformanceTestBuilder(PGBuilder.Algorithm algo) {
            PGBuilder builder = new PGBuilder(PERF_INPUTS, PERF_OUTPUTS).algorithm(algo);
            int d = 32; // A bit more capacity for the learning task
            builder.policy(p -> p.hiddenLayers(d, d).optimizer(o -> o.learningRate(3e-4f)));
            switch (algo) {
                case PPO, VPG, REINFORCE ->
                        builder.value(v -> v.hiddenLayers(d, d).optimizer(o -> o.learningRate(1e-3f)));
                case SAC -> builder.qNetworks(q -> q.hiddenLayers(d, d).optimizer(o -> o.learningRate(1e-3f)));
                case DDPG -> builder.value(v -> v.hiddenLayers(d, d).optimizer(o -> o.learningRate(1e-3f)))
                        .action(a -> a.distribution(PGBuilder.ActionConfig.Distribution.DETERMINISTIC));
            }
            // A lower gamma emphasizes immediate rewards, which is suitable for this one-step task.
            builder.hyperparams(h -> h.epochs(4).policyUpdateFreq(1).gamma(0.9f));
            builder.memory(m -> m
                    .episodeLength(1) // For on-policy, update after each step/episode
                    .replayBuffer(rb -> rb.capacity(10000).batchSize(128))
            );
            return builder;
        }

        private double evaluatePerformance(AbstrPG model, int episodes) {
            double totalReward = 0;
            Random envRandom = new Random(42); // Use a fixed seed for consistent evaluation

            for (int e = 0; e < episodes; e++) {
                double[] state = envRandom.doubles(PERF_INPUTS, -1.0, 1.0).toArray();
                // Use deterministic action for evaluation
                double[] action = model._action(Tensor.row(state), true);
                double mse = 0;
                for (int i = 0; i < PERF_OUTPUTS; i++) {
                    mse += Math.pow(state[i] - action[i], 2);
                }
                totalReward += (-mse / PERF_OUTPUTS); // Reward is negative MSE
            }
            return totalReward / episodes;
        }

        @ParameterizedTest(name = "Algorithm: {0}")
        @EnumSource(value = PGBuilder.Algorithm.class, names = {"REINFORCE", "PPO", "SAC", "DDPG"})
        @DisplayName("Learns to match state vector as action")
        void testLearningOnMatchingTask(PGBuilder.Algorithm algo) {
            AbstrPG model = createPerformanceTestBuilder(algo).build();

            int totalSteps = 18000;
            int evalInterval = 1000;
            int evalEpisodes = 50;
            List<Double> rewardsHistory = new ArrayList<>();
            var trainRandom = new Random(123);

            // Initial performance
            rewardsHistory.add(evaluatePerformance(model, evalEpisodes));

            double[] state = trainRandom.doubles(PERF_INPUTS, -1.0, 1.0).toArray();
            double lastReward = 0.0;

            for (int step = 1; step <= totalSteps; step++) {

                // --- FIX: Signal `done=true` after each step to create 1-step episodes. ---
                // This is crucial for on-policy algorithms like REINFORCE to trigger their update logic.
                // The `act` call processes the reward from the *previous* step's action.
                double[] action = model.act(state, lastReward, true);

                // Calculate reward for the action we just took.
                double mse = 0;
                for (int i = 0; i < PERF_OUTPUTS; i++) {
                    mse += Math.pow(state[i] - action[i], 2);
                }
                lastReward = -mse / PERF_OUTPUTS;

                // Get a new state for the next independent episode
                state = trainRandom.doubles(PERF_INPUTS, -1.0, 1.0).toArray();

                if (step % evalInterval == 0) {
                    rewardsHistory.add(evaluatePerformance(model, evalEpisodes));
                }
            }

            String learningCurve = rewardsHistory.stream()
                    .map(r -> String.format("%.4f", r))
                    .collect(Collectors.joining(", "));

            System.out.printf("Performance for %s: [%s]%n", algo, learningCurve);

            // Assert that the agent has learned. Check if the avg of the last 2 evals is better than the first.
            int historySize = rewardsHistory.size();
            double initialPerf = rewardsHistory.get(0);
            double finalPerf = (rewardsHistory.get(historySize - 1) + rewardsHistory.get(historySize - 2)) / 2.0;

            assertTrue(finalPerf > initialPerf,
                    "Final performance should be better than initial. Learning curve: " + learningCurve);
            assertTrue(finalPerf > -0.1,
                    "Model " + algo + " failed to learn the task adequately. Final avg reward: " + finalPerf + ". Learning curve: " + learningCurve);
        }
    }


    // ===================================================================================
    // Configuration and Validation Tests
    // ===================================================================================
    @Nested
    @DisplayName("L3: Builder Configuration and Validation")
    class BuilderConfigTests {
        @Test
        @DisplayName("Fails for PPO/VPG without a value network")
        void testPPO_failsWithoutValueNetwork() {
            PGBuilder builder = new PGBuilder(INPUTS, OUTPUTS)
                    .algorithm(PGBuilder.Algorithm.PPO)
                    .policy(p -> p.hiddenLayers(10));
            var e = assertThrows(IllegalStateException.class, builder::build);
            assertTrue(e.getMessage().contains("requires a value network"));
        }

        @Test
        @DisplayName("Fails for DDPG without a critic (value) network")
        void testDDPG_failsWithoutCriticNetwork() {
            PGBuilder builder = new PGBuilder(INPUTS, OUTPUTS)
                    .algorithm(PGBuilder.Algorithm.DDPG)
                    .policy(p -> p.hiddenLayers(10));
            var e = assertThrows(IllegalStateException.class, builder::build);
            assertTrue(e.getMessage().contains("requires a critic (value) network"));
        }

        @Test
        @DisplayName("Fails for SAC without Q-networks")
        void testSAC_failsWithoutQNetworks() {
            PGBuilder builder = new PGBuilder(INPUTS, OUTPUTS)
                    .algorithm(PGBuilder.Algorithm.SAC)
                    .policy(p -> p.hiddenLayers(10));
            var e = assertThrows(IllegalStateException.class, builder::build);
            assertTrue(e.getMessage().contains("requires two Q-networks"));
        }

        @Test
        @DisplayName("Fails for DDPG with non-deterministic action distribution")
        void testDDPG_failsWithInvalidDistribution() {
            // --- FIX: This test now asserts that an exception is thrown, ---
            // --- matching the corrected, stricter validation logic. ---
            var builder = createTestBuilder(PGBuilder.Algorithm.DDPG)
                    // Attempt to configure with the wrong distribution type
                    .action(a -> a.distribution(PGBuilder.ActionConfig.Distribution.GAUSSIAN));

            // Assert that the build fails with the expected exception
            var e = assertThrows(IllegalArgumentException.class, builder::build);
            assertTrue(e.getMessage().contains("DETERMINISTIC action distribution"));
        }

        @Test
        @DisplayName("Fails with invalid hyperparameters")
        void testInvalidHyperparameters() {
            assertThrows(IllegalArgumentException.class, () -> createTestBuilder(PGBuilder.Algorithm.PPO).hyperparams(h -> h.gamma(1.1f)).build());
            assertThrows(IllegalArgumentException.class, () -> createTestBuilder(PGBuilder.Algorithm.PPO).hyperparams(h -> h.lambda(-0.1f)).build());
            assertThrows(IllegalArgumentException.class, () -> createTestBuilder(PGBuilder.Algorithm.PPO).hyperparams(h -> h.tau(2.0f)).build());
            assertThrows(IllegalArgumentException.class, () -> createTestBuilder(PGBuilder.Algorithm.PPO).hyperparams(h -> h.policyLR(0f)).build());
        }

        @Test
        @DisplayName("ActionFilter correctly modifies action")
        void testActionFilter() {
            AtomicBoolean filterWasCalled = new AtomicBoolean(false);
            AbstrPG model = createTestBuilder(PGBuilder.Algorithm.PPO)
                    .actionFilter(a -> {
                        a[0] = 999.0;
                        filterWasCalled.set(true);
                    })
                    .build();

            double[] action = model.act(random.doubles(INPUTS).toArray(), 0.0, false);
            assertTrue(filterWasCalled.get(), "Action filter was not called");
            assertEquals(999.0, action[0], EPSILON, "Action filter did not modify the action");
        }

    }

    // ===================================================================================
    // Model Architecture Tests
    // ===================================================================================
    @Nested
    @DisplayName("L2: Model Architectures")
    class ModelArchitectureTests {
        @Test
        @DisplayName("QNetwork rejects single tensor input")
        void testQNetworkApi() {
            var config = new PGBuilder.NetworkConfig.Builder().hiddenLayers(10).build(0.001f);
            var qNet = new PGBuilder.QNetwork(config, INPUTS, OUTPUTS);
            assertThrows(UnsupportedOperationException.class, () -> qNet.apply(Tensor.randGaussian(1, INPUTS)));
            assertDoesNotThrow(() -> qNet.apply(Tensor.randGaussian(1, INPUTS), Tensor.randGaussian(1, OUTPUTS)));
        }

        @Test
        @DisplayName("GaussianPolicy rejects direct apply call")
        void testGaussianPolicyApi() {
            var config = new PGBuilder.NetworkConfig.Builder().hiddenLayers(10).build(0.001f);
            var policy = new PGBuilder.GaussianPolicy(config, INPUTS, OUTPUTS);
            assertThrows(UnsupportedOperationException.class, () -> policy.apply(Tensor.randGaussian(1, INPUTS)));
            assertDoesNotThrow(() -> policy.getDistribution(Tensor.randGaussian(1, INPUTS), 0.1f, 1.0f));
        }
    }

    // ===================================================================================
    // Component and Utility Tests
    // ===================================================================================
    @Nested
    @DisplayName("L1: Low-Level Components")
    class ComponentTests {
        @Test
        @DisplayName("ReplayBuffer correctly stores, samples, and cycles")
        void testReplayBuffer() {
            var buffer = new ReplayBuffer(5);
            for (int i = 0; i < 7; i++) {
                buffer.add(new Experience(Tensor.scalar(i), null, 0, null, false));
            }
            assertEquals(5, buffer.size(), "Buffer should be at full capacity");
            List<Experience> all = buffer.getAll();
            assertFalse(all.stream().anyMatch(e -> e.state().scalar() < 2.0), "Oldest elements should be overwritten");
            assertTrue(all.stream().anyMatch(e -> e.state().scalar() == 6.0), "Newest element should be present");

            List<Experience> sample = buffer.sample(3);
            assertEquals(3, sample.size(), "Sample should have the requested batch size");

            buffer.clear();
            assertEquals(0, buffer.size(), "Buffer should be empty after clear");
        }

        @Test
        @DisplayName("OnPolicyEpisodeBuffer correctly stores and clears")
        void testOnPolicyBuffer() {
            var buffer = new OnPolicyEpisodeBuffer(5);
            for (int i = 0; i < 7; i++) {
                buffer.add(new Experience(Tensor.scalar(i), null, 0, null, false));
            }
            assertEquals(5, buffer.size(), "Buffer should not exceed capacity");
            List<Experience> all = buffer.getAll();
            assertFalse(all.stream().anyMatch(e -> e.state().scalar() > 4.0), "Elements beyond capacity should be ignored");

            buffer.clear();
            assertEquals(0, buffer.size(), "Buffer should be empty after clear");
        }

        @Test
        @DisplayName("RLUtils.softUpdate correctly blends parameters")
        void testSoftUpdate() {
            var sourceNet = new Models.Layers(null, null, true, 2, 2);
            var targetNet = new Models.Layers(null, null, true, 2, 2);

            Tensor.parameters(sourceNet).forEach(p -> p.setData(Tensor.ones(p.shape())));
            Tensor.parameters(targetNet).forEach(p -> p.setData(Tensor.zeros(p.shape())));

            RLUtils.softUpdate(sourceNet, targetNet, 0.1f);

            Tensor.parameters(targetNet).forEach(p -> {
                assertTrue(p.sub(Tensor.scalar(0.1)).sumAbs() < EPSILON, "Target params should be 0.1");
            });

            RLUtils.hardUpdate(sourceNet, targetNet);

            Tensor.parameters(targetNet).forEach(p -> {
                assertTrue(p.sub(Tensor.scalar(1.0)).sumAbs() < EPSILON, "Target params should be 1.0 after hard update");
            });
        }

        @Test
        @DisplayName("PGAgent wraps model and processes data")
        void testPGAgent() {
            AbstrPG model = createTestBuilder(PGBuilder.Algorithm.PPO).build();
            AbstrPG.PGAgent agent = model.agent();
            assertNotNull(agent);

            double[] input = new double[INPUTS];
            double[] action = new double[OUTPUTS];

            agent.apply(null, null, 1.0f, input, action);

            assertTrue(DoubleStream.of(action).anyMatch(v -> v != 0.0), "Agent did not populate action array");
        }
    }
}