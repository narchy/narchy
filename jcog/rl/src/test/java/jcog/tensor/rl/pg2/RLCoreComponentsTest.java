package jcog.tensor.rl.pg2;

import jcog.tensor.Models;
import jcog.tensor.Tensor;
// import jcog.tensor.rl.pg.PolicyGradientModel; // Removed
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg.util.ReplayBuffer2;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
// import org.jetbrains.annotations.NotNull; // Removed
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
// import org.junit.jupiter.params.ParameterizedTest; // Removed
// import org.junit.jupiter.params.provider.EnumSource; // Removed

// import java.util.Random; // Removed
// import java.util.concurrent.atomic.AtomicBoolean; // Removed
// import java.util.function.Consumer; // Removed
// import java.util.stream.DoubleStream; // No longer needed after removing testPGAgent

// import static jcog.tensor.rl.pg2.DDPGStrategy.ddpgStrategy; // Ensure fully removed
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("RL Core Components Test Suite")
class RLCoreComponentsTest {

    private static final int INPUTS = 4; // Kept for potential use in component tests
    private static final int OUTPUTS = 2; // Kept for potential use in component tests
    private static final double EPSILON = 1e-6; // Kept for potential use in component tests
    // private static final Random random = new Random(12345); // Will be removed if no tests use it.

    // TestAlgoType enum removed as it was used by createTestModel and testBuildAndRunModel.
    // PGBuilder related helper methods (createTestModel, xxxStrategy methods) removed.
    // PGBuilder related config fields (validHyperparams etc.) removed with ComponentConfigTests.
    // L4 End-to-End Model Build and Run (testBuildAndRunModel) removed.
    // L3 Component Configuration and Validation (ComponentConfigTests) removed.

    // ===================================================================================
    // Model Architecture Tests
    // ===================================================================================
    @Nested
    @DisplayName("L2: PG3 Network Architectures") // Updated display name
    class ModelArchitectureTests {
        @Test
        @DisplayName("ValueNet (pg3) API")
        void testValueNetApi() {
            var config = new NetworkConfig(0.001f, 10); // learning rate, hidden layers
            // INPUTS is a static final int from the outer class RLCoreComponentsTest
            var valueNet = new jcog.tensor.rl.pg3.networks.ValueNet(config, INPUTS);

            Tensor stateInput = Tensor.randGaussian(1, INPUTS);
            Tensor valueOutput = assertDoesNotThrow(() -> valueNet.apply(stateInput), "ValueNet.apply(state) should not throw");
            assertNotNull(valueOutput);
            assertEquals(1, valueOutput.shape()[0], "Batch size of value output should be 1");
            assertEquals(1, valueOutput.shape()[1], "Output dimension of value V(s) should be 1");
        }

        @Test
        @DisplayName("GaussianPolicyNet (pg3) API")
        void testGaussianPolicyNetApi() {
            var config = new NetworkConfig(0.001f, 10); // learning rate, hidden layers
            // INPUTS and OUTPUTS are static final ints from the outer class
            var policyNet = new jcog.tensor.rl.pg3.networks.GaussianPolicyNet(config, INPUTS, OUTPUTS);

            Tensor stateInput = Tensor.randGaussian(1, INPUTS);

            // Test apply() method - it should return the mean (mu)
            Tensor muOutput = assertDoesNotThrow(() -> policyNet.apply(stateInput), "GaussianPolicyNet.apply(state) should not throw");
            assertNotNull(muOutput);
            assertEquals(1, muOutput.shape()[0], "Batch size of mu output should be 1");
            assertEquals(OUTPUTS, muOutput.shape()[1], "Output dimension of mu should match action dimension");

            // Test getDistribution() method
            jcog.tensor.rl.pg3.util.AgentUtils.GaussianDistribution dist =
                assertDoesNotThrow(() -> policyNet.getDistribution(stateInput, 0.1f, 1.0f), "getDistribution should not throw");
            assertNotNull(dist);
            assertNotNull(dist.mu());
            assertNotNull(dist.sigma());
            assertEquals(OUTPUTS, dist.mu().shape()[1]);
            assertEquals(OUTPUTS, dist.sigma().shape()[1]);
        }
    }

    // ===================================================================================
    // Component and Utility Tests
    // ===================================================================================
    @Nested
    @DisplayName("L1: Low-Level Components")
    class ComponentTests {
        @Test
        @DisplayName("ReplayBuffer2 (pg.util) correctly stores, samples, and cycles")
        void testReplayBuffer() {
            var buffer = new ReplayBuffer2(5);
            for (var i = 0; i < 7; i++) {
                // For Experience2, ensure all required fields are non-null if its constructor enforces it.
                // Assuming null action, null nextState, null oldLogProb are acceptable for this test's purpose.
                buffer.add(new Experience2(Tensor.scalar(i), new double[]{0}, 0, null, false, null));
            }
            Assertions.assertEquals(5, buffer.size(), "Buffer should be at full capacity");
            var all = buffer.getAll();
            assertFalse(all.stream().anyMatch(e -> e.state().scalar() < 2.0), "Oldest elements should be overwritten");
            assertTrue(all.stream().anyMatch(e -> e.state().scalar() == 6.0), "Newest element should be present");

            var sample = buffer.sample(3);
            assertEquals(3, sample.size(), "Sample should have the requested batch size");

            buffer.clear();
            Assertions.assertEquals(0, buffer.size(), "Buffer should be empty after clear");
        }

        @Test
        @DisplayName("OnPolicyBuffer (pg3.memory) correctly stores and clears")
        void testPg3OnPolicyBuffer() {
            // Test jcog.tensor.rl.pg3.memory.OnPolicyBuffer
            var buffer = new jcog.tensor.rl.pg3.memory.OnPolicyBuffer(5);
            for (var i = 0; i < 7; i++) {
                buffer.add(new Experience2(Tensor.scalar(i), new double[]{0}, 0, null, false, null));
            }
            // pg3.memory.OnPolicyBuffer does not automatically discard if full, it just stops adding.
            // This behavior is different from the old test's expectation for OnPolicyEpisodeBuffer.
            assertEquals(5, buffer.size(), "Buffer should be at capacity and not exceed it.");

            var all = buffer.getAll();
            // Check that the first 5 elements are 0,1,2,3,4 and elements 5,6 were not added.
            assertTrue(all.stream().anyMatch(e -> e.state().scalar() == 0.0), "Element 0 should be present");
            assertTrue(all.stream().anyMatch(e -> e.state().scalar() == 4.0), "Element 4 should be present");
            assertFalse(all.stream().anyMatch(e -> e.state().scalar() > 4.0), "Elements beyond capacity should be ignored");


            buffer.clear();
            assertEquals(0, buffer.size(), "Buffer should be empty after clear");
        }

        @Test
        @DisplayName("AgentUtils (pg3.util) softUpdate correctly blends parameters")
        void testAgentUtilsSoftUpdate() {
            // Using Models.Layers as concrete UnaryOperator<Tensor> for testing
            var sourceNet = new Models.Layers(null, null, true, 2, 2);
            var targetNet = new Models.Layers(null, null, true, 2, 2);

            Tensor.parameters(sourceNet).forEach(p -> p.setData(Tensor.ones(p.shape())));
            Tensor.parameters(targetNet).forEach(p -> p.setData(Tensor.zeros(p.shape())));

            // Test with pg3.util.AgentUtils
            jcog.tensor.rl.pg3.util.AgentUtils.softUpdate(sourceNet, targetNet, 0.1f);

            Tensor.parameters(targetNet).forEach(p -> {
                assertTrue(p.sub(Tensor.scalar(0.1)).sumAbs().scalar() < EPSILON, "Target params should be 0.1");
            });

            jcog.tensor.rl.pg3.util.AgentUtils.hardUpdate(sourceNet, targetNet);

            Tensor.parameters(targetNet).forEach(p -> {
                assertTrue(p.sub(Tensor.scalar(1.0)).sumAbs().scalar() < EPSILON, "Target params should be 1.0 after hard update");
            });
        }

        // testPGAgent is removed as it tests a wrapper around pg2 strategies,
        // and pg3 agents are tested directly.
        // @Test
        // @DisplayName("PGAgent wraps model and processes data")
        // void testPGAgent() { ... }
    }

    // L5 RLPerformanceTests section removed.
}