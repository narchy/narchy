package jcog.tensor.rl.pg3;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg3.configs.*;
import jcog.tensor.rl.pg3.memory.OnPolicyBuffer;
import jcog.tensor.rl.pg3.networks.GaussianPolicyNet;
import jcog.tensor.rl.pg3.networks.ValueNet;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.*;

class PPOAgentTest {

    private PPOAgent agent;
    private PPOAgentConfig config;
    private GaussianPolicyNet mockPolicyNet;
    private ValueNet mockValueNet;
    private OnPolicyBuffer mockMemory;

    // Define dimensions
    private final int stateDim = 4;
    private final int actionDim = 2;
    private final int episodeLength = 10;

    @BeforeEach
    void setUp() {
        // Mock configurations
        ActionConfig actionConfig = new ActionConfig(0.1, 0.5, "some_action_space"); // sigmaMin, sigmaMax
        NetworkConfig policyNetConfig = new NetworkConfig(new OptimizerConfig("Adam", 0.001,0,0,0,0), null, 0, List.of(64, 64), "relu", "tanh", false);
        NetworkConfig valueNetConfig = new NetworkConfig(new OptimizerConfig("Adam", 0.001,0,0,0,0), null, 0, List.of(64, 64), "relu", null, false);
        MemoryConfig memoryConfig = new MemoryConfig((long) episodeLength, 1L); // episodeLength, numEpisodes
        HyperparamConfig hyperparams = new HyperparamConfig(0.99, 0.95, 0.2, 10, 0.01, true, false); // gamma, lambda, ppoClip, epochs, entropyBonus, normalizeAdvantages, normalizeReturns

        config = new PPOAgentConfig(actionConfig, policyNetConfig, valueNetConfig, memoryConfig, hyperparams);

        // Mock networks and memory
        mockPolicyNet = Mockito.mock(GaussianPolicyNet.class);
        mockValueNet = Mockito.mock(ValueNet.class);
        mockMemory = Mockito.mock(OnPolicyBuffer.class); // Mocking memory to intercept calls

        // Stub network behaviors
        // GaussianPolicyNet.getDistribution returns an AgentUtils.GaussianDistribution
        // AgentUtils.GaussianDistribution then has sample() and logProb()
        // This is getting complex to mock deeply. For now, let's assume selectActionWithLogProb works.

        // Create agent instance, injecting mocks where possible or using real ones and spying.
        // For simplicity in this setup, we'll let the agent create its own networks and memory for now,
        // but ideally, they'd be injectable for easier mocking.
        // We can spy on the agent to verify interactions if needed.
        agent = new PPOAgent(config, stateDim, actionDim);
        // Replace the agent's actual memory with the mock for verification
        // This requires PPOAgent.memory to be non-final or have a setter, or use reflection.
        // For this example, let's assume we can modify it or the test focuses on what it calls.
        // agent.memory = mockMemory; // This would be ideal if memory field was accessible.
        // Instead, we will have to rely on the state of the agent's own memory if not mockable.
    }

    @Test
    void testApply_firstStep() {
        // Given: First step, so inputPrev and actionPrev are null
        double[] inputPrev = null;
        double[] actionPrev = null;
        float reward = 0.0f;
        double[] input = new double[stateDim]; // s_t
        java.util.Arrays.fill(input, 0.5);
        double[] actionNext = new double[actionDim]; // Output parameter

        // Mock the policy interaction for PPOAgent's selectActionWithLogProb
        // This part is tricky because selectActionWithLogProb is on the agent itself.
        // We would need to spy on the agent or refactor PPOAgent to make policy injectable.
        // For now, let's assume selectActionWithLogProb internally calls the policy and works.
        // We can verify that lastActionLogProb is set.

        PPOAgent spiedAgent = spy(agent);
        PPOAgent.ActionWithLogProb dummyActionWithLogProb = new PPOAgent.ActionWithLogProb(
            new double[]{0.1, -0.1}, Tensor.scalar( -0.5f)
        );
        // It's hard to mock selectActionWithLogProb if it's not broken out.
        // So we test the effect: lastActionLogProb should be set.

        // When: apply is called
        spiedAgent.apply(inputPrev, actionPrev, reward, input, actionNext);

        // Then:
        // 1. No experience should be recorded yet with the mockMemory approach (if it were injected)
        // verify(mockMemory, times(0)).add(any(Experience2.class));
        // Since mockMemory isn't easily injected without reflection/refactor, check agent's actual memory
        assertEquals(0, spiedAgent.memory.size(), "Memory should be empty on the first step before recording.");

        // 2. actionNext should be populated
        assertNotEquals(0.0, actionNext[0], "actionNext should be populated - x component");
        assertNotEquals(0.0, actionNext[1], "actionNext should be populated - y component");
        // A more specific check would require knowing the policy's output, which is hard with mocks here.

        // 3. agent.lastActionLogProb should be set by PPOAgent
        assertNotNull(spiedAgent.lastActionLogProb, "lastActionLogProb should be set for PPOAgent.");
        // We can't easily verify the exact value without deeper mocking of policy.getDistribution().logProb()

        // 4. Ensure selectActionWithLogProb was called (if using spy)
        // verify(spiedAgent, times(1)).selectActionWithLogProb(any(Tensor.class), anyBoolean());
        // This direct verification is problematic if selectActionWithLogProb is called by the apply method
        // of the *base* class, and spiedAgent.apply calls super.apply().
        // The effect (lastActionLogProb set, actionNext populated) is a more robust check here.
    }

     @Test
    void testApply_subsequentStep() {
        // Given: a previous step has occurred.
        // Simulate that lastActionLogProb was set in the previous step.
        agent.lastActionLogProb = Tensor.scalar(-0.25f); // Dummy log prob for a_t-1

        double[] inputPrev = new double[stateDim]; // s_t-1
        java.util.Arrays.fill(inputPrev, 0.1);
        double[] actionPrev = new double[actionDim]; // a_t-1
        java.util.Arrays.fill(actionPrev, 0.2);
        float reward = 1.0f; // r_t-1
        double[] input = new double[stateDim]; // s_t
        java.util.Arrays.fill(input, 0.3);
        double[] actionNext = new double[actionDim]; // a_t (output)


        // When: apply is called for the subsequent step
        // Using a spy to verify internal calls or state of the real memory
        PPOAgent spiedAgent = spy(agent);
        // If we could inject mockMemory:
        // doNothing().when(mockMemory).add(any(Experience2.class));

        spiedAgent.apply(inputPrev, actionPrev, reward, input, actionNext);

        // Then:
        // 1. Experience should have been recorded
        // If using mockMemory: verify(mockMemory, times(1)).add(any(Experience2.class));
        // Check agent's actual memory:
        assertEquals(1, spiedAgent.memory.size(), "One experience should be in memory.");
        Experience2 recordedExp = spiedAgent.memory.getAll().get(0);
        assertArrayEquals(inputPrev, recordedExp.state().array(), 0.001);
        assertArrayEquals(actionPrev, recordedExp.action(), 0.001);
        assertEquals(reward, recordedExp.reward(), 0.001);
        assertArrayEquals(input, recordedExp.nextState().array(), 0.001);
        assertFalse(recordedExp.done(), "Done should be false as per guidance.");
        assertNotNull(recordedExp.oldLogProb(), "oldLogProb (for a_t-1) should have been recorded.");
        assertEquals(-0.25f, recordedExp.oldLogProb().scalar(), 0.001);


        // 2. actionNext should be populated for s_t
        // (similar checks as first step, value depends on policy)
        assertNotEquals(0.0, actionNext[0], "actionNext should be populated for s_t");

        // 3. agent.lastActionLogProb should be updated for the new action a_t
        assertNotNull(spiedAgent.lastActionLogProb, "lastActionLogProb should be updated for a_t.");
        // Ensure it's a different tensor instance or value if policy is stochastic / state changed
        // For this test, just checking notNull is fine. A more complex test could mock policy output.
    }

    // TODO: Test for apply when input is null (terminal state)
    // TODO: Test for update triggering when buffer is full
    // TODO: Test training mode vs evaluation mode action selection (stochastic vs deterministic)
}

// Minimal stubs for NetworkConfig, OptimizerConfig etc. if not using full records
// For this example, assuming the actual records are available in the classpath.
// If not, they'd need to be defined or mocked.
// For GaussianPolicyNet, ValueNet, and Tensor.Optimizer, Mockito.mock is used.
// Tensor class itself is harder to mock effectively due to its fluent API and static methods.
// Tests often rely on using the real Tensor class.
```

This is a starting point. `GaussianPolicyNet` and `ValueNet` would ideally be injected into `PPOAgent` to allow for easier mocking of their `apply` and `getDistribution` methods. Without that, testing the precise numerical outputs or internal calls to these networks is harder. The current tests focus more on the interaction logic of `apply`.

I'll also create stubs for `ReinforceAgentTest.java` and `VPGAgentTest.java`.
