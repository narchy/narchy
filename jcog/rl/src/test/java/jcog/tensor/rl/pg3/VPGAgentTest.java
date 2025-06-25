package jcog.tensor.rl.pg3;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg3.configs.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.spy;

class VPGAgentTest {

    private VPGAgent agent;
    private VPGAgentConfig config;

    private final int stateDim = 2;
    private final int actionDim = 2;
    private final int episodeLength = 6;

    @BeforeEach
    void setUp() {
        ActionConfig actionConfig = new ActionConfig(0.1, 0.5, "Continuous");
        NetworkConfig policyNetConfig = new NetworkConfig(new OptimizerConfig("Adam", 0.0003,0,0,0,0), null, 0, List.of(48), "tanh", "tanh", false);
        NetworkConfig valueNetConfig = new NetworkConfig(new OptimizerConfig("Adam", 0.001,0,0,0,0), null, 0, List.of(48), "tanh", null, false);
        MemoryConfig memoryConfig = new MemoryConfig((long) episodeLength, 1L);
        HyperparamConfig hyperparams = new HyperparamConfig(0.98, 0, 0, 1, 0.01, true, true); // gamma, lambda (not used by VPG), ppoClip (not used), epochs (1 for VPG), entropyBonus, normalizeAdvantages, normalizeReturns

        config = new VPGAgentConfig(actionConfig, policyNetConfig, valueNetConfig, memoryConfig, hyperparams);
        agent = new VPGAgent(config, stateDim, actionDim);
    }

    @Test
    void testApply_firstStep_vpg() {
        double[] inputPrev = null;
        double[] actionPrev = null;
        float reward = 0.0f;
        double[] input = new double[stateDim]; Arrays.fill(input, 0.7);
        double[] actionNext = new double[actionDim];

        VPGAgent spiedAgent = spy(agent);
        spiedAgent.apply(inputPrev, actionPrev, reward, input, actionNext);

        assertEquals(0, spiedAgent.memory.size());
        assertTrue(Arrays.stream(actionNext).anyMatch(x -> x != 0.0));
        assertNull(spiedAgent.lastActionLogProb, "lastActionLogProb should be null for VPGAgent.");
    }

    @Test
    void testApply_subsequentStep_vpg() {
        agent.lastActionLogProb = null;

        double[] inputPrev = new double[stateDim]; Arrays.fill(inputPrev, -0.5);
        double[] actionPrev = new double[actionDim]; Arrays.fill(actionPrev, 0.1);
        float reward = -0.5f;
        double[] input = new double[stateDim]; Arrays.fill(input, 0.0);
        double[] actionNext = new double[actionDim];

        VPGAgent spiedAgent = spy(agent);
        spiedAgent.apply(inputPrev, actionPrev, reward, input, actionNext);

        assertEquals(1, spiedAgent.memory.size());
        Experience2 recordedExp = spiedAgent.memory.getAll().get(0);
        assertArrayEquals(inputPrev, recordedExp.state().array(), 0.001);
        assertNull(recordedExp.oldLogProb(), "oldLogProb should be null for VPG.");

        assertTrue(Arrays.stream(actionNext).anyMatch(x -> x != 0.0));
        assertNull(spiedAgent.lastActionLogProb, "lastActionLogProb should remain null for VPGAgent.");
    }

    @Test
    void testUpdateTriggeredWhenBufferFull_vpg() {
        VPGAgent spiedAgent = spy(agent);
        Mockito.doNothing().when(spiedAgent).update(Mockito.anyLong());

        for (int i = 0; i < episodeLength; i++) {
            double[] sPrev = (i == 0) ? null : new double[stateDim];
            double[] aPrev = (i == 0) ? null : new double[actionDim];
            if (sPrev != null) Arrays.fill(sPrev, 0.1 * i);
            if (aPrev != null) Arrays.fill(aPrev, 0.2 * i);

            double[] sCurr = new double[stateDim]; Arrays.fill(sCurr, 0.3 * i);
            double[] aNext = new double[actionDim];
            spiedAgent.apply(sPrev, aPrev, (float)(0.1 * i), sCurr, aNext);
        }

        Mockito.verify(spiedAgent, Mockito.times(1)).update(Mockito.anyLong());
        assertEquals(0, spiedAgent.memory.size(), "Memory should be cleared after update.");
    }
}
