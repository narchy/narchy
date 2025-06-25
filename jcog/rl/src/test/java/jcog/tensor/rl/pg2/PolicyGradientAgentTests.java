package jcog.tensor.rl.pg2;

import jcog.agent.Agent;
import jcog.tensor.Tensor;
import jcog.tensor.rl.RLAgentTestBase;
import jcog.tensor.rl.env.ContinuousSeekEnv;
import jcog.tensor.rl.env.MatchingEnv;
import jcog.tensor.rl.env.SyntheticEnv;
import jcog.tensor.rl.pg.*;
// PG3 imports
import jcog.tensor.rl.pg3.PolicyGradientAgent;
import jcog.tensor.rl.pg3.ReinforceAgent;
import jcog.tensor.rl.pg3.VPGAgent;
import jcog.tensor.rl.pg3.PPOAgent;
import jcog.tensor.rl.pg3.configs.*;
import jcog.tensor.rl.pg.util.Experience2; // Already exists but good to note its use for pg3
import jcog.tensor.rl.pg3.networks.GaussianPolicyNet; // For focused tests

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer; // For focused tests
import java.util.stream.Collectors;


import static org.junit.jupiter.api.Assertions.*; // More specific assertions for unit tests

public class PolicyGradientAgentTests extends RLAgentTestBase {

    // Local enum for test parameterization, similar to PGBuilderTest.TestAlgoType
    // This helps in replacing PGBuilder.Algorithm
    enum TestAlgoType {
        REINFORCE, PPO, SAC, DDPG
        // VPG tests will use PPO strategy as per old PGBuilder mapping
    }

    int brains = 4; // Used for hidden layer size calculation
    float gamma = 0.9f;
    int iter = 1 * 1024;
    int trials = 3;
    int episodeLen = 1; // For on-policy agents, this means update after every step if done

    // @Deprecated // Uses old Reinforce class, new tests should use PGBuilder.ReinforceStrategy
    @Test void REINFORCE_Matching1() {
        var env = new MatchingEnv(1);
        int stateDim = env.stateDimension();
        int actionDim = env.actionDimension();
        int hiddenSize = stateDim * brains;

        var policyNetConfig = new NetworkConfig(1e-3f, hiddenSize).withActivation(Tensor.RELU);
        var hyperparamConfig = new HyperparamConfig().withGamma(this.gamma).withEntropyBonus(0.001f); // Added small entropy bonus
        var actionConfig = new ActionConfig(); // Defaults are fine
        var memoryConfig = new MemoryConfig((long) this.episodeLen); // episodeLen for on-policy

        var agentConfig = new ReinforceAgentConfig(policyNetConfig, hyperparamConfig, actionConfig, memoryConfig);
        var pg3Agent = new ReinforceAgent(agentConfig, stateDim, actionDim);

        evalPg3Agent(pg3Agent, env);
    }

    // @Deprecated // Uses old Reinforce class, new tests should use PGBuilder.ReinforceStrategy
    @Test void REINFORCE_Matching2() {
        var env = new MatchingEnv(2); // Difference: MatchingEnv(2)
        int stateDim = env.stateDimension();
        int actionDim = env.actionDimension();
        int hiddenSize = stateDim * brains;

        var policyNetConfig = new NetworkConfig(1e-3f, hiddenSize).withActivation(Tensor.RELU);
        var hyperparamConfig = new HyperparamConfig().withGamma(this.gamma).withEntropyBonus(0.001f);
        var actionConfig = new ActionConfig();
        var memoryConfig = new MemoryConfig((long) this.episodeLen);

        var agentConfig = new ReinforceAgentConfig(policyNetConfig, hyperparamConfig, actionConfig, memoryConfig);
        var pg3Agent = new ReinforceAgent(agentConfig, stateDim, actionDim);

        evalPg3Agent(pg3Agent, env);
    }

    // @Disabled("Needs refactor to pg3.VPGAgent")
    // @Deprecated // Uses old VPG class, new tests should use PGBuilder.PPOStrategy or a dedicated VPG-like strategy
    @Test void VPG_Matching1() {
        var env = new MatchingEnv(1);
        int stateDim = env.stateDimension();
        int actionDim = env.actionDimension();
        int hiddenSize = stateDim * brains;

        var policyNetConfig = new NetworkConfig(1e-3f, hiddenSize).withActivation(Tensor.RELU);
        var valueNetConfig = new NetworkConfig(1e-3f, hiddenSize).withActivation(Tensor.RELU); // For VPG's baseline
        var hyperparamConfig = new HyperparamConfig()
            .withGamma(this.gamma)
            .withEntropyBonus(0.001f)
            .withNormalizeAdvantages(true); // VPG typically uses advantage normalization
        var actionConfig = new ActionConfig();
        var memoryConfig = new MemoryConfig((long) this.episodeLen);

        var agentConfig = new VPGAgentConfig(policyNetConfig, valueNetConfig, hyperparamConfig, actionConfig, memoryConfig);
        var pg3Agent = new VPGAgent(agentConfig, stateDim, actionDim);

        evalPg3Agent(pg3Agent, env);
    }

    @Disabled("StreamAC is not part of pg3 refactor focus")
    @Test void StreamAC_Matching1() { // Assuming StreamAC is a current/valid agent not part of PGBuilder refactor
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new StreamAC(e.stateDimension(), e.actionDimension(), hidden, hidden);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Disabled("DDPG is not in pg3, needs different handling or removal from this test suite if focusing on pg3")
    @Deprecated // Uses old DDPG class, new tests should use PGBuilder.DDPGStrategy
    @Test void DDPG_Matching() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new DDPG(e.stateDimension(), e.actionDimension(), hidden, hidden);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    // @Disabled("Needs refactor to pg3.ReinforceAgent")
    // @Deprecated // Uses old Reinforce class
    @Test void REINFORCE_Target1D() {
        var env = new ContinuousSeekEnv();
        int stateDim = env.stateDimension();
        int actionDim = env.actionDimension();
        int hiddenSize = stateDim * brains * 2; // Increased hidden size slightly for potentially harder task

        var policyNetConfig = new NetworkConfig(1e-3f, hiddenSize, hiddenSize).withActivation(Tensor.RELU);
        var hyperparamConfig = new HyperparamConfig().withGamma(this.gamma).withEntropyBonus(0.01f); // Higher entropy for exploration
        var actionConfig = new ActionConfig();
        var memoryConfig = new MemoryConfig((long) this.episodeLen * 5); // Longer episodes for continuous tasks

        var agentConfig = new ReinforceAgentConfig(policyNetConfig, hyperparamConfig, actionConfig, memoryConfig);
        var pg3Agent = new ReinforceAgent(agentConfig, stateDim, actionDim);

        // Use more iterations for continuous seek tasks
        evalPg3Agent(pg3Agent, env, iter * 4, trials, random);
    }

    // @Disabled("Needs refactor to pg3.ReinforceAgent")
    // @Deprecated // Uses old Reinforce class
    @Test void REINFORCE_Target2D() {
        var env = ContinuousSeekEnv.W2D();
        int stateDim = env.stateDimension();
        int actionDim = env.actionDimension();
        int hiddenSize = stateDim * brains * 2; // Increased hidden size

        var policyNetConfig = new NetworkConfig(1e-3f, hiddenSize, hiddenSize).withActivation(Tensor.RELU);
        var hyperparamConfig = new HyperparamConfig().withGamma(this.gamma).withEntropyBonus(0.01f);
        var actionConfig = new ActionConfig();
        var memoryConfig = new MemoryConfig((long) this.episodeLen * 5); // Longer episodes

        var agentConfig = new ReinforceAgentConfig(policyNetConfig, hyperparamConfig, actionConfig, memoryConfig);
        var pg3Agent = new ReinforceAgent(agentConfig, stateDim, actionDim);

        evalPg3Agent(pg3Agent, env, iter * 4, trials, random);
    }

//    @Test void REINFORCE_GridWorld() {
//        var e = new SimpleGridWorld();
//        var hidden = e.stateDimension() * brains;
//        var a = new Reinforce(e.stateDimension(), e.actionDimension(), hidden, episodeLen);
//        eval(a.agent(), e);
//    }

    // @Disabled("Needs refactor to pg3.PPOAgent")
    // @Deprecated // Uses old PPO class
    @Test void PPO_Original_Matching() {
        var env = new MatchingEnv(1);
        int stateDim = env.stateDimension();
        int actionDim = env.actionDimension();
        int hiddenSize = stateDim * brains;

        var policyNetConfig = new NetworkConfig(3e-4f, hiddenSize, hiddenSize).withActivation(Tensor.RELU);
        var valueNetConfig = new NetworkConfig(1e-3f, hiddenSize, hiddenSize).withActivation(Tensor.RELU);
        var hyperparamConfig = new HyperparamConfig()
            .withGamma(this.gamma)
            .withLambda(0.95f) // GAE lambda
            .withPpoClip(0.2f)
            .withEpochs((long)4) // PPO specific: multiple epochs
            .withEntropyBonus(0.005f)
            .withNormalizeAdvantages(true);
        var actionConfig = new ActionConfig();
        // PPO typically updates based on a filled buffer rather than just episode length for single step 'done'
        // Let's use a memory config that reflects collecting a number of steps.
        // If episodeLen = 1 (as in this test class default), PPOAgent updates after 1 step if memoryConfig.episodeLength=1.
        // For more standard PPO, memoryConfig.episodeLength would be larger, e.g., 128, 256 etc.
        var memoryConfig = new MemoryConfig((long) this.episodeLen * 16); // Collect a bit more data before PPO update

        var agentConfig = new PPOAgentConfig(policyNetConfig, valueNetConfig, hyperparamConfig, actionConfig, memoryConfig);
        var pg3Agent = new PPOAgent(agentConfig, stateDim, actionDim);

        evalPPOAgent(pg3Agent, env); // Use PPO-specific eval
    }

    // This evalStrategyForEnv is based on pg2 strategies, will be removed or heavily adapted for pg3 agents.
    @Deprecated
    private void evalStrategyForEnv(MatchingEnv e, TestAlgoType algoType) {
        // ... existing code ...
        // This method will likely be removed as we test pg3 agents directly.
        throw new UnsupportedOperationException("This method is deprecated and pg3 agents should be tested directly.");
    }


    @Deprecated // Old eval method for jcog.agent.Agent
    private void eval(Agent a, SyntheticEnv e) {
        eval(a, e, iter, trials);
    }

    // New eval method for pg3.PolicyGradientAgent
    private void evalPg3Agent(PolicyGradientAgent pg3Agent, SyntheticEnv env) {
        evalPg3Agent(pg3Agent, env, iter, trials, random); // Pass RLAgentTestBase.random
    }

    // Overloaded version for more control if needed, and to pass Random
    protected void evalPg3Agent(PolicyGradientAgent pg3Agent, SyntheticEnv env, int iterations, int numTrials, Random evalRandom) {
        var episodesForTrend = 4; // How many segments to divide iterations into for trend calculation

        List<Double> trends = new ArrayList<>(numTrials);

        for (int t = 0; t < numTrials; t++) {
            List<Double> rewards = new ArrayList<>(iterations);
            pg3Agent.setTrainingMode(true); // Ensure agent is in training mode for each trial
            pg3Agent.clearMemory(); // Clear memory at the start of each trial

            Tensor currentStateTensor = Tensor.row(env.reset(evalRandom));
            double rewardSignal = 0.0; // Initial reward
            // boolean doneSignal = false; // Initial done signal - not needed as var, obtained from env step

            for (int i = 0; i < iterations; i++) {
                // Select action
                double[] actionArray = pg3Agent.selectAction(currentStateTensor, !pg3Agent.isTrainingMode());

                // Step environment
                SyntheticEnv.StepResult result = env.step(actionArray);
                Tensor nextStateTensor = result.nextState != null ? Tensor.row(result.nextState) : null;
                rewardSignal = result.reward;
                boolean currentStepDone = result.done;


                // For REINFORCE and VPG, oldLogProb is not strictly needed in Experience2 if not used by agent.
                // This eval loop is NOT suitable for PPOAgent as it requires oldLogProb.
                // PPOAgent tests should use evalPPOAgent.
                if (pg3Agent instanceof PPOAgent) {
                    throw new IllegalArgumentException("PPOAgent must be tested with evalPPOAgent to handle logProbs correctly.");
                }
                Experience2 exp = new Experience2(currentStateTensor, actionArray, rewardSignal, nextStateTensor, currentStepDone, null);
                pg3Agent.recordExperience(exp);

                currentStateTensor = nextStateTensor;

                rewards.add(rewardSignal);

                if (currentStepDone) {
                    if (i < iterations - 1) {
                        currentStateTensor = Tensor.row(env.reset(evalRandom));
                    }
                }
                if (currentStateTensor == null && i < iterations -1) {
                     System.err.println("Warning: null state encountered mid-trial for " + pg3Agent.getClass().getSimpleName() + ", resetting.");
                     currentStateTensor = Tensor.row(env.reset(evalRandom));
                }
            }

            double trend = rewardTrend(rewards, episodesForTrend);
            trends.add(trend);
        }

        var trendMean = trends.stream().mapToDouble(x -> x).average().orElse(0.0);
        System.out.println(pg3Agent.getClass().getSimpleName() + " on " + env.getClass().getSimpleName() +
                           ": mean trend=" + String.format("%.4f", trendMean) + ", trends: " + trends);
        assertTrue(trendMean > -0.1, pg3Agent.getClass().getSimpleName() + " mean trend (" + trendMean + ") should be positive or near zero for simple tasks.");
    }

    // Specific eval method for PPOAgent to handle log probabilities
    private void evalPPOAgent(PPOAgent ppoAgent, SyntheticEnv env) {
        evalPPOAgent(ppoAgent, env, iter, trials, random);
    }

    protected void evalPPOAgent(PPOAgent ppoAgent, SyntheticEnv env, int iterations, int numTrials, Random evalRandom) {
        var episodesForTrend = 4;

        List<Double> trends = new ArrayList<>(numTrials);

        for (int t = 0; t < numTrials; t++) {
            List<Double> rewards = new ArrayList<>(iterations);
            ppoAgent.setTrainingMode(true);
            ppoAgent.clearMemory();

            Tensor currentStateTensor = Tensor.row(env.reset(evalRandom));
            double rewardSignal = 0.0;

            for (int i = 0; i < iterations; i++) {
                PPOAgent.ActionWithLogProb actionWithLogProb = ppoAgent.selectActionWithLogProb(currentStateTensor, !ppoAgent.isTrainingMode());
                double[] actionArray = actionWithLogProb.action();
                Tensor oldLogProb = actionWithLogProb.logProb();

                SyntheticEnv.StepResult result = env.step(actionArray);
                Tensor nextStateTensor = result.nextState != null ? Tensor.row(result.nextState) : null;
                rewardSignal = result.reward;
                boolean currentStepDone = result.done;

                Experience2 exp = new Experience2(currentStateTensor, actionArray, rewardSignal, nextStateTensor, currentStepDone, oldLogProb);
                ppoAgent.recordExperience(exp);

                currentStateTensor = nextStateTensor;
                rewards.add(rewardSignal);

                if (currentStepDone) {
                    if (i < iterations - 1) {
                        currentStateTensor = Tensor.row(env.reset(evalRandom));
                    }
                }
                 if (currentStateTensor == null && i < iterations -1) {
                     System.err.println("Warning: null state encountered mid-trial for PPO, resetting.");
                     currentStateTensor = Tensor.row(env.reset(evalRandom));
                }
            }
            double trend = rewardTrend(rewards, episodesForTrend);
            trends.add(trend);
        }

        var trendMean = trends.stream().mapToDouble(x -> x).average().orElse(0.0);
        System.out.println(ppoAgent.getClass().getSimpleName() + " on " + env.getClass().getSimpleName() +
                           ": mean trend=" + String.format("%.4f", trendMean) + ", trends: " + trends);
        assertTrue(trendMean > -0.1, ppoAgent.getClass().getSimpleName() + " mean trend (" + trendMean + ") should be positive or near zero for simple tasks.");
    }


    public static double rewardTrend(List<Double> rewards, int numEpisodes) {
        if (rewards == null || rewards.isEmpty() || numEpisodes <= 0) {
            // throw new IllegalArgumentException("Invalid input: rewards must not be null/empty, numEpisodes must be positive");
            return 0.0; // Return 0 if no rewards to analyze
        }

        int actualNumEpisodes = Math.min(numEpisodes, rewards.size());
        if (actualNumEpisodes == 0) return 0.0;
        int episodeSize = rewards.size() / actualNumEpisodes;
        if (episodeSize == 0 && rewards.size() > 0) { // if iterations < numEpisodes, treat whole thing as one segment
            episodeSize = rewards.size();
            actualNumEpisodes = 1;
        } else if (episodeSize == 0) {
            return 0.0;
        }


        // Calculate first and last episode means
        double firstMean = 0.0;
        int firstCount = 0;
        for (int i = 0; i < episodeSize; i++) {
            if (i < rewards.size()) {
                firstMean += rewards.get(i);
                firstCount++;
            }
        }
        if (firstCount > 0) firstMean /= firstCount; else firstMean = 0.0;

        double lastMean = 0.0;
        int lastCount = 0;

        if (actualNumEpisodes == 1) { // Only one segment
            lastMean = firstMean;
        } else {
            int start = (actualNumEpisodes - 1) * episodeSize;
            for (int i = start; i < rewards.size(); i++) {
                 if (i < rewards.size()) {
                    lastMean += rewards.get(i);
                    lastCount++;
                }
            }
            if (lastCount > 0) lastMean /= lastCount; else lastMean = 0.0;
        }


        // Return difference as trend indicator
        return lastMean - firstMean;
    }

    @Deprecated // Old eval method signature
    protected void eval(Agent agent, SyntheticEnv env, int iterations, int trials) {
        // This method is now deprecated in favor of evalPg3Agent
        throw new UnsupportedOperationException("Use evalPg3Agent for pg3 agents.");
    }

    // ===================================================================================
    // Focused Unit Tests for pg3 Agents
    // ===================================================================================
    @Nested
    @DisplayName("ReinforceAgent Unit Tests")
    class ReinforceAgentUnitTests {

        private ReinforceAgent createTestReinforceAgent(int stateDim, int actionDim, Consumer<ReinforceAgentConfig.Builder> configCustomizer) {
            ReinforceAgentConfig.Builder configBuilder = ReinforceAgentConfig.newBuilder()
                .policyNetworkConfig(new NetworkConfig(1e-3f, 8).withOptimizerType(Tensor.Optimizer.Type.ADAM)) // Simple network
                .hyperparamConfig(new HyperparamConfig().withGamma(0.99f))
                .actionConfig(new ActionConfig())
                .memoryConfig(new MemoryConfig(10L)); // Default episode length for buffer

            if (configCustomizer != null) {
                configCustomizer.accept(configBuilder);
            }
            // The ReinforceAgentConfig.Builder now requires stateDim and actionDim at build time.
            return new ReinforceAgent(configBuilder.build(stateDim, actionDim), stateDim, actionDim);
        }

        // Helper to get parameters from a policy network
        private List<Tensor> getPolicyParams(PolicyGradientAgent agent) {
            return ((GaussianPolicyNet) agent.getPolicy()).parameters().collect(Collectors.toList());
        }

        @Test
        @DisplayName("computeReturns calculates discounted returns correctly (indirect test via update)")
        void testComputeReturnsIndirect() {
            int stateDim = 1, actionDim = 1;
            // We need a way to check the returns. Since computeReturns is private,
            // we can check its effect on the loss or gradients.
            // This requires a very controlled setup.
            // For now, this test will ensure an update happens and parameters change,
            // implying returns were computed and used. A more direct test of values is hard without access.

            ReinforceAgent agent = createTestReinforceAgent(stateDim, actionDim, cfg ->
                cfg.hyperparamConfig(new HyperparamConfig().withGamma(0.9f).withNormalizeReturns(false).withEntropyBonus(0.0f))
                   .policyNetworkConfig(new NetworkConfig(1e-2f, 4).withOptimizerType(Tensor.Optimizer.Type.SGD)) // SGD, higher LR for visible change
            );
            agent.setTrainingMode(true);

            // Make policy deterministic for predictable logProbs if needed, though REINFORCE samples.
            // The actual logProb value will depend on the random sample and network state.
            // We mainly check that *some* update happens based on *some* computed returns.

            List<Tensor> paramsBefore = getPolicyParams(agent).stream().map(Tensor::detachCopy).collect(Collectors.toList());

            agent.recordExperience(new Experience2(Tensor.scalar(0.1f), new double[]{0.5f}, 1.0, Tensor.scalar(0.2f), false, null));
            agent.recordExperience(new Experience2(Tensor.scalar(0.2f), new double[]{0.5f}, 1.0, null, true, null)); // done

            assertEquals(1, agent.getUpdateCount());
            List<Tensor> paramsAfter = getPolicyParams(agent);

            boolean changed = false;
            for (int i = 0; i < paramsBefore.size(); i++) {
                if (!paramsBefore.get(i).equals(paramsAfter.get(i))) {
                    changed = true;
                    break;
                }
            }
            assertTrue(changed, "Policy parameters should change after update, implying returns were used.");
        }


        @Test
        @DisplayName("update method processes a simple trajectory and changes parameters")
        void testUpdateSimpleTrajectory() {
            int stateDim = 1, actionDim = 1;
            ReinforceAgent agent = createTestReinforceAgent(stateDim, actionDim, cfg ->
                cfg.hyperparamConfig(new HyperparamConfig().withGamma(0.9f).withNormalizeReturns(false).withEntropyBonus(0.0f))
                   .policyNetworkConfig(new NetworkConfig(1e-2f, 4).withOptimizerType(Tensor.Optimizer.Type.SGD)) // simple net, SGD, higher LR
            );

            Tensor state0 = Tensor.scalar(0.5f);
            double[] action0 = {0.1f}; // Action doesn't matter much if we don't check logProb value
            double reward0 = 1.0;
            Tensor state1 = Tensor.scalar(0.6f);
            double[] action1 = {0.2f};
            double reward1 = 1.0;

            agent.setTrainingMode(true);

            List<Tensor> paramsBefore = getPolicyParams(agent).stream().map(Tensor::detachCopy).collect(Collectors.toList());

            agent.recordExperience(new Experience2(state0, action0, reward0, state1, false, null));
            agent.recordExperience(new Experience2(state1, action1, reward1, null, true, null)); // done=true triggers update

            assertEquals(1, agent.getUpdateCount(), "Update count should be 1 after one episode");

            List<Tensor> paramsAfter = getPolicyParams(agent);

            boolean changed = false;
            for (int i = 0; i < paramsBefore.size(); i++) {
                if (!paramsBefore.get(i).equals(paramsAfter.get(i))) {
                    changed = true;
                    break;
                }
            }
            assertTrue(changed, "Policy parameters should change after update.");
        }

        @Test
        @DisplayName("setTrainingMode propagates to network and clears memory on eval")
        void testSetTrainingMode() {
            ReinforceAgent agent = createTestReinforceAgent(1, 1, null);
            GaussianPolicyNet policyNet = (GaussianPolicyNet) agent.getPolicy();

            agent.setTrainingMode(true);
            assertTrue(agent.isTrainingMode(), "Agent should be in training mode");
            assertTrue(policyNet.isTraining(), "Policy network should be in training mode");

            // Add some experience
            agent.recordExperience(new Experience2(Tensor.scalar(0), new double[]{0}, 1.0, Tensor.scalar(1), false, null));
            assertFalse(agent.memory.isEmpty(), "Memory should not be empty before switching to eval");

            agent.setTrainingMode(false); // This should clear memory for on-policy agents like Reinforce
            assertFalse(agent.isTrainingMode(), "Agent should be in eval mode");
            assertFalse(policyNet.isTraining(), "Policy network should be in eval mode");
            assertTrue(agent.memory.isEmpty(), "Memory should be cleared when switching to eval mode for ReinforceAgent");
        }
    }

    @Nested
    @DisplayName("VPGAgent Unit Tests")
    class VPGAgentUnitTests {

        private VPGAgent createTestVPGAgent(int stateDim, int actionDim, Consumer<VPGAgentConfig.Builder> configCustomizer) {
            VPGAgentConfig.Builder configBuilder = VPGAgentConfig.newBuilder()
                .policyNetworkConfig(new NetworkConfig(1e-3f, 8).withOptimizerType(Tensor.Optimizer.Type.ADAM))
                .valueNetworkConfig(new NetworkConfig(1e-3f, 8).withOptimizerType(Tensor.Optimizer.Type.ADAM))
                .hyperparamConfig(new HyperparamConfig().withGamma(0.99f))
                .actionConfig(new ActionConfig())
                .memoryConfig(new MemoryConfig(10L));

            if (configCustomizer != null) {
                configCustomizer.accept(configBuilder);
            }
            return new VPGAgent(configBuilder.build(stateDim, actionDim), stateDim, actionDim);
        }

        private List<Tensor> getValueNetParams(PolicyGradientAgent agent) {
            return ((jcog.tensor.rl.pg3.networks.ValueNet) agent.getValueFunction()).parameters().collect(Collectors.toList());
        }


        @Test
        @DisplayName("update method processes trajectory and updates both networks")
        void testUpdateSimpleTrajectoryVPG() {
            int stateDim = 1, actionDim = 1;
            VPGAgent agent = createTestVPGAgent(stateDim, actionDim, cfg ->
                cfg.hyperparamConfig(new HyperparamConfig().withGamma(0.9f).withNormalizeReturns(false).withNormalizeAdvantages(false).withEntropyBonus(0.0f))
                   .policyNetworkConfig(new NetworkConfig(1e-2f, 4).withOptimizerType(Tensor.Optimizer.Type.SGD))
                   .valueNetworkConfig(new NetworkConfig(1e-2f, 4).withOptimizerType(Tensor.Optimizer.Type.SGD))
            );

            agent.setTrainingMode(true);

            List<Tensor> policyParamsBefore = getPolicyParams(agent).stream().map(Tensor::detachCopy).collect(Collectors.toList());
            List<Tensor> valueParamsBefore = getValueNetParams(agent).stream().map(Tensor::detachCopy).collect(Collectors.toList());

            Tensor state0 = Tensor.scalar(0.5f);
            double[] action0 = {0.1f};
            double reward0 = 1.0;
            Tensor state1 = Tensor.scalar(0.6f);
            double[] action1 = {0.2f};
            double reward1 = 1.0;

            agent.recordExperience(new Experience2(state0, action0, reward0, state1, false, null));
            agent.recordExperience(new Experience2(state1, action1, reward1, null, true, null)); // done

            assertEquals(1, agent.getUpdateCount(), "Update count should be 1");

            List<Tensor> policyParamsAfter = getPolicyParams(agent);
            List<Tensor> valueParamsAfter = getValueNetParams(agent);

            boolean policyChanged = false;
            for (int i = 0; i < policyParamsBefore.size(); i++) {
                if (!policyParamsBefore.get(i).equals(policyParamsAfter.get(i))) {
                    policyChanged = true;
                    break;
                }
            }
            assertTrue(policyChanged, "Policy parameters should change after update.");

            boolean valueChanged = false;
            for (int i = 0; i < valueParamsBefore.size(); i++) {
                if (!valueParamsBefore.get(i).equals(valueParamsAfter.get(i))) {
                    valueChanged = true;
                    break;
                }
            }
            assertTrue(valueChanged, "Value function parameters should change after update.");
        }

        @Test
        @DisplayName("setTrainingMode propagates to both networks for VPG")
        void testSetTrainingModeVPG() {
            VPGAgent agent = createTestVPGAgent(1, 1, null);
            GaussianPolicyNet policyNet = (GaussianPolicyNet) agent.getPolicy();
            jcog.tensor.rl.pg3.networks.ValueNet valueNet = (jcog.tensor.rl.pg3.networks.ValueNet) agent.getValueFunction();

            agent.setTrainingMode(true);
            assertTrue(agent.isTrainingMode());
            assertTrue(policyNet.isTraining());
            assertTrue(valueNet.isTraining());

            agent.recordExperience(new Experience2(Tensor.scalar(0), new double[]{0}, 1.0, Tensor.scalar(1), false, null));
            assertFalse(agent.memory.isEmpty());

            agent.setTrainingMode(false);
            assertFalse(agent.isTrainingMode());
            assertFalse(policyNet.isTraining());
            assertFalse(valueNet.isTraining());
            assertTrue(agent.memory.isEmpty(), "Memory should be cleared for VPG on eval mode switch");
        }
    }

    @Nested
    @DisplayName("PPOAgent Unit Tests")
    class PPOAgentUnitTests {

        private PPOAgent createTestPPOAgent(int stateDim, int actionDim, Consumer<PPOAgentConfig.Builder> configCustomizer) {
            PPOAgentConfig.Builder configBuilder = PPOAgentConfig.newBuilder()
                .policyNetworkConfig(new NetworkConfig(1e-3f, 8).withOptimizerType(Tensor.Optimizer.Type.ADAM))
                .valueNetworkConfig(new NetworkConfig(1e-3f, 8).withOptimizerType(Tensor.Optimizer.Type.ADAM))
                .hyperparamConfig(new HyperparamConfig().withGamma(0.99f).withLambda(0.95f).withPpoClip(0.2f).withEpochs(2L)) // 2 epochs for faster test
                .actionConfig(new ActionConfig())
                .memoryConfig(new MemoryConfig(4L)); // Small buffer for faster filling in tests

            if (configCustomizer != null) {
                configCustomizer.accept(configBuilder);
            }
            return new PPOAgent(configBuilder.build(stateDim, actionDim), stateDim, actionDim);
        }

        @Test
        @DisplayName("selectActionWithLogProb returns action and logProb")
        void testSelectActionWithLogProb() {
            PPOAgent agent = createTestPPOAgent(2, 2, null);
            Tensor testState = Tensor.rand(1, 2);
            PPOAgent.ActionWithLogProb result = agent.selectActionWithLogProb(testState, false);

            assertNotNull(result.action());
            assertEquals(2, result.action().length);
            assertNotNull(result.logProb());
            assertEquals(1, result.logProb().dims()[0]); // Assuming logProb is a scalar or [1,1]
        }

        @Test
        @DisplayName("update method processes a buffer and updates networks over epochs")
        void testUpdateProcessesBufferPPO() {
            int stateDim = 1, actionDim = 1;
            long bufferSize = 4L; // Must match memoryConfig in createTestPPOAgent or be passed
            long numEpochs = 2L;  // Must match hyperparamConfig

            PPOAgent agent = createTestPPOAgent(stateDim, actionDim, cfg ->
                cfg.memoryConfig(new MemoryConfig(bufferSize))
                   .hyperparamConfig(new HyperparamConfig().withGamma(0.9f).withLambda(0.95f).withPpoClip(0.2f).withEpochs(numEpochs).withEntropyBonus(0.0f))
                   .policyNetworkConfig(new NetworkConfig(1e-2f, 4).withOptimizerType(Tensor.Optimizer.Type.SGD))
                   .valueNetworkConfig(new NetworkConfig(1e-2f, 4).withOptimizerType(Tensor.Optimizer.Type.SGD))
            );
            agent.setTrainingMode(true);

            List<Tensor> policyParamsBefore = getPolicyParams(agent).stream().map(Tensor::detachCopy).collect(Collectors.toList());
            List<Tensor> valueParamsBefore = getValueNetParams(agent).stream().map(Tensor::detachCopy).collect(Collectors.toList());

            for (int i = 0; i < bufferSize; i++) {
                Tensor state = Tensor.scalar(0.1f * i);
                // For PPO, logProb is crucial. We need to use selectActionWithLogProb.
                PPOAgent.ActionWithLogProb awlp = agent.selectActionWithLogProb(state, false);
                agent.recordExperience(new Experience2(state, awlp.action(), 0.1, Tensor.scalar(0.1f * (i + 1)), i == bufferSize - 1, awlp.logProb()));
            }
            // The PPOAgent's recordExperience calls update when buffer is full.
            // So, one update call (which internally runs numEpochs) should have happened.
            assertEquals(1, agent.getUpdateCount(), "Update count should be 1 after buffer fills");


            List<Tensor> policyParamsAfter = getPolicyParams(agent);
            List<Tensor> valueParamsAfter = getValueNetParams(agent);

            boolean policyChanged = false;
            for (int i = 0; i < policyParamsBefore.size(); i++) {
                if (!policyParamsBefore.get(i).equals(policyParamsAfter.get(i))) {
                    policyChanged = true;
                    break;
                }
            }
            assertTrue(policyChanged, "Policy parameters should change after PPO update.");

            boolean valueChanged = false;
            for (int i = 0; i < valueParamsBefore.size(); i++) {
                if (!valueParamsBefore.get(i).equals(valueParamsAfter.get(i))) {
                    valueChanged = true;
                    break;
                }
            }
            assertTrue(valueChanged, "Value function parameters should change after PPO update.");
        }


        @Test
        @DisplayName("setTrainingMode propagates to both networks for PPO")
        void testSetTrainingModePPO() {
            PPOAgent agent = createTestPPOAgent(1, 1, null);
            GaussianPolicyNet policyNet = (GaussianPolicyNet) agent.getPolicy();
            jcog.tensor.rl.pg3.networks.ValueNet valueNet = (jcog.tensor.rl.pg3.networks.ValueNet) agent.getValueFunction();

            agent.setTrainingMode(true);
            assertTrue(agent.isTrainingMode());
            assertTrue(policyNet.isTraining());
            assertTrue(valueNet.isTraining());

            // Add some experience (not enough to fill buffer to avoid update)
            PPOAgent.ActionWithLogProb awlp = agent.selectActionWithLogProb(Tensor.scalar(0), false);
            agent.recordExperience(new Experience2(Tensor.scalar(0), awlp.action(), 1.0, Tensor.scalar(1), false, awlp.logProb()));
            assertFalse(agent.memory.isEmpty());


            agent.setTrainingMode(false);
            assertFalse(agent.isTrainingMode());
            assertFalse(policyNet.isTraining());
            assertFalse(valueNet.isTraining());
            assertTrue(agent.memory.isEmpty(), "Memory should be cleared for PPO on eval mode switch");
        }
    }
}
