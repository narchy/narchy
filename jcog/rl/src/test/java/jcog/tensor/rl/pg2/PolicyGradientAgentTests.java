package jcog.tensor.rl.pg2;

import jcog.agent.Agent;
import jcog.tensor.Tensor;
import jcog.tensor.rl.RLAgentTestBase;
import jcog.tensor.rl.env.ContinuousSeekEnv;
import jcog.tensor.rl.env.MatchingEnv;
import jcog.tensor.rl.env.SyntheticEnv;
import jcog.tensor.rl.pg.*;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class PolicyGradientAgentTests extends RLAgentTestBase {

    // Local enum for test parameterization, similar to PGBuilderTest.TestAlgoType
    // This helps in replacing PGBuilder.Algorithm
    enum TestAlgoType {
        REINFORCE, PPO, SAC, DDPG
        // VPG tests will use PPO strategy as per old PGBuilder mapping
    }

    int brains = 4;
    float gamma = 0.9f;
    int iter = 1 * 1024;
    int trials = 3;
    int episodeLen = 1;

    @Deprecated // Uses old Reinforce class, new tests should use PGBuilder.ReinforceStrategy
    @Test void REINFORCE_Matching1() {
        var e = new MatchingEnv(1);
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Deprecated // Uses old Reinforce class, new tests should use PGBuilder.ReinforceStrategy
    @Test void REINFORCE_Matching2() {
        var e = new MatchingEnv(2);
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Deprecated // Uses old VPG class, new tests should use PGBuilder.PPOStrategy or a dedicated VPG-like strategy
    @Test void VPG_Matching1() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new VPG(e.stateDimension(), e.actionDimension(), hidden, hidden, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
    @Test void StreamAC_Matching1() { // Assuming StreamAC is a current/valid agent not part of PGBuilder refactor
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new StreamAC(e.stateDimension(), e.actionDimension(), hidden, hidden);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
    @Deprecated // Uses old DDPG class, new tests should use PGBuilder.DDPGStrategy
    @Test void DDPG_Matching() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new DDPG(e.stateDimension(), e.actionDimension(), hidden, hidden);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Deprecated // Uses old Reinforce class
    @Test void REINFORCE_Target1D() {
        var e = new ContinuousSeekEnv();
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
    @Deprecated // Uses old Reinforce class
    @Test void REINFORCE_Target2D() {
        var e = ContinuousSeekEnv.W2D();
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

//    @Test void REINFORCE_GridWorld() {
//        var e = new SimpleGridWorld();
//        var hidden = e.stateDimension() * brains;
//        var a = new Reinforce(e.stateDimension(), e.actionDimension(), hidden, episodeLen);
//        eval(a.agent(), e);
//    }

    @Deprecated // Uses old PPO class
    @Test void PPO_Original_Matching() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new PPO(e.stateDimension(), e.actionDimension(), hidden, hidden, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    // Refactored helper method to use direct DI
    private void evalStrategyForEnv(MatchingEnv e, TestAlgoType algoType) {
        var hidden = e.stateDimension() * brains;
        int inputs = e.stateDimension();
        int outputs = e.actionDimension();

        PGBuilder.HyperparamConfig hyperparams = new PGBuilder.HyperparamConfig()
                .withGamma(this.gamma) // Use gamma from test class
                .withPolicyLR(3e-4f)
                .withValueLR(1e-3f) // Relevant for PPO/SAC/DDPG critics
                .withPpoClip(0.2f)   // Relevant for PPO
                .withEpochs(1)
                .withEntropyBonus(0.01f);

        PGBuilder.ActionConfig actionConfig = new PGBuilder.ActionConfig(); // Default

        // MemoryConfig: episodeLength for on-policy, ReplayBufferConfig for off-policy (though not used by MatchingEnv tests for off-policy yet)
        PGBuilder.MemoryConfig memoryConfig = new PGBuilder.MemoryConfig(this.episodeLen);


        NetworkConfig policyNetConfig = new NetworkConfig(hyperparams.policyLR(), hidden).withActivation(Tensor.RELU);
        NetworkConfig valueNetConfig = new NetworkConfig(hyperparams.valueLR(), hidden).withActivation(Tensor.RELU); // For PPO


        PGStrategy strategy;

        switch (algoType) {
            case PPO: // VPG tests will also use PPO strategy
                PGBuilder.GaussianPolicy ppoPolicy = new PGBuilder.GaussianPolicy(policyNetConfig, inputs, outputs);
                Tensor.Optimizer ppoPolicyOpt = policyNetConfig.optimizer().build();
                PGBuilder.ValueNetwork ppoValueNet = new PGBuilder.ValueNetwork(valueNetConfig, inputs);
                Tensor.Optimizer ppoValueOpt = valueNetConfig.optimizer().build();
                OnPolicyEpisodeBuffer ppoMemory = new OnPolicyEpisodeBuffer(memoryConfig.episodeLength());
                strategy = new PPOStrategy(hyperparams, actionConfig, memoryConfig, ppoMemory, ppoPolicy, ppoValueNet, ppoPolicyOpt, ppoValueOpt);
                break;
            case REINFORCE: // Example if we wanted to test Reinforce this way
                 PGBuilder.GaussianPolicy reinforcePolicy = new PGBuilder.GaussianPolicy(policyNetConfig, inputs, outputs);
                 Tensor.Optimizer reinforcePolicyOpt = policyNetConfig.optimizer().build();
                 OnPolicyEpisodeBuffer reinforceMemory = new OnPolicyEpisodeBuffer(memoryConfig.episodeLength());
                 strategy = new ReinforceStrategy(hyperparams, actionConfig, memoryConfig, reinforceMemory, reinforcePolicy, reinforcePolicyOpt);
                 break;
            // SAC and DDPG cases would need more setup (Q-networks, different policy types, replay buffers)
            // and are not currently tested by the MatchingEnv tests using evalPGBuilder.
            // If they were, their setup would mirror createTestModel from PGBuilderTest.
            default:
                throw new UnsupportedOperationException("Algorithm type " + algoType + " not fully configured in this test helper for MatchingEnv.");
        }


        eval(new PolicyGradientModel(inputs, outputs, strategy).agent(), e);
    }


    private void eval(Agent a, SyntheticEnv e) {
        eval(a, e, iter, trials);
    }

    public static double rewardTrend(List<Double> rewards, int numEpisodes) {
        if (rewards == null || rewards.isEmpty() || numEpisodes <= 0) {
            throw new IllegalArgumentException("Invalid input: rewards must not be null/empty, numEpisodes must be positive");
        }

        numEpisodes = Math.min(numEpisodes, rewards.size());
        int episodeSize = rewards.size() / numEpisodes;

        // Calculate first and last episode means
        double firstMean = 0.0;
        int firstCount = 0;
        for (int i = 0; i < episodeSize; i++) {
            firstMean += rewards.get(i);
            firstCount++;
        }
        firstMean /= firstCount;

        double lastMean = 0.0;
        int lastCount = 0;
        int start = (numEpisodes - 1) * episodeSize;
        for (int i = start; i < rewards.size(); i++) {
            lastMean += rewards.get(i);
            lastCount++;
        }
        lastMean /= lastCount;

        // Return difference as trend indicator
        return lastMean - firstMean;
    }

    protected void eval(Agent agent, SyntheticEnv env, int iterations, int trials) {
        var episodes = 4;

        double[] actionVec = new double[agent.actions], actionPrev = new double[agent.actions];
        List<Double> trends = new ArrayList(trials);

        for (int t = 0; t < trials; t++) {
            List<Double> rewards = new ArrayList<>(iterations);

            double reward = 0; // Initial reward signal for the first step
            boolean doneSignal = false;    // Initial done signal
            double[] currentState = env.reset(random);

            for (int i = 0; i < iterations; i++) {
                agent.act(actionPrev, (float) reward, currentState, actionVec);

                SyntheticEnv.StepResult result;
                result = env.step(actionVec);

                reward = result.reward;
                rewards.add(reward);
                currentState = result.nextState;
            }

            double trend = rewardTrend(rewards, episodes);
            trends.add(trend);
        }

        var trendMean = trends.stream().mapToDouble(x -> x).average().getAsDouble();
        System.out.println(agent + ": mean=" + trendMean + " " + trends);
        //for (var trend : trends) assertTrue(trend > 0, agent + " rewardTrend=" + trend);
        assertTrue(trendMean > 0);
    }
}
