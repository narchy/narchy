package jcog.tensor.rl.pg;

import jcog.agent.Agent;
import jcog.tensor.Tensor;
import jcog.tensor.rl.RLAgentTestBase;
import jcog.tensor.rl.env.ContinuousSeekEnv;
import jcog.tensor.rl.env.MatchingEnv;
import jcog.tensor.rl.env.SyntheticEnv;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class PolicyGradientAgentTests extends RLAgentTestBase {

    int brains = 4;
    float gamma = 0.9f;
    int iter = 1 * 1024;
    int trials = 3;
    int episodeLen = 1;

    @Test void REINFORCE_Matching1() {
        var e = new MatchingEnv(1);
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
    @Test void REINFORCE_Matching2() {
        var e = new MatchingEnv(2);
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Test void VPG_Matching1() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new VPG(e.stateDimension(), e.actionDimension(), hidden, hidden, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
    @Test void StreamAC_Matching1() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new StreamAC(e.stateDimension(), e.actionDimension(), hidden, hidden);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
    @Test void DDPG_Matching() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new DDPG(e.stateDimension(), e.actionDimension(), hidden, hidden);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Test void REINFORCE_Target1D() {
        var e = new ContinuousSeekEnv();
        var a = new Reinforce(e.stateDimension(), e.actionDimension(), e.stateDimension()*brains, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }
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

    @Test void PPO_Original_Matching() {
        var e = new MatchingEnv(1);
        var hidden = e.stateDimension() * brains;
        var a = new PPO(e.stateDimension(), e.actionDimension(), hidden, hidden, episodeLen);
        a.gamma.set(gamma);
        eval(a.agent(), e);
    }

    @Test void PPO_PGBuilder_Matching1() {
        evalPGBuilder(new MatchingEnv(1), PGBuilder.Algorithm.PPO);
    }

    @Test void VPG_PGBuilder_Matching1() {
        evalPGBuilder(new MatchingEnv(1), PGBuilder.Algorithm.VPG);
    }

    @Test void VPG_PGBuilder_Matching2() {
        evalPGBuilder(new MatchingEnv(2), PGBuilder.Algorithm.VPG);
    }

    private void evalPGBuilder(MatchingEnv e, PGBuilder.Algorithm alg) {
        var hidden = e.stateDimension() * brains;
        var a = new PGBuilder(e.stateDimension(), e.actionDimension())
                .algorithm(alg)
                .policy(p -> p.hiddenLayers(hidden).activation(Tensor.RELU))
                .value(v -> v.hiddenLayers(hidden).activation(Tensor.RELU))
                .hyperparams(h -> h.gamma(gamma).policyLR(3e-4f).valueLR(1e-3f).ppoClip(0.2f).epochs(1).entropyBonus(0.01f))
                .memory(m -> m.episodeLength(episodeLen))
                .build();
        eval(a.agent(), e);
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
