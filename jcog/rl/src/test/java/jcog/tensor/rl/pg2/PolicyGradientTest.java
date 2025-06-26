package jcog.tensor.rl.pg2;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg2.configs.PPOAgentConfig;
import jcog.tensor.rl.pg2.configs.VPGAgentConfig;
import jcog.tensor.rl.pg2.stats.DefaultMetricCollector;
import jcog.tensor.rl.pg2.stats.MetricCollector;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.round;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PolicyGradientTest {

    // Simple environment for testing: PointNavigation1D
    static class PointNavigation1D {
        private double currentState;
        private final double targetState = 0.0;
        private final double stepSize = 0.1; // Max action magnitude
        private final double stateMin = -1.0;
        private final double stateMax = 1.0;
        private final int maxStepsPerEpisode = 50;
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

            var done = this.currentStepInEpisode >= this.maxStepsPerEpisode || Math.abs(this.currentState - targetState) < 0.01;

            if (Math.abs(this.currentState - targetState) < 0.01) {
                reward += 10.0; // Bonus for reaching target
            }

            // Small penalty for existing
            // reward -= 0.01;

            return new Triplet<>(Tensor.scalar(this.currentState), reward, done);
        }

        public int getStateDim() {
            return 1;
        }

        public int getActionDim() {
            return 1;
        }
    }

    // Helper record for environment step results
    record Triplet<S, R, D>(S state, R reward, D done) {}

    @Test
    void vpg_AgentLearnsOnPointNavigation() {
        var env = new PointNavigation1D();
        var stateDim = env.getStateDim();
        var actionDim = env.getActionDim();
        eval(vpg(stateDim, actionDim), env);
    }
    @Test
    void ppo_AgentLearnsOnPointNavigation() {
        var env = new PointNavigation1D();
        var stateDim = env.getStateDim();
        var actionDim = env.getActionDim();
        eval(ppo(stateDim, actionDim), env);
    }

    private void eval(AbstractPolicyGradientAgent agent, PointNavigation1D env) {
        agent.setTrainingMode(true);

        MetricCollector metrics = new DefaultMetricCollector();
        agent.setMetricCollector(metrics);

        var params = Tensor.parameters(agent).toList();
        assertFalse(params.isEmpty());
        var paramCount = params.stream().mapToInt(z -> z.volume()).sum();
        assertTrue(paramCount > 0);

        var numEpisodes = 200;

        List<Double> episodeRewards = new ArrayList<>();
        double totalRewardLast50Episodes = 0;

        System.out.println("Starting training on PointNavigation1D...");

        double[] actionNext = new double[agent.actions];
        double[] actionPrev = new double[agent.actions];

        for (var episode = 0; episode < numEpisodes; episode++) {
            var state = env.reset();
            double currentEpisodeReward = 0;
            Tensor oldLogProb = null; // For PPO, this should come from selectActionWithLogProb

            for (var t = 0; t < env.maxStepsPerEpisode; t++) {
                var action = agent.selectAction(state, false);//selectActionWithLogProb(state, false);
                Tensor currentLogProb = null; //PPO: actionWithLogProb.logProb();

                var stepResult = env.step(action);
                var nextState = stepResult.state();
                double reward = stepResult.reward();
                boolean done = stepResult.done();

                // Store experience: state, action, reward, nextState, done, oldLogProb
                // Note: Experience2 needs state, action, reward, nextState, done, and optionally oldLogProb.
                //agent.memory.add(new Experience2(state, action, reward, nextState, done, currentLogProb)); //TODO agent.act()...
                agent.act(actionPrev, (float)reward, nextState.array(), actionNext);
                System.arraycopy(actionNext, 0, actionPrev, 0, agent.actions);

                state = nextState;
                currentEpisodeReward += reward;

//                if (agent.memory.size() >= agent.config.memoryConfig().episodeLength().intValue()) {
//                    agent.update(episode * env.maxStepsPerEpisode + t); // Pass global step, or agent.getUpdateCount()
//                    agent.memory.clear(); // For on-policy
//                }

                if (done) break;
            }
            episodeRewards.add(currentEpisodeReward);
            if (episode >= numEpisodes - 50) {
                totalRewardLast50Episodes += currentEpisodeReward;
            }

            if ((episode + 1) % 20 == 0) {
                System.out.printf("Episode %d, Avg Reward (last 20): %.2f, Total Updates: %d%n",
                        episode + 1,
                        episodeRewards.subList(Math.max(0, episodeRewards.size()-20), episodeRewards.size())
                                .stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN),
                        agent.getUpdateCount());

                metrics.getSummary("policy_loss").ifPresent(s -> System.out.printf("  Policy Loss: %.4f%n", s.getMean()));
                metrics.getSummary("value_loss").ifPresent(s -> System.out.printf("  Value Loss: %.4f%n", s.getMean()));
                metrics.getSummary("entropy").ifPresent(s -> System.out.printf("  Entropy: %.4f%n", s.getMean()));

            }
        }

        System.out.println("Training finished.");
        System.out.println("Total updates: " + agent.getUpdateCount());

        // Assertions
        assertTrue(agent.getUpdateCount() > 0, "Agent should have performed updates.");

        var policyLossSummaryOpt = metrics.getSummary("policy_loss");
        assertTrue(policyLossSummaryOpt.isPresent(), "Policy loss should have been recorded.");
        System.out.printf("Policy Loss Summary: %s%n", policyLossSummaryOpt.get());

        var valueLossSummaryOpt = metrics.getSummary("value_loss");
        assertTrue(valueLossSummaryOpt.isPresent(), "Value loss should have been recorded.");
        System.out.printf("Value Loss Summary: %s%n", valueLossSummaryOpt.get());

        // Check that average reward in the last 50 episodes is significantly better than the first 50
        // This is a basic check for learning. More sophisticated checks might look at loss curves.
        var avgRewardFirst50 = episodeRewards.subList(0, Math.min(50, episodeRewards.size()))
                .stream().mapToDouble(r -> r).average().orElse(Double.NEGATIVE_INFINITY);
        var avgRewardLast50 = totalRewardLast50Episodes / Math.min(50, episodeRewards.size() - (numEpisodes - 50) );

        System.out.printf("Avg Reward First 50 Episodes: %.2f%n", avgRewardFirst50);
        System.out.printf("Avg Reward Last 50 Episodes: %.2f%n", avgRewardLast50);

        // A simple environment like PointNavigation should show clear improvement.
        // The exact values depend heavily on reward structure and hyperparameters.
        // We expect last 50 to be substantially higher than first 50 if learning occurred.
        // Initial rewards are around -25 to -50 (avg -0.5 to -1 per step for 50 steps).
        // Optimal reward is close to 10 (reaching target quickly).
        assertTrue(avgRewardLast50 > avgRewardFirst50 + 5, // Expect at least some improvement. This threshold might need tuning.
                "Average reward in later episodes should be significantly higher than in early episodes." +
                        " Last50: " + avgRewardLast50 + ", First50: " + avgRewardFirst50);

        // Check if loss values are sensible (not NaN or Infinity)
        // Further checks could involve storing initial vs final loss values and asserting decrease.
        // For now, checking presence and getting the mean is a start.
        assertFalse(Double.isNaN(policyLossSummaryOpt.get().getMean()), "Policy loss mean should not be NaN.");
        assertFalse(Double.isNaN(valueLossSummaryOpt.get().getMean()), "Value loss mean should not be NaN.");
    }

    private static VPGAgent vpg(int i, int o) {
        float s = 4;
        var h = round(i * s);
        var episodeLen = 2;
        return new VPGAgent(new VPGAgentConfig(new int[]{
                h, h
        }, episodeLen), i, o);
    }
    private static PPOAgent ppo(int i, int o) {
        float s = 4;
        var h = round(i * s);
        var episodeLen = 2;
        return new PPOAgent(new PPOAgentConfig(new int[]{
                h, h
        }, episodeLen), i, o);
    }

}
