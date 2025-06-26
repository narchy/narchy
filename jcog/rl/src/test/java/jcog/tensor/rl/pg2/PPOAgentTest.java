package jcog.tensor.rl.pg2;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg2.configs.*;
import jcog.tensor.rl.pg2.stats.DefaultMetricCollector;
import jcog.tensor.rl.pg2.stats.MetricCollector;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PPOAgentTest {

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

        public Triplet<Tensor, Double, Boolean> step(double[] action) {
            var move = Util.clamp(action[0], -stepSize, stepSize);
            this.currentState = Util.clamp(this.currentState + move, stateMin, stateMax);
            this.currentStepInEpisode++;
            var reward = -Math.abs(this.currentState - this.targetState);
            var done = this.currentStepInEpisode >= this.maxStepsPerEpisode || Math.abs(this.currentState - targetState) < 0.01;
            if (Math.abs(this.currentState - targetState) < 0.01) {
                reward += 10.0; // Bonus for reaching target
            }
            return new Triplet<>(Tensor.scalar(this.currentState), reward, done);
        }

        public int getStateDim() { return 1; }
        public int getActionDim() { return 1; }
    }

    record Triplet<S, R, D>(S state, R reward, D done) {}

    private PPOAgent createPPOAgent(int stateDim, int actionDim, MetricCollector metricCollector) {
        var policyNetConfig = new NetworkConfig(new int[]{64, 64}, "relu", "tanh", new OptimizerConfig("adam", 3e-4, 0.9, 0.999, 1e-7, 0.0));
        var valueNetConfig = new NetworkConfig(new int[]{64, 64}, "relu", null, new OptimizerConfig("adam", 1e-3, 0.9, 0.999, 1e-7, 0.0)); // Value net output is scalar
        var actionConfig = new ActionConfig(0.05, 0.5, "gaussian"); // Adjusted sigma for potentially more stable exploration
        var memoryConfig = new MemoryConfig(256, "on_policy"); // Buffer size for PPO updates
        var hyperparamConfig = new HyperparamConfig(
            0.99,  // gamma
            0.95,  // lambda (for GAE)
            0.2,   // ppoClip
            0.01,  // entropyBonus
            10,    // epochs for PPO update
            true,  // normalizeReturns (PPO often normalizes advantages, which implies returns are processed for GAE)
            true   // normalizeAdvantages
        );

        var agentConfig = new PPOAgentConfig(policyNetConfig, valueNetConfig, actionConfig, memoryConfig, hyperparamConfig);
        // PPOAgent constructor takes MetricCollector
        return new PPOAgent(agentConfig, stateDim, actionDim, metricCollector);
    }

    @Test
    void testPPOAgentLearnsOnPointNavigation() {
        var env = new PointNavigation1D();
        var stateDim = env.getStateDim();
        var actionDim = env.getActionDim();
        MetricCollector metrics = new DefaultMetricCollector();

        var agent = createPPOAgent(stateDim, actionDim, metrics);
        // MetricCollector is passed to PPOAgent, so 'metrics' object will be populated.

        agent.setTrainingMode(true);

        var numEpisodes = 200; // PPO might learn faster or with fewer episodes than REINFORCE

        List<Double> episodeRewards = new ArrayList<>();
        double totalRewardLast50Episodes = 0;

        System.out.println("Starting PPO Agent training on PointNavigation1D...");
        long globalStep = 0;

        for (var episode = 0; episode < numEpisodes; episode++) {
            Tensor currentStateTensor = env.reset();
            double currentEpisodeReward = 0;

            Tensor previousStateTensor = null;
            double[] previousActionArray = null;
            double rewardFromPreviousStep = 0.0;
            boolean firstStepInEpisode = true;

            for (var t = 0; t < env.maxStepsPerEpisode; t++) {
                globalStep++;
                double[] currentActionArray = new double[actionDim];

                // agent.apply will use the PPOAgent's overridden act method,
                // which handles selectActionWithLogProb and stores experience with logProb.
                agent.apply(
                    firstStepInEpisode ? null : previousStateTensor,
                    firstStepInEpisode ? null : previousActionArray,
                    (float) rewardFromPreviousStep,
                    currentStateTensor,
                    currentActionArray
                );

                Triplet<Tensor, Double, Boolean> stepResult = env.step(currentActionArray);
                Tensor nextStateTensor = stepResult.state();
                double currentReward = stepResult.reward();
                boolean done = stepResult.done();

                previousStateTensor = currentStateTensor;
                previousActionArray = currentActionArray.clone();
                rewardFromPreviousStep = currentReward;

                currentStateTensor = nextStateTensor;
                currentEpisodeReward += currentReward;
                firstStepInEpisode = false;

                // PPO updates when its buffer (memory) is full.
                if (agent.memory.size() >= agent.config.memoryConfig().episodeLength().intValue()) {
                    if (agent.memory.size() > 0) {
                        agent.update(globalStep);
                        agent.memory.clear(); // Standard for on-policy PPO
                    }
                }

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
                // PPOAgent is expected to record these metrics.
                metrics.getSummary("policy_loss").ifPresent(s -> System.out.printf("  Policy Loss: %.4f%n", s.getMean()));
                metrics.getSummary("value_loss").ifPresent(s -> System.out.printf("  Value Loss: %.4f%n", s.getMean()));
                metrics.getSummary("entropy").ifPresent(s -> System.out.printf("  Entropy: %.4f%n", s.getMean()));
                metrics.getSummary("gae_advantages_raw_mean").ifPresent(s -> System.out.printf("  GAE Raw Mean: %.4f%n", s.getMean()));
                metrics.getSummary("gae_advantages_normalized_mean").ifPresent(s -> System.out.printf("  GAE Norm Mean: %.4f%n", s.getMean()));
                metrics.getSummary("policy_sigma_mean").ifPresent(s -> System.out.printf("  Sigma Mean: %.4f%n", s.getMean()));


            }
        }

        System.out.println("Training finished.");
        System.out.println("Total updates: " + agent.getUpdateCount());

        assertTrue(agent.getUpdateCount() > 0, "Agent should have performed updates.");

        var policyLossSummaryOpt = metrics.getSummary("policy_loss");
        assertTrue(policyLossSummaryOpt.isPresent(), "Policy loss should have been recorded.");
        assertFalse(Double.isNaN(policyLossSummaryOpt.get().getMean()), "Policy loss mean should not be NaN.");

        var valueLossSummaryOpt = metrics.getSummary("value_loss");
        assertTrue(valueLossSummaryOpt.isPresent(), "Value loss should have been recorded.");
        assertFalse(Double.isNaN(valueLossSummaryOpt.get().getMean()), "Value loss mean should not be NaN.");

        var entropySummaryOpt = metrics.getSummary("entropy");
        assertTrue(entropySummaryOpt.isPresent(), "Entropy should have been recorded if entropy bonus > 0.");
        if (agent.config.hyperparams().entropyBonus() > 0) {
            assertFalse(Double.isNaN(entropySummaryOpt.get().getMean()), "Entropy mean should not be NaN.");
        }


        var avgRewardFirst50 = episodeRewards.subList(0, Math.min(50, episodeRewards.size()))
                                    .stream().mapToDouble(r -> r).average().orElse(Double.NEGATIVE_INFINITY);
        int last50StartIndex = Math.max(0, numEpisodes - 50);
        int numActualLastEpisodes = episodeRewards.size() - last50StartIndex;
        var avgRewardLast50 = totalRewardLast50Episodes / Math.max(1, numActualLastEpisodes);

        System.out.printf("Avg Reward First 50 Episodes: %.2f%n", avgRewardFirst50);
        System.out.printf("Avg Reward Last 50 Episodes: %.2f%n", avgRewardLast50);

        assertTrue(avgRewardLast50 > avgRewardFirst50 + 10, // PPO might show more significant improvement
                "Average reward in later episodes should be significantly higher. Last50: " + avgRewardLast50 + ", First50: " + avgRewardFirst50);
    }
}
