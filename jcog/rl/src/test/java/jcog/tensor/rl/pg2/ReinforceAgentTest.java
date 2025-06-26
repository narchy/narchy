package jcog.tensor.rl.pg2;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg2.configs.ActionConfig;
import jcog.tensor.rl.pg2.configs.HyperparamConfig;
import jcog.tensor.rl.pg2.configs.MemoryConfig;
import jcog.tensor.rl.pg2.configs.NetworkConfig;
import jcog.tensor.rl.pg2.configs.OptimizerConfig;
import jcog.tensor.rl.pg2.configs.ReinforceAgentConfig;
import jcog.tensor.rl.pg2.stats.DefaultMetricCollector;
import jcog.tensor.rl.pg2.stats.MetricCollector;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.Math.round;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ReinforceAgentTest {

    // Simple environment for testing: PointNavigation1D
    // Copied from PolicyGradientTest - consider refactoring to a shared test utility if used by more tests.
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

    private ReinforceAgent createReinforceAgent(int stateDim, int actionDim, MetricCollector metricCollector) {
        var netConfig = new NetworkConfig(new int[]{64, 64}, "relu", "tanh", new OptimizerConfig("adam", 1e-3, 0.9, 0.999, 1e-7, 0.0));
        var actionConfig = new ActionConfig(0.1, 1.0, "gaussian"); // sigmaMin, sigmaMax might need tuning for REINFORCE
        var memoryConfig = new MemoryConfig(256, "on_policy"); // episodeLength
        var hyperparamConfig = new HyperparamConfig(
            0.99, // gamma
            null, // lambda (not used by REINFORCE)
            null, // ppoClip (not used by REINFORCE)
            0.01, // entropyBonus
            null, // epochs (not used by REINFORCE in the same way as PPO)
            true, // normalizeReturns
            false // normalizeAdvantages (REINFORCE uses returns, not advantages in the same way as A2C/PPO)
        );

        var agentConfig = new ReinforceAgentConfig(netConfig, actionConfig, memoryConfig, hyperparamConfig);
        return new ReinforceAgent(agentConfig, stateDim, actionDim); // MetricCollector is handled by AbstractPolicyGradientAgent if null
    }

    @Test
    void testReinforceAgentLearnsOnPointNavigation() {
        var env = new PointNavigation1D();
        var stateDim = env.getStateDim();
        var actionDim = env.getActionDim();
        MetricCollector metrics = new DefaultMetricCollector(); // Agent will use its own if this is passed as null to constructor

        var agent = createReinforceAgent(stateDim, actionDim, metrics);
        // If createReinforceAgent doesn't take metrics, get it from agent:
        // MetricCollector actualMetrics = agent.getMetricCollector();


        agent.setTrainingMode(true);

        var numEpisodes = 300; // REINFORCE might need more episodes or different hyperparams

        List<Double> episodeRewards = new ArrayList<>();
        double totalRewardLast50Episodes = 0;

        System.out.println("Starting REINFORCE Agent training on PointNavigation1D...");
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

                // For REINFORCE, update typically happens at the end of an episode,
                // or when memory buffer (if sized for multiple episodes) is full.
                // The current ReinforceAgent.update() processes whatever is in memory.
                // If memoryConfig.episodeLength is small, it might update mid-episode.
                // For classic REINFORCE, memory should store one full episode.
                // Let's assume memoryConfig.episodeLength is set to handle this, or update at 'done'.
                // The current ReinforceAgent clears memory after update in its `update` method if it processes full buffer.
                // However, the AbstractPolicyGradientAgent.recordExperience now doesn't auto-update.
                // The current ReinforceAgent.update() is designed to process the whole buffer.
                // So, we should fill the buffer and then update.
                // If memoryConfig.episodeLength is, say, 256, and maxStepsPerEpisode is 50,
                // it will collect a few episodes before updating.

                if (done || agent.memory.size() >= agent.config.memoryConfig().episodeLength().intValue()) {
                     if (agent.memory.size() > 0) { // Ensure there's something to update
                        agent.update(globalStep);
                        // REINFORCE typically clears memory after each update (episode)
                        // The provided ReinforceAgent might do this internally, or we do it here.
                        // VPGAgent and PPOAgent clear memory in the test loop after update.
                        // ReinforceAgent.java: "Memory is cleared in recordExperience after 'done' for REINFORCE." -> This is old comment.
                        // Let's clear it here to be consistent with other on-policy tests.
                        agent.memory.clear();
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
                // REINFORCE might not log value_loss. Check for policy_loss and entropy if configured.
                // Assuming ReinforceAgent records "policy_loss" and "entropy" via recordMetric in AbstractPolicyGradientAgent
                // For now, let's assume ReinforceAgent itself needs to call recordMetric for these.
                // Checking AbstractPolicyGradientAgent.recordMetric - it exists.
                // ReinforceAgent.java currently doesn't explicitly call recordMetric.
                // This needs to be added to ReinforceAgent.update() to see these metrics.
                // For now, the test will pass if learning occurs, but metric checks might fail or be absent.
                // TODO: Add metric recording to ReinforceAgent.update()
                 agent.getMetricCollector().getSummary("policy_loss").ifPresent(s -> System.out.printf("  Policy Loss: %.4f%n", s.getMean()));
                 agent.getMetricCollector().getSummary("entropy").ifPresent(s -> System.out.printf("  Entropy: %.4f%n", s.getMean()));
            }
        }

        System.out.println("Training finished.");
        System.out.println("Total updates: " + agent.getUpdateCount());

        assertTrue(agent.getUpdateCount() > 0, "Agent should have performed updates.");

        // Check that average reward in the last 50 episodes is significantly better than the first 50
        var avgRewardFirst50 = episodeRewards.subList(0, Math.min(50, episodeRewards.size()))
                                    .stream().mapToDouble(r -> r).average().orElse(Double.NEGATIVE_INFINITY);
        // Correct calculation for avgRewardLast50 if numEpisodes < 50 or numEpisodes-50 < 0
        int last50StartIndex = Math.max(0, numEpisodes - 50);
        int numActualLastEpisodes = episodeRewards.size() - last50StartIndex;
        var avgRewardLast50 = totalRewardLast50Episodes / Math.max(1, numActualLastEpisodes);


        System.out.printf("Avg Reward First 50 Episodes: %.2f%n", avgRewardFirst50);
        System.out.printf("Avg Reward Last 50 Episodes: %.2f%n", avgRewardLast50);

        assertTrue(avgRewardLast50 > avgRewardFirst50 + 5,
                "Average reward in later episodes should be significantly higher. Last50: " + avgRewardLast50 + ", First50: " + avgRewardFirst50);

        // Optional: Check policy_loss if it's being recorded by ReinforceAgent
        var policyLossSummaryOpt = agent.getMetricCollector().getSummary("policy_loss");
        // If ReinforceAgent is updated to record metrics:
        // assertTrue(policyLossSummaryOpt.isPresent(), "Policy loss should have been recorded if agent implements it.");
        // if (policyLossSummaryOpt.isPresent()) {
        //     assertFalse(Double.isNaN(policyLossSummaryOpt.get().getMean()), "Policy loss mean should not be NaN.");
        // }
        System.out.println("Note: ReinforceAgent metric recording (policy_loss, entropy) needs to be implemented in its update() method for detailed checks.");
    }
}
