package jcog.tensor.rl.blackbox;

import jcog.Util;
import jcog.signal.FloatRange;
import jcog.tensor.rl.dqn.Policy;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Policy implementation using blackbox optimization with adaptive evaluation periods.
 * Uses exponential moving statistics and sequential analysis for robust evaluation timing.
 * Implements confidence-based early termination and precision-targeted period adjustment.
 */
public abstract class BlackboxPolicy implements Policy {

    public final FloatRange exploit = FloatRange.unit(0);
    private boolean exploring;
    public final int outputs;
    protected final int populationSize;
    protected final double[] rewards;

    // Adaptive evaluation configuration
    private final int minPeriod;
    private final int maxPeriod;

    private final double confidenceThreshold;  // Statistical confidence (recommended: 2.0)
    private final double earlyStopThreshold;  // Early termination threshold (recommended: 1.5)
    private final StatTracker rewardStats;

    private int currentPeriod;
    public double[] policy;
    final double[] actions;
    public double reward = Double.NaN;
    int currentIndividual;
    int ttl;

    // Constants for numerical stability
    private static final double EPSILON = 1e-10;
    private static final double MIN_RELATIVE_CHANGE = 0.5;
    private static final int MIN_SAMPLES = 3;

    public BlackboxPolicy(int numOutputs, int populationSize, int minPeriod, int maxPeriod) {
        this(numOutputs, populationSize, minPeriod, maxPeriod, 0.1, 0.1f);
    }
    public BlackboxPolicy(int numOutputs, int populationSize, int minPeriod, int maxPeriod,
                          double emaAlpha, double confidenceThreshold) {
        if (minPeriod <= 0 || maxPeriod < minPeriod || emaAlpha <= 0 || emaAlpha > 1 ||
                confidenceThreshold <= 0) {
            throw new IllegalArgumentException("Invalid parameter values");
        }

        this.outputs = numOutputs;
        this.populationSize = populationSize;
        this.minPeriod = minPeriod;
        this.maxPeriod = maxPeriod;

        this.confidenceThreshold = confidenceThreshold;
        this.earlyStopThreshold = confidenceThreshold * 0.75;

        this.currentPeriod = minPeriod;
        this.policy = new double[numOutputs];
        this.rewards = new double[populationSize];
        this.actions = new double[numOutputs];
        this.rewardStats = new StatTracker(emaAlpha);

        clear(ThreadLocalRandom.current());
    }

    @Override
    public void clear(Random rng) {
        Arrays.fill(rewards, 0);
        Arrays.fill(actions, 0);
        rewardStats.reset();
        currentPeriod = minPeriod;
        currentIndividual = -1;
        ttl = currentPeriod;
        exploring = false;
        evalIterations = 0;
    }

    protected void nextIndividual() {
        if (exploring)
            processCurrentIndividual();

        exploring = !shouldExploit();
        ttl = currentPeriod;

        currentIndividual = exploring ?
            (currentIndividual == -1) ? 0 :
            (currentIndividual + 1) % populationSize : -1;

        System.arraycopy(exploring ? next() : best(), 0, policy, 0, policy.length);

        evalIterations = 0;
    }

    private boolean shouldExploit() {
        return ThreadLocalRandom.current().nextFloat() < exploit.floatValue();
    }

    private void processCurrentIndividual() {
        double evalPeriod = evalIterations;
        if (evalIterations<1) throw new UnsupportedOperationException();

        var avgReward = rewardMean(evalPeriod);

        commitActions(evalPeriod, avgReward);

        commitReward(avgReward);

        adjustEvaluationPeriod();
    }

    private void commitReward(double avgReward) {
        rewardStats.update(avgReward);
        rewards[currentIndividual] = 0;
    }

    private void commitActions(double evalPeriod, double avgReward) {
        Util.mul(actions, 1.0/ evalPeriod);
        commitIndividual(actions, avgReward);
    }

    private double rewardMean(double evalPeriod) {
        return rewards[currentIndividual] / evalPeriod;
    }

    private void adjustEvaluationPeriod() {
        if (!rewardStats.statsValid())
            return;

        var standardError = rewardStats.standardErr();
        var mean = rewardStats.ema;
        var relativeError = Math.abs(standardError / (Math.abs(mean) + EPSILON));
        var e = relativeError * Math.sqrt(currentPeriod);

        // If relative error is high, we need MORE samples (longer period)
        if (e > confidenceThreshold)
            increasePeriod();  // More samples needed for higher precision

        // If relative error is very low, we can get by with FEWER samples
        else if (e < confidenceThreshold/2)
            decreasePeriod();  // Fewer samples needed since precision is good
    }

    private void increasePeriod() {
        var delta = Math.max(1, (int)(currentPeriod * MIN_RELATIVE_CHANGE));
        currentPeriod = Math.min(currentPeriod + delta, maxPeriod);
    }

    private void decreasePeriod() {
        var delta = Math.max(1, (int)(currentPeriod * MIN_RELATIVE_CHANGE));
        currentPeriod = Math.max(currentPeriod - delta, minPeriod);
    }

    private int evalIterations;

    @Override
    public double[] learn(double[] xPrev_ignored, double[] actionPrev, double reward,
                          double[] x, float pri) {
        if (exploring) {
            rewards[currentIndividual] += (this.reward = reward);
            Util.addTo(actions, actionPrev);
            evalIterations++;
        }

        if (--ttl <= 0 || (exploring && earlyStop()))
            nextIndividual();

        if (exploring && currentIndividual >= populationSize) {
            commit();
            update(policy);
        }

        return policy;
    }

    protected boolean earlyStop() {
        if (!exploring || ttl < minPeriod/2 || !rewardStats.statsValid())
            return false;

        double evalPeriod = evalIterations;
        if (evalPeriod < MIN_SAMPLES)
            return false;

        var currentMean = rewardMean(evalPeriod);
        var meanDiff = Math.abs(currentMean - rewardStats.ema);
        var threshold = rewardStats.standardErr() * earlyStopThreshold;
        return meanDiff < threshold;
    }

    protected void update(double[] policy) {}

    protected void commit() {
        commitPopulation(rewards);
        Arrays.fill(rewards, 0);
        currentIndividual = 0;
    }

    public int parameters() {
        return outputs;
    }

    public double[] policy() {
        return policy;
    }

    protected abstract double[] next();
    protected abstract double[] best();
    protected abstract void commitPopulation(double[] rewards);
    protected void commitIndividual(double[] actualActions, double reward) {}

    /**
     * Maintains numerically stable exponential moving statistics
     * TODO use Hipparchus's stats or something
     */
    @Deprecated private static class StatTracker {
        private double ema;      // Exponential moving average
        private double emv;      // Exponential moving variance
        private int count;

        /** EMA decay factor (recommended: 0.1) */
        private final double alpha;

        StatTracker(double alpha) {
            this.alpha = alpha;
            reset();
        }

        void reset() {
            ema = emv = 0;
            count = 0;
        }

        void update(double value) {
            count++;

            if (count == 1) {
                ema = value;
                emv = 0;
            } else {
                // Numerically stable EMA update
                var delta = value - ema;
                ema += alpha * delta;
                // Welford's online algorithm variant for variance
                emv = (1 - alpha) * (emv + alpha * delta * delta);
            }
        }

        boolean statsValid() {
            return count >= MIN_SAMPLES;
        }

        double standardErr() {
            return Math.sqrt(Math.max(EPSILON, emv / count));
        }
    }
}