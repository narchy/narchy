package jcog.decide;

import jcog.Util;
import jcog.signal.FloatRange;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import static jcog.Str.n4;

/**
 * priority calculation using the UCB1 algorithm principle
 */
public abstract class UCBPrioritizer<X,Y> {

    public final FloatRange explore = new FloatRange(1, 0, 1);

    private final Map<Y, Stats> stats = new HashMap<>();
    private long totalUpdates = 0;

    public double pri(X x) {
        var stats = this.stats.computeIfAbsent(type(x), k -> new Stats());
        double exploitability = stats.score();
        double explorability = Math.sqrt(
            2 * Util.log1p(totalUpdates) / (stats.count + 1)
        );
        return exploitability + this.explore.doubleValue() * explorability;
    }

    public void update(X x, double score) {
        stats.get(type(x)).update(score);
        totalUpdates++;
    }

    abstract protected Y type(X x);

    public String status() {
        return stats.entrySet().stream().map(e ->
            e.getKey() + "=" + n4(e.getValue().score())
        ).collect(Collectors.joining(" "));
    }

    public void clear() {
        stats.clear();
        totalUpdates = 0;
    }
    private static class Stats {
        long count = 0;
        double scoreSum = 0;

        void update(double score) {
            count++;
            scoreSum += score;
        }

        double score() {
            return count > 0 ? scoreSum / count : 0;
        }

        void clear() {
            count = 0;
            scoreSum = 0;
        }

    }
}