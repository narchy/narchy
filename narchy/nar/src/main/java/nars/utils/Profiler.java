package nars.utils;

import org.HdrHistogram.Histogram;
import org.eclipse.collections.api.map.primitive.MutableObjectLongMap;
import org.eclipse.collections.impl.factory.primitive.ObjectLongMaps;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

public class Profiler {

    public static boolean PROFILING_ENABLED = false;

    // Using ConcurrentHashMap for thread safety, as different parts of NAR might be concurrent
    private static final Map<String, LongAdder> counters = new ConcurrentHashMap<>();
    private static final Map<String, Histogram> timers = new ConcurrentHashMap<>();
    private static final Map<String, LongAdder> totalTimes = new ConcurrentHashMap<>(); // For simpler total time tracking along with histograms

    // Histogram configuration (example, can be adjusted)
    private static final long HIGHEST_TRACKABLE_VALUE = 3_600_000_000_000L; // 1 hour in nanoseconds
    private static final int NUMBER_OF_SIGNIFICANT_DIGITS = 3;

    public static void enable() {
        PROFILING_ENABLED = true;
    }

    public static void disable() {
        PROFILING_ENABLED = false;
    }

    public static void reset() {
        counters.clear();
        timers.clear();
        totalTimes.clear();
    }

    public static void incrementCounter(String name) {
        if (!PROFILING_ENABLED) return;
        counters.computeIfAbsent(name, k -> new LongAdder()).increment();
    }

    public static long getCounter(String name) {
        LongAdder adder = counters.get(name);
        return adder != null ? adder.sum() : 0;
    }

    public static long startTime() {
        if (!PROFILING_ENABLED) return 0;
        return System.nanoTime();
    }

    public static void recordTime(String name, long startTimeNanos) {
        if (!PROFILING_ENABLED) return;
        if (startTimeNanos == 0) return; // Profiling was disabled when startTime was called

        long durationNanos = System.nanoTime() - startTimeNanos;

        timers.computeIfAbsent(name, k -> {
            Histogram histogram = new Histogram(HIGHEST_TRACKABLE_VALUE, NUMBER_OF_SIGNIFICANT_DIGITS);
            // It's good practice to enable auto-resize on histograms that might see a wide range of values
            // histogram.setAutoResize(true); // Requires HdrHistogram version that supports it
            return histogram;
        }).recordValue(durationNanos);

        totalTimes.computeIfAbsent(name, k -> new LongAdder()).add(durationNanos);
    }

    public static Histogram getHistogram(String name) {
        return timers.get(name);
    }

    public static long getTotalTimeNanos(String name) {
        LongAdder adder = totalTimes.get(name);
        return adder != null ? adder.sum() : 0;
    }

    public static String getStats() {
        if (!PROFILING_ENABLED && counters.isEmpty() && timers.isEmpty()) {
            return "Profiling is disabled or no data recorded.";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("--- Profiling Stats ---\n");

        sb.append("\n--- Counters ---\n");
        if (counters.isEmpty()) {
            sb.append("No counters recorded.\n");
        } else {
            counters.forEach((name, value) -> {
                sb.append(String.format("%s: %d\n", name, value.sum()));
            });
        }

        sb.append("\n--- Timers (Nanoseconds) ---\n");
        if (timers.isEmpty()) {
            sb.append("No timers recorded.\n");
        } else {
            timers.forEach((name, histogram) -> {
                sb.append(String.format("Timer: %s\n", name));
                sb.append(String.format("  Count: %d\n", histogram.getTotalCount()));
                sb.append(String.format("  Total Time (ms): %.3f\n", getTotalTimeNanos(name) / 1_000_000.0));
                if (histogram.getTotalCount() > 0) {
                    sb.append(String.format("  Min: %.3f ms\n", histogram.getMinValue() / 1_000_000.0));
                    sb.append(String.format("  Max: %.3f ms\n", histogram.getMaxValue() / 1_000_000.0));
                    sb.append(String.format("  Mean: %.3f ms\n", histogram.getMean() / 1_000_000.0));
                    sb.append(String.format("  50th Pctl (Median): %.3f ms\n", histogram.getValueAtPercentile(50) / 1_000_000.0));
                    sb.append(String.format("  90th Pctl: %.3f ms\n", histogram.getValueAtPercentile(90) / 1_000_000.0));
                    sb.append(String.format("  99th Pctl: %.3f ms\n", histogram.getValueAtPercentile(99) / 1_000_000.0));
                }
                sb.append("\n");
            });
        }
        sb.append("-----------------------\n");
        return sb.toString();
    }
}
