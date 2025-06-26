package jcog.tensor.rl.pg2.stats;

import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.tuple.Tuples;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A default implementation of {@link MetricCollector}.
 * It stores the latest value of each metric and summary statistics (mean, min, max, count).
 * It does not store the full history of metrics by default to save memory, but can notify a
 * {@link MetricListener} for custom processing or storage of all values.
 * This implementation is thread-safe for recording and retrieval.
 */
public class DefaultMetricCollector implements MetricCollector {

    private final Map<String, Pair<Long, Double>> latestMetrics;
    private final Map<String, MetricSummary> metricSummaries;
    private MetricListener metricListener;

    public DefaultMetricCollector() {
        // Use ConcurrentHashMap for thread-safety, as agents might be updated or queried from different threads.
        this.latestMetrics = new ConcurrentHashMap<>();
        this.metricSummaries = new ConcurrentHashMap<>();
        this.metricListener = null;
    }

    @Override
    public void record(String metricName, double value, long step) {
        if (metricName == null || metricName.trim().isEmpty()) {
            // Or throw IllegalArgumentException, depending on desired strictness
            System.err.println("Metric name cannot be null or empty.");
            return;
        }
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            // Or throw IllegalArgumentException
            System.err.println("Metric value cannot be NaN or Infinite for metric: " + metricName);
            return;
        }

        latestMetrics.put(metricName, Tuples.pair(step, value));

        MetricSummary summary = metricSummaries.computeIfAbsent(metricName, k -> new MetricSummary());
        summary.update(value);

        // Notify listener if one is set
        MetricListener currentListener = this.metricListener; // Read volatile/final field once
        if (currentListener != null) {
            try {
                currentListener.onMetricRecorded(metricName, step, value);
            } catch (Exception e) {
                // Log or handle listener exceptions appropriately
                System.err.println("MetricListener threw an exception: " + e.getMessage());
                e.printStackTrace(System.err);
            }
        }
    }

    @Override
    public void setMetricListener(MetricListener listener) {
        this.metricListener = listener;
    }

    @Override
    public Optional<Pair<Long, Double>> getLatestMetric(String metricName) {
        return Optional.ofNullable(latestMetrics.get(metricName));
    }

    @Override
    public Optional<MetricSummary> getSummary(String metricName) {
        return Optional.ofNullable(metricSummaries.get(metricName));
    }

    @Override
    public Map<String, MetricSummary> getAllSummaries() {
        // Return an immutable copy to prevent external modification
        return Collections.unmodifiableMap(new HashMap<>(metricSummaries));
    }

    @Override
    public void reset() {
        latestMetrics.clear();
        metricSummaries.clear();
        // Do not reset the listener, as it might be intended to persist across resets of data.
        // If listener reset is desired, a separate method or explicit null set is better.
        // Based on plan "Resets the collector, clearing all recorded metrics, summaries, and removing any listener."
        // -> so, listener should be removed.
        this.metricListener = null;

        // If MetricSummary objects themselves need reset (if they were reused from a pool, not here)
        // metricSummaries.values().forEach(MetricSummary::reset); // Not needed with current new MetricSummary()
    }
}
