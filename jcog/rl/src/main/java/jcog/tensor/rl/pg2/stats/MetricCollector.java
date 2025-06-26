package jcog.tensor.rl.pg2.stats;


import org.eclipse.collections.api.tuple.Pair;

import java.util.Map;
import java.util.Optional;

/**
 * Interface for collecting and retrieving metrics during an agent's training or evaluation.
 * Implementations can choose how to store and process these metrics.
 */
public interface MetricCollector {

    /**
     * Records a metric value at a specific step.
     *
     * @param metricName The name of the metric (e.g., "policy_loss", "reward").
     * @param value      The value of the metric.
     * @param step       The step at which the metric was recorded (e.g., training update count).
     */
    void record(String metricName, double value, long step);

    /**
     * Sets a listener to be notified whenever a metric is recorded.
     * Only one listener can be active at a time. Setting a new listener
     * will replace the previous one.
     *
     * @param listener The {@link MetricListener} to be notified. Can be null to remove the listener.
     */
    void setMetricListener(MetricListener listener);

    /**
     * Retrieves the latest recorded value and step for a specific metric.
     *
     * @param metricName The name of the metric.
     * @return An {@link Optional} containing a {@link Pair} of (step, value) if the metric exists,
     *         otherwise an empty Optional.
     */
    Optional<Pair<Long, Double>> getLatestMetric(String metricName);

    /**
     * Retrieves the summary statistics for a specific metric.
     *
     * @param metricName The name of the metric.
     * @return An {@link Optional} containing the {@link MetricSummary} if the metric has been recorded,
     *         otherwise an empty Optional.
     */
    Optional<MetricSummary> getSummary(String metricName);

    /**
     * Retrieves a map of all metric names to their corresponding summary statistics.
     *
     * @return A map where keys are metric names and values are their {@link MetricSummary}.
     *         Returns an empty map if no metrics have been recorded.
     */
    Map<String, MetricSummary> getAllSummaries();

    /**
     * Resets the collector, clearing all recorded metrics, summaries, and removing any listener.
     * This is useful for starting fresh, e.g., between independent test runs.
     */
    void reset();
}
