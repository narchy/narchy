package jcog.tensor.rl.pg2.stats;

/**
 * A listener interface for receiving notifications when a metric is recorded.
 * Implementations can choose to store, log, or process these metrics in real-time.
 */
@FunctionalInterface
public interface MetricListener {

    /**
     * Called when a metric is recorded.
     *
     * @param metricName The name of the metric (e.g., "policy_loss", "reward").
     * @param step       The step at which the metric was recorded (e.g., training update count, episode number).
     * @param value      The value of the metric.
     */
    void onMetricRecorded(String metricName, long step, double value);
}
