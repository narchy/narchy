package jcog.tensor.rl.pg2.stats;

import java.util.Locale;

/**
 * Holds summary statistics for a specific metric.
 * This includes the sum of values, count of recordings, minimum value, and maximum value.
 * The mean can be calculated from the sum and count.
 */
public class MetricSummary {
    private double sum = 0.0;
    private long count = 0;
    private double min = Double.MAX_VALUE;
    private double max = Double.MIN_VALUE;

    /**
     * Default constructor.
     */
    public MetricSummary() {
    }

    /**
     * Updates the summary statistics with a new value.
     *
     * @param value The new metric value to incorporate into the summary.
     */
    public void update(double value) {
        this.sum += value;
        this.count++;
        this.min = Math.min(this.min, value);
        this.max = Math.max(this.max, value);
    }

    /**
     * Gets the sum of all recorded values for this metric.
     *
     * @return The sum of values.
     */
    public double getSum() {
        return sum;
    }

    /**
     * Gets the total number of times this metric has been recorded.
     *
     * @return The count of recordings.
     */
    public long getCount() {
        return count;
    }

    /**
     * Gets the minimum value recorded for this metric.
     * Returns {@link Double#MAX_VALUE} if no values have been recorded.
     *
     * @return The minimum value.
     */
    public double getMin() {
        return count == 0 ? Double.MAX_VALUE : min;
    }

    /**
     * Gets the maximum value recorded for this metric.
     * Returns {@link Double#MIN_VALUE} if no values have been recorded.
     *
     * @return The maximum value.
     */
    public double getMax() {
        return count == 0 ? Double.MIN_VALUE : max;
    }

    /**
     * Calculates the mean (average) of all recorded values for this metric.
     * Returns {@link Double#NaN} if no values have been recorded (count is zero).
     *
     * @return The mean of the values, or NaN if count is zero.
     */
    public double getMean() {
        if (count == 0) {
            return Double.NaN;
        }
        return sum / count;
    }

    /**
     * Resets the summary statistics to their initial state.
     */
    public void reset() {
        this.sum = 0.0;
        this.count = 0;
        this.min = Double.MAX_VALUE;
        this.max = Double.MIN_VALUE;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "MetricSummary{count=%d, mean=%.4f, min=%.4f, max=%.4f, sum=%.4f}",
                count, getMean(), getMin(), getMax(), sum);
    }
}
