package jcog.tensor.rl.pg3.util;

import jcog.math.FloatMeanWindow;
import jcog.tensor.Tensor;

/**
 * Normalizes observations using a running mean and variance calculated via Welford's algorithm
 * maintained by {@link FloatMeanWindow}.
 * This is useful for RL agents to stabilize learning when input features have varying scales or shift over time.
 */
public class RunningObservationNormalizer {

    private final FloatMeanWindow[] dimensionNormalizers;
    private final int dimensions;
    private final Tensor meanTensor; // Cached tensor for mean
    private final Tensor stddevTensor; // Cached tensor for stddev
    private boolean initialized; // Becomes true after first update for each dimension

    private static final double EPSILON = 1e-8; // For numerical stability when dividing by stddev

    /**
     * Constructs a running observation normalizer.
     *
     * @param dimensions The number of dimensions (features) in the observations.
     * @param history    The window size for calculating running mean and variance.
     */
    public RunningObservationNormalizer(int dimensions, int history) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("Number of dimensions must be positive.");
        }
        if (history <= 1) {
            // Welford's needs at least 2 samples for variance, FloatMeanWindow might handle history=1 for mean only.
            // For normalization, variance is key.
            throw new IllegalArgumentException("History window must be greater than 1 to compute variance.");
        }
        this.dimensions = dimensions;
        this.dimensionNormalizers = new FloatMeanWindow[dimensions];
        for (int i = 0; i < dimensions; i++) {
            this.dimensionNormalizers[i] = new FloatMeanWindow(history);
        }
        this.meanTensor = Tensor.zeros(1, dimensions);
        this.stddevTensor = Tensor.ones(1, dimensions); // Initialize stddev to 1 to avoid division by zero before updates
        this.initialized = false;
    }

    /**
     * Updates the running statistics with a new observation and returns the normalized observation.
     * The input observation tensor should be a row vector (1 x dimensions).
     *
     * @param observation The observation tensor to update with and normalize.
     * @return The normalized observation tensor.
     */
    public Tensor normalize(Tensor observation) {
        if (observation.rows() != 1 || observation.cols() != dimensions) {
            throw new IllegalArgumentException("Observation tensor must have shape (1, " + dimensions + "). " +
                                               "Got: (" + observation.rows() + ", " + observation.cols() + ")");
        }

        double[] obsArray = observation.array(); // Assumes Tensor.array() returns a flat array for a row vector

        for (int i = 0; i < dimensions; i++) {
            double[] meanVar = dimensionNormalizers[i].acceptAndGetMeanAndVariance((float) obsArray[i]);
            meanTensor.data(i, meanVar[0]);
            // Ensure variance is non-negative; stddev is sqrt(variance)
            double variance = Math.max(0.0, meanVar[1]);
            stddevTensor.data(i, Math.sqrt(variance));
        }

        if (!initialized && dimensionNormalizers[0].count() > 1) {
             // Considered initialized once there's enough data for variance in at least one normalizer
            initialized = true;
        }

        // Normalize: (observation - mean) / (stddev + epsilon)
        // Only apply normalization if initialized (i.e., have seen enough samples for stable variance)
        if (initialized) {
            return observation.sub(meanTensor).div(stddevTensor.add(EPSILON));
        } else {
            // Return original observation if not enough data yet for stable normalization
            return observation;
        }
    }

    /**
     * Returns the current running mean for each dimension.
     * @return A Tensor (1 x dimensions) of means.
     */
    public Tensor getMean() {
        return meanTensor.copy(); // Return a copy
    }

    /**
     * Returns the current running standard deviation for each dimension.
     * @return A Tensor (1 x dimensions) of standard deviations.
     */
    public Tensor getStddev() {
        return stddevTensor.copy(); // Return a copy
    }

    /**
     * Resets the statistics for all dimensions.
     */
    public void reset() {
        for (int i = 0; i < dimensions; i++) {
            this.dimensionNormalizers[i].clear(); // Assuming FloatMeanWindow has a clear/reset method
            meanTensor.data(i, 0.0);
            stddevTensor.data(i, 1.0); // Reset stddev to 1
        }
        this.initialized = false;
    }
}
