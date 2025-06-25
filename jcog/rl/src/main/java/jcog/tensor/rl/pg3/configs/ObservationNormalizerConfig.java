package jcog.tensor.rl.pg3.configs;

import jcog.Util;

/**
 * Configuration for running observation normalization.
 *
 * @param enabled Whether observation normalization is enabled.
 * @param history The window size for calculating running mean and variance.
 */
public record ObservationNormalizerConfig(
    boolean enabled,
    int history
) {
    public static final int DEFAULT_HISTORY = 10000; // Default window size

    public ObservationNormalizerConfig {
        if (enabled) {
            Util.validate(history, h -> h > 1, "History window must be greater than 1 for normalization.");
        }
    }

    /**
     * Creates a default disabled ObservationNormalizerConfig.
     */
    public ObservationNormalizerConfig() {
        this(false, DEFAULT_HISTORY);
    }

    /**
     * Creates an enabled ObservationNormalizerConfig with default history.
     */
    public static ObservationNormalizerConfig createEnabled() {
        return new ObservationNormalizerConfig(true, DEFAULT_HISTORY);
    }

    /**
     * Creates an enabled ObservationNormalizerConfig with specified history.
     */
    public static ObservationNormalizerConfig createEnabled(int history) {
        return new ObservationNormalizerConfig(true, history);
    }
}
