package jcog.tensor.rl.agents.pg.configs;

import jcog.signal.FloatRange; // Assuming this exists

import java.util.Objects;

/**
 * Configuration for action selection properties of an RL agent.
 * This includes the type of action distribution, parameters for stochastic policies (like sigma bounds),
 * and configuration for action noise.
 *
 * @param distribution The type of probability distribution used by the policy (e.g., Gaussian for continuous actions).
 * @param sigmaMin Minimum standard deviation for Gaussian policies. Ensures exploration doesn't collapse.
 * @param sigmaMax Maximum standard deviation for Gaussian policies. Prevents overly erratic exploration.
 * @param noise Configuration for adding noise to actions, often used for exploration in deterministic policies.
 */
public record ActionConfig(
    Distribution distribution,
    FloatRange sigmaMin,
    FloatRange sigmaMax,
    NoiseConfig noise
) {
    /**
     * Defines the type of probability distribution the policy will use.
     */
    public enum Distribution {
        /** For continuous action spaces, typically outputting mean and standard deviation. */
        GAUSSIAN,
        /** For policies that output a single action value directly, without a stochastic component from a distribution. */
        DETERMINISTIC
    }

    /**
     * Configuration for action noise, typically used for exploration with deterministic policies
     * or sometimes to augment stochastic policies.
     *
     * @param type The type of noise to apply (e.g., Gaussian, Ornstein-Uhlenbeck).
     * @param stddev The standard deviation of the noise. For OU noise, this might be an initial magnitude.
     */
    public record NoiseConfig(
        Type type,
        FloatRange stddev
    ) {
        /**
         * Defines the type of noise process.
         */
        public enum Type {
            /** No action noise. */
            NONE,
            /** Simple Gaussian (normal) noise. */
            GAUSSIAN,
            /** Ornstein-Uhlenbeck process, often used for temporally correlated noise in continuous control. */
            OU
        }

        /** Default noise type: Gaussian. */
        public static final Type DEFAULT_NOISE_TYPE = Type.GAUSSIAN;
        /** Default noise standard deviation: 0.1, range [0, 1]. */
        public static final FloatRange DEFAULT_NOISE_STDDEV = new FloatRange(0.1f, 0.0f, 1.0f, 0.1f);

        /**
         * Validates noise configuration.
         */
        public NoiseConfig {
            Objects.requireNonNull(type, "Noise type cannot be null");
            Objects.requireNonNull(stddev, "Noise stddev cannot be null");
            if (stddev.floatValue() < 0 && type != Type.NONE) {
                throw new IllegalArgumentException("Noise stddev must be non-negative for GAUSSIAN or OU noise.");
            }
        }

        /**
         * Creates a default NoiseConfig (Gaussian noise with stddev 0.1).
         */
        public NoiseConfig() {
            this(DEFAULT_NOISE_TYPE, DEFAULT_NOISE_STDDEV);
        }

        /** Returns a new NoiseConfig with the specified type. */
        public NoiseConfig withType(Type type) { return new NoiseConfig(type, this.stddev); }
        /** Returns a new NoiseConfig with the specified stddev range. */
        public NoiseConfig withStddev(FloatRange stddev) { return new NoiseConfig(this.type, stddev); }
        /** Returns a new NoiseConfig with the specified stddev value, retaining existing min/max from the current stddev range. */
        public NoiseConfig withStddev(float stddevValue) {
            return new NoiseConfig(this.type, new FloatRange(stddevValue, this.stddev.min(), this.stddev.max(), stddevValue));
        }
    }

    /** Default action distribution: Gaussian. */
    public static final Distribution DEFAULT_ACTION_DISTRIBUTION = Distribution.GAUSSIAN;
    /** Default minimum sigma for Gaussian policies: 0.001, range [1e-6, 1.0]. */
    public static final FloatRange DEFAULT_SIGMA_MIN = new FloatRange(1e-3f, 1e-6f, 1.0f, 1e-3f);
    /** Default maximum sigma for Gaussian policies: 1.0, range [0.1, 10.0]. */
    public static final FloatRange DEFAULT_SIGMA_MAX = new FloatRange(1.0f, 0.1f, 10.0f, 1.0f);
    /** Default noise configuration. */
    public static final NoiseConfig DEFAULT_NOISE_CONFIG = new NoiseConfig();

    /**
     * Validates action configuration.
     */
    public ActionConfig {
        Objects.requireNonNull(distribution, "Action distribution cannot be null");
        Objects.requireNonNull(sigmaMin, "sigmaMin cannot be null");
        Objects.requireNonNull(sigmaMax, "sigmaMax cannot be null");
        Objects.requireNonNull(noise, "Noise config cannot be null");

        if (distribution == Distribution.GAUSSIAN) {
            if (sigmaMin.floatValue() <= 0) throw new IllegalArgumentException("sigmaMin must be positive for GAUSSIAN distribution");
            if (sigmaMax.floatValue() <= 0) throw new IllegalArgumentException("sigmaMax must be positive for GAUSSIAN distribution");
            if (sigmaMin.floatValue() >= sigmaMax.floatValue()) {
                throw new IllegalArgumentException("sigmaMin must be less than sigmaMax for GAUSSIAN distribution");
            }
        }
    }

    /**
     * Creates a default ActionConfig.
     * Uses Gaussian distribution, default sigma ranges, and default noise settings.
     */
    public ActionConfig() {
        this(DEFAULT_ACTION_DISTRIBUTION, DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_NOISE_CONFIG);
    }

    // Withers for ActionConfig
    /** Returns a new ActionConfig with the specified distribution. */
    public ActionConfig withDistribution(Distribution distribution) { return new ActionConfig(distribution, sigmaMin, sigmaMax, noise); }
    /** Returns a new ActionConfig with the specified sigmaMin range. */
    public ActionConfig withSigmaMin(FloatRange sigmaMin) {
        if (this.distribution == Distribution.GAUSSIAN && sigmaMin.floatValue() >= this.sigmaMax.floatValue()) {
            throw new IllegalArgumentException("sigmaMin must be less than current sigmaMax");
        }
        return new ActionConfig(distribution, sigmaMin, sigmaMax, noise);
    }
    /** Returns a new ActionConfig with the specified sigmaMin value, retaining current min/max from its range. */
    public ActionConfig withSigmaMin(float sigmaMinValue) { return withSigmaMin(new FloatRange(sigmaMinValue, this.sigmaMin.min(), this.sigmaMin.max(), sigmaMinValue)); }
    /** Returns a new ActionConfig with the specified sigmaMax range. */
    public ActionConfig withSigmaMax(FloatRange sigmaMax) {
         if (this.distribution == Distribution.GAUSSIAN && this.sigmaMin.floatValue() >= sigmaMax.floatValue()) {
            throw new IllegalArgumentException("current sigmaMin must be less than sigmaMax");
        }
        return new ActionConfig(distribution, sigmaMin, sigmaMax, noise);
    }
    /** Returns a new ActionConfig with the specified sigmaMax value, retaining current min/max from its range. */
    public ActionConfig withSigmaMax(float sigmaMaxValue) { return withSigmaMax(new FloatRange(sigmaMaxValue, this.sigmaMax.min(), this.sigmaMax.max(), sigmaMaxValue)); }
    /** Returns a new ActionConfig with the specified noise configuration. */
    public ActionConfig withNoiseConfig(NoiseConfig noise) { return new ActionConfig(distribution, sigmaMin, sigmaMax, noise); }
    /** Returns a new ActionConfig with the specified noise type. */
    public ActionConfig withNoiseType(NoiseConfig.Type noiseType) { return new ActionConfig(distribution, sigmaMin, sigmaMax, this.noise.withType(noiseType)); }
    /** Returns a new ActionConfig with the specified noise stddev range. */
    public ActionConfig withNoiseStddev(FloatRange noiseStddevRange) { return new ActionConfig(distribution, sigmaMin, sigmaMax, this.noise.withStddev(noiseStddevRange)); }
    /** Returns a new ActionConfig with the specified noise stddev value. */
    public ActionConfig withNoiseStddev(float noiseStddevValue) { return new ActionConfig(distribution, sigmaMin, sigmaMax, this.noise.withStddev(noiseStddevValue)); }
}
