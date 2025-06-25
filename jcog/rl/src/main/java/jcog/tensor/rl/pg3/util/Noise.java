package jcog.tensor.rl.pg3.util;

import jcog.Util;
import jcog.math.Range;
import jcog.tensor.rl.pg3.configs.ActionConfig;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Utility class for generating noise, typically used for exploration in continuous action spaces
 * by algorithms like DDPG.
 */
public final class Noise {

    private Noise() {
        // Utility class
    }

    /**
     * Interface for different noise generation strategies.
     */
    public interface NoiseStrategy {
        /**
         * Applies noise to the given action array in-place.
         *
         * @param action The action array to add noise to.
         * @param rng    A random number generator.
         */
        void apply(double[] action, RandomGenerator rng);

        /**
         * Resets the internal state of the noise process, if any.
         */
        void reset();

        /**
         * Creates a NoiseStrategy based on the provided configuration.
         * @param config The noise configuration.
         * @param actionDim The dimensionality of the action space.
         * @return A configured NoiseStrategy instance.
         */
        static NoiseStrategy create(ActionConfig.NoiseConfig config, int actionDim) {
            Objects.requireNonNull(config, "NoiseConfig cannot be null");
            Util.validate(actionDim, ad -> ad > 0, "Action dimension must be positive");

            // Assuming PerlinNoiseConfig might be part of ActionConfig.NoiseConfig in the future,
            // or these parameters are added to NoiseConfig directly.
            // For now, using some defaults for Perlin timeStep if not in config.
            // A more robust solution would be to have Perlin-specific params in NoiseConfig.
            float perlinTimeStep = config.ouDt() != null ? config.ouDt().floatValue() : 0.1f; // Reuse ouDt for now, or add dedicated param

            return switch (config.type()) {
                case OU -> new OrnsteinUhlenbeckNoise(actionDim, config.stddev().floatValue(),
                                                      config.ouTheta().floatValue(), config.ouDt().floatValue());
                case GAUSSIAN -> new GaussianNoise(config.stddev().floatValue());
                case PERLIN -> new PerlinNoise.PerlinNoiseStrategy(actionDim, config.stddev().floatValue(), perlinTimeStep);
                case NONE -> new NoneNoise();
            };
        }
    }

    /**
     * Ornstein-Uhlenbeck noise process.
     * This process generates temporally correlated noise, which can be beneficial for exploration
     * in physical environments with momentum.
     * dxt = theta * (mu - xt) * dt + sigma * sqrt(dt) * Wt
     */
    public static class OrnsteinUhlenbeckNoise implements NoiseStrategy {
        private final int size;
        private final double mu;
        private final double theta;
        private final double sigma;
        private final double dt;
        private final double[] state;

        public OrnsteinUhlenbeckNoise(int size, double sigma, double theta, double dt) {
            this(size, 0.0, theta, sigma, dt);
        }

        public OrnsteinUhlenbeckNoise(int size, double mu, double theta, double sigma, double dt) {
            this.size = size;
            this.mu = mu;
            this.theta = theta;
            this.sigma = sigma;
            this.dt = dt;
            this.state = new double[size];
            reset();
        }

        @Override
        public void apply(double[] action, RandomGenerator rng) {
            if (action.length != size) {
                throw new IllegalArgumentException("Action dimension mismatch. Expected " + size + ", got " + action.length);
            }
            double sqrtDt = Math.sqrt(dt);
            for (int i = 0; i < size; i++) {
                double dx = theta * (mu - state[i]) * dt + sigma * sqrtDt * rng.nextGaussian();
                state[i] += dx;
                action[i] += state[i]; // Add noise to the action
            }
        }

        @Override
        public void reset() {
            Arrays.fill(state, mu);
        }

        @Override
        public String toString() {
            return "OrnsteinUhlenbeckNoise{" +
                "size=" + size +
                ", mu=" + mu +
                ", theta=" + theta +
                ", sigma=" + sigma +
                ", dt=" + dt +
                '}';
        }
    }

    /**
     * Simple Gaussian noise.
     * Adds uncorrelated noise sampled from a Gaussian distribution N(0, stddev^2).
     */
    public static class GaussianNoise implements NoiseStrategy {
        private final double stddev;

        public GaussianNoise(double stddev) {
            Util.validate(stddev, s -> s >= 0, "Standard deviation must be non-negative");
            this.stddev = stddev;
        }

        @Override
        public void apply(double[] action, RandomGenerator rng) {
            if (stddev == 0) return; // No noise to add
            for (int i = 0; i < action.length; i++) {
                action[i] += rng.nextGaussian(0, stddev);
            }
        }

        @Override
        public void reset() {
            // Gaussian noise is stateless
        }

        @Override
        public String toString() {
            return "GaussianNoise{stddev=" + stddev + '}';
        }
    }

    /**
     * A no-operation noise strategy that applies no noise.
     */
    public static class NoneNoise implements NoiseStrategy {
        @Override
        public void apply(double[] action, RandomGenerator rng) {
            // No noise is applied
        }

        @Override
        public void reset() {
            // Stateless
        }

        @Override
        public String toString() {
            return "NoneNoise{}";
        }
    }
}
