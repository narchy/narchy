package jcog.tensor.rl.pg3.util; // Changed package

import jcog.Util; // For validation if needed

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator; // For NoiseStrategy

/**
 * Classic Perlin Noise implementation.
 * This class can be used to generate coherent, natural-looking noise.
 * The static part of this class is the original Perlin noise generator.
 * The inner class PerlinNoiseStrategy adapts it for use with RL agents.
 */
public class PerlinNoise { // Outer class remains for the core static Perlin logic
    private static final int PERMUTATION_SIZE = 256;
    // Ensure permutations is initialized only once and is effectively final after static block
    private static final int[] P; // Renamed for clarity, made final

    static {
        P = new int[PERMUTATION_SIZE * 2];
        Random random = ThreadLocalRandom.current(); // Use a local Random for static init

        for (int i = 0; i < PERMUTATION_SIZE; i++)
            P[i] = i;

        for (int i = 0; i < PERMUTATION_SIZE; i++) {
            int j = random.nextInt(PERMUTATION_SIZE);
            int temp = P[i];
            P[i] = P[j];
            P[j] = temp;
        }

        // Duplicate the permutation array to avoid buffer overflow with P[X+1], etc.
        System.arraycopy(P, 0, P, PERMUTATION_SIZE, PERMUTATION_SIZE);
    }

    // Static Perlin noise generation method (original logic)
    public static double noise(double x, double y, double z) {
        int X = (int) Math.floor(x) & (PERMUTATION_SIZE -1); // Use PERMUTATION_SIZE-1 for mask
        int Y = (int) Math.floor(y) & (PERMUTATION_SIZE -1);
        int Z = (int) Math.floor(z) & (PERMUTATION_SIZE -1);

        x -= Math.floor(x);
        y -= Math.floor(y);
        z -= Math.floor(z);

        double u = fade(x);
        double v = fade(y);
        double w = fade(z);

        int A = P[X] + Y;
        int AA = P[A] + Z;
        int AB = P[A + 1] + Z;
        int B = P[X + 1] + Y;
        int BA = P[B] + Z;
        int BB = P[B + 1] + Z;

        return lerp(w, lerp(v, lerp(u, grad(P[AA], x, y, z),
                                       grad(P[BA], x - 1, y, z)),
                                lerp(u, grad(P[AB], x, y - 1, z),
                                       grad(P[BB], x - 1, y - 1, z))),
                       lerp(v, lerp(u, grad(P[AA + 1], x, y, z - 1),
                                       grad(P[BA + 1], x - 1, y, z - 1)),
                                lerp(u, grad(P[AB + 1], x, y - 1, z - 1),
                                       grad(P[BB + 1], x - 1, y - 1, z - 1))));
    }

    private static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    private static double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }

    private static double grad(int hash, double x, double y, double z) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : (h == 12 || h == 14 ? x : z); // Corrected condition
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

    /**
     * Implements Noise.NoiseStrategy using Perlin noise.
     * Generates correlated noise over time for each action dimension.
     */
    public static class PerlinNoiseStrategy implements Noise.NoiseStrategy {
        private final int actionDim;
        private final double scale; // Standard deviation / magnitude of noise
        private final double timeStep;
        private final double[] timeOffsets; // Per-dimension offset to get different noise patterns
        private double currentTime;

        public PerlinNoiseStrategy(int actionDim, double scale, double timeStep) {
            Util.validate(actionDim, ad -> ad > 0, "Action dimension must be positive.");
            Util.validate(scale, s -> s >= 0, "Noise scale (stddev) must be non-negative.");
            Util.validate(timeStep, ts -> ts > 0, "Time step for Perlin noise must be positive.");

            this.actionDim = actionDim;
            this.scale = scale;
            this.timeStep = timeStep;
            this.currentTime = 0;
            this.timeOffsets = new double[actionDim];
            // Initialize with random offsets to ensure different noise for each dimension
            Random random = ThreadLocalRandom.current();
            for (int i = 0; i < actionDim; i++) {
                this.timeOffsets[i] = random.nextDouble() * 1000; // Large random offset
            }
        }

        @Override
        public void apply(double[] action, RandomGenerator rng) {
            if (action.length != actionDim) {
                throw new IllegalArgumentException("Action dimension mismatch. Expected " + actionDim + ", got " + action.length);
            }
            if (scale == 0) return; // No noise to add

            for (int i = 0; i < actionDim; i++) {
                // Use currentTime for x, action_dimension_offset for y, and a constant for z
                // Perlin noise output is roughly in [-1, 1], scale it by `this.scale`
                double noiseVal = PerlinNoise.noise(currentTime, timeOffsets[i], 0.0);
                action[i] += noiseVal * scale;
            }
            currentTime += timeStep;
        }

        @Override
        public void reset() {
            currentTime = 0;
            // Optionally, re-randomize timeOffsets if a completely new noise pattern is desired on reset
            // Random random = ThreadLocalRandom.current();
            // for (int i = 0; i < actionDim; i++) {
            //     this.timeOffsets[i] = random.nextDouble() * 1000;
            // }
        }

        @Override
        public String toString() {
            return "PerlinNoiseStrategy{actionDim=" + actionDim + ", scale=" + scale + ", timeStep=" + timeStep + "}";
        }
    }
}