package jcog.tensor.rl.pg.util;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public enum PerlinNoise {
    ;
    private static final int PERMUTATION_SIZE = 256;
    private static final int[] permutations = new int[PERMUTATION_SIZE * 2];

    static {
        Random random = ThreadLocalRandom.current();

        for (int i = 0; i < PERMUTATION_SIZE; i++)
            permutations[i] = i;


        for (int i = 0; i < PERMUTATION_SIZE; i++) {
            int j = random.nextInt(PERMUTATION_SIZE);
            int temp = permutations[i];
            permutations[i] = permutations[j];
            permutations[j] = temp;
        }

        for (int i = 0; i < PERMUTATION_SIZE; i++)
            permutations[PERMUTATION_SIZE + i] = permutations[i];
    }

    public static double perlinNoise(double x, double y, double z) {
        int X = (int) Math.floor(x) & 255;
        int Y = (int) Math.floor(y) & 255;
        int Z = (int) Math.floor(z) & 255;

        x -= Math.floor(x);
        y -= Math.floor(y);
        z -= Math.floor(z);

        double u = fade(x);
        double v = fade(y);
        double w = fade(z);

        int A = permutations[X] + Y;
        int AA = permutations[A] + Z;
        int AB = permutations[A + 1] + Z;
        int B = permutations[X + 1] + Y;
        int BA = permutations[B] + Z;
        int BB = permutations[B + 1] + Z;

        return lerp(w, lerp(v, lerp(u, grad(permutations[AA], x, y, z),
                                       grad(permutations[BA], x - 1, y, z)),
                                lerp(u, grad(permutations[AB], x, y - 1, z),
                                       grad(permutations[BB], x - 1, y - 1, z))),
                       lerp(v, lerp(u, grad(permutations[AA + 1], x, y, z - 1),
                                       grad(permutations[BA + 1], x - 1, y, z - 1)),
                                lerp(u, grad(permutations[AB + 1], x, y - 1, z - 1),
                                       grad(permutations[BB + 1], x - 1, y - 1, z - 1))));
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
        double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }


}