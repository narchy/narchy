package jcog.random;

/*		 
 * DSI utilities
 *
 * Copyright (C) 2013-2015 Sebastiano Vigna 
 *
 *  This library is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by the Free
 *  Software Foundation; either version 3 of the License, or (at your option)
 *  any later version.
 *
 *  This library is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
 *  for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, see <http:
 *
 */


import java.util.Random;


/**
 * An unbelievably fast, top-quality {@linkplain Random pseudorandom number generator} that
 * returns the sum of consecutive outputs of a Marsaglia Xorshift generator (described in <a
 * href="http:
 * Statistical Software</i>, 8:1&minus;6, 2003) with 128 bits of state.
 * <p>
 * <p>More details can be found on the <a href="http:
 * The basic idea is taken from Mutsuo Saito and Makuto Matsumoto's
 * <a href="http:
 * is however based on 32-bit shifts and fails several statistical tests when reversed.
 * <p>
 * <p>Note that this is
 * <strong>not</strong> a cryptographic-strength pseudorandom number generator, but its quality is
 * preposterously higher than {@link Random}'s, and its cycle length is
 * 2<sup>128</sup>&nbsp;&minus;&nbsp;1, which is more than enough for any single-thread application.
 *
 * @see it.unimi.dsi.util
 * @see Random
 * @see XorShift128PlusRandomGenerator
 */
public class XorShift128PlusRandom extends Random {

    /**
     * 2<sup>-53</sup>.
     */
    private static final double NORM_53 = 1.0 / (1L << 53);
    /**
     * 2<sup>-24</sup>.
     */
    private static final double NORM_24 = 1.0 / (1L << 24);

    /**
     * The internal state of the algorithm.
     */
    private long s0;
    private long s1;


    /**
     * Creates a new generator using a given seed.
     *
     * @param seed a nonzero seed for the generator (if zero, the generator will be seeded with -1).
     */
    public XorShift128PlusRandom(long seed) {
        setSeed(seed);
    }

    public XorShift128PlusRandom() {
        this(System.nanoTime());
    }

    @Override
    public long nextLong() {
        long s1 = s0;
        long s0 = this.s1;
        this.s0 = s0;
        s1 ^= s1 << 23;
        return (this.s1 = (s1 ^ s0 ^ (s1 >>> 17) ^ (s0 >>> 26))) + s0;
    }

    @Override
    public int nextInt() {
        return (int) nextLong();
    }

    @Override
    public int nextInt(int n) {

        int v = Math.abs((int) (nextLong())) % n;
        
        
        return v;
    }

    /**
     * Returns a pseudorandom uniformly distributed {@code long} value
     * between 0 (inclusive) and the specified value (exclusive), drawn from
     * this random number generator's sequence. The algorithm used to generate
     * the value guarantees that the result is uniform, provided that the
     * sequence of 64-bit values produced by this generator is.
     *
     * @param n the positive bound on the random number to be returned.
     * @return the next pseudorandom {@code long} value between {@code 0} (inclusive) and {@code n} (exclusive).
     */
    public long nextLong(long n) {
        if (n <= 0)
            throw new IllegalArgumentException();
        
        while (true) {
            long bits = nextLong() >>> 1;
            long value = bits % n;
            if (bits - value + (n - 1) >= 0)
                return value;
        }
    }

    @Override
    public double nextDouble() {
        return (nextLong() >>> 11) * NORM_53;
    }

    @Override
    public float nextFloat() {
        return (float) ((nextLong() >>> 40) * NORM_24);
    }

    @Override
    public boolean nextBoolean() {
        return nextLong() < 0;
    }

    @Override
    public void nextBytes(byte[] bytes) {
        int i = bytes.length, n = 0;
        while (i != 0) {
            n = Math.min(i, 8);
            for (long bits = nextLong(); n-- != 0; bits >>= 8) bytes[--i] = (byte) bits;
        }
    }

    /**
     * Sets the seed of this generator.
     * <p>
     * <p>The seed will be passed twice through {@link HashCommon#murmurHash3(long)}. In this way, if the
     * user passes a small value we will avoid the short irregular transient associated
     * with states with a very small number of bits setAt.
     *
     * @param seed a nonzero seed for this generator (if zero, the generator will be seeded with {@link Long#MIN_VALUE}).
     */
    @Override
    public void setSeed(long seed) {
        s0 = SplitMix64Random.murmurHash3(seed == 0 ? Long.MIN_VALUE : seed);
        s1 = SplitMix64Random.murmurHash3(s0);
    }















}
