package jcog.random;

import java.util.Random;
import java.util.random.RandomGenerator;

import static java.lang.Float.MIN_NORMAL;
import static jcog.data.bit.FixedPoint.*;

/** buffers 64-bits at a time from a delegate Random generator.
 *  for efficient sub-word generation.
 *  ex: if you only need nextBoolean()'s, then it can generate 64 results from each internal RNG call.
 *  not thread-safe.
 *
 *  also provides extra utility methods not present in java.util.Random
 */
public class RandomBits implements RandomGenerator {

    public final Random rng;

    /** up to 64 bits to be dequeued */
    private long buffer;

    /** bits remaining */
    byte bit = 0;

    public RandomBits(Random rng) {
        this.rng = rng;
    }

    @Override public String toString() {
        return Long.toBinaryString(buffer) + " @ " + bit;
    }

    @Override public boolean nextBoolean() {
        return _next(1) != 0;
    }

    int _next(int bits) {
        assert(bits < 31 /* maybe 32 */ && bits > 0);
        if (bit < bits)
            refresh();
        int r = (int) (buffer & ((1 << bits) - 1));
        buffer >>>= bits;
        bit -= bits;
        return r;
    }

    private void refresh() {
        buffer = rng.nextLong();
        bit = 64;
    }

    @Override public int nextInt(int i) {
        return i < 2 ? 0 : _next(bits(i)) % i;
    }

    private static int bits(int i) {
        return 32 - Integer.numberOfLeadingZeros(i);
    }

    public void setSeed(long seed) {
        rng.setSeed(seed);
        bit = 0; //force refresh
    }


    /** lower-precision boolean probability */
    public final boolean nextBooleanFast(float probTrue, int bits) {
        return switch (bits) {
            case 8 -> nextBooleanFast8(probTrue);
            case 16 -> nextBooleanFast16(probTrue);
            case 32 -> nextBoolean(probTrue);
            default -> throw new UnsupportedOperationException();
        };
    }

    public boolean nextBooleanFast8(float probTrue) {
        int a = nextBoolAbs(probTrue); return a!=0 ? (a == +1) :
                _nextBooleanFast8(probTrue);
    }

    public boolean _nextBooleanFast8(float probTrue) {
        return probTrue >= unitByteToFloat(_next(8));
    }

    public boolean nextBooleanFast16(float probTrue) {
        int a = nextBoolAbs(probTrue); return a!=0 ? (a==+1) :
                _nextBooleanFast16(probTrue);
    }

    public boolean _nextBooleanFast16(float probTrue) {
        return probTrue >= unitShortToFloat(_next(16));
    }

    public boolean nextBooleanFast24(float probTrue) {
        int a = nextBoolAbs(probTrue); return a!=0 ? (a==+1) :
                _nextBooleanFast24(probTrue);
    }

    public boolean _nextBooleanFast24(float probTrue) {
        return probTrue >= unit24ToFloat(_next(24));
    }

    public final boolean nextBoolean(float probTrue) {
        int a = nextBoolAbs(probTrue); return a!=0 ? (a==+1) :
                _nextBoolean(probTrue);
    }

    public boolean _nextBoolean(float probTrue) {
        return probTrue >= nextFloat();
    }

    /** absolute special cases */
    private int nextBoolAbs(float probTrue) {
        if (probTrue == 0.5f) {
            return nextBoolean() ? +1 : -1;
        } else if (probTrue <= MIN_NORMAL) {
            return -1;
        } else if (probTrue >= 1.0f - MIN_NORMAL) {
            return +1;
        } else {
            return 0;
        }
    }

    public final int nextBooleanAsInt() {
         return nextBoolean() ? 0 : 1;
    }

    /** fractional ceiling; any integer remainder decides randomly to increment */
    public int nextFloor(float max) {
        int i = (int) max;
        float fract = max - i;
        return nextBoolean(fract) ? i + 1 : i;
    }

    public static int nextFloor(float max, RandomGenerator rng) {
        int i = (int) max;
        return rng.nextFloat() < (/*fraction:*/max - i) ? i + 1 : i;
    }

//    @Nullable public <X> X get(X[] x) {
//        int n = x.length;
//        return switch (n) {
//            case 0 -> null;
//            case 1 -> x[0];
//            default -> x[nextInt(n)];
//        };
//    }

    public long nextLong(long a, long b) {
        long r = b - a;
        if (r == 0) return a;
        else if (r < 0) throw new UnsupportedOperationException();
        else return a + (r < Integer.MAX_VALUE ?
            nextInt((int) r) :
            Math.abs(rng.nextLong()) % r);
    }


    @Override
    public long nextLong() {
        return rng.nextLong();
    }

    @Override
    public int nextInt() {
        return rng.nextInt();
    }

    @Override
    public double nextDouble() {
        return rng.nextDouble();
    }

    @Override public float nextFloat() {
        return rng.nextFloat();
    }

    /** from Random.nextFloat():
     *     return (nextInt() >>> (Float.SIZE - Float.PRECISION)) * 0x1.0p-24f; */
    public final float nextFloatFast8() {
        return nextFloatFast(8);
    }

    public final float nextFloatFast16() {
        return nextFloatFast(16);
    }

    public float nextFloatFast(int bits) {
        return (nextIntFast(bits) >>> (Float.SIZE - Float.PRECISION)) * 0x1.0p-24f;
    }

    /** maps lower random bits to the full 32-bit range by bit shifting and zeroing the lower bits */
    public int nextIntFast(int bits) {
        return _next(bits) << (32-bits);
    }

}