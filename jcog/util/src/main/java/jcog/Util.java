package jcog;

import com.google.common.base.Joiner;
import com.google.common.collect.Iterators;
import com.google.common.primitives.*;
import jcog.data.byt.ByteSequence;
import jcog.data.list.Lst;
import jcog.data.set.ArrayUnenforcedSet;
import jcog.math.NumberException;
import jcog.util.ArrayUtil;
import jcog.util.KahanSum;
import org.eclipse.collections.api.block.function.primitive.*;
import org.eclipse.collections.api.block.predicate.primitive.FloatPredicate;
import org.eclipse.collections.api.list.primitive.ByteList;
import org.eclipse.collections.api.list.primitive.ImmutableByteList;
import org.eclipse.collections.api.map.primitive.ObjectIntMap;
import org.eclipse.collections.api.set.MutableSet;
import org.eclipse.collections.api.set.primitive.IntSet;
import org.eclipse.collections.impl.EmptyIterator;
import org.eclipse.collections.impl.bag.mutable.HashBag;
import org.eclipse.collections.impl.factory.primitive.ByteLists;
import org.eclipse.collections.impl.factory.primitive.IntSets;
import org.eclipse.collections.impl.list.mutable.primitive.DoubleArrayList;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;
import org.hipparchus.fitting.PolynomialCurveFitter;
import org.hipparchus.fitting.WeightedObservedPoint;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.net.URL;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.StampedLock;
import java.util.function.*;
import java.util.function.IntFunction;
import java.util.function.IntToDoubleFunction;
import java.util.random.RandomGenerator;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.lang.Double.isFinite;
import static java.lang.Math.ceil;
import static java.lang.Thread.onSpinWait;
import static java.lang.invoke.MethodHandles.lookup;

//import jcog.pri.Prioritizable;

/**
 *
 */
public enum Util {
    ;

    public static final Iterator emptyIterator = EmptyIterator.getInstance();
    public static final List emptyIterable = Collections.EMPTY_LIST;

    public static final double PHI = (1 + Math.sqrt(5)) / 2;
    public static final float PHIf = (float) PHI;

    /**
     * phi-1 == 1/phi
     */
    public static final double PHI_min_1 = PHI - 1;
    public static final float PHI_min_1f = (float) PHI_min_1;

    public static final int PRIME3 = 524287;
    public static final int PRIME2 = 92821;
    public static final int PRIME1 = 31;

    public static final ImmutableByteList EmptyByteList = ByteLists.immutable.empty();

    public static final double log2 = Math.log(2);
    public static final float MIN_NORMALsqrt = (float) Math.sqrt(Float.MIN_NORMAL);
    public static final Consumer[] EmptyConsumerArray = new Consumer[0];

    /**
     * high precision fused-multiply-add SIMD computation
     * disable if VM doesnt impl fma intrinsics for the CPU.
     * TODO determine what is best for the current system.
     * FMA may be time and/or energy expensive on certain systems
     */
    private static final boolean MATH_FMA = Config.IS("math_fma",
        //true
        false
    );

    public static double logFast(double x) {
        return 6 * (x - 1) / (x + 1 + 4 * Math.sqrt(x));
    }


    /**
     * It is basically the same as a lookup table with 2048 entries and linear interpolation between the entries, but all this with IEEE floating point tricks.
     * https://stackoverflow.com/questions/66402/faster-math-exp-via-jni/424985#424985
     */
    public static double expFast(double val) {
        return Double.longBitsToDouble((long)
                (1512775 * val + (1072693248 - 60801)) << 32);
    }

    public static double fma(double a, double b, double c) {
        return MATH_FMA ? Math.fma(a, b, c) : a * b + c;
    }

    public static float fma(float a, float b, float c) {
        return MATH_FMA ? Math.fma(a, b, c) : a * b + c;
    }

    /** intrinsic support seems to be missing, but maybe some day it won't */
    public static double log1p(double x) {
        return Math.log(1 + x);
        //return Math.log1p(x);
    }

    public static String UUIDbase64() {
        var uuid = UUID.randomUUID();
        var low = uuid.getLeastSignificantBits();
        var high = uuid.getMostSignificantBits();
        return new String(Base64.getEncoder().encode(
            Bytes.concat(
                Longs.toByteArray(low),
                Longs.toByteArray(high)
            )
        ));
    }

    public static int hash(byte b) {
        return b;
    }

    public static int hash(byte[] bytes) {
        return hash(bytes, 0, bytes.length);
    }

    /** f = from, t = to */
    public static int hash(ByteSequence x, int f, int t) {
        return switch (t - f) {
            case 0 -> 1;
            case 1 -> x.at(f);
            case 2 -> Shorts.fromBytes(x.at(f++), x.at(f));
            case 3, 4 -> Ints.fromBytes(x.at(f++), x.at(f++), x.at(f++), f >= t ? (byte) 0 : x.at(f));
            default -> hashFNV(x, f, t);
        };
    }

    /** f = from, t = to */
    public static int hash(byte[] x, int f, int t) {
        return switch (t - f) {
            case 0 -> 1;
            case 1 -> x[f];
            case 2 -> Shorts.fromBytes(x[f++], x[f]);
            case 3, 4 -> Ints.fromBytes(x[f++], x[f++], x[f++], f>=t ? (byte) 0 : x[f]);
            default -> hashFNV(x, f, t);
        };
    }

    public static int hashJava(byte[] bytes, int len) {
        var result = 1;
        for (var i = 0; i < len; ++i)
            result = 31 * result + bytes[i];
        return result;
    }


//    public static <X> Predicate<X> limit(Predicate<X> x, int max) {
//        if (max <= 0)
//            throw new WTF();
//
//        if (max == 1) {
//            return z -> {
//                x.test(z);
//                return false;
//            };
//        } else {
//            int[] remain = {max};
//            return z -> {
//                boolean next = (--remain[0] > 0);
//                return x.test(z) && next;
//            };
//        }
//    }

    //	public static int hashFNV(byte[] bytes, int from, int to) {
//		int h = 0x811c9dc5;
//		for (int i = from; i < to; i++)
//			h = (h * 16777619) ^ bytes[i];
//		return h;
//	}
    public static int hashFNV(ByteSequence bytes, int from, int to) {
        var h = 0x811c9dc5;
        for (var i = from; i < to; i++)
            h = (h * 16777619) ^ bytes.at(i);
        return h;
    }

    public static int hashFNV(byte[] bytes, int from, int to) {
        var h = 0x811c9dc5;
        for (var i = from; i < to; i++)
            h = (h * 16777619) ^ bytes[i];
        return h;
    }

    public static void assertNotNull(Object test, String varName) {
        if (test == null) {
            throw new NullPointerException(varName);
        }
    }


//
//    public static <E> void assertNotEmpty(Collection<E> test, String varName) {
//        if (test == null) {
//            throw new NullPointerException(varName);
//        }
//        if (test.isEmpty()) {
//            throw new IllegalArgumentException("empty " + varName);
//        }
//    }
//    /* End Of  P. J. Weinberger Hash Function */

    public static void assertNotEmpty(Object[] test, String varName) {
        if (test == null) {
            throw new NullPointerException(varName);
        }
        if (test.length == 0) {
            throw new IllegalArgumentException("empty " + varName);
        }
    }

    public static String globToRegEx(String line) {

        line = line.trim();
        var strLen = line.length();
        var sb = new StringBuilder(strLen);

        if (strLen > 0 && line.charAt(0) == '*') {
            line = line.substring(1);
            strLen--;
        }
        if (strLen > 0 && line.charAt(strLen - 1) == '*') {
            line = line.substring(0, strLen - 1);
            strLen--;
        }
        var escaping = false;
        var inCurlies = 0;
        for (var currentChar : line.toCharArray()) {
            switch (currentChar) {
                case '*':
                    if (escaping)
                        sb.append("\\*");
                    else
                        sb.append(".*");
                    escaping = false;
                    break;
                case '?':
                    if (escaping)
                        sb.append("\\?");
                    else
                        sb.append('.');
                    escaping = false;
                    break;
                case '.':
                case '(':
                case ')':
                case '+':
                case '|':
                case '^':
                case '$':
                case '@':
                case '%':
                    sb.append('\\');
                    sb.append(currentChar);
                    escaping = false;
                    break;
                case '\\':
                    if (escaping) {
                        sb.append("\\\\");
                        escaping = false;
                    } else
                        escaping = true;
                    break;
                case '{':
                    if (escaping) {
                        sb.append("\\{");
                    } else {
                        sb.append('(');
                        inCurlies++;
                    }
                    escaping = false;
                    break;
                case '}':
                    if (inCurlies > 0 && !escaping) {
                        sb.append(')');
                        inCurlies--;
                    } else if (escaping)
                        sb.append("\\}");
                    else
                        sb.append('}');
                    escaping = false;
                    break;
                case ',':
                    if (inCurlies > 0 && !escaping) {
                        sb.append('|');
                    } else if (escaping)
                        sb.append("\\,");
                    else
                        sb.append(',');
                    break;
                default:
                    escaping = false;
                    sb.append(currentChar);
            }
        }
        return sb.toString();
    }

    public static long hashPJW(String str) {
        long BitsInUnsignedInt = (4 * 8);
        var ThreeQuarters = (BitsInUnsignedInt * 3) / 4;
        var OneEighth = BitsInUnsignedInt / 8;
        var HighBits = (0xFFFFFFFFL) << (BitsInUnsignedInt - OneEighth);
        long hash = 0;
        long test;

        var n = str.length();
        for (var i = 0; i < n; i++) {
            hash = (hash << OneEighth) + str.charAt(i);

            if ((test = hash & HighBits) != 0)
                hash = ((hash ^ (test >> ThreeQuarters)) & (~HighBits));
        }

        return hash;
    }

    public static long hashELF(String str) {
        long hash = 0;
        long x = 0;

        var l = str.length();
        for (var i = 0; i < l; i++) {
            hash = (hash << 4) + str.charAt(i);

            if ((x = hash & 0xF0000000L) != 0) {
                hash ^= (x >> 24);
            }
            hash &= ~x;
        }

        return hash;
    }

    public static int hashJava(int a, int b) {
        return a * 31 + b;
    }

    public static int hashJavaX(int a, int b) {
        return a * PRIME2 + b;
    }

    /**
     * from clojure.Util - not tested
     * also appears in https://www.boost.org/doc/libs/1_35_0/doc/html/boost/hash_combine_id241013.html
     */
    public static int hashCombine(int a, int b) {
        return hashCombineXX(a, b);
        //return boostHashCombine(a, b);
    }

    public static int hashCombineBoost(int a, int b) {
        return a ^ ((a << 6) + (a >> 2) + b + 0x9e3779b9);
    }
    // CityHash 2-ary combine function
    public static int hashCombineCity(int h1, int h2) {
        var kMul = 0x9DDfea08eb382d69L;
        var a = (h1 ^ h2) * kMul;
        a ^= (a >>> 47); // Right shift
        var b = (h2 ^ a) * kMul;
        b ^= (b >>> 47);
        return (int) b;
    }

    // XXHash 2-ary combine function
    public static int hashCombineXX(int h1, int h2) {
        final var prime = 0x9E3779B1; // Prime constant used by XXHash
        return Integer.rotateLeft(h1 ^ h2 * prime, 13) * prime;
    }

    public static int hashCombine(int a, int b, int c) {
        return hashCombine(hashCombine(a, b), c);
    }

    public static int hashCombine(int a, int b, int c, int d) {
        return hashCombine(hashCombine(hashCombine(a, b), c), d);
    }

    public static int hashCombine(int i, long x, long y) {
        return hashCombine(hashCombine(i, x), y);
    }

    public static int hashCombine(long x, long y) {
        return hashCombine(longUpper(x), longLower(x), longUpper(y), longLower(y));
        //return hashCombine(Long.hashCode(y), x);
    }

    public static int hashCombine(int x, long y) {
        return hashCombine(x, longUpper(y), longLower(y));
    }

    private static int longLower(long y) {
        return (int)y;
        //return (int) (y & 0xffff);
    }

    private static int longUpper(long y) {
        return (int) (y >> 32);
    }

    public static int hashCombine(int a, long... b) {
        var x = a;
        for (var l : b)
            x = hashCombine(x, l);
        return x;
    }

    public static int hashCombine(int a, Object b) {
        return hashCombine(a, b.hashCode());
    }

    public static int hashCombine(Object a, Object b, Object c) {
        return hashCombine(hashCombine(a, b), c);
    }

    public static int hashCombine(Object a, Object b) {
        if (a != b) {
            return hashCombine(a.hashCode(), b.hashCode());
        } else {
            var ah = a.hashCode();
            return hashCombine(ah, ah);
        }
    }

    /**
     * hashCombine(1, b)
     */
    public static int hashCombine1(Object x) {
        return hashCombine(1, x.hashCode());
    }

    /**
     * linear interpolate between target & current, factor is between 0 and 1.0
     * targetFactor=1:   full target
     * targetfactor=0.5: average
     * targetFactor=0:   full current
     */
    public static float lerp(float x, float min, float max) {
        return lerpSafe(unitize(x), min, max);
    }

    public static double lerp(double x, double min, double max) {
        return lerpSafe(unitize(x), min, max);
    }

    public static float lerpSafe(float x, float min, float max) {
        //return min + x * (max - min);
        return fma(x, max - min, min);
    }

    public static double lerpSafe(double x, double min, double max) {
//		return min + x * (max - min);
        return fma(x, max - min, min);
    }

    public static long lerpLong(float x, long min, long max) {
        return min == max ? min : Math.round(
                //min + (max - min) * unitize((double) x)
                fma(max - min, unitize((double) x), min)
        );
    }

    public static int lerpInt(float x, int min, int max) {
        return min == max ? min : Math.round(
                //min + (max - min) * unitize(x)
                fma(max - min, unitize(x), min)
        );
    }

    public static float max(float a, float b, float c) {
        return Math.max(Math.max(a, b), c);
    }


    public static float mean(float a, float b) {
        return (float)mean((double)a, b);
    }

    public static double mean(double a, double b) {
        return sumPrecise(a,b)/2;
    }

    public static double mean(double... d) {
        return mean(d, 0, d.length);
    }

    public static double mean(float... d) {
        return mean(d, 0, d.length);
    }

    public static double mean(float[] d, int s, int e) {
        return switch (e - s) {
            case 0 -> throw new UnsupportedOperationException();
            case 1 -> d[s];
            default -> sum(d, s, e) / (e - s);
        };
    }

    public static double mean(double[] d, int s, int e) {
        return switch (e - s) {
            case 0 -> throw new UnsupportedOperationException();
            case 1 -> d[s];
            default -> sum(d, s, e) / (e - s);
        };
    }

    /**
     * clamps a value to 0..1 range
     */
    public static double unitize(double x) {
        assertFinite(x);
        return unitizeSafe(x);
    }

    /**
     * clamps a value to 0..1 range
     */
    public static float unitize(float x) {
        assertFinite(x);
        return unitizeSafe(x);
    }

    public static float unitizeSafe(float x) {
        return clampSafe(x, 0, 1);
    }

    public static double unitizeSafe(double x) {
        return clampSafe(x, 0, 1);
    }

    public static float assertFinite(float x) throws NumberException {
        if (!Float.isFinite(x))
            throw new NumberException("non-finite", x);
        return x;
    }

    public static double assertFinite(double x) throws NumberException {
        if (!isFinite(x))
            throw new NumberException("non-finite", x);
        return x;
    }

    public static float assertNotNaN(float x) throws NumberException {
        if (x != x) throw NumberException.NaN(x);
        return x;
    }

//    /**
//     * clamps a value to -1..1 range
//     */
//    public static float clampBi(float p) {
//        if (p > 1f)
//            return 1f;
//        return Math.max(p, -1f);
//        return p;
//    }

    public static double assertNotNaN(double x) throws NumberException {
        if (x != x) throw NumberException.NaN(x);
        return x;
    }

    /**
     * discretizes values to nearest finite resolution real number determined by epsilon spacing
     */
    public static float round(float value, float epsilon) {
        if (value != value || epsilon <= Float.MIN_NORMAL) return value;
        assertFinite(epsilon);
        assertFinite(value);
        return roundSafe(value, epsilon);
    }

    public static double round(double value, double epsilon) {
        if (value != value || epsilon <= Double.MIN_NORMAL) return value;
        assertFinite(epsilon);
        assertFinite(value);
        return roundSafe(value, epsilon);
    }

    public static float roundSafe(float value, float epsilon) {
        return (float)roundSafe((double)value, epsilon);
    }

    public static double roundSafe(double value, double epsilon) {
        return Math.round(value / epsilon) * epsilon;
    }

    /**
     * rounds x to the nearest multiple of the dither parameter
     */
    public static long round(long x, int m) {
//		assert(x!=Long.MIN_VALUE && x!=Long.MAX_VALUE): "possibly reserved Long values"; //HACK

        if (m <= 1 || x == 0) return x;
        else return x < 0 ? -_round(-x, m) : _round(x, m);

//		//long lo = (x / dither) * dither,  hi = lo + dither;
//		long lo = roundDown(x, m), hi = roundUp(x, m);
//		return (x - lo > hi - x)? hi : lo; //closest
    }

//	/**
//	 * round n down to nearest multiple of m
//	 * from: https://gist.github.com/aslakhellesoy/1134482 */
//	public static long roundDown(long n, int m) {
//		return n >= 0 ? (n / m) * m : ((n - m + 1) / m) * m;
//	}
//
//	/**
//	 * round n up to nearest multiple of m
//	 * from: https://gist.github.com/aslakhellesoy/1134482
//	 */
//	public static long roundUp(long n, int m) {
//		return n >= 0 ? ((n + m - 1) / m) * m : (n / m) * m;
//	}

    private static long _round(long x, int m) {
        var base = x - x % m;
        x -= base;
        var lo = (x / m) * m;
        var hi = lo + m;
        return base + (x - lo > hi - x ? hi : lo); //closest
    }

//	public static long toInt(double f, int discretness) {
//		return Math.round(f * discretness);
//	}

    public static int toInt(double f, int discretness) {
        return (int) Math.round(f * discretness);
    }

    public static float toFloat(int i, int discretness) {
        return (float) toDouble(i, discretness);
    }

//	public static @Nullable <X> X get(@Nullable Supplier<X> s) {
//		return s != null ? s.get() : null;
//	}

    public static double toDouble(int i, int discretness) {
        return ((double) i) / discretness;
    }

    public static @Nullable <X> X get(Object xOrSupplierOfX) {
        return (X) (xOrSupplierOfX instanceof Supplier s ? s.get() : xOrSupplierOfX);
    }

    public static boolean equals(float a, float b) {
        return equals(a, b, Float.MIN_NORMAL);
    }

//	/**
//	 * tests equivalence (according to epsilon precision)
//	 */
//	public static boolean equals(float a, float b, float epsilon) {
////		if (a == b)
////			return true;
//		//if (Float.isFinite(a) && Float.isFinite(b))
//		return Math.abs(a - b) < epsilon;
////        else
////            return (a != a) && (b != b); //both NaN
//	}

    public static boolean equals(long a, long b, int tolerance) {
        //assert(tolerance > 0);
        return Math.abs((float) (a - b)) < tolerance;
    }

    public static boolean equals(double a, double b) {
        return equals(a, b, Double.MIN_NORMAL);
    }

    /**
     * tests equivalence (according to epsilon precision)
     */
    public static boolean equals(double a, double b, double epsilon) {
        return Math.abs(a - b) < epsilon;
//		if (a == b)
//			return true;
//        if (Double.isFinite(a) && Double.isFinite(b))
//        else
//            return (a != a) && (b != b); //both NaN
    }

    public static boolean equals(float a, float b, float epsilon) {
        return Math.abs(a - b) < epsilon;
    }

    public static boolean equals(float[] a, float[] b, float epsilon) {
        if (a == b) return true;
        var l = a.length;
        for (var i = 0; i < l; i++) {
            if (!equals(a[i], b[i], epsilon))
                return false;
        }
        return true;
    }

//	public static byte[] intAsByteArray(int index) {
//
//		if (index < 36) {
//			byte x = base36(index);
//			return new byte[]{x};
//		} else if (index < (36 * 36)) {
//			byte x1 = base36(index % 36);
//			byte x2 = base36(index / 36);
//			return new byte[]{x2, x1};
//		} else {
//			throw new RuntimeException("variable index out of range for this method");
//		}
//
//
//	}

    /**
     * applies a quick, non-lexicographic ordering compare
     * by first testing their lengths
     */
    public static int compare(long[] x, long[] y) {
        if (x == y) return 0;

        var xlen = x.length;

        var yLen = y.length;
        if (xlen != yLen) {
            return Integer.compare(xlen, yLen);
        } else {

            for (var i = 0; i < xlen; i++) {
                var c = Long.compare(x[i], y[i]);
                if (c != 0)
                    return c;
            }

            return 0;
        }
    }

//    /**
//     * bins a priority value to an integer
//     */
//    public static int decimalize(float v) {
//        return bin(v, 10);
//    }
//
//	/**
//	 * finds the mean value of a given bin
//	 */
//	public static float unbinCenter(int b, int bins) {
//		return ((float) b) / bins;
//	}

//    public static <D> D runProbability(Random rng, float[] probs, D[] choices) {
//        float tProb = 0;
//        for (int i = 0; i < probs.length; i++) {
//            tProb += probs[i];
//        }
//        float s = rng.nextFloat() * tProb;
//        int c = 0;
//        for (int i = 0; i < probs.length; i++) {
//            s -= probs[i];
//            if (s <= 0) {
//                c = i;
//                break;
//            }
//        }
//        return choices[c];
//    }

    public static int bin(float x, int bins) {
        //assertFinite(x); assertUnitized(x); assert(bins > 0);

        return clampSafe((int) (x * bins), 0, bins - 1);
        //return Math.round(0.5f + x * (bins-1));

        //return (int) Math.floor(x * bins);
        //return (int) (x  * bins);

        //return Math.round(x * (bins - 1));
        //return Util.clamp((int)((x * bins) + 0.5f/bins), 0, bins-1);


        //return (int) ((x + 0.5f/bins) * (bins-1));
        //        return (int) Math.floor((x + (0.5 / bins)) * bins);
        //        return Util.clamp(b, 0, bins-1);
    }

    public static MethodHandle mhRef(Class<?> type, String name) {
        try {
            for (var m : type.getMethods()) {
                if (m.getName().equals(name)) {
                    return lookup()
                            .unreflect(m);
                }
            }
            return null;
        } catch (IllegalAccessException e) {
            throw new Error(e);
        }
    }

    public static <F> MethodHandle mh(String name, F fun) {
        return mh(name, fun.getClass(), fun);
    }

    public static <F> MethodHandle mh(String name, Class<? extends F> type, F fun) {
        return mhRef(type, name).bindTo(fun);
    }

    @SafeVarargs
    public static <F> MethodHandle mh(String name, F... fun) {
        var fun0 = fun[0];
        var m = mh(name, fun0.getClass(), fun0);
        for (var i = 1; i < fun.length; i++)
            m = m.bindTo(fun[i]);

        return m;
    }

    public static byte base36(int index) {
        if (index < 10)
            return (byte) ('0' + index);
        else if (index < (10 + 26))
            return (byte) ((index - 10) + 'a');
        else
            throw new RuntimeException("out of bounds");
    }

    /**
     * clamps output to 0..+1.  y=0.5 at x=0
     */
    public static float sigmoid(float x) {
        return (float) sigmoid((double) x);
    }

//	public static float sigmoidDiff(float a, float b) {
//		float sum = a + b;
//		float delta = a - b;
//		float deltaNorm = delta / sum;
//		return sigmoid(deltaNorm);
//	}

//	public static float sigmoidDiffAbs(float a, float b) {
//		float sum = a + b;
//		float delta = Math.abs(a - b);
//		float deltaNorm = delta / sum;
//		return sigmoid(deltaNorm);
//	}

//	public static List<String> inputToStrings(InputStream is) throws IOException {
//		List<String> x = CharStreams.readLines(new InputStreamReader(is, Charsets.UTF_8));
//		Closeables.closeQuietly(is);
//		return x;
//	}
//
//	public static String inputToString(InputStream is) throws IOException {
//		String s = CharStreams.toString(new InputStreamReader(is, Charsets.UTF_8));
//		Closeables.closeQuietly(is);
//		return s;
//	}

//	public static int[] reverse(IntArrayList l) {
//		return switch (l.size()) {
//			case 0 -> throw new UnsupportedOperationException();
//			case 1 -> new int[]{l.get(0)};
//			case 2 -> new int[]{l.get(1), l.get(0)};
//			case 3 -> new int[]{l.get(2), l.get(1), l.get(0)};
//			default -> l.asReversed().toArray();
//		};
//	}

//	public static byte[] reverse(ByteArrayList l) {
//		int s = l.size();
//		switch (s) {
//			case 0:
//				return ArrayUtil.EMPTY_BYTE_ARRAY;
//			case 1:
//				return new byte[]{l.get(0)};
//			case 2:
//				return new byte[]{l.get(1), l.get(0)};
//			default:
//				byte[] b = new byte[s];
//				for (int i = 0; i < s; i++)
//					b[i] = l.get(--s);
//				return b;
//		}
//	}

//	public static String s(String s, int maxLen) {
//		if (s.length() < maxLen) return s;
//		return s.substring(0, maxLen - 2) + "..";
//	}

//	public static void writeBits(int x, int numBits, float[] y, int offset) {
//		for (int i = 0, j = offset; i < numBits; i++, j++) {
//			y[j] = ((x & 1 << i) == 1) ? 1 : 0;
//		}
//	}

//	/**
//	 * a and b must be instances of input, and output must be of size input.length-2
//	 */
//	public static <X> X[] except(X[] input, X a, X b, X[] output) {
//		int targetLen = input.length - 2;
//		if (output.length != targetLen) {
//			throw new RuntimeException("wrong size");
//		}
//		int j = 0;
//		for (X x : input) {
//			if ((x != a) && (x != b))
//				output[j++] = x;
//		}
//
//		return output;
//	}

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoid(double x, double sharp) {
        return 1 / (1 + Math.exp(-x * sharp));
    }

    /**
     * to unit interval: [0, 1]
     */
    public static float[] normalize(float[] x) {
        var minmax = minmax(x);
        return normalize(x, minmax[0], minmax[1]);
    }

    public static double[] normalizeUnit(double... x) {
        return normalize(x, x.length);
    }

    public static double[] normalize(double[] x) {
        return normalize(x, x.length);
    }


//	public static float[] normalizeCartesian(float[] x, float epsilon) {
//		return normalizeCartesian(x, x.length, epsilon);
//	}
//
//	public static float[] normalizeCartesian(float[] x, int n, float epsilon) {
//
//		double magSq = 0;
//		for (int i = 0; i < n; i++) magSq += sqr(x[i]);
//
//		if (magSq >= sqr(((double) epsilon * n))) {
//			double mag = Math.sqrt(magSq);
//			for (int i = 0; i < n; i++) x[i] /= mag;
//		} else {
//			Arrays.fill(x, 0, n, 0); //zero vector
//		}
//
//		return x;
//	}
//
//	public static double[] normalizeCartesian(double[] x, double epsilon) {
//		return normalizeCartesian(x, x.length, epsilon);
//	}

    public static double[] normalize(double[] x, int n) {
        var minmax = minmax(x, 0, n);
        return normalize(x, 0, n, minmax[0], minmax[1]);
    }
//
//	public static float[] normalizeMargin(float lowerPct, float upperPct, float[] x) {
//		float[] minmax = minmax(x);
//		float range = minmax[1] - minmax[0];
//		return normalize(x, minmax[0] - lowerPct * range, minmax[1] + upperPct * range);
//	}

    public static double[] normalizeCartesian(double[] x, int n, double epsilon) {
        double magSq = sumSqr(x);
        if (magSq < sqr(epsilon * n)) {
            Arrays.fill(x, 0);
        } else {
            var mag = Math.sqrt(magSq);
            for (var i = 0; i < n; i++) x[i] /= mag;
        }
        return x;
    }
    public static double[] normalizeCartesianIfMagGt1(double[] x, int n, double epsilon) {
        double magSq = sumSqr(x);
        if (magSq < sqr(epsilon * n)) {
            Arrays.fill(x, 0);
        } else {
            var mag = Math.sqrt(magSq);
            if (mag > 1)
                for (var i = 0; i < n; i++) x[i] /= mag;
        }
        return x;
    }

//	public static double[] normalizeSubArray(double[] x, int s, int e) {
//		//final double min = Util.min(s, e, x);
//		//return normalize(x, s, e, min, min + Util.sum(x, s, e));
//		return normalize(x, s, e, min(x, s, e), max(x, s, e));
//	}

//	public static double[] normalizeSubArraySum1(double[] x, int s, int e) {
//		double min = min(x, s, e);
//		return normalize(x, s, e, min, min + sum(x, s, e));
//	}

//	public static float[] normalizeSubArray(float[] x, int s, int e) {
//		return normalize(x, s, e, Util.min(s, e, x), Util.max(s, e, x));
//	}

    public static float[] normalize(float[] x, float min, float max) {
        return normalize(x, 0, x.length, min, max);
    }

    /**
     * https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
     */
    @Is("Feature_scaling")
    public static float[] normalize(float[] x, int s, int e, float min, float max) {
        var n = e - s;
        var range = max - min;
        if (range < Float.MIN_NORMAL * n) {
            for (var i = s; i < e; i++) {
                if (x[i] == x[i]) //skip NaN's
                    x[i] = 0.5f;
            }
        } else {
            for (var i = s; i < e; i++)
                x[i] = normalizeSafer(x[i], min, range);
        }
        return x;
    }

    /**
     * normalize to unit min/max range.
     * modifies input array
     */
    public static double[] normalize(double[] x, int s, int e, double min, double max) {
        var n = e - s;
        var range = max - min;
        if (range < Double.MIN_NORMAL * n) {
            for (var i = s; i < e; i++) {
                if (x[i] == x[i]) //skip NaN's
                    x[i] = 0.5;
            }
        } else {
            for (var i = s; i < e; i++)
                x[i] = normalizeRange(x[i], min, range);
        }
        return x;
    }

    public static double normalize(double x, double min, double max) {
        assertFinite(x);
        assertFinite(min);
        assertFinite(max);
        assert (max >= min);
        return normalizeSafe(x, min, max);
    }

    public static float normalizeSafe(float x, float min, float max) {
        return ((max - min) <= Float.MIN_NORMAL) ? 0.5f : normalizeSafer(x, min, max - min);
    }

    private static float normalizeSafer(float x, float min, float range) {
        return (x - min) / range;
    }

    public static double normalizeSafe(double x, double min, double max) {
        return ((max - min) <= Double.MIN_NORMAL) ? 0 : normalizeRange(x, min, max - min);
    }

    public static double normalizeRange(double x, double min, double range) {
        return (x - min) / range;
    }

    public static float normalize(float x, float min, float max) {
        assertFinite(x);

        assertFinite(min);
        assertFinite(max);
        assert (max >= min);

        return normalizeSafe(x, min, max);
    }

    public static float variance(float[] x) {
        var mean = mean(x);

        double variance = 0;
        for (var p : x)
            variance += sqr(p - mean);

        return (float) (variance / x.length);
    }

    public static double[] variance(double[] x) {
        var mean = mean(x);

        var variance = 0.0;
        for (var p : x)
            variance += sqr(p - mean);

        return new double[] { mean, variance / x.length };
    }
    public static double[] variance(DoubleStream s) {
        var dd = new DoubleArrayList();
        s.forEach(dd::add);
        if (dd.isEmpty())
            return null;

        var mean = dd.average();

        double variance = 0;
        var n = dd.size();
        for (var i = 0; i < n; i++)
            variance += sqr(dd.get(i) - mean);

        variance /= n;

        return new double[]{mean, variance};
    }

    public static double meanWeighted(float[] data, float[] weights) {
        double sum = 0, sumWeights = 0;
        var n = data.length;
        for (var i = 0; i < n; i++) {
            double w = weights[i];
            sumWeights += w;
            sum += data[i] * w;
        }
        return sum / sumWeights;
    }

    /**
     * When you calculate variance, whether using the traditional formula or a weighted variance as previously discussed, you are primarily interested in the magnitude of the result. The sign (positive or negative) of the variance is typically not emphasized because it represents the direction of deviation from the mean.
     * So, yes, you can interpret the overall variance by considering its absolute value. The absolute value of the variance tells you the amount of variation or dispersion in your data, without concern for whether that variation is in the positive or negative direction.
     * In practice, when people discuss variance, they often refer to the positive value (the absolute value) of the variance because it provides a straightforward measure of the data's spread or variability. The larger the absolute value of the variance, the more spread out the data points are from the mean.
     * However, it's important to note that sometimes the sign of the variance might carry specific meaning or implications in certain statistical analyses or contexts, so it's always a good practice to consider both the magnitude (absolute value) and sign when interpreting the variance, depending on the particular problem or question you are addressing.
     * https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
     * <p>
     * from: https://github.com/Hipparchus-Math/hipparchus/blob/master/hipparchus-stat/src/main/java/org/hipparchus/stat/descriptive/moment/Variance.java
     */
    @Is("Weighted_arithmetic_mean#Frequency_weights")
    public static double[] weightedVariance(float[] data, float[] weights) {
        int begin = 0, end = data.length;
        var mean = meanWeighted(data, weights);
        return new double[]{mean, weightedVariance(data, weights, begin, end, mean)};
    }

    public static double weightedVariance(float[] data, float[] weights, int begin, int end, double mean) {
        double a = 0;
        double b = 0;
        for (var i = begin; i < end; i++) {
            var d = data[i] - mean;
            double w = weights[i];
            a += w * (d * d);
            b += w * d;
        }

        double sumWts = 0;
        for (var i = begin; i < end; i++)
            sumWts += weights[i];

//        if (isBiasCorrected) {
        return (a - (b * b / sumWts)) / (sumWts - 1); //SAMPLE
//        } else {
//            return (a - (b * b / sumWts)) / sumWts; //POPULATION
//        }
    }

    public static float[] toFloat(double[] d) {
        return toFloat(d, new float[d.length]);
    }

    public static float[] toFloat(double[] from, float[] to) {
        var l = from.length;
        for (var i = 0; i < l; i++)
            to[i] = (float) from[i];
        return to;
    }

    public static double[] toDouble(float[] d) {
        return toDouble(d, new double[d.length]);
    }

    public static double[] toDouble(float[] from, @Nullable double[] to) {
        var l = from.length;
        if (to == null || to.length < l)
            to = new double[l];
        for (var i = 0; i < l; i++)
            to[i] = from[i];
        return to;
    }

    public static float[]  minmax(float[] x)  { return minmax(x, 0, x.length); }
    @Contract(value = "_ -> new", pure = true)
    public static double[] minmax(double[] x) {
        return minmax(x, 0, x.length);
    }

    public static float[] minmax(float[] x, int from, int to) {
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (var i = from; i < to; i++) {
            var y = x[i];
            if (y < min) min = y; else if (y > max) max = y;
        }
        return new float[]{min, max};
    }

    public static double[] minmax(double[] x, int from, int to) {
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (var i = from; i < to; i++) {
            var xi = x[i];
            if (xi < min) min = xi; else if (xi > max) max = xi;
        }
        return new double[]{min, max};
    }

    public static void time(String name, Logger logger, Runnable r) {
        if (!logger.isInfoEnabled()) {
            r.run();
        } else {
            var dtNS = timeNS(r);
            logger.info("{} {}", name, Str.timeStr(dtNS));
        }
    }

    public static long timeNS(Runnable r) {
        var start = System.nanoTime();

        r.run();

        return System.nanoTime() - start;
    }

    public static String tempDir() {
        return System.getProperty("java.io.tmpdir");
    }

    public static <X, Y> Y[] map(Function<X, Y> f, Y[] y, Collection<X> c) {
        var s = c.size();
        if (y.length<s) y = Arrays.copyOf(y, s);
        var j = 0;
        for (var x : c)
            y[j++] = f.apply(x);
        return y;
    }

    /**
     * TODO make a version of this which can return the input array if no modifications occurr either by .equals() or identity
     */
    @SafeVarargs
    public static <X, Y> Y[] map(Function<X, Y> f, Y[] y, X... x) {
        return map(f, y, Math.min(y.length, x.length), x);
    }

    @SafeVarargs
    public static <X, Y> Y[] map(Function<X, Y> f, Y[] y, int size, X... x) {
        return map(f, y, 0, x, 0, size);
    }

    public static <X, Y> Y[] map(Function<X, Y> f, Y[] y, int targetOffset, X[] x, int srcFrom, int srcTo) {
        //assert (x.length > 0);
        for (var i = srcFrom; i < srcTo; i++)
            y[targetOffset++] = f.apply(x[i]);
        return y;
    }

    /**
     * TODO make a version of this which can return the input array if no modifications occurr either by .equals() or identity
     */
    @SafeVarargs
    public static <X, Y> Y[] map(Function<X, Y> f, IntFunction<Y[]> y, X... x) {
        var i = 0;
        var target = y.apply(x.length);
        for (var xx : x)
            target[i++] = f.apply(xx);
        return target;
    }

    @SafeVarargs
    public static <X> X[] mapIfChanged(UnaryOperator<X> f, X... src) {
        X[] target = null;
        for (int i = 0, srcLength = src.length; i < srcLength; i++) {
            var x = src[i];
            var y = f.apply(x);
            if (y != x) {
                if (target == null)
                    target = src.clone();

                target[i] = y;
            }
        }
        return target == null ? src : target;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> double sum(FloatFunction<X> value, X... xx) {
        double y = 0;
        for (var x : xx)
            y += value.floatValueOf(x);
        return y;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> double sum(ToDoubleFunction<X> value, X... xx) {
        var y = 0.0;
        for (var x : xx)
            y += value.applyAsDouble(x);
        return y;
    }

//	@SuppressWarnings("LambdaUnfriendlyMethodOverload")
//	public static <X> float sum(FloatFunction<X> value, Iterable<X> xx) {
//		float y = 0;
//		for (X x : xx)
//			y += value.floatValueOf(x);
//		return y;
//	}
//	@SuppressWarnings("LambdaUnfriendlyMethodOverload")
//	public static <X> double sum(ToDoubleFunction<X> value, Iterable<X> xx) {
//		double y = 0;
//		for (X x : xx)
//			y += value.applyAsDouble(x);
//		return y;
//	}

//	public static <X> float mean(FloatFunction<X> value, Iterable<X> xx) {
//		float y = 0;
//		int count = 0;
//		for (X x : xx) {
//			y += value.floatValueOf(x);
//			count++;
//		}
//		return y / count;
//	}

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public static <X> int sum(ToIntFunction<X> value, Iterable<X> xx) {
        var y = 0;
        for (var x : xx)
            y += value.applyAsInt(x);
        return y;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> int sum(ToIntFunction<X> value, X... xx) {
        return sum(value, 0, xx.length, xx);
    }

//	@SuppressWarnings("LambdaUnfriendlyMethodOverload")
//	@SafeVarargs
//    public static <X> long sum(ToLongFunction<X> value, X... xx) {
//		long y = 0;
//		for (X x : xx)
//			y += value.applyAsLong(x);
//		return y;
//	}

//	@SuppressWarnings("LambdaUnfriendlyMethodOverload")
//	@SafeVarargs
//    public static <X> long min(ToLongFunction<X> value, X... xx) {
//		long y = Long.MAX_VALUE;
//		for (X x : xx)
//			y = Math.min(y, value.applyAsLong(x));
//		return y;
//	}

    @SafeVarargs
    public static <X> int sum(ToIntFunction<X> value, int from, int to, X... xx) {
        var len = to - from;
        var y = 0;
        for (var i = from; i < len; i++)
            y += value.applyAsInt(xx[i]);
        return y;
    }

    @Is("Smoothstep")
    public static double smoothstep(double x) {
        x = unitizeSafe(x);
        return x * x * (3 - 2 * x);
    }

    @Is("Smoothstep")
    public static double smootherstep(double x) {
        x = unitizeSafe(x);
        return x * x * x * (x * (x * 6 - 15) + 10);
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> long max(ToLongFunction<X> value, X... xx) {
        var y = Long.MIN_VALUE;
        for (var x : xx)
            y = Math.max(y, value.applyAsLong(x));
        return y;
    }

    public static <X> long max(ToLongFunction<X> value, X[] xx, int n) {
        var y = Long.MIN_VALUE;
        for (var i = 0; i < n; i++)
            y = Math.max(y, value.applyAsLong(xx[i]));
        return y;
    }
    public static <X> long maxIgnoreLongMax(ToLongFunction<X> value, X[] xx, int n) {
        var y = Long.MIN_VALUE;
        for (var i = 0; i < n; i++) {
            var l = value.applyAsLong(xx[i]);
            if (l!=Long.MAX_VALUE)
                y = Math.max(y, l);
        }
        return y;
    }

    public static double max(IntToDoubleFunction value, int start, int end) {
        var max = Double.NEGATIVE_INFINITY;
        for (var i = start; i < end; i++) {
            var v = value.applyAsDouble(i);
            if (v > max) max = v;
        }
        return max;
    }

    public static double min(IntToDoubleFunction value, int start, int end) {
        var min = Double.POSITIVE_INFINITY;
        for (var i = start; i < end; i++) {
            var v = value.applyAsDouble(i);
            if (v < min) min = v;
        }
        return min;
    }

    @SafeVarargs
    public static <X> boolean sumBetween(ToIntFunction<X> value, int min, int max, X... xx) {
        var y = 0;
        for (var x : xx) {
            if ((y += value.applyAsInt(x)) > max)
                return false;
        }
        return (y >= min);
    }

    @SafeVarargs
    public static <X> boolean sumExceeds(ToIntFunction<X> value, int max, X... xx) {
        var y = 0;
        for (var x : xx) {
            if ((y += value.applyAsInt(x)) > max)
                return true;
        }
        return false;
    }

    public static <X> boolean sumExceeds(ToDoubleFunction<X> value, double thresh, Iterable<X> xx) {
        double y = 0;
        for (var x : xx) {
            y += value.applyAsDouble(x);
            if (y > thresh) return true;
        }
        return false;
    }

    /**
     * ignores NaN
     */
    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> double mean(FloatFunction<X> value, X... xx) {
        double y = 0;
        var count = 0;
        for (var x : xx) {
            var v = value.floatValueOf(x);
            if (v == v) {
                y += v;
                count++;
            }
        }
        return count > 0 ? (y / count) : Double.NaN;
    }

    /**
     * ignores NaN
     */
    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> double mean(ToDoubleFunction<X> value, X... xx) {
        double y = 0;
        var count = 0;
        for (var x : xx) {
            var v = value.applyAsDouble(x);
            if (v == v) {
                y += v;
                count++;
            }
        }
        return count > 0 ? (y / count) : Double.NaN;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> float max(FloatFunction<X> value, X... xx) {
        var y = Float.NEGATIVE_INFINITY;
        for (var x : xx)
            y = Math.max(y, value.floatValueOf(x));
        return y;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public static <X> float max(FloatFunction<X> value, Iterable<X> xx) {
        var y = Float.NEGATIVE_INFINITY;
        for (var x : xx)
            y = Math.max(y, value.floatValueOf(x));
        return y;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    @SafeVarargs
    public static <X> float min(FloatFunction<X> value, X... xx) {
        var y = Float.POSITIVE_INFINITY;
        for (var x : xx)
            y = Math.min(y, value.floatValueOf(x));
        return y;
    }

    public static long sum(int[] x, int from, int to) {
        long y = 0;
        for (var j = from; j < to; j++)
            y += x[j];
        return y;
    }

    public static double sum(float[] x, int from, int to) {
        var b = KahanSum.sum(x, from, to);
//        double a = sumNaive(x, from, to); if (a!=b)
//            Util.nop(); else
//            Util.nop();
        return b;
//        double y = 0;
//        for (var j = from; j < to; j++)
//            y += x[j];
//        return y;
    }

    public static double sum(double[] x, int from, int to) {
        var b = KahanSum.sum(x, from, to);
//        double a = sumNaive(x, from, to); if (a!=b)
//            Util.nop(); else
//            Util.nop();
        return b;
    }
    private static double sumPairwiseRecursive(double[] x, int start, int end) {
        var n = end - start;
        return switch (n) {
            case 0 -> 0;
            case 1 -> x[start];
            case 2 -> x[start] + x[start + 1];
            default -> {
                var mid = start + n / 2;
                yield sumPairwiseRecursive(x, start, mid) + sumPairwiseRecursive(x, mid, end);
            }
        };
    }
    public static double sumNaive(double[] x, int from, int to) {
        double y = 0;
        for (var j = from; j < to; j++)
            y += x[j];
        return y;
    }

    public static double max(double... x) {
        return max(x, 0, x.length);
    }

    public static byte max(byte... x) {
        var y = Byte.MIN_VALUE;
        for (var f : x) {
            if (f > y) y = f;
        }
        return y;
    }

    public static int max(int... x) {
        var y = Integer.MIN_VALUE;
        for (var f : x)
            y = Math.max(y, f);
        return y;
    }

    public static short max(short... x) {
        var y = Short.MIN_VALUE;
        for (var f : x) {
            if (f > y) y = f;
        }
        return y;
    }

    public static float max(float... x) {
        return max(x, 0, x.length);
    }

    public static long max(long[] x, int s, int e) {
        var y = Long.MIN_VALUE;
        for (var i = s; i < e; i++)
            y = Math.max(y, x[i]);
        return y;
    }

    public static float max(float[] x, int s, int e) {
        var y = Float.NEGATIVE_INFINITY;
        for (var i = s; i < e; i++)
            y = Math.max(y, x[i]);
        return y;
    }

    public static double max(double[] x, int s, int e) {
        var y = Double.NEGATIVE_INFINITY;
        for (var i = s; i < e; i++)
            y = Math.max(y, x[i]);
        return y;
    }

    public static double min(double... x) {
        return min(x, 0, x.length);
    }

    public static int compare(double x, double y) {
        return y > x ? -1 : (y < x ? +1 : 0);
    }


    public static float min(float[] x, int s, int e) {
        var y = Float.POSITIVE_INFINITY;
        for (var i = s; i < e; i++)
            y = Math.min(y, x[i]);
        return y;
    }

    public static double min(double[] x, int s, int e) {
        var y = Double.POSITIVE_INFINITY;
        for (var i = s; i < e; i++)
            y = Math.min(y, x[i]);
        return y;
    }

    public static double sum(float... x) {
        return sum(x, 0, x.length);
    }

    public static double sum(double... x) {
        return sum(x, 0, x.length);
    }
    public static double sum(double[][] xx) {
        var s = new KahanSum();
        s.add(xx);
        return s.value();
    }

    public static double sumAbs(float... x) {
        var s = new KahanSum();
        s.addAbs(x);
        return s.value();
    }
//        double y = 0;
//        for (var f : x) {
//            if (f == f)
//                y += abs(f);
//        }
//        return y;
//    }

    public static double sumAbs(double... x) {
        var s = new KahanSum();
        s.addAbs(x);
        return s.value();
    }

    public static double sumSqr(double[][] x) {
        var s = new KahanSum();
        for (var xx : x)
            s.add(sumSqr(xx));
        return s.value();
    }

    public static double sumSqr(double... x) {
        return sumSqr(new KahanSum(), x).value();
    }

    public static KahanSum sumSqr(KahanSum s, double... x) {
        for (var f : x)
            s.addSqr(f); // y+=f*f
        return s;
    }

    /**
     * TODO fair random selection when exist equal values
     */
    public static int argmax(double[] x) {
        var result = -1;
        var max = Double.NEGATIVE_INFINITY;
        for (int i = 0, l = x.length; i < l; i++) {
            var xi = x[i];
            if (xi > max) {
                max = xi;
                result = i;
            }
        }
        return result;
    }

    /**
     * TODO fair random selection when exist equal values
     */
    public static int argmax(float... vec) {
        var result = -1;
        var max = Float.NEGATIVE_INFINITY;
        for (int i = 0, l = vec.length; i < l; i++) {
            var v = vec[i];
            if (v > max) {
                max = v;
                result = i;
            }
        }
        return result;
    }

    public static int argmin(float... vec) {
        var result = -1;
        var max = Float.POSITIVE_INFINITY;
        for (int i = 0, l = vec.length; i < l; i++) {
            var v = vec[i];
            if (v < max) {
                max = v;
                result = i;
            }
        }
        return result;
    }

    public static int argmax(int a, int b, float... vec) {
        var result = -1;
        var max = Float.NEGATIVE_INFINITY;
        for (var i = a; i < b; i++) {
            var v = vec[i];
            if (v > max) {
                max = v;
                result = i;
            }
        }
        return result;
    }

//    public static Pair tuple(Object a, Object b) {
//        return Tuples.pair(a, b);
//    }
//
//    public static Pair tuple(Object a, Object b, Object c) {
//        return tuple(tuple(a, b), c);
//    }
//
//    public static Pair tuple(Object a, Object b, Object c, Object d) {
//        return tuple(tuple(a, b, c), d);
//    }
//
//    public static int argmax(RandomGenerator random, float... vec) {
//        var result = -1;
//        var max = Float.NEGATIVE_INFINITY;
//
//        var l = vec.length;
//        var start = random.nextInt(l);
//        for (var i = 0; i < l; i++) {
//            var ii = (i + start) % l;
//            var v = vec[ii];
//            if (v > max) {
//                max = v;
//                result = ii;
//            }
//        }
//        return result;
//    }

    /**
     * min is inclusive, max is exclusive: [min, max)
     */
    public static int unitize(int x, int min, int max) {
        if (x < min) x = min;
        else if (x > max) x = max;
        return x;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public static float sum(int count, IntToFloatFunction values) {
        double weightSum = 0;
        for (var i = 0; i < count; i++)
            weightSum += values.valueOf(i);
        return (float) weightSum;
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public static double sum(int count, IntToDoubleFunction values) {
        double weightSum = 0;
        for (var i = 0; i < count; i++)
            weightSum += values.applyAsDouble(i);
        return weightSum;
    }

    public static double sumIfPositive(int count, IntToFloatFunction values) {
        double weightSum = 0;
        for (var i = 0; i < count; i++) {
            var w = values.valueOf(i);
            //assert (w == w);
            if (/*w == w && */w > Float.MIN_NORMAL)
                weightSum += w;
        }
        return weightSum;
    }

    public static float clamp(float x, float min, float max) {
        //assertFinite(f);
        assertNotNaN(min);
        assertNotNaN(max);
        assert (min <= max);
        return clampSafe(x, min, max);
    }

    public static float clampSafe(float x, float min, float max) {
        return (x > max) ? max : ((x < min) ? min : x);
//        return max(min(f, max), min);
//        if (f > max) return max;
//        if (f < min) return min;
//        return f;
    }
    public static double clampSafe(double x, double min, double max) {
        return (x > max) ? max : ((x < min) ? min : x);
        //return Util.max(Util.min(X, max), min);
        //return Math.max(Math.min(X, max), min);
    }
    public static double clamp(double x, double min, double max) {
        assertFinite(x);
        //notNaN(min);
        //notNaN(max);
        //assert (min <= max);
        return clampSafe(x, min, max);
    }

    public static float clampSafePolar(float X) {
        return clampSafe(X, -1f, +1f);
    }

    public static double clampSafePolar(double X) {
        return clampSafePolar(X, 1);
    }

    public static double clampSafePolar(double X, double minmax) {
        return clampSafe(X, -minmax, +minmax);
    }
    public static float clampSafePolar(float X, float minmax) {
        return clampSafe(X, -minmax, +minmax);
    }


    public static int clamp(int i, int min, int max) {
        if (max < min)
            throw new IllegalArgumentException();
        return clampSafe(i, min, max);
    }

    public static int clampSafe(int i, int min, int max) {
        return Math.min(Math.max(i, min), max);
    }

    public static long clamp(long i, long min, long max) {
        if (max < min)
            throw new IllegalArgumentException();
        return clampSafe(i, min, max);
    }

    public static long clampSafe(long i, long min, long max) {
        return Math.min(Math.max(i, min), max);
    }

    public static double sqr(long x) {
        return x * x;
    }

    public static int sqr(int x) {
        return x * x;
    }

    public static int cube(int x) {
        return x * x * x;
    }

    public static float sqr(float x) {
        return (float)sqr((double)x);
    }

    public static float sqrt(float v) {
        return (float) Math.sqrt(v);
    }

    public static float cube(float x) {
        return x * x * x;
    }

    public static double cube(double x) {
        return x * x * x;
    }

    public static double sqr(double x) {
        return x * x;
    }

    /**
     * adaptive spinlock behavior
     * see: https:
     * TODO tune
     * TODO randomize?
     */
    public static void pauseSpin(int previousContiguousPauses) {
        if (previousContiguousPauses < 2)
            return; //immediate

        if (previousContiguousPauses < 4096) {
            pauseSpinning(previousContiguousPauses);
            return;
        }

        Thread.yield();
    }

    /*
        static final long PARK_TIMEOUT = 50L;
    static final int MAX_PROG_YIELD = 2000;
            if(n > 500) {
            if(n<1000) {
                // "randomly" yield 1:8
                if((n & 0x7) == 0) {
                    LockSupport.parkNanos(PARK_TIMEOUT);
                } else {
                    onSpinWait();
                }
            } else if(n<MAX_PROG_YIELD) {
                // "randomly" yield 1:4
                if((n & 0x3) == 0) {
                    Thread.yield();
                } else {
                    onSpinWait();
                }
            } else {
                Thread.yield();
                return n;
            }
        } else {
            onSpinWait();
        }
        return n+1;

     */

//    /**
//     * adaptive spinlock behavior
//     * see: https:
//     */
//    public static void pauseNextCountDown(long timeRemainNS) {
//        if (timeRemainNS < 10 * (1000 /* uS */))
//            onSpinWait();
//        else
//            Thread.yield();
//    }

    private static void pauseSpinning(int previousContiguousPauses) {
        if (previousContiguousPauses > 512 && (previousContiguousPauses % 1024) == 0) {
            Thread.yield();
        } else {
            onSpinWait();
        }
    }

    public static void sleepMS(long milliseconds) {
        sleepNS(milliseconds * 1_000_000);
    }

    public static void sleepS(int seconds) { sleepNS(seconds * 1_000_000_000L); }

    public static void sleepNS(long nanoseconds) {
//        int sleepThreshNS =
//            //1
//            1000 //1uS
//            //50 * 1000 //50uS https://hazelcast.com/blog/locksupport-parknanos-under-the-hood-and-the-curious-case-of-parking/
//        ;
//
//        if (nanoseconds > sleepThreshNS) {
            LockSupport.parkNanos(nanoseconds);
            //U.park(false, nanoseconds);
        //} else {
            //Thread.onSpinWait();
            //Thread.yield();
//        }

    }

//	/**
//	 * https://hazelcast.com/blog/locksupport-parknanos-under-the-hood-and-the-curious-case-of-parking/
//	 * expect ~50uSec resolution on linux
//	 */
//	private static void sleepNS(long sleepNS, int epsilonNS) {
//
//
//		if (sleepNS <= epsilonNS) return;
//
//		long end = System.nanoTime() + sleepNS;
//
//		do {
//
////			if (sleepNS >= spinThresholdNS) {
////				if (sleepNS >= 1_000_000 /* 1ms */) {
////					try {
////						Thread.sleep(sleepNS/1_000_000);
////					} catch (InterruptedException e) { }
////				} else {
//					LockSupport.parkNanos(sleepNS);
////				}
////			} else {
////				//Thread.onSpinWait();
////				Thread.yield();
////			}
//
//		} while ((sleepNS = end - System.nanoTime()) > epsilonNS);
//
//	}

//    public static void sleepNS(long periodNS) {
//        if (periodNS > 1_000_000_000 / 1000 / 2  /*0.5ms */) {
//            LockSupport.parkNanos(periodNS);
//            return;
//        }
//
//        final long thresholdNS = 1000; /** 1uS = 0.001ms */
//        if (periodNS <= thresholdNS)
//            return;
//
//        long end = System.nanoTime() + periodNS;
//        //long remainNS = end - System.nanoTime();
//        int pauses = 0;
//        long now;
//        while ((now = System.nanoTime()) < end) {
//            Util.pauseNextCountDown(end - now);
//            //while (remainNS > thresholdNS) {
//
////            if (remainNS <= 500000 /** 100uS = 0.5ms */) {
////                Thread.yield();
////            } else {
////                Thread.onSpinWait();
////            }
//            //Util.pauseNextIterative(pauses++);
//
//            //remainNS = end - System.nanoTime();
//        }
//
//
//    }


//	public static void sleepNSwhile(long periodNS, long napTimeNS, BooleanSupplier keepSleeping) {
//		if (!keepSleeping.getAsBoolean())
//			return;
//
//		if (periodNS <= napTimeNS) {
//			sleepNS(periodNS);
//		} else {
//			long now = System.nanoTime();
//			long end = now + periodNS;
//			do {
//				sleepNS(Math.min(napTimeNS, end - now));
//			} while (((now = System.nanoTime()) < end) && keepSleeping.getAsBoolean());
//		}
//	}

    public static int largestPowerOf2NoGreaterThan(int v) {
        if (v < 1) throw new IllegalArgumentException("x must be greater or equal 1");

        if ((v & v - 1) == 0) return v;

        v |= v >>> 1;
        v |= v >>> 2;
        v |= v >>> 4;
        v |= v >>> 8;
        v |= v >>> 16;
        return v + 1;

//		if (isPowerOf2(i))
//			return i;
//		else {
//			while (--i > 0) {
//				if (isPowerOf2(i))
//					return i;
//			}
//			return 0;
//		}
    }

    public static boolean isPowerOf2(int n) {
        if (n < 1) return false;

        var p_of_2 = (Math.log(n) / log2);
        return Math.abs(p_of_2 - Math.round(p_of_2)) == 0;
    }

    /**
     * http:
     * calculate height on a uniform grid, by splitting a quad into two triangles:
     */
    public static float lerp2d(float x, float z, float nw, float ne, float se, float sw) {

        x -= (int) x;
        z -= (int) z;


        if (x > z)
            sw = nw + se - ne;
        else
            ne = se + nw - sw;


        var n = lerp(x, ne, nw);
        var s = lerp(x, se, sw);
        return lerp(z, s, n);
    }

    public static String secondStr(double s) {
        int decimals;
        if (s >= 0.01) decimals = 0;
        else if (s >= 0.00001) decimals = 3;
        else decimals = 6;

        return secondStr(s, decimals);
    }

    public static String secondStr(double s, int decimals) {
        if (decimals < 0)
            return secondStr(s);
        else {
            return switch (decimals) {
                case 0 -> Str.n2(s) + 's';
                case 3 -> Str.n2(s * 1000) + "ms";
                case 6 -> Str.n2(s * 1.0E6) + "us";
                default -> throw new UnsupportedOperationException("TODO");
            };
        }
    }

    /**
     * A function where the output is disjunctively determined by the inputs
     *
     * @param arr The inputs, each in [0, 1]
     * @return The output that is no smaller than each input
     */

    public static <X> X[] sortUniquely(X[] arg) {
        var len = arg.length;
        Arrays.sort(arg);
        for (var i = 0; i < len - 1; i++) {
            var dups = 0;
            while (arg[i].equals(arg[i + 1])) {
                dups++;
                if (++i == len - 1)
                    break;
            }
            if (dups > 0) {
                System.arraycopy(arg, i, arg, i - dups, len - i);
                len -= dups;
            }
        }

        return len == arg.length ? arg : Arrays.copyOfRange(arg, 0, len);
    }

    public static boolean calledBySomethingContaining(String s) {
        return Joiner.on(' ').join(Thread.currentThread().getStackTrace()).contains(s);
    }

    public static <X> int count(Predicate<X> p, X[] xx) {
        var i = 0;
        for (var x : xx) if (p.test(x)) i++;
        return i;
    }

    public static int count(FloatPredicate p, float[] xx) {
        var i = 0;
        for (var x : xx) if (p.accept(x)) i++;
        return i;
    }

    public static int count(DoublePredicate p, double[] xx) {
        var i = 0;
        for (var x : xx) if (p.test(x)) i++;
        return i;
    }

    public static <X> boolean and(Predicate<X> p, int from, int to, X[] xx) {
        for (var i = from; i < to; i++)
            if (!p.test(xx[i]))
                return false;
        return true;
    }

    public static <X> boolean or(Predicate<X> p, int from, int to, X[] xx) {
        for (var i = from; i < to; i++)
            if (p.test(xx[i]))
                return true;
        return false;
    }

    public static <X> boolean and(Predicate<X> p, X[] xx) {
        return and(p, 0, xx.length, xx);
    }

    public static <X> boolean or(Predicate<X> p, X[] xx) {
        return or(p, 0, xx.length, xx);
    }

    public static <X> boolean and(X x, Iterable<Predicate<? super X>> p) {
        for (Predicate pp : p) {
            if (!pp.test(x))
                return false;
        }
        return true;
    }

    public static <X> boolean and(Predicate<? super X> p, Iterable<X> xx) {
        for (var x : xx) {
            if (!p.test(x))
                return false;
        }
        return true;
    }

    public static <X> boolean or(Predicate<? super X> p, Iterable<X> xx) {
        for (var x : xx) {
            if (p.test(x))
                return true;
        }
        return false;
    }

    @Nullable
    public static <X> X first(Predicate<? super X> p, Iterable<X> xx) {
        for (var x : xx) {
            if (p.test(x))
                return x;
        }
        return null;
    }

    public static <X> int count(Predicate<? super X> p, Iterable<X> xx) {
        var c = 0;
        for (var x : xx) {
            if (p.test(x))
                c++;
        }
        return c;
    }


    /**
     * x in -1..+1, y in -1..+1.   typical value for sharpen will be ~ >5
     * http:
     */
    public static float sigmoidBipolar(float x, float sharpen) {
        return (float) ((1 / (1 + Math.exp(-sharpen * x)) - 0.5) * 2);
    }


    public static float[] toFloat(double[] a, int from, int to, DoubleToFloatFunction df) {
        var result = new float[to - from];
        for (int j = 0, i = from; i < to; i++, j++) {
            result[j] = df.valueOf(a[i]);
        }
        return result;
    }

    public static float[] toFloat(double[] a, int from, int to) {
        var result = new float[to - from];
        for (int j = 0, i = from; i < to; i++, j++) {
            result[j] = (float) a[i];
        }
        return result;
    }

    public static void mul(float[] f, float scale) {
        var n = f.length;
        for (var i = 0; i < n; i++)
            f[i] *= scale;
    }

    public static void mul(double[] f, double x) {
        if (x==1) return;
        var n = f.length;
        for (var i = 0; i < n; i++)
            f[i] *= x;
    }

    public static void pow(double[] f, double x) {
        var n = f.length;
        for (var i = 0; i < n; i++)
            f[i] = Math.pow(f[i], x);
    }

    public static <X> X[] arrayOf(IntFunction<X> f, X[] a) {
        return arrayOf(f, 0, a.length, a);
    }

    public static <X> X[] arrayOf(IntFunction<X> f, int from, int to, IntFunction<X[]> arrayizer) {
        assert (to >= from);
        return arrayOf(f, from, to, arrayizer.apply(to - from));
    }

    public static <X> X[] arrayOf(IntFunction<X> f, int from, int to, X[] x) {
        for (int i = from, j = 0; i < to; )
            x[j++] = f.apply(i++);
        return x;
    }


//    /**
//     * builds a MarginMax weight array, which can be applied in a Roulette decision
//     * a lower margin > 0 controls the amount of exploration while values
//     * closer to zero prefer exploitation of provided probabilities
//     */
//    @Paper
//    public static float[] marginMax(int num, IntToFloatFunction build, float lower, float upper) {
//        float[] minmax = {Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
//
//        float[] w = Util.map(num, i -> {
//            float v = build.valueOf(i);
//            if (v < minmax[0]) minmax[0] = v;
//            if (v > minmax[1]) minmax[1] = v;
//            return v;
//        });
//
//        if (Util.equals(minmax[0], minmax[1], Float.MIN_NORMAL * 2)) {
//            Arrays.fill(w, 0.5f);
//        } else {
//
//
//            Util.normalize(w, minmax[0], minmax[1]);
//            Util.normalize(w, 0 - lower, 1 + upper);
//        }
//        return w;
//    }


    public static double[] softmax(double[] x, double temperature) {
        var y = new double[x.length];
        for (var i = 0; i < x.length; i++)
            y[i] = softmax(x[i], temperature);
        return y;
    }

    public static double softmax(double x, double temperature) {
        if (temperature < Double.MIN_NORMAL)
            throw new IllegalArgumentException("Temperature must be positive.");

        return Math.exp(x / temperature);
        //return assertFinite(f);
    }

    public static double invSoftmax(double x, double temperature) {
        return Math.log(x) + temperature;
    }

    public static float[] arrayOf(IntToFloatFunction build, @Nullable float[] target) {
        return arrayOf(build, target.length, target);
    }

    public static double[] arrayOf(IntToDoubleFunction build, @Nullable double[] target) {
        Arrays.setAll(target, build);
        return target;
    }

//	public static double[] arrayOf(IntToDoubleFunction build, int num, @Nullable double[] reuse) {
//		double[] f = (reuse != null && reuse.length == num) ? reuse : new double[num];
//		for (int i = 0; i < num; i++)
//			f[i] = build.applyAsDouble(i);
//		return f;
//	}

    public static float[] arrayOf(IntToFloatFunction build, int num, @Nullable float[] reuse) {
        var f = (reuse != null && reuse.length == num) ? reuse : new float[num];
        for (var i = 0; i < num; i++)
            f[i] = build.valueOf(i);
        return f;

    }

    public static float[] floatArrayOf(IntToFloatFunction build, int range) {
        return arrayOf(build, range, null);
    }


    public static <X> float[] floatArrayOf(X[] what, FloatFunction<X> value) {
        var num = what.length;
        var f = new float[num];
        for (var i = 0; i < num; i++) {
            f[i] = value.floatValueOf(what[i]);
        }
        return f;
    }

    /**
     * returns amount of memory used as a value between 0 and 100% (1.0)
     */
    public static float memoryUsed() {
        var r = Runtime.getRuntime();
        var total = r.totalMemory();
        var free = r.freeMemory();
        var max = r.maxMemory();
        var usedMemory = total - free;
        var availableMemory = max - usedMemory;
        return (float) (1 - ((double) availableMemory) / max);
    }


//	public static void toMap(Frequency f, String header, BiConsumer<String, Object> x) {
//		toMap(f.entrySetIterator(), header, x);
//	}

    public static void toMap(HashBag<?> f, String header, BiConsumer<String, Object> x) {
        f.forEachWithIndex((e, n) -> x.accept(header + ' ' + e, n));
    }

    public static void toMap(ObjectIntMap<?> f, String header, BiConsumer<String, Object> x) {
        f.forEachKeyValue((e, n) -> x.accept(header + ' ' + e, n));
    }

    public static void toMap(Iterator<? extends Map.Entry<?, ?>> f, String header, BiConsumer<String, Object> x) {
        f.forEachRemaining((e) -> x.accept(header + ' ' + e.getKey(), e.getValue()));
    }


    /**
     * pretty close
     */
    public static float tanhFast(float x) {
        if (x <= -3) return -1f;
        if (x >= 3f) return +1f;
        return x * (27 + x * x) / (27 + 9 * x * x);
    }

    /**
     * exponential unit-scaled function, take-off curve, x in 0..1, y in 0..1
     * http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiIoMl54LTEpLygyLTEpIiwiY29sb3IiOiIjOTYyQjFCIn0seyJ0eXBlIjowLCJlcSI6IigyXih4KjYpLTEpLygyXig2KS0xKSIsImNvbG9yIjoiIzkxQjgyNSJ9LHsidHlwZSI6MCwiZXEiOiIoM14oeCo4KS0xKS8oM14oOCktMSkiLCJjb2xvciI6IiMwRjU5QTMifSx7InR5cGUiOjAsImVxIjoiKDNeKHgqMTYpLTEpLygzXigxNiktMSkiLCJjb2xvciI6IiM4NDE2QjMifSx7InR5cGUiOjEwMDAsIndpbmRvdyI6WyItMS40NDQwNDQ3OTk5OTk5OTk3IiwiMS45NjM4MjcyIiwiLTAuNDE2MzU4Mzk5OTk5OTk4OSIsIjEuNjgwNzkzNTk5OTk5OTk5OSJdfV0-
     * https://demo2.yunser.com/math/fooplot/#W3sidHlwZSI6MCwiZXEiOiIoMl54LTEpLygyLTEpIiwiY29sb3IiOiIjOTYyQjFCIn0seyJ0eXBlIjowLCJlcSI6IigyXih4KjYpLTEpLygyXig2KS0xKSIsImNvbG9yIjoiIzkxQjgyNSJ9LHsidHlwZSI6MCwiZXEiOiIoM14oeCo4KS0xKS8oM14oOCktMSkiLCJjb2xvciI6IiMwRjU5QTMifSx7InR5cGUiOjAsImVxIjoiKDNeKHgqMTYpLTEpLygzXigxNiktMSkiLCJjb2xvciI6IiM4NDE2QjMifSx7InR5cGUiOjEwMDAsIndpbmRvdyI6WyItMS40NDQwNDQ3OTk5OTk5OTk3IiwiMS45NjM4MjcyIiwiLTAuNDE2MzU4Mzk5OTk5OTk4OSIsIjEuNjgwNzkzNTk5OTk5OTk5OSJdfV0-
     * https://pfortuny.net/fooplot.com/#W3sidHlwZSI6MCwiZXEiOiIoMl54LTEpLygyLTEpIiwiY29sb3IiOiIjOTYyQjFCIn0seyJ0eXBlIjowLCJlcSI6IigyXih4KjYpLTEpLygyXig2KS0xKSIsImNvbG9yIjoiIzkxQjgyNSJ9LHsidHlwZSI6MCwiZXEiOiIoM14oeCo4KS0xKS8oM14oOCktMSkiLCJjb2xvciI6IiMwRjU5QTMifSx7InR5cGUiOjAsImVxIjoiKDNeKHgqMTYpLTEpLygzXigxNiktMSkiLCJjb2xvciI6IiM4NDE2QjMifSx7InR5cGUiOjEwMDAsIndpbmRvdyI6WyItMS40NDQwNDQ3OTk5OTk5OTk3IiwiMS45NjM4MjcyIiwiLTAuNDE2MzU4Mzk5OTk5OTk4OSIsIjEuNjgwNzkzNTk5OTk5OTk5OSJdfV0-
     */
    public static double expUnit(double x, float sharpness /* # decades */) {
        float base = 2; //TODO this affects the shape too
        return powMin1(base, x * sharpness) / powMin1(base, sharpness);
    }

    /**
     * logarithmic saturation curve
     * expUnit vs. logUnit: http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiIoMl4oeCo2KS0xKS8oMl4oNiktMSkiLCJjb2xvciI6IiMxMTE2QUIifSx7InR5cGUiOjAsImVxIjoiMS0oMl4oKDEteCkqNiktMSkvKDJeKDYpLTEpIiwiY29sb3IiOiIjQzQxMzEzIn0seyJ0eXBlIjoxMDAwLCJ3aW5kb3ciOlsiMCIsIjEiLCIwIiwiMSJdfV0-
     */
    public static double logUnit(double x, float sharpness) {
//		float base = 2; //TODO this affects the shape too
//		return 1 - powMin1(base, (1-x)*sharpness) / powMin1(base, sharpness);
        return 1 - expUnit(1 - x, sharpness);
    }

    private static double powMin1(float base, double v) {
        return Math.pow(base, v) - 1;
    }

    public static Object toString(Object x) {
        return x.getClass() + "@" + System.identityHashCode(x);
    }

    /**
     * @noinspection ArrayEquality
     */
    public static int compare(byte[] a, byte[] b) {
        if (a == b) return 0;
        var al = a.length;
        var l = Integer.compare(al, b.length);
        if (l != 0)
            return l;
        for (var i = 0; i < al; i++) {
            var d = a[i] - b[i];
            if (d != 0) return d < 0 ? -1 : +1;
        }
        return 0;
    }

    public static <X> Supplier<Stream<X>> buffer(Stream<X> x) {
        var buffered = x.toList();
        return buffered::stream;
    }

    /**
     * creates an immutable sublist from a ByteList, since this isnt implemented yet in Eclipse collections
     */
    public static ImmutableByteList subList(ByteList x, int a, int b) {
        var size = b - a;
        if (a == 0 && b == x.size())
            return x.toImmutable();

        return switch (size) {
            case 0 -> ByteLists.immutable.empty();
            case 1 -> ByteLists.immutable.of(x.get(a));
            case 2 -> ByteLists.immutable.of(x.get(a++), x.get(a));
            case 3 -> ByteLists.immutable.of(x.get(a++), x.get(a++), x.get(a));
            case 4 -> ByteLists.immutable.of(x.get(a++), x.get(a++), x.get(a++), x.get(a));
            case 5 -> ByteLists.immutable.of(x.get(a++), x.get(a++), x.get(a++), x.get(a++), x.get(a));
            case 6 -> ByteLists.immutable.of(x.get(a++), x.get(a++), x.get(a++), x.get(a++), x.get(a++), x.get(a));
            default -> ByteLists.immutable.of(ArrayUtil.subarray(x.toArray(), a, b));
        };
    }

    public static <X> X first(X[] x) {
        return x[0];
    }

    public static <X> X last(X[] x) {
        return x[x.length - 1];
    }


//    /* domain: [0..1], range: [0..1] */
//    public static float smoothDischarge(float x) {
//        x = unitize(x);
//        return 2 * (x - 1) / (x - 2);
//    }

    /**
     * Get the location from which the supplied object's class was loaded.
     *
     * @param object the object for whose class the location should be retrieved
     * @return an {@code Optional} containing the URL of the class' location; never
     * {@code null} but potentially empty
     */
    public static @Nullable URL locate(ClassLoader loader, String className) {


        if (loader == null) {
            loader = ClassLoader.getSystemClassLoader();
            while (loader != null && loader.getParent() != null) {
                loader = loader.getParent();
            }
        }

        if (loader != null) {


            try {
                return (loader.getResource(className));
            } catch (RuntimeException ignore) {
                /* ignore */
            }
        }


        return null;
    }

    public static int concurrency() {
        return concurrencyExcept(0);
    }

    public static int concurrencyExcept(int reserve) {
        var specifiedThreads = System.getenv("threads");
        int threads;
        threads = specifiedThreads != null ? Str.i(specifiedThreads) : Runtime.getRuntime().availableProcessors() - reserve;

        var maxThreads = Integer.MAX_VALUE;
        var minThreads = 2;
        return clamp(
                threads, minThreads, maxThreads);
    }

    /**
     * modifies the input; instance compare, not .equals
     */
    public static <X> X[] replaceDirect(X[] xx, X from, X to) {
        for (int i = 0, xxLength = xx.length; i < xxLength; i++) {
            var x = xx[i];
            if (x == from)
                xx[i] = to;
        }
        return xx;
    }

    public static <X> X[] replaceDirect(X[] xx, UnaryOperator<X> f) {
        return replaceDirect(xx, 0, xx.length, f);
    }

    public static <X> X[] replaceDirect(X[] xx, int start, int end, UnaryOperator<X> f) {
        for (var i = start; i < end; i++) {
            var x = xx[i];
            xx[i] = f.apply(x);
        }
        return xx;
    }

    public static void assertUnitized(float... f) {
        for (var x : f)
            assertUnitized(x);
    }

    public static float assertUnitized(float x) {
        if (x != x || x < 0 || x > 1)
            throw new NumberException("non-unitized value: ", x);
        return x;
    }

    public static double assertUnitized(double x) {
        if (x != x || x < 0 || x > 1)
            throw new NumberException("non-unitized value: ", x);
        return x;
    }


    /**
     * tests if the array is already in natural order
     */
    public static <X extends Comparable> boolean isSorted(X[] x) {
        var n = x.length;
        if (n > 1) {
            for (var i = 1; i < n; i++) {
                if (x[i - 1].compareTo(x[i]) > 0)
                    return false;
            }
        }
        return true;
    }

    public static int[] bytesToInts(byte[] x) {
        var n = x.length;
        if (n == 0)
            return ArrayUtil.EMPTY_INT_ARRAY;
        var y = new int[n];
        for (var i = 0; i < n; i++)
            y[i] = x[i];
        return y;
    }

    public static Class[] typesOfArray(Object[] orgs) {
        return typesOfArray(orgs, 0, orgs.length);
    }

    public static Class[] typesOfArray(Object[] orgs, int from, int to) {
        return orgs.length == 0 ? ArrayUtil.EMPTY_CLASS_ARRAY : map(x -> Primitives.unwrap(x.getClass()),
                new Class[to - from], 0, orgs, from, to);
    }

    public static Lst<Class<?>> typesOf(Object[] orgs, int from, int to) {
        return new Lst<>(typesOfArray(orgs, from, to));
    }


    /**
     * fits a polynomial curve to the specified points and compiles an evaluator for it
     */
    public static <X> ToIntFunction<X> curve(ToIntFunction<X> toInt, int... pairs) {
        var c = curve(pairs);
        return x -> c.applyAsInt(toInt.applyAsInt(x));
    }

    public static IntToIntFunction curve(int... pairs) {
        if (pairs.length % 2 != 0)
            throw new RuntimeException("must be even # of arguments");

        var points = pairs.length / 2;
        if (points < 2) {
            //TODO return constant function
            throw new RuntimeException("must provide at least 2 points");
        }

        //https://commons.apache.org/proper/commons-math/userguide/fitting.html
        List<WeightedObservedPoint> obs = new Lst<>(points);
        int yMin = Integer.MAX_VALUE, yMax = Integer.MIN_VALUE;
        for (var i = 0; i < pairs.length; ) {
            int y;
            obs.add(new WeightedObservedPoint(1, pairs[i++], y = pairs[i++]));
            if (y < yMin) yMin = y;
            if (y > yMax) yMax = y;
        }
        //TODO if yMin==yMax return constant function

        var degree =
                points - 1;
        //points;

        var coefficients = toFloat(PolynomialCurveFitter.create(degree).fit(obs));

        /* adapted from: PolynomialFunction
           https://en.wikipedia.org/wiki/Horner%27s_method
           */
        int YMin = yMin, YMax = yMax;
//		assert (yMin < yMax);
        return x -> {
            var n = coefficients.length;
            double y = coefficients[n - 1];
            for (var j = n - 2; j >= 0; j--)
                y = x * y + coefficients[j];

            return clampSafe((int) Math.round(y), YMin, YMax);
        };
    }

    public static int sqrtInt(float x) {
        assertPositiveForSqrt(x);
        return (int) Math.round(Math.sqrt(x));
    }

    public static int sqrtIntFloor(float x) {
        assertPositiveForSqrt(x);
        return (int) (Math.sqrt(x));
    }

    public static int sqrtIntCeil(float x) {
        assertPositiveForSqrt(x);
        return (int) (ceil(Math.sqrt(x)));
    }

    public static int cbrtInt(float x) {
        assertPositiveForSqrt(x);
        return (int) Math.round(Math.pow(x, 1 / 3.0));
    }

    private static void assertPositiveForSqrt(float x) {
        if (x < 0) throw new NumberException("sqrt of negative value", x);
    }

    public static int logInt(float x) {
        return (int) Math.round(Math.log(x));
    }


    /**
     * scan either up or down within a capacity range
     */
    public static int next(int current, boolean direction, int cap) {
        if (direction) {
            if (++current == cap) return 0;
        } else {
            if (--current == -1) return cap - 1;
        }
        return current;
    }

    /**
     * if the collection is known to be of size==1, get that item in a possibly better-than-default way
     * according to the Collection's implementation
     */
    public static @Nullable <X> X only(Collection<X> next) {
        return (X) switch (next) {
            case List list -> list.get(0);
            case MutableSet set -> set.getOnly();
            case SortedSet set -> set.first();
            case null, default -> next.iterator().next();
        };
        //TODO SortedSet.getFirst() etc
    }

    @SafeVarargs
    public static <X> IntSet intSet(ToIntFunction<X> f, X... items) {
        switch (items.length) {
            case 0:
                return IntSets.immutable.empty();
            case 1:
                return IntSets.immutable.of(f.applyAsInt(items[0]));
            case 2:
                return IntSets.immutable.of(f.applyAsInt(items[0]), f.applyAsInt(items[1]));
            //...
            default:
                var i = new IntHashSet(items.length);
                for (var x : items) {
                    i.add(f.applyAsInt(x));
                }
                return i;
        }
    }


//	public static double interpSum(float[] data, double sStart, double sEnd) {
//		return interpSum((i) -> data[i], data.length, sStart, sEnd, false);
//	}

    /**
     * TODO not 100% working
     */
    public static double interpMean(IntToFloatFunction data, int capacity, double sStart, double sEnd, @Deprecated boolean wrap) {
        sStart = Math.max(0, sStart);
        sEnd = Math.min(capacity - 1, sEnd);

        var iStart = (int) ceil(sStart);
        var iEnd = (int) (sEnd);

        if (iEnd == iStart - 1) {
            if (iEnd >= 0 && iEnd < capacity)
                return data.valueOf(iEnd);//single unit
            else
                return Double.NaN;
        }
//
//		int i = iStart;
//		if (i < 0) {
//			if (wrap)
//				while (i < 0) i += capacity;
//			else
//				i = 0;
//		} else if (i >= capacity) {
//			i = 0; //wrap?
//		}

        if (iEnd < 0 || iStart >= capacity)
            return 0; //OOB

        iStart = Math.max(0, iStart);
        iEnd = Math.min(capacity - 1, iEnd);

        double sum = 0;

//		if (iStart < sStart)
//			sum += (iStart - sStart) * data.valueOf(iStart-1);

        if (iStart == iEnd)
            iEnd = iStart + 1; //only HACK
        for (var k = iStart; k < iEnd; k++) {
            sum += data.valueOf(k);
        }
        return sum / (iEnd - iStart);

//		if (iEnd < sEnd)
//			sum += (sEnd - iEnd) * data.valueOf(iEnd);

//		return sum/(sEnd-sStart);
    }


    public static int longToInt(long x) {
        //if (x > Integer.MAX_VALUE/4  || x < Integer.MIN_VALUE/4) //TEMPORARY
        if (x > Integer.MAX_VALUE || x < Integer.MIN_VALUE)
            throw new NumberException("long exceeds int capacity", x);
        return (int) x;
    }

    public static void normalizeHamming(float[] v, float epsilon) {
        normalizeHamming(v, 1, epsilon);
    }

    public static void normalizeHamming(float[] v, float target, float epsilon) {
        float current = 0;
        for (var value : v)
            current += Math.abs(value);

        if (current < epsilon) {
            Arrays.fill(v, target / v.length);
        } else {
            var scale = target / current;
            for (var i = 0; i < v.length; i++)
                v[i] *= scale;
        }
    }

    public static void normalizeHamming(double[] v, double epsilon) {
        double current = 0;
        for (var value : v)
            current += Math.abs(value);

        if (current < epsilon) {
            Arrays.fill(v, 1.0 / v.length);
        } else {
            var scale = 1.0 / current;
            for (var i = 0; i < v.length; i++)
                v[i] *= scale;
        }
    }

    public static long readToWrite(long l, StampedLock lock) {
        return readToWrite(l, lock, true);
    }

    public static long readToWrite(long l, StampedLock lock, boolean strong) {

        if (l != 0) {
            if (StampedLock.isWriteLockStamp(l))
                return l;

            var ll = lock.tryConvertToWriteLock(l);
            if (ll != 0) return ll;

            if (!strong) return 0;

            lock.unlockRead(l);
        }

        return strong ? lock.writeLock() : lock.tryWriteLock();
    }

    public static long writeToRead(long l, StampedLock lock) {
        if (l != 0) {
            if (StampedLock.isReadLockStamp(l))
                return l;

            var ll = lock.tryConvertToReadLock(l);
            if (ll != 0) return ll;

            lock.unlockWrite(l);
        }

        return lock.readLock();
    }

    /**
     * selects the previous instance if equal
     */
    public static <X, Y extends X, Z extends X> X maybeEqual(@Nullable Z next, @Nullable Y prev) {
        return Objects.equals(next, prev) ? prev : next;
    }

    public static long[] maybeEqual(long[] next, @Nullable long[] prev) {
        return next == prev || (prev != null && ArrayUtil.equals(next, prev)) ?
                prev :
                next;
    }

    public static <X> X maybeEqual(X next, X prevA, X prevB) {
        if (Objects.equals(next, prevA)) return prevA;
        else if (Objects.equals(next, prevB)) return prevB;
        else return next;
    }

    /**
     * NOP harness, useful for debugging
     */
    public static void nop() {
        //Thread.onSpinWait();
    }

    /**
     * untested
     */
    public static float lerpInverse(float x, float min, float max) {
        return 1 / lerp(x, 1 / max, 1 / min);
    }

    /**
     * compose filter from one or two filters
     */
    public static <X> Predicate<X> and(@Nullable Predicate<X> a, @Nullable Predicate<X> b) {
        if (b == null || a == b) return a;
        else if (a == null) return b;
        else return a.and(b);
    }

    public static <X,Y> Map.Entry<X, Y> firstEntry(Iterable<Map.Entry<X, Y>> m) {
        return m instanceof ArrayUnenforcedSet as ?
            ((ArrayUnenforcedSet<Map.Entry<X, Y>>) m).items[0] :
            m.iterator().next();
    }

    /**
     * 2 pairs of key, values pairs
     */
    public static <X> X[] firstTwoEntries(X[] target, Set<? extends Map.Entry<? extends X, X>> m) {
        X a, aa, b, bb;
        if (m instanceof ArrayUnenforcedSet) {
            //direct access
            var ee = ArrayUnenforcedSet.toArrayShared((ArrayUnenforcedSet<Map.Entry<X, X>>) m);
            a = ee[0].getKey();
            aa = ee[0].getValue();
            b = ee[1].getKey();
            bb = ee[1].getValue();
        } else {
            var ii = m.iterator();
            var aaa = ii.next();
            var bbb = ii.next();
            a = aaa.getKey();
            aa = aaa.getValue();
            b = bbb.getKey();
            bb = bbb.getValue();
        }
        target[0] = a;
        target[1] = aa;
        target[2] = b;
        target[3] = bb;
        return target;
    }

    public static MethodHandle sequence(MethodHandle[] mh) {
        var result = mh[mh.length - 1];
        for (var i = mh.length - 2; i >= 0; i--)
            result = MethodHandles.foldArguments(result, mh[i]);
        return result;
    }

    /**
     * Wang-Jenkings Hash Spreader
     * <p>
     * Applies a supplemental hash function to a given hashCode, which
     * defends against poor quality hash functions.  This is critical
     * because we use power-of-two length hash tables, that otherwise
     * encounter collisions for hashCodes that do not differ in lower
     * or upper bits.
     * <p>
     * from: ConcurrentReferenceHashMap.java found in Hazelcast
     */
    public static int spreadHash(int h) {
        h += (h << 15) ^ 0xffffcd7d;
        h ^= (h >>> 10);
        h += (h << 3);
        h ^= (h >>> 6);
        h += (h << 2) + (h << 14);
        return h ^ (h >>> 16);
    }


    public static VarHandle VAR(Class c, String field, Class<?> type) {
        try {
            return MethodHandles.privateLookupIn(c, lookup())
                    .findVarHandle(c, field, type)
                    .withInvokeExactBehavior();
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static <X> X ifEqualThen(@Nullable X x, X maybeEqualX, X ifEqualsXThenThis) {
        return Objects.equals(x, maybeEqualX) ? ifEqualsXThenThis : x;
    }

    public static boolean inIncl(int x, int min, int max) {
        return x >= min && x <= max;
    }

    public static boolean inIncl(long x, long min, long max) {
        return x >= min && x <= max;
    }

    public static boolean inIncl(float x, float min, float max) {
        return x >= min && x <= max;
    }


    public static <X> Consumer<X> compose(Consumer<X> a, @Nullable Consumer<X> b) {
        return b == null ? a : a.andThen(b);
    }

    public static <X> Iterator<X> nonNull(Iterator<X> i) {
        return Iterators.filter(i, Objects::nonNull);
    }

    /**
     * returns the ith byte in the 4 bytes of int x
     */
    public static int intByte(int x, int i) {
        return switch (i) {
            case 0 -> (x & 0x000000ff);
            case 1 -> (x & 0x0000ff00) >> 8;
            case 2 -> (x & 0x00ff0000) >> 16;
            case 3 -> (x & 0xff000000) >> 24;
            default -> throw new UnsupportedOperationException();
        };
    }

    /**
     * returns a value between 0 and 1.0f for the proportion of the 8-bit range covered by the ith sub-byte of integer x
     */
    public static float intBytePct(int x, int i) {
        return intByte(x, i) / 256f;
    }

    @Is("Relative_change_and_difference")
    public static double pctDiff(double x, double y) {
        return x == y ? 0 : Math.abs(x - y) /
                Math.max(Math.abs(x), Math.abs(y));
                //mean(abs(x), abs(y));
                //min(abs(x), abs(y));
    }

    public static void clampSafe(double[] x, double min, double max) {
        var n = x.length;
        for (var a = 0; a < n; a++)
            x[a] = clampSafe(x[a], min, max);
    }

    public static double[] normalizePolar(double[] x, double lenThresh) {
        var minmax = minmax(x);
        var rad = Math.max(Math.abs(minmax[0]), Math.abs(minmax[1]));
        return normalize(x, 0, x.length, -rad, +rad);
    }

    /**
     * https://en.wikipedia.org/wiki/LogSumExp
     * https://pytorch.org/docs/stable/generated/torch.logsumexp.html
     * https://www.deeplearningbook.org/slides/04_numerical.pdf (page 31)
     */
    @Is("LogSumExp")
    public static double logsumexp(double[] x, double innerPlus, double innerProd) {
        var mx = innerProd * (max(x) + innerPlus);
        double s = 0;
        for (var xx : x) {
            //s += Math.exp(innerProd * (xx + innerPlus) - mx);
            s += Math.exp(fma(innerProd, xx + innerPlus, -mx));
        }
        var y = Math.log(s) + mx;
        //System.out.println(logsumexp_simple(x, innerPlus, innerProd) + " "  + y);
        return y;
    }

    /**
     * for reference
     */
    @Deprecated
    private static double logsumexp_simple(double[] x, double innerPlus, float innerMult) {
        double s = 0;
        for (var xx : x)
            s += Math.exp(innerMult * (xx + innerPlus));
        return Math.log(s);
    }

    public static int shortUpper(int x) {
        return x >> 16;
    }

    public static int shortLower(int x) {
        return x & 0xffff;
    }

    public static int shortToInt(short high, short low) {
        return high << 16 | low;
    }

    public static int shortToInt(int high, int low) {
        return high << 16 | low;
    }

    public static short shortToInt(int x, boolean high) {
        return (short) (high ? x >> 16 : x & 0xffff);
    }

    public static <X> Supplier<X> once(Supplier<X> f) {
        return new Supplier<>() {
            X x;

            @Override
            public X get() {
                var x = this.x;
                return x == null ? (this.x = f.get()) : x;
            }
        };
    }

    public static double powAbs(double x, float p) {
        return x >= 0 ? Math.pow(x, p) : -Math.pow(-x, p);
    }

    public static double halflifeRate(float period) {
        //half-life
        //instant
        return period < Double.MIN_NORMAL ? 1 : Math.min(1, Math.log(2) / period);
    }



    public static DoubleToDoubleFunction lerpLog(double min, double max) {
        double minLog = Math.log(min), maxLog = Math.log(max);
        return x -> Math.exp(lerpSafe(x, minLog, maxLog));
    }

    public static float[] normalizeToSum(double[] x, int start, int end, double sumTarget) {
        var sumInput = sum(x, start, end);
        var s = sumTarget / sumInput;
        var y = new float[end - start];
        for (int i = start, j = 0; i < end; i++, j++)
            y[j] = (float) (x[i] * s);
        return y;
    }

    /** use with next(): sets the flag for a next() iteration */
    public static void once(AtomicBoolean changed) {
        changed.setRelease(true);
    }
    /** use with once(): acquires the next iteration caused by once() */
    public static boolean next(AtomicBoolean changed) {
        return changed.compareAndExchangeAcquire(true, false);
    }

    /** use with exitAlone() */
    public static boolean enterAlone(AtomicBoolean busy) {
        return !busy.compareAndExchangeAcquire(false, true);
    }
    /** use with enterAlone() */
    public static void exitAlone(AtomicBoolean busy) {
        busy.setRelease(false);
    }

    public static double[] abs(double[] x) {
        var n = x.length;
        for (var i = 0; i < n; i++)
            x[i] = Math.abs(x[i]);
        return x;
    }

    public static void addTo(double[] x, double y) {
        var n = x.length;
        for (var j = 0; j < n; j++) x[j] += y;
    }

    /** x += y */
    public static void addTo(double[] x, double[] y) {
        var n = x.length;
        if (n != y.length)
            throw new UnsupportedOperationException();
        for (var j = 0; j < n; j++)
            x[j] += y[j];
    }
    /** x +=  * m */
    public static void addTo(double[] x, double[] y, double m) {
        var n = x.length;
        if (n!=y.length) throw new UnsupportedOperationException();
        if (m == 0) return;

        for (var j = 0; j < n; j++)
            x[j] += y[j] * m; //TODO fma
    }

    public static void addTo(double[][] xx, double y) {
        var h = xx.length;
        for (var x : xx)
            addTo(x, y);
    }

    public static double sumAbs(double[][] x) {
        var s = new KahanSum();
        var n = x.length;
        for (var xx : x)
            s.add(sumAbs(xx));
        return s.value();
    }

    public static double maxAbs(double[][] gg) {
        double m = 0;
        var I = gg.length;
        for (var g : gg) {
            double x = maxAbs(g);
            m = Math.max(x, m);
        }
        return m;
    }

    public static double maxAbs(double[] g) {
        var J = g.length;
        double m = 0;
        for (var v : g)
            m = Math.max(Math.abs(v), m);
        return m;
    }

    public static void unitizeSafe(double[] x) {
        for (int i = 0, length = x.length; i < length; i++)
            x[i] = unitizeSafe(x[i]);
    }

    /** TODO Kahan sum */
    public static double dot(double[] a, double[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Vectors are not of equal length");

        var res = 0;
        for (var i = 0; i < b.length; i++)
            res += a[i] * b[i];

        return res;
    }

    public static double dot(double[] a, double b) {

        var n = a.length;
        var s = new KahanSum();
        for (var i = 0; i < n; i++)
            s.add(a[i] * b);
        return s.value();
    }

    public static double[] sub(double[] a, double[] b) {
        var y = a.clone();
        for (var i = 0; i < y.length; i++)
            y[i] -= b[i];
        return y;
    }

    public static void randomGaussian(double[] data, double stddev, RandomGenerator rng) {
        var n = data.length;
        for (var i = 0; i < n; i++)
            data[i] = rng.nextGaussian() * stddev;
    }

    public static void randomGaussian(double[][] data, double stddev, RandomGenerator rng) {
        var I = data.length;
        for (var i = 0; i < I; i++)
            randomGaussian(data[i], stddev, rng);
    }

    public static void clamp(double[] x, double min, double max) {
        clamp(x, x, min, max);
    }

    public static void clamp(double[] x, double[] y, double min, double max) {
        if (min == Double.NEGATIVE_INFINITY && max == Double.POSITIVE_INFINITY) return;
        var I = y.length;
        for (var i = 0; i < I; i++) {
            var xij = x[i];
            //if (!isFinite(xij)) xij = 0; //repair
            y[i] = clamp(xij, min, max);
        }
    }
    public static void clamp(float[] x, float[] y, float min, float max) {
        if (min == Float.NEGATIVE_INFINITY && max == Float.POSITIVE_INFINITY) return;
        var I = y.length;
        for (var i = 0; i < I; i++) {
            var xij = x[i];
            //if (!isFinite(xij)) xij = 0; //repair
            y[i] = clamp(xij, min, max);
        }
    }
    
    public static void clamp(double[][] x, double min, double max) {
        clamp(x, x, min, max);
    }
    public static void clamp(double[][] x, double[][] y, double min, double max) {
        if (min == Double.NEGATIVE_INFINITY && max == Double.POSITIVE_INFINITY) return;
        var I = y.length;
        for (var i = 0; i < I; i++)
            clamp(x[i], y[i], min, max);
    }

    public static void fill(double[][] data, double x) {
        for (var d : data)
            Arrays.fill(d, x);
    }

    public static double sumPrecise(double a, double b) {
        var s = a + b;
        var v = s - a;
        var e = (a - (s - v)) + (b - v);
        return s + e;
    }

    public static double signum(double x) {
        if (x > 0) return +1;
        else if (x < 0) return -1;
        else return 0;
    }

    public static double replaceNaNwithRandom(@Nullable double x, RandomGenerator random) {
        var xx = new double[]{x};
        replaceNaNwithRandom(xx, random);
        return xx[0];
    }

    public static void replaceNaNwithRandom(@Nullable double[] x, RandomGenerator random) {
        if (x==null) return;

        for (var j = 0; j < x.length; j++) {
            var i = x[j];
            if (i!=i)
                x[j] = random.nextFloat();
        }
    }

}