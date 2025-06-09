package jcog.math;

import jcog.Research;
import jcog.Util;
import org.jetbrains.annotations.Nullable;

import static java.lang.Math.*;
import static jcog.math.LongInterval.ETERNAL;
import static jcog.math.LongInterval.TIMELESS;


/** utility class for working with time interval (number pairs) */
public enum Intervals { ;

    public static int compare(LongInterval a, LongInterval b) {
        if (a == b) {
            return 0;
        } else {
            long as = a.start(), bs = b.start();
            // Compare start times
            if (as < bs) return -1;
            if (as > bs) return 1;

            long ae = a.end(), be = b.end();
            // If start times are equal, compare end times
            if (ae < be) return -1;
            if (ae > be) return 1;

            // If both start and end times are equal
            return 0;

//            long as = a.start(), ae = a.end(), bs = b.start(), be = b.end();
//            long d = as - bs + ae - be;
//            if (d == 0) return 0;
//            else if (d > 0) return +1;
//            else return -1;
        }
    }

    public static String tStr(long s, long e) {
        if (s == ETERNAL)
            return "ETE";
        else if (s == TIMELESS)
            return "TIMELESS";
        else
            return s + ".." + e;
    }

    /**
     * returns -1 if no intersection; 0 = adjacent, > 0 = non-zero interval in common
     */
    public static long intersectLength(long as, long ae, long bs, long be) {
        if (as == TIMELESS || bs == TIMELESS)
            throw new UnsupportedOperationException();
        if (as == ETERNAL) {
            if (bs == ETERNAL) throw new UnsupportedOperationException();
            return be - bs;
        } else if (bs == ETERNAL)
            return ae - as;
        else
            return intersectLengthRaw(as, ae, bs, be);
    }

    @Nullable
    public static long[] intersectionRaw(long as, long ae, long bs, long be) {
        long a = max(as, bs), b = min(ae, be);
        return a <= b ? new long[] { a, b } : null;
    }

    private static long intersectLengthRaw(long as, long ae, long bs, long be) {
        long a = max(as, bs), b = min(ae, be);
        return a <= b ? b - a : -1;
    }

    private static double intersectLengthRaw(double as, double ae, double bs, double be) {
        double a = max(as, bs), b = min(ae, be);
        return a <= b ? b - a : 0;
    }

    private static boolean intersects(double as, double ae, double bs, double be) {
        return max(as, bs) <= min(ae, be);
    }

    /**
     * cs,ce = container;   xs,xe = possibly contained
     */
    public static boolean containsRaw(long cs, long ce, long xs, long xe) {
        return cs <= xs && xe <= ce;
    }

    public static boolean contains(double cs, double ce, double xs, double xe) {
        return cs <= xs && xe <= ce;
    }

    public static double diffTotal(double fs, double fe, double ts, double te) {
        return abs(fs - ts) + abs(fe - te);
    }
    public static long diffTotal(long fs, long fe, long ts, long te) {
        return abs(fs - ts) + abs(fe - te);
    }
    public static long diffSep(long fs, long fe, long ts, long te) {
        return min(min(abs(fe - ts), abs(fs - ts)), min(abs(fe - te), abs(fs - te)));
    }

    public static double diffSep(double[] f, double[] t) {
        return diffSep(f[0], f[1], t[0], t[1]);
    }

    public static double diffSep(double fs, double fe, double ts, double te) {
        return min(min(abs(fe - ts), abs(fs - ts)), min(abs(fe - te), abs(fs - te)));
    }

    public static long[] range(long center, float diameter) {
        long rad = (long) ceil(diameter / 2.0);
        return new long[]{
            center - rad,
            center + rad
        };
    }

    public static long[] unionArray(long xs, long xe, long ys, long ye) {
        return (ys == ETERNAL ? new long[]{xs, xe} :
                (xs == ETERNAL ? new long[]{ys, ye} :
                        new long[]{min(xs, ys), max(xe, ye)})
        );
    }

    public static long[] shrink(long cs, long ce, long os, long oe) {
        long[] cse = {cs, ce};
        shrink(cse, os, oe);
        return cse;
    }
    /**
     * Shrinks the target range to match the size of the limiting range, while keeping its position.
     * Decides which side(s) to shrink from based on the relative positions of the ranges.
     * @param x target range to adjust, represented as [start, end]
     * @param os start of the limiting interval
     * @param oe end of the limiting interval
     */
    public static void shrink(long[] x, long os, long oe) {
        long xs = x[0], xe = x[1];
        long xr = xe - xs, or = oe - os;

        // If target range is already smaller or equal to limit, do nothing
        if (xr > or) {
            if (xs <= os && oe <= xe) {
                //if Target contains limit, shrink to limit
                x[0] = os;
                x[1] = oe;
            } else {
                long pre = os - xs, post = xe - oe;
                if (pre >= post) {
                    // If target starts more before limit than after, shrink from start
                    x[0] = xe - or;
                } else {
                    // If target ends more after limit than before, shrink from end
                    x[1] = xs + or;
                }
            }
        }
    }
    /**
     * Adjusts the target range to be entirely within the limiting interval,
     * positioning it as close as possible to the original target range.
     *
     * @param x  The target range to adjust, represented as [start, end].
     * @param os The start of the limiting interval.
     * @param oe The end of the limiting interval.
     * @throws IllegalArgumentException if input ranges are invalid.
     */
    public static void fitWithin(long[] x, long os, long oe) {
        // Input validation
//        if (x == null || x.length != 2)
//            throw new IllegalArgumentException("Array x must have exactly two elements.");
        long xs = x[0], xe = x[1];
        if (xs == ETERNAL || xs == TIMELESS) {
            x[0] = os; x[1] = oe;
            return;
        }
//        if (xs > xe)
//            throw new IllegalArgumentException("Start of target range must be <= end.");
//        if (os > oe)
//            throw new IllegalArgumentException("Start of limiting interval must be <= end.");

        long targetSize = xe - xs;
        long limitSize = oe - os;

        if (targetSize > limitSize) {
            // If target is larger than limit, set to limit
            x[0] = os;
            x[1] = oe;
        } else {
            // Attempt to keep the original position as much as possible
            long newStart = xs;
            long newEnd = xe;

            // If target starts before the limit, shift right
            if (newStart < os) {
                newStart = os;
                newEnd = newStart + targetSize;
            }

            // If target ends after the limit, shift left
            if (newEnd > oe) {
                newEnd = oe;
                newStart = newEnd - targetSize;
            }

            // In case shifting causes the start to go before the limit's start
            if (newStart < os) {
                newStart = os;
                newEnd = os + targetSize;
            }

            x[0] = newStart;
            x[1] = newEnd;
        }
    }

    /**
     * true if [as..ae] intersects [bs..be]
     */
    public static boolean intersects(long as, long ae, long bs, long be) {
        //assert (as != TIMELESS && bs != TIMELESS);
        return intersectsSafe(as, ae, bs, be);
    }

    public static boolean intersectsSafe(long as, long ae, long bs, long be) {
        return (as == ETERNAL) || (bs == ETERNAL) || intersectsRaw(as, ae, bs, be);
    }

    public static boolean intersectsRaw(long[] A, long[] B) {
        return intersectsRaw(A[0], A[A.length-1], B[0], B[B.length-1]);
        //return intersectsRaw(A[0], A[1], B[0], B[1]);
    }

    public static boolean intersectsRaw(long as, long ae, long bs, long be) {
        return ae >= bs && be >= as;
        //return max(as, bs) <= min(ae, be);
    }

    public static boolean intersectsRaw(double as, double ae, double bs, double be) {
        return ae >= bs && be >= as;
    }

    public static boolean intersectsOrAdjacentRaw(long as, long ae, long bs, long be, int pad) {
        return max(as, bs) <= min(ae, be) + pad;
    }

    public static boolean intersectsRaw(long a, long bs, long be) {
        return a >= bs && be >= a;
        //return max(a, bs) <= min(a, be);
    }

    /**
     * does not test for eternal, timeless
     */
    public static long diffSep(long x, long s, long e) {
        return containsRaw(x, s, e) ? 0 :
                min(abs(x - s), abs(x - e));
    }

    /**
     * warning: does not test for eternal, timeless
     */
    public static long maxTimeToRaw(long x, long s, long e) {
        return max(abs(x - s), abs(x - e));
    }

    public static double meanTimeToRaw(long when, long s, long e) {
        return (abs(s - when) + abs(e - when)) / 2.0;
    }

    static boolean containsRaw(long w, long s, long e) {
        return w >= s && w <= e;
    }

    static boolean _during(long x, long start, long end) {
        return start <= x && end >= x;
    }

    public static double dSepNorm(double[] a, double[] b) {
        return dSepNorm(a[0], a[1], b[0], b[1]);
    }

    public static double dSepFraction(double[] a, double[] b) {
        return dSepFraction(a[0], a[1], b[0], b[1]);
    }

    public static double dSepNorm(double a0, double a1, double b0, double b1) {
        double aRange = a1 - a0, bRange = b1 - b0;
        return abs(aRange-bRange) + diffSep(a0, a1, b0, b1) / (1 + min(aRange, bRange));
        //return diffTotal(a0, a1, b0, b1) / (1 + Util.min(aRange, bRange));
        //return diffTotal(a0, a1, b0, b1) / (1 + Util.mean(aRange, bRange));

    }

    /**
     * ratio of absolute extrema distance to intersection range
     */
    @Research
    public static double dSepFraction(double a0, double a1, double b0, double b1) {
        return (diffTotal(a0, a1, b0, b1))
                / (1 + intersectLengthRaw(a0, a1, b0, b1));
//        return ((abs(a0 - b0) + abs(a1 - b1))
//                / (1 + intersectLengthRaw(a0, a1, b0, b1))
//                / (1 + Fuzzy.mean(a1-a0, b1-b0))) //normalize to mean range
//                ;
    }

    /**
     * ds + de
     */
    public static double diffTotal(double[] a, double[] b) {
        return diffTotal(a[0], a[1], b[0], b[1]);
    }

    public static boolean intersects(LongInterval x, long[] y) {
        return x.intersects(y[0], y[1]);
    }

    public static double diffMid(double[] a, double[] b) {
        return abs((a[1] + a[0]) - (b[1] + b[0])) / 2;
    }

    public static double diffRangeRaw(double[] a, double[] b) {
        return abs((a[1] - a[0]) - (b[1] - b[0]));
    }

    public static long timeBetween(long as, long ae, long bs, long be) {
        //assert(!(as ==TIMELESS || bs == TIMELESS));
        return as == ETERNAL || bs == ETERNAL ? 0 : timeBetweenRaw(as, ae, bs, be);
    }

    public static long timeBetweenRaw(long as, long ae, long bs, long be) {
        return intersectsRaw(as, ae, bs, be) ? 0 : as > be ? as - be : bs - ae;
    }

    public static double timeBetweenRaw(double[] a, double[] b) {
        return timeBetweenRaw(a[0], a[1], b[0], b[1]);
    }

    public static double timeBetweenRaw(double as, double ae, double bs, double be) {
        return intersectsRaw(as, ae, bs, be) ? 0 : ((as > be) ? (as - be) : (bs - ae));
    }

    public static double timeBetweenPct(double[] a, double[] b) {
        return pctRange(a, b, timeBetweenRaw(a, b));
    }

    public static double diffTotalPct(double[] a, double[] b) {
        return pctRange(a, b, diffTotal(a, b));
    }

    private static double pctRange(double[] a, double[] b, double dt) {
        return dt <= 0 ? 0 : dt / (1 +
                Util.mean(a[1] - a[0], b[1] - b[0])
                //Util.max(a[1] - a[0], b[1] - b[0])
        );
    }
}
