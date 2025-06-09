package jcog.math;

import org.eclipse.collections.impl.block.factory.Comparators;
import org.jetbrains.annotations.Nullable;

import java.util.Comparator;

import static java.lang.Math.*;
import static jcog.Fuzzy.mean;

/**
 * pair of 64-bit signed long integers representing an interval.
 * a special 'ETERNAL' value represents (-infinity,+infinity)
 * <p>
 * TODO allow (-infinity, ..x) and (x, +infinity)
 */
public interface LongInterval extends LongIntervalArray {

    Comparator<LongInterval> comparator2 = Comparators
            .byLongFunction(LongInterval::start)
            .thenComparingLong(LongInterval::end);

    long ETERNAL = Long.MIN_VALUE;
    LongInterval Eternal = new ImmLongInterval(ETERNAL, ETERNAL);

    long TIMELESS = Long.MAX_VALUE;
    LongInterval Timeless = new ImmLongInterval(TIMELESS, TIMELESS);

    int MIN = -1;
    int MEAN = 0;
    int MAX = 1;
    int SUM = 2;


    default long diffTotal(LongInterval x) {
        long xs = x.start();
        return xs==ETERNAL ? 0 : abs(start() - xs) + abs(end() - x.end());
    }

    long start();

    long end();

    @Override
    default long[] startEndArray() {
        return new long[]{start(), end()};
    }

    default long mid() {
        long s = start();
        return (s == ETERNAL || s == TIMELESS) ? s : mean(s, end());
    }

    /**
     * return number of elements between a and b inclusively. x..x is length 1.
     * if b &lt; a, then length is 0.  9..10 has length 2.
     */
    default long range() {
        long s = start();
        if (s == ETERNAL || s == TIMELESS)
            throw new ArithmeticException("ETERNAL range calculated");
        return 1 + (end() - s);
    }

    default /* final */ long rangeElse(long ifEternal) {
        long s = start();
        return s == ETERNAL ? ifEternal : 1 + (end() - s);
    }

    default /* final */ long rangeElseTimeless() {
        return rangeElse(TIMELESS);
    }

    /**
     * finds the nearest point within the provided interval relative to some point in this interval
     */
    @Deprecated
    default long nearestPointExternal(long a, long b) {
        if (a == b || a == ETERNAL)
            return a;

        long s = start();
        if (s == ETERNAL) {
            return mean(a,b); // Use unsigned shift for averaging
        }

        long e = end();

        long mid = mean(s, e);
        if (s >= a && e <= b)
            return mid;

        return abs(mid - a) <= abs(mid - b) ? a : b;
    }

    /**
     * finds the nearest point inside this interval to the provided range, which may be
     * inside, intersecting, or disjoint from this interval.
     */
    @Deprecated
    default long nearestPointInternal(long a, long b) {

        //assert (b >= a && (a != ETERNAL || a == b));

        if (a == ETERNAL)
            return mid();

        long s = this.start();
        if (s == ETERNAL)
            return ETERNAL;

        long e = this.end();
        if (s == e)
            return s;

        if ((a >= s) && (b <= e)) {
            return mean(a, b);
        } else if (a < s && b > e) {
            return mean(s, e);
        } else {
            long se = mean(s, e);
            long ab = mean(a, b);
            return se <= ab ? e : s;
        }
    }

    default /*final*/ long timeBetweenTo(LongInterval l) {
        return timeBetweenTo(l.start(), l.end());
    }

    /**
     * if the task intersects (ex: occurrs during) the specified interval,
     * returned time distance is zero, regardless of how far it may extend before or after it
     */
    default long timeBetweenTo(long a, long b) {
        return Intervals.timeBetween(start(), end(), a, b);
    }

    default long timeBetweenTo(long w) {
        return timeBetweenTo(w, w);
    }

    default long diff(long w) {
        long s = start();
        if (s == ETERNAL) return 0;

        if (w < s)
            return s - w;

        long e = end();
        if (w > e)
            return w - e;
        else
            return 0;  // inside interval => distance 0
    }

    default long timeMeanTo(long x) {
        return timeTo(x, false, MEAN);
    }

    /**
     * Calculates the distance between two intervals using their centers and radii.
     * Optimized version focusing on reducing unnecessary calculations.
     *
     * @param b The interval to compare against
     * @param rangeDiff Whether to include range difference in the calculation
     * @return The calculated distance metric
     */
    default long diffCenterRadii(LongInterval b, boolean rangeDiff) {
        long as = start(), bs = b.start();
        if (as == ETERNAL || bs == ETERNAL)
            return 0;

        long ae = end(), be = b.end();

        long gap = Intervals.timeBetweenRaw(as, ae, bs, be);

        return rangeDiff ? rangeDiff(as, ae, bs, be) + gap : gap; // Only calculate radii difference if needed
    }

    private static long rangeDiff(long as, long ae, long bs, long be) {
        return abs((ae - as) - (be - bs));
    }

//    /**
//     * Calculates the distance between two intervals using their centers and radii.
//     * Uses overflow-safe mean calculation.
//     *
//     * @param b The interval to compare against
//     * @param rangeDiff Whether to include range difference in the calculation
//     * @return The calculated distance metric
//     */
//    default long diffCenterRadii0(LongInterval b, boolean rangeDiff) {
//        long as = start(), bs = b.start();
//        if (as == ETERNAL || bs == ETERNAL)
//            return 0;
//
//        long ae = end(), be = b.end();
//
//        // Radii
//        long ar = ae - as, br = be - bs;
//
//        // Adjust distance by radii sum
//        long dt = max(0, abs(mean(as, ae) - mean(bs, be)) - ((ar/2) + (br/2)));
//
//        return rangeDiff ? dt + abs(ar - br) : dt;
//    }

    private static long rangeDiffDT(long ae, long as, long be, long bs) {
        long rangeA = ae - as, rangeB = be - bs;
        var rangeDiffDT = rangeA >= rangeB ? rangeA - rangeB : rangeB - rangeA;
        return rangeDiffDT;
    }


//    /**
//     * distance metric effectively captures the relationship between intervals by considering both their central positions and their sizes. It avoids the need for explicit case handling of overlap vs. non-overlap, making the function more uniform in its behavior.
//     */
//    default long diffCenterRadii(LongInterval b, boolean rangeDiff) {
//        long as = start();    if (as == ETERNAL) return 0;
//        long bs = b.start();  if (bs == ETERNAL) return 0;
//
//        long ae = end(), be = b.end();
//
//        //Center Distance
//        long dt = abs(mean(as, ae) - mean(bs, be));
//
//        //Adjusted for radii
//        long rangeA = ae - as, rangeB = be - bs;
//        long radiusA = rangeA /2, radiusB = rangeB /2;
//        dt = max(0, dt - (radiusA + radiusB) );
//
//        if (rangeDiff)
//            dt += abs(rangeA - rangeB);
//
//        return dt;
//    }

    default long timeMeanDuringOrTo(long x) {
        return timeTo(x, true, MEAN);
    }

    default long timeTo(long x, boolean zeroIfDuring, int mode) {
        if (x == ETERNAL) return 0;
        long s = start();
        if (s == ETERNAL) return 0;

        long e = end();

        if (zeroIfDuring && Intervals._during(x, s, e))
            return 0; //contained

        long ds = abs(s - x);
        return s == e ? ds : timeTo(mode, ds, abs(e - x));
    }

    private static long timeTo(int mode, long ds, long de) {
        return switch (mode) {
            case MIN -> min(ds, de);
            case MEAN -> mean(ds, de);
            case MAX -> max(ds, de);
            case SUM -> ds + de;
            default -> throw new UnsupportedOperationException();
        };
    }


    default @Nullable LongInterval intersection(LongInterval l) {
        return intersection(l.start(), l.end());
    }

    default @Nullable LongInterval intersection(long bs, long be) {
        long as = this.start();
        if (as == TIMELESS || bs == TIMELESS)
            throw new UnsupportedOperationException();
        else if (as == ETERNAL || bs == ETERNAL)
            return Eternal;
        else {
            long ae = this.end();
            long s = max(as, bs), e = min(ae, be);
            return s > e ? null : (s == as && e == ae ? this : new ImmLongInterval(s, e));
        }
    }

    default boolean intersects(LongInterval i) {
        return this == i || intersects(i.start(), i.end());
    }

    default boolean intersectsRaw(LongInterval i) {
        return this == i || intersectsRaw(i.start(), i.end());
    }

    default long intersectLength(LongInterval i) {
        return Intervals.intersectLength(start(), end(), i.start(), i.end());
    }

    default boolean intersects(long s, long e) {
        // Simplified conditional check
        return s != TIMELESS && (s == ETERNAL || start() == ETERNAL || intersectsRaw(s, e));
    }

    default boolean intersectsRaw(long s, long e) {
        return Intervals.intersectsRaw(s, e, start(), end());
        //return (e >= start() && s <= end());
    }

    default boolean contains(long s, long e) {
        assert (s != TIMELESS);
        long start = start();
        return start == ETERNAL || (s != ETERNAL && Intervals.containsRaw(start, end(), s, e));
    }

    default boolean containsRaw(LongInterval b) {
        return this == b || Intervals.containsRaw(start(), end(), b.start(), b.end());
    }

    default boolean containedBy(long cs, long ce) {
        // Simplified conditional check
        return cs == ETERNAL || (start() != ETERNAL && Intervals.containsRaw(cs, ce, start(), end()));
    }

    /**
     * eternal contains itself
     */
    default boolean contains(LongInterval b) {
        if (this == b) return true;

        long as = start();
        // Simplified conditional check and inlined static method call
        return as == ETERNAL || (b.start() != ETERNAL && Intervals.containsRaw(as, end(), b.start(), b.end()));
    }

    default long diffRaw(long a, long b) {
        return abs(a - start()) + abs(b - end());
    }
    default long diffRaw(long w) {
        return abs(w - start()) + abs(w - end());
    }

}