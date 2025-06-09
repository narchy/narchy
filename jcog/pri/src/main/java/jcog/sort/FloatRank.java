package jcog.sort;

import org.eclipse.collections.api.block.function.primitive.FloatFunction;
import org.jetbrains.annotations.Nullable;

@FunctionalInterface
public interface FloatRank<X> extends FloatFunction<X> {
    /**
     * @param min this value which may be NEGATIVE_INFINITY, is a value that the rank must exceed to matter.
     *            so if a scoring function can know that, before completing,
     *            it wont meet this threshold, it can fail fast (by returning NaN).
     */
    float rank(X x, float min);

    default float rank(X x) {
        return rank(x, Float.NEGATIVE_INFINITY);
    }

    /**
     * adapter which ignores the minimum
     */
    static <X> FloatRank<X> the(FloatFunction<X> f) {
        return f instanceof FloatRank<X> fr ? fr : ((x, min) -> f.floatValueOf(x));
    }


    @Override
    default float floatValueOf(X x) {
        return rank(x, Float.NEGATIVE_INFINITY);
    }

//    default FloatRank<X> filter(@Nullable Predicate<X> filter) {
//        return filter == null ? this : new FilteredFloatRank<>(filter, this);
//    }

//    /**
//     * assumes f returns values in 0..1, to enable early exit condition
//     */
//    default FloatRank<X> mulUnit(FloatFunction<? super X> f) {
//        return (x, min) -> {
//            float y = rank(x, min);
//            return /*y != y || [covered]*/ y < min ? Float.NaN : y * f.floatValueOf(x);
//        };
//    }

    default FloatRank<X> mulUnitPow(@Nullable FloatFunction<? super X> base, float power) {
        if (base == null) return this;

        //TODO shortcut for pow=2, 1/2, etc..
        FloatFunction<? super X> f =
            power == 1 ?
                base :
                x -> (float) Math.pow(base.floatValueOf(x), power);

        return mul(f);
    }

    default FloatRank<X> mul(@Nullable FloatFunction<? super X> f) {
        if (f == null) return this;
        return (x, min) -> {
            float y = rank(x, min);
            return y != y ? Float.NaN :
                    (float) (((double) y) * f.floatValueOf(x));
        };
    }

    @Nullable default X better(@Nullable X x, @Nullable X y) {
        return x==null ? y : ((y == null || (floatValueOf(x) >= floatValueOf(y)) ? x : y));
    }



//    record FilteredFloatRank<X>(@Nullable Predicate<X> filter, FloatRank<X> rank) implements FloatRank<X> {
//
//        public FilteredFloatRank(@Nullable Predicate<X> filter, FloatRank<X> rank) {
//            this.filter = filter;
//            this.rank = rank;
//        }
//
//        @Override
//        public float rank(X t, float m) {
//            return filter != null && !filter.test(t) ? Float.NaN : rank.rank(t, m);
//        }
//    }
}