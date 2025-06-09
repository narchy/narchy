package jcog.pri.op;

import jcog.Fuzzy;
import jcog.Util;
import jcog.pri.Prioritizable;
import jcog.pri.Prioritized;
import jcog.util.PriReturn;
import org.eclipse.collections.api.block.function.primitive.DoubleDoubleToDoubleFunction;
import org.jetbrains.annotations.Nullable;

import java.util.function.BiConsumer;

/**
 * Budget merge function, with input scale factor
 */

public enum PriMerge implements BiConsumer<Prioritizable, Prioritized>, DoubleDoubleToDoubleFunction {

    plus {
        @Override
        public double valueOf(double e, double i) {
            return e + i;
        }
    },
    minus {
        @Override
        public double valueOf(double e, double i) {
            return e - i;
        }
    },
    mean {
        @Override
        public double valueOf(double e, double i) {
            return Util.mean(e, i);
        }
    },
    meanGeo {
        @Override
        public double valueOf(double e, double i) {
            return Fuzzy.meanGeo(e, i);
        }
    },
    and {
        @Override
        public double valueOf(double e, double i) {
            return Fuzzy.and(e, i);
        }
    },
    meanAndMean {
        @Override
        public double valueOf(double e, double i) {
            return Fuzzy.meanAndMean(e, i);
        }
    },
    or {
        @Override
        public double valueOf(double e, double i) {
            return Fuzzy.or(e, i);
        }
    },
    min {
        @Override
        public double valueOf(double e, double i) {
            return Math.min(e, i); }
    },
    max {
        @Override
        public double valueOf(double e, double i) {
            return Math.max(e, i); }
    },
    einsteinSum {
        @Override public double valueOf(double e, double i) {
            return Fuzzy.einsteinSum(e,i);
        }
    },
    replace {
        @Override
        public double valueOf(double e, double i) {
            return i;
        }
    }
    //    AVG_GEO,
    //    AVG_GEO_SLOW, //adds momentum by includnig the existing priority as a factor twice against the new value once
    //    AVG_GEO_FAST,

    ;

//    /** either double or double form must be implemented */
//    @Override public double valueOf(double e, double i) {
//        return apply(e, i);
//    }



    @Override
    public final void accept(Prioritizable existing, Prioritized incoming) {
        apply(existing, incoming.pri());
    }

    /**
     * merge 'incoming' budget (scaled by incomingScale) into 'existing'
     */
    public final float apply(Prioritizable x, float pri, @Nullable PriReturn mode) {
        float pBefore = x.priElseZero();
        x.pri((float) valueOf(pBefore, pri));
        return mode !=null ? mode.apply(pri, pBefore,
            x.priElseZero()
            //existing.pri()
            //pNext
        ) : Float.NaN;
    }

    public final void apply(Prioritizable x, float pri) {
        x.pri((float) valueOf(x.priElseZero(), pri));
    }

//    protected boolean commutative() {
//        throw new TODO();
//    }

    /**
     * merges for non-NaN 0..1.0 range
     */
    public final float mergeUnitize(float x, float incoming) {
        if (x != x)
            x = 0;
        float next = (float) valueOf(x, incoming);
        if (next == next) {
            if (next > 1) next = 1;
            else if (next < 0) next = 0;
        } else
            next = 0;
        return next;
    }


//
//    /**
//     * sum priority
//     */
//    PriMerge<Prioritizable,Prioritized> plus = (tgt, src) -> merge(tgt, src, PLUS);
//
//    /**
//     * avg priority
//     */
//    PriMerge<Prioritizable,Prioritized> avg = (tgt, src) -> merge(tgt, src, AVG);
//
//    PriMerge<Prioritizable,Prioritized> or = (tgt, src) -> merge(tgt, src, OR);
//
//
//    PriMerge<Prioritizable,Prioritized> max = (tgt, src) -> merge(tgt, src, MAX);
//
//    /**
//     * avg priority
//     */
//    PriMerge<Prioritizable,Prioritized> replace = (tgt, src) -> tgt.pri((doubleSupplier)()-> src.pri());


//    PriMerge<Prioritizable,Prioritized> avgGeoSlow = (tgt, src) -> merge(tgt, src, AVG_GEO_SLOW);
//    PriMerge<Prioritizable,Prioritized> avgGeoFast = (tgt, src) -> merge(tgt, src, AVG_GEO_FAST);
//    PriMerge<Prioritizable,Prioritized> avgGeo = (tgt, src) -> merge(tgt, src, AVG_GEO); //geometric mean


}