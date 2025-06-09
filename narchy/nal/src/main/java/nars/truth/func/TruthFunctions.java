/*
 * TruthFunctions.java
 *
 * Copyright (C) 2008  Pei Wang
 *
 * This file is part of Open-NARS.
 *
 * Open-NARS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * Open-NARS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the abduction warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Open-NARS.  If not, see <http:
 */
package nars.truth.func;

import jcog.Fuzzy;
import jcog.Is;
import jcog.Util;
import nars.NAL;
import nars.truth.AbstractMutableTruth;
import nars.truth.Truth;
import org.jetbrains.annotations.Nullable;

import static java.lang.Math.abs;
import static jcog.Fuzzy.and;
import static jcog.Fuzzy.or;
import static jcog.Util.assertUnitized;
import static nars.$.tt;
import static nars.NAL.HORIZON;
import static nars.Op.ETERNAL;
import static nars.truth.func.TruthFunctions2.confComposeFactor;
import static nars.truth.func.TruthFunctions2.confComposeFactorWeak;

/**
 * All truth-value (and desire-value) functions used in logic rules
 */
public enum TruthFunctions {
    ;

    /**
     * commutative
     * {<M --> P>, <S --> M>} |- <S --> P>
     * {<S --> M>, <M --> P>} |- <P --> S>
     * https://groups.google.com/g/open-nars/c/ILfG8OFVxN8/m/33toHH4rrxYJ?pli=1
     */
    public static @Nullable Truth deduction(Truth x, Truth y, boolean strong, float minConf) {
        double cxy = confCompose(x, y);
        if (cxy < minConf) return null;

        double fxy = Fuzzy.intersect(x.freq(), y.freq());

        double c = and(fxy, cxy);
        if (c < minConf) return null;

        if (!strong) {
            c = weak(c); if (c < minConf) return null;
        }

        return c < minConf ? null : tt(fDoubt(fxy), c);
    }



    @Deprecated
    private static double fDoubt(double x) {
        return NAL.truthFn.OPEN_WORLD ? Util.lerpSafe(x, 0.5f, 1) : x;
    }

    /**
     * {<S ==> M>, <M <=> P>} |- <S ==> P>
     * <p>
     * stronger than deduction such that A's frequency does not reduce the output confidence
     */
    public static @Nullable Truth analogy(Truth a, float bf, double bc, float minConf) {
        double c = and(confCompose(a, bc), bf);
        return c < minConf ? null : tt(fDoubt(and(a.freq(), bf)), c);
    }


    /**
     * {<S ==> M>, <P ==> M>} |- <S ==> P>
     *
     * @param a Truth value of the first premise
     * @param b Truth value of the second premise
     * @return Truth value of the conclusion, or null if either truth is analytic already
     */
    public static @Nullable Truth induction(Truth a, Truth b, double minConf) {
        double c = confComposeFactorWeak(a, b, b.freq());
        return c < minConf ? null : tt(a.freq(), c);
    }

    public static @Nullable Truth comparison(Truth a, Truth b, float minConf) {
        double fA = a.freq(), fB = b.freq();

        double orAB = or(fA, fB);
        double c = weak(confCompose(orAB, confCompose(a, b)));
        if (c < minConf) return null;

        double f = (orAB < NAL.truth.FREQ_EPSILON) ? 0 : (and(fA, fB) / orAB);
        return tt(f, c);
    }

    /**
     * {<S <=> M>, <M <=> P>} |- <S <=> P>
     *
     * @param v1 Truth value of the first premise
     * @param v2 Truth value of the second premise
     * @return Truth value of the conclusion
     * <p>
     * like intersection, but confidence is factored by or(f1,f2)
     */
    static @Nullable Truth resemblance(Truth v1, Truth v2, float minConf) {
        float f1 = v1.freq(), f2 = v2.freq();
        double c = confComposeFactor(v1, v2, or((double) f1, f2));
        return c < minConf ? null : tt(and(f1, f2), c);
    }


    /**
     * TODO return double
     */
    public static double confCompose(Truth a, double b) {
        return confCompose(a.conf(), b);
    }

    /**
     * TODO return double
     */
    public static double confCompose(Truth a, Truth b) {
        return confCompose(a.conf(), b.conf());
    }
//    public static double confRevise(Truth a, Truth b) {
//        return eviReviseToConf(a.evi(), b.evi());
//    }

    /**
     * TODO return double
     */
    public static double confCompose(double cx, double cy) {
        return NAL.truthFn.CONF_COMPOSITION.valueOf(cx, cy);
    }

//    private static double confRevise(double cx, double cy) {
//        //return confCompose(cx, cy);
//        return eviReviseToConf(c2e(cx),c2e(cy));
//    }
//    private static double eviReviseToConf(double ex, double ey) {
//        return e2c(ex+ey);
//    }




    /**
     * {<M --> S>, <M <-> P>} |- <M --> (S&P)>
     *
     * @param x Truth value of the first premise
     * @param y Truth value of the second premise
     * @return Truth value of the conclusion
     * <p>
     * In the confidence functions, each case for the conclusion to reach its
     * maximum is separately considered. The plus operator is used in place of an
     * or operator, because the two cases involved are mutually exclusive, rather
     * than independent of each other.
     * <p>
     * Fint : Intersection
     * f = and(f1, f2)
     * c = or(and(not(f1), c1), and(not(f2), c2)) + and(f1, c1, f2, c2)
     * Funi : Union
     * f = or(f1, f2)
     * c = or(and(f1, c1), and(f2, c2)) + and(not(f1), c1, not(f2), c2)
     * Fdif : Difference
     * f = and(f1, not(f2))
     * c = or(and(not(f1), c1), and(f2, c2)) + and(f1, c1, not(f2), c2)
     * <p>
     */
    public static @Nullable Truth intersect(Truth x, boolean negX, Truth y, boolean negY, double minConf) {
        double c = confCompose(x, y); if (c < minConf) return null;

        //        if (NAL.truth.INTERSECTION_FADE_NONPOLAR) {
//            //EXPERIMENTAL - TODO needs modification in the divide function
//            f = Util.lerpSafe(pxy, 0.5, f);
//        }

        return tt(intersectFreq(x, negX, y, negY), c);
    }

    private static double intersectFreq(Truth x, boolean negX, Truth y, boolean negY) {
        return Fuzzy.intersect(negIf(x.freq(), negX), negIf(y.freq(), negY));
    }

    private static double negIf(double f, boolean neg) {
        return neg ? (1 - f) : f;
    }


    /**
     * original OpenNARS desire function
     */
    @Deprecated
    public static Truth desireClassic(Truth a, Truth b, float minConf, boolean strong) {
        float f1 = a.freq();
        float f2 = b.freq();
        float f = and(f1, f2);
        double c12 = confCompose(a, b);
        double c = and(c12, f2) * (strong ? 1 : weak(1));
        return c < minConf ? null : tt(f, c);
    }


    /**
     * conf -> evidence
     */
    public static double c2e(double c) {
        return c2e(c, HORIZON);
    }


    /**
     * conf -> evidence
     * <p>
     * http://www.wolframalpha.com/input/?i=x%2F(1-x)
     *
     * @param c confidence, in [0, 1)
     * @return The corresponding weight of evidence, a non-negative real number
     */
    public static double c2e(double c, double horizon) {
        return horizon * c / (1 - c);
    }

    /**
     * evidence -> conf
     * http://www.wolframalpha.com/input/?i=x%2F(x%2B1)
     */
    public static double e2c(double e, double horizon) {
        return e / (e + horizon);
    }

    /**
     * evidence -> conf
     */
    public static double e2c(double e) {
        return e2c(e, HORIZON);
    }


    public static float originality(int stampLen) {
        return stampLen <= 1 ?
                1 : 1 / (1 + ((float)(stampLen - 1)) / (NAL.STAMP_CAPACITY - 1));
    }

    /** a scalar decomposition of a truth value, in the range [0..1] involving both frequency and confidence */
    public static double expectation(float frequency, double confidence) {
        //return confidence * (frequency - 0.5) + 0.5;
        return Util.fma(confidence, frequency - 0.5, 0.5);
    }

    public static double eternalize(double evi) {
        return weak(evi);
    }

    /**
     * point-segment-segment relative projection formula
     * TODO test
     * @param ete eternalization
     */
    public static double projectMid(long now, long fs, long fe, long ts, long te, float dur, float ete) {
        if (fs == ETERNAL)
            return 1;
        assert (ts != ETERNAL);

        /* source surrounds or equals target */
        if (fs <= ts && fe >= te)
            return 1;

        assertUnitized(ete);

        //mid->mid
        //TODO minshift?
        double p = project((fs+fe)/2, (ts+te)/2, now, dur);

        double dilution = rangeDilution(fs, fe, ts, te);

        return (1 - p * (1-ete)) * dilution;
    }

    /**
     * point-segment-segment relative projection formula
     * TODO test
     */
    public static double projectTrapezoidal(long now, long fs, long fe, long ts, long te, float dur, double ete, int n) {
        if (fs == ETERNAL || (fs==ts && fe==te) || ete>=1) return 1;
        assert (ts != ETERNAL);

        assert(n>=2);
        double nmin1 = n - 1;
        double df = (fe - fs) / nmin1;
        double dt = (te - ts) / nmin1;
        double xx = 0;

        for (int i = 0; i < n; i++) {
            long t = ts + (i > 0 ? (i < n - 1 ? Math.round(dt * i) : te) : 0);
            double x = ((t >= fs && t <= fe) ? 0 :
                project(fs + (i > 0 ? (i < n - 1 ? Math.round(df*i) : fe) : 0), t, now, dur));
            xx += x;
        }

        double xMean = xx / n;
        double z =
            /* 1 - (1 - ete) * xMean; */
            Util.fma(ete-1, xMean, +1);

        //assertUnitized(z);

        double dilution = rangeDilution(fs, fe, ts, te);

        return z * dilution;

//2ary: TODO unit test that the above is equivalent:
//        double ps = (ts >= fs && ts <= fe) ? 0 : project(fs, ts, now, dur);
//        double pe = (te >= fs && te <= fe) ? 0 : project(fe, te, now, dur);
//        return 1 - mean(ps, pe)*(1-ete);
    }

    private static double rangeDilution(long fs, long fe, long ts, long te) {
        return Math.min(1, ((double) (1 + fe - fs)) / (1 + te - ts));
    }

    /** "triangular" temporal projection discount */
    public static double project(long from, long to, long now, float dur) {
        if (from == to) {
            return 0;
        } else {
            float y = dur +
                    (NAL.revision.RELATIVE_PROJECTION_MODE_CLASSIC_OR_ISOSCELES ?
                        abs(now - from) + abs(now - to)          //classic
                        :
                        Math.max(abs(now - from), abs(now - to))*2 //isosceles
                    );
            //classic
            //isosceles
            return ((double) abs(from-to)) / Math.max((float) 1, y);
        }
    }


//    /**
//     * point-segment-segment relative projection formula
//     * TODO test
//     */
//    public static double project0(long now, long fs, long fe, long ts, long te, float dur, double memoryFactor) {
//        if (fs == ETERNAL) return 1;
//
//        assert (ts != ETERNAL);
//
//        /* source surrounds or equals target */
//        if (fs <= ts && fe >= te)
//            return 1;
//
//        //long ft = LongInterval.minTimeShiftTo(fs, fe, ts, te);
//        //long ft = abs( (fs + fe)/2 - (ts + te)/2 ); //midpoint delta
//        long ft;
//                //(minTimeToRaw(ts, fs, fe) + minTimeToRaw(te, fs, fe))/2.0;
//                //min(minTimeToRaw(ts, fs, fe), minTimeToRaw(te, fs, fe));
//        //if (LongInterval.containsRaw(ts, te, fs, fe)) { // target surrounds source
//            ft = abs( ((fs+fe)/2 - (ts+te)/2)); //midpoint difference
////        } else {
////            ft = min(abs(ts - fs), abs(te - fe));
////        }
//
//        double factor;
//        if (ft == 0) {
//            factor = 1;
//        } else {
//            //TODO better precision (0.5 cycles)?
//            //long F = (fs + fe) / 2L, T = (ts + te) / 2L;
//            //factor = 1 - unitizeSafe(ft / ((abs(now - F) + abs(now - T)) * memoryFactor));
//            //final double nf = meanTimeToRaw(now, fs, fe), nt = meanTimeToRaw(now, ts, te);
//            final long nf = maxTimeToRaw(now, fs, fe), nt = maxTimeToRaw(now, ts, te);
//
//            factor = 1 - unitizeSafe(
//                ft / ((nf + nt + dur) * memoryFactor));
//        }
//
//
//        /* dilution factor: temporal range stretch, when target range is larger */
//        final long fes = fe - fs;
//        final long tes = te - ts;
//        double dilution =
//                (fes>=tes) ? 1 : fes / ((double) tes);
//                //1;
//
//
//        return factor * dilution;
//    }


//    /**
//     * 3-point relative projection formula
//    */
//    public static double project(long now, long f, long t, double memoryFactor) {
//
//
//        //long ft = LongInterval.minTimeShiftTo(fs, fe, ts, te);
//        //long ft = abs( (fs + fe)/2 - (ts + te)/2 ); //midpoint delta
//        long ft =
//                //(minTimeToRaw(ts, fs, fe) + minTimeToRaw(te, fs, fe))/2.0;
//                abs(f-t);
//
//        double factor;
//        if (ft == 0) {
//            factor = 1;
//        } else {
//            final long nf = abs(now - f), nt = abs(now - t);
//
//            final double p = ft / ((nf + nt) * memoryFactor);
//
//            factor = 1 - unitizeSafe(p);
//        }
//
//        return factor;
//
//    }

    public static @Nullable Truth neg(@Nullable Truth t) {
        return t instanceof AbstractMutableTruth T ? T.negThis() : t == null ? null : t.neg();
    }

    //    /**
//     * {<M ==> S>, <M ==> P>} |- <S <=> P>
//     *
//     * @param a Truth value of the first premise
//     * @param b Truth value of the second premise
//     * @return Truth value of the conclusion
//     */
//    @Nullable
//    private static Truth comparison(Truth a, boolean negA, Truth b, float minConf) {
//        float cc = TruthFunctions.confCompose(a, b);
//        if (cc < minConf) return null;
//
//        float f1 = a.freq();
//        if (negA) f1 = 1 - f1;
//
//        float f2 = b.freq();
//
//
//        float f0 =
//                //or(f1, f2);
//                Math.max(and(f1, f2), and(1 - f1, 1 - f2));
//        float c = w2cSafe(and(f0, cc));
//        if (!(c >= minConf))
//            return null;
//
//        //float f = (Util.equals(f0, 0, NAL.truth.TRUTH_EPSILON)) ? 0 : (and(f1, f2) / f0);
//        return tt(f0, c);
//    }

    /**
     * {(A ==> B)} |- (--B ==> --A)
     *
     * @param t Truth value of the premise
     * @return Truth value of the conclusion
     */
    @Is("Contraposition") public static Truth contraposition(Truth t, double minConf) {
        double c = weak(t.freq() * t.conf());
        return c < minConf ? null : tt(0, c);
    }

    /**
     * {<A ==> B>} |- <B ==> A>
     *
     * @param t Truth value of the premise
     * @return Truth value of the conclusion
     */
    public static Truth conversion(Truth t, float minConf) {
        double c = weak(t.freq() * t.conf());
        return c < minConf ? null : tt(1, c);
    }



    @Nullable public static Truth negIf(@Nullable Truth x, boolean neg) {
        return x == null || !neg ? x : neg(x);
    }

    public static double weak(double c) {
        return e2c(c);
        //return c * w2cSafe(1);
    }

    @Nullable @Deprecated
    static Truth weak(@Nullable Truth t, float confMin) {
        if (t == null) return null;

        double c = weak(t.conf());
        return c < confMin ? null :
            (t instanceof AbstractMutableTruth T ? T.conf(c) : tt(t.freq(), c));
    }

    public static Truth decompose(Truth xy, Truth x, float confMin) {
        double f = ((double)xy.freq()) * x.freq();
        double conf = confComposeFactor(xy, x, f);
        return conf >= confMin ? tt(f, conf) : null;
    }

    public static Truth weak(Truth t) {
        var oldEvi = t.evi();
        var newEvi = c2e(weak(e2c(oldEvi)));
        double eviScale = newEvi/oldEvi;
        return t.cloneEviMult(eviScale, 0);
    }
}