package nars.truth.func;

import jcog.Fuzzy;
import jcog.Is;
import jcog.TODO;
import jcog.Util;
import nars.NAL;
import nars.truth.Truth;
import org.jetbrains.annotations.Nullable;

import static jcog.Fuzzy.*;
import static jcog.Util.lerpSafe;
import static nars.$.tt;
import static nars.NAL.truth.FREQ_EPSILON;
import static nars.NAL.truth.FREQ_EPSILON_half;
import static nars.truth.func.TruthFunctions.*;

public enum TruthFunctions2 {
    ;

    /** equality / frequency alignment function - determines how to compare frequencies for alignment, match, correlation, etc... */
    private static double alignment(float x, float y) {
        return NAL.truth.ALIGNMENT_ABSOLUTE ?
            Fuzzy.equals(x, y) :
            Fuzzy.alignment(x, y, FREQ_EPSILON/2);
            //Fuzzy.linearEqualsNormalized(x, y);
            //Math.pow(Fuzzy.equals(x, y), NAL.GRIP/*>=1*/);
            //Fuzzy.xnr(x, y);
            //Fuzzy.eqNorm(x, y);
            //Fuzzy.xnrNorm(x, y);
            //Fuzzy.equalsPolar(x, y, 4);
    }

    /**
     *   X, (  X ==> Y) |- Y
     * --X, (--X ==> Y) |- Y
     * frequency determined by impl
     */
    public static Truth pre(Truth X, Truth XimplY, boolean strong, float confMin) {

        double alignment = alignment(1, X.freq()), xy = XimplY.freq();

        double f, c;

        f = xy; c = alignment; //direct

        //f = xy; c = alignment * alignment; //direct with quadratic doubt

        //f = lerpSafe(x, 1-xy, xy); c = x; //crossover (presumptuous)
        //f = lerpSafe(x, 0.5, xy); c = _c * x; //fade to maybe + doubt
        //f = lerpSafe(x, 0.5, xy); c = _c; //fade to maybe
//            f = lerpSafe(x, 1-xy, xy); c = _c * x; //crossover, with doubt
//            f = lerpSafe(x, 0.5, xy); c = _c;
//            f = lerpSafe(x, 1-xy, xy); c = _c * polarity(x); //crossover

        c = confComposeLinear(X, XimplY, c, !strong); if (c < confMin) return null;

        return tt(f, c);
    }


    /**
     *   Y. , (X ==>   Y). |- X.
     * --Y. , (X ==> --Y). |- X.
     */
    public static Truth post(Truth Y, Truth XimplY, boolean beliefOrGoal, boolean strong, float confMin) {

        var y = Y.freq();
        var xy = XimplY.freq();

        var alignment = alignment(y, xy);

        double f, c;
        if (beliefOrGoal) {

            f = 1; c = alignment; //full-wave

            //f = 1; c = alignment*alignment; //full-wave quadratic

            //f = 0.5f + alignment/2; c = alignment; //fade to maybe with doubt

            //f = 0.5f + alignment/2; c = alignment*alignment; //fade to maybe with doubt (quadratic)

            //f = alignment; c = alignment; //bipolar with doubt


            //f = Util.lerp(alignment, 0.5, 1); c *= alignment; //half-wave: smooth bipolar with doubt

        } else {
            //TODO abstract these modes to enum parameter
            //f = 1; c = alignment; //full-wave

            //f = Util.sigmoid(polarize(alignment), 8); c = polarity(alignment);  //absolute_analog squashed

            f = Util.sigmoid(polarize(alignment), 8); c = 4 * Util.sqr(alignment - 0.5);  //quadratic_absolute_analog squashed

            //f = alignment; c = polarity(alignment); //Absolute Analog; neutral alignment -> inaction

            //f = alignment; c = 4 * Util.sqr(alignment - 0.5);  //quadratic_absolute_analog: Scaled parabola, peaks at 0 and 1, zero at 0.5

            //f = alignment; c = 0.5 + 2 * Util.sqr(alignment - 0.5);  //semiquadratic_absolute_analog: Scaled parabola, peaks at 0 and 1, 1/2 at 0.5


            //f = alignment; c = alignment; //crossfade

//            f = alignment; c = alignment >= 0.5 ?  //'reflecting dilemma': quadratic_asymmetric_analog: Scaled parabola, peaks at 0 and 1, zero at 0.5
//                   4 * Util.sqr(alignment - 0.5) : // Positive polarity: full range
//                   2 * Util.sqr(alignment - 0.5);  // Negative polarity: half range

              //f = Math.sqrt(alignment); c = Math.sqrt(polarity(alignment)); //Absolute Analog Sqrt

              //f = 0.5f + alignment/2; c = alignment; //crossfade maybe

              //f = 1; c = alignment*alignment; //full-wave quadratic

              //f = alignment; c = 1; //full preference: freq only

              //f = alignment; c = 0.5f + alignment/2; //Absolute Analog + half doubt

//            //Smooth Binary via squashing function, with confidence drop
//            f = PostGoalFn_SquashFactor!=0 ? fSquash(alignment, PostGoalFn_SquashFactor) : alignment;
//            c = lerpSafe(polarity(alignment),
//                //1/4f //paraboloid?
//                1/8f
//                //0
//                //Util.PHI_min_1f
//                //1/2f
//                //1/3f
//                //1/10f
//                //1/100f
//                //TRUTH_EPSILON
//            , 1);
//
//            //Smooth Binary via squashing function
//            //f = fSquash(alignment, NAL.truth.PostGoalFn_SquashFactor); c = polarity(alignment);
//
//
//            //Fading Absolute Analog; neutral alignment -> semi-inaction (positive noise floor)
////            f = alignment; c = Util.lerpSafe(polarity(alignment),
////                    1/4f
////                    //1/2f
////                    //NAL.truth.TRUTH_EPSILON_sqrt
////                , 1);
//
//
//            //Binary - not susceptible to 'vanishing polarity'
//            //f = alignment >= 0.5f ? 1 : 0; c = polarity(alignment);
//
//            //f = alignment; c = polarity(xy);
//            //f = alignment; c = (polarity(xy) + 1 /* exploration 'benefit of the doubt' */ )/2;
//
//            //f = alignment; c = min(polarity(y), polarity(xy));
//            //f = alignment; c = max(polarity(y), polarity(xy));
//            //f = alignment; c = polarity(y) * polarity(xy);
//
//            //f = alignment; c = polarity(alignment); //absolute; neutral alignment -> inaction
        }


        c = confComposeLinear(Y, XimplY, c, !strong); if (c < confMin) return null;

        return tt(f, c);
    }



//    private static double freqExpandTanh(double f, float contrast, float margin) {
//        return Fuzzy.unpolarize(
//                Math.tanh(Fuzzy.polarize(f) * contrast) * (1.0 + margin)
//        );
//    }

//    private static double freqExpandSigmoid(double f, float contrast) {
//        return Util.unitizeSafe(Util.sigmoid(Fuzzy.polarize(f) * contrast));
//    }


//    public static @Nullable Truth divideSym(Truth XY, Truth X, float confMin) {
//        if (XY.NEGATIVE())
//            return neg(divide(XY.neg(), X.neg(), confMin));
//        else
//            return divide(XY, X, confMin);
//    }

    public static @Nullable Truth divide(Truth XY, Truth X, double confMin) {
        return divide/*Sym*/(XY, X,
                true,
            //false,
                true,
            confMin);
    }

//    /** untested */
//    public static @Nullable Truth divideSym(Truth XY, Truth X, boolean overflowDoubt, boolean underflowDoubt, float confMin) {
//        if (XY.NEGATIVE()) {
//            XY = XY.neg();
//            var y = divide(XY, X.neg(), overflowDoubt, underflowDoubt, confMin);
//            if (y == null)
//                return null;
//
//            return y.negThis();
//        } else
//            return divide(XY, X, overflowDoubt, underflowDoubt, confMin);
//    }

    public static @Nullable Truth divide(Truth XY, Truth X, boolean overflowDoubt, boolean underflowDoubt, double confMin) {
        var x = X.freq(); if (x < FREQ_EPSILON) /* DIVIDE BY ZERO */ return null;
        var xy = XY.freq();

        double cFactor = 1;


        //if (...) c *= xy;

        var y = Fuzzy.divide(xy, x);
        if (y > 1) {
            if (overflowDoubt)
                cFactor /= y;
            y = 1;
        } else if (y < FREQ_EPSILON) {
            if (underflowDoubt)
                cFactor *= x;
            y = 0;
        }
        var c = confComposeFactor(XY, X, cFactor); if (c < confMin) return null;
        return tt(y, c);
    }

//    /**
//     * EXPERIMENTAL
//     * variation of the original union truth function that
//     * decreases confidence in proportion to the information lost
//     * by one component's higher frequency masking the other
//     * component's lower frequency.  if the components
//     * have equal frequencies then no loss is involved.
//     */
//    public static Truth unionFair(Truth t, Truth b, float confMin) {
//        double c = confCompose(t, b); if (c < confMin) return null;
//
//        float tf = t.freq(), bf = b.freq();
//        float f = Fuzzy.or(tf, bf);
//        if (f < TRUTH_EPSILON)
//            return null;
//
//        float loss = abs((f - tf) - (f - bf));
//        float lossFraction = loss / f;
//        c *= 1 - lossFraction;
//        return c < confMin ? null : tt(f, c);
//    }


    /**
     * commutative
     * {<M --> S>, <P --> M>} |- <S --> P>
     * {<M ==> S>, <P ==> M>} |- <S ==> P>
     */
    static Truth exemplification(Truth a, Truth b, boolean weak, float confMin) {
        var af = a.freq();
        if (af < FREQ_EPSILON) return null;
        var bf = b.freq();
        if (bf < FREQ_EPSILON) return null;

        var abf = ((double)af) * bf;
        var c =
            confComposeLinear(a, b, abf, weak);
            //confComposeFactor(a, b, abf, weak);

        return c < confMin ? null : tt(1, c);
    }

    /**
     * commutative
     * a 'half-way' exemplification that zeros at freq=0.5
     * @param ap 'a' positive-only filter
     * @param bp 'b' positive-only filter
     */
    static Truth suppose(Truth a, boolean ap, Truth b, boolean bp, boolean weak, float confMin) {

        var af = a.freq();
        if (ap) {
            af = polarize(af);
            if (af <= FREQ_EPSILON_half) return null;
        }
        var bf = b.freq();
        if (bp) {
            bf = polarize(bf);
            if (bf <= FREQ_EPSILON_half) return null;
        }

        var c = confComposeLinear(a, b, af * bf, weak);
        return c < confMin ? null : tt(1, c);
    }

    @Nullable
    public static Truth mix2(Truth t, Truth b, boolean weak, float confMin) {
        return mix2(t, b, NAL.truthFn.MIX_EVI_OR_CONF, weak, confMin);
    }

    /**
     * like mixing two frequencies of light or sound to produce a 3rd that is a compromise of both
     */
    @Nullable
    public static Truth mix2(Truth t, Truth b, boolean eviOrConf, boolean weak, float confMin) {
        float tf = t.freq(), bf = b.freq();

        var alignment = alignment(tf, bf);

        var c = confComposeLinear(t, b, alignment, weak); if (c < confMin) return null;

        double T, B;
        if (eviOrConf) { T = t.evi();  B = b.evi();  }
        else           { T = t.conf(); B = b.conf(); }

        var tb = T / (T + B);
        var f = lerpSafe(tb, bf, tf); //linear interpolate weighed by conf

        return tt(f, c);
    }

    @Deprecated private static double confComposeLinear(Truth x, Truth y, double f, boolean weak) {
        if (!NAL.truthFn.CONF_COMPOSE_LINEAR)
            return confCompose(x, y, f, weak);

        var xy = TruthFunctions.confCompose(x, y);

        if (weak) xy = weak(xy);

        return f == 1 ? xy : e2c(c2e(xy) * f);
    }

    private static double confCompose(Truth x, Truth y, double f, boolean weak) {
        var xy = TruthFunctions.confCompose(x, y);

        if (weak) xy = weak(xy);

        return
//          NAL.truthFn.COMPOSE_CONF_CLASSIC ?
            xy * f;
//            :
//            f == 1 ? xy : e2c(c2e(xy) * f);
    }

    public static double confComposeFactor(Truth x, Truth y, double f) {
        return confCompose(x, y, f, false);
    }

    public static double confComposeFactorWeak(Truth x, Truth y, double f) {
        return confCompose(x, y, f, true);
    }

    /** propagates frequency, attenuates conf */
    public static Truth conduct(Truth dir, Truth mag, boolean weak, float confMin) {
        var mf = mag.freq();
        var c = confComposeLinear(mag, dir, mf, weak);
        return c < confMin ? null :
                tt(dir.freq(), c);
    }

    /**
     * SuperConduct - Analogous to a transistor circuit where:
     *    - 'dir' is the incoming signal (input at the base/emitter),
     *    - 'mag' is the gate voltage controlling signal flow (akin to the gate in a FET or base bias in a BJT),
     *    - full/weak flags adjust transistor operation mode,
     *    - and the output signal's confidence can decay over propagation,
     *      similar to signal loss over distance or through imperfect amplification.
     *
     * The parameter {@code full} determines how the nature of the 'dir' signal
     * (unipolar vs. bipolar) influences the output confidence:
     * - When {@code full} is true, the method treats 'dir' as a bipolar signal,
     *   meaning it considers both positive and negative components of the frequency
     *   (similar to a bipolar junction transistor (BJT) that can handle both
     *   positive and negative swings more symmetrically).
     * - When {@code full} is false, it treats 'dir' as a unipolar signal,
     *   using only its magnitude in one direction, which might be likened
     *   to a unipolar field-effect transistor (FET) operation that only responds
     *   to one polarity of voltage at the gate or source.
     *
     * As in real-world electronics, different transistor types or configurations
     * (bipolar vs. unipolar) respond differently to input signals, affecting
     * amplification and propagation characteristics. Here, {@code full} chooses
     * the mode of operation, influencing how strongly 'dir' impacts the confidence.
     *
     * @param dir  The incoming signal's truth, analogous to the input at a transistor's base.
     * @param mag  The gate voltage or bias controlling the transistor,
     *             affecting how 'dir' is allowed to pass or be amplified.
     * @param full When true, treat 'dir' as a bipolar signal (both positive and negative
     *             frequency components influence confidence), similar to a BJT's behavior.
     *             When false, treat 'dir' as a unipolar signal, analogous to a unipolar FET's response.
     * @param weak A flag that mimics lower bias conditions or weaker input
     *             scenarios, affecting the transistor's response.
     * @param confMin The minimum confidence threshold for a successful output,
     *                akin to a cutoff below which the transistor cannot sustain meaningful amplification.
     *
     * @return A new {@code Truth} representing the propagated signal if the confidence
     *         exceeds {@code confMin}, otherwise {@code null}.
     */
    public static Truth superConduct(Truth dir, Truth mag, boolean full, boolean weak, float confMin) {
        // Signal (dir) and Gate (mag) frequencies
        float df = dir.freq(), mf = mag.freq();

        /*
         * Determine how 'dir' contributes to the confidence based on the 'full' flag.
         * - In full mode (bipolar operation), use the polarity-adjusted frequency to consider
         *   both positive and negative influences, akin to a BJT handling bipolar signals.
         * - In non-full mode (unipolar operation), use the raw frequency of 'dir',
         *   similar to a unipolar FET reacting to a single polarity.
         */
        var c = confComposeLinear(mag, dir, (full ? polarity(df) : df) * mf, weak);

        // If the confidence (signal strength) falls below the threshold,
        // the transistor's output is effectively cut off.
        if (c < confMin) return null;

        /*
         * Polarize the directional frequency, shaping the input signal based on gate control.
         * The sigmoid function models the transistor's threshold behavior— a smooth yet sharp
         * transition from off to on, mirroring transistor switching characteristics.
         *
         * Applies sharpness parameter to determine how abruptly the transistor transitions.
         */
        var ss = NAL.truthFn.SUPERCONDUCT_SHARPNESS;
        var F = ss>0 ? Util.sigmoid(polarize(df), ss) : df;

        return tt(F, c);
    }

    /**
     * soft conduct/semi-conduct
     * non-inverting analogy / bi-polarized deduction
     */
    @Deprecated public static Truth biduct(Truth dir, Truth mag, boolean weak, float confMin) {
        float mf = mag.freq(), df = dir.freq();
        var c = confCompose(mag, dir, mf, weak);
        return c < confMin ? null :
                tt(lerpSafe(mf, 0.5, df), c);
    }

    @Is("Ising_model") public static Truth polarduct(Truth dir, Truth mag, boolean weak, float confMin) {
        double df = dir.freq(), mf = mag.freq();
        var c = confCompose(mag, dir, polarity(mf), weak);
        if (c < confMin) return null;

        var f =
            lerpSafe(mf, 1 - df, df) //soft
            //mf >= 0.5 ? df : 1 - df //hard
        ;

        return tt(f, c);
    }


//    /**
//     * new bipolar analogy
//     */
//    @Nullable
//    public static Truth analogyBipolar(Truth x, Truth a, float confMin) {
//        double af = a.freq();
//
//        double c = confCompose(x, a) * polarity(af);
//        if (c < confMin) return null;
//
//        double xf = x.freq();
//        double F = lerpSafe(af, 1 - xf, xf);
//        return tt(F, c);
//    }

    public static double confReduce(Truth t) {
        return TruthFunctions.confCompose(t, NAL.truthFn.GULLIBILITY);
        //return confCompose(t, (float) t.conf()); //too strict
    }

    public static @Nullable Truth structuralReduction(Truth T, float confMin) {
        var c = confReduce(T);
        return c < confMin ? null : tt(T.freq(), c);
    }

    @Nullable
    static Truth deductionSym(Truth T, Truth B, boolean strong, float confMin) {
        double tf = T.freq(), bf = B.freq();
        var neg = (1 - tf) * (1 - bf) > tf * bf;
        if (neg) {
            T = T.neg();
            B = B.neg();
        }
        return negIf(deduction(T, B, strong, confMin), neg);
    }

    public static Truth diff(Truth a, Truth b, double confMin) {
        throw new TODO();
//        double c =
//            TruthFunctions.confCompose(a, b);
//            //confComposeFactor(a, b, polarity(sim));
//        return c < confMin ? null :
//            tt(diffCommutative(a.freq(), b.freq()), c);
    }

    /**
      a!, (a <~> b) |- b!

      b_goal(diff, a_goal) has the following properties:

            diff ~= 0:   //freq(a) < freq(b)
               b_goal = a_goal lerp to 0 proportionally to how close df is to 0

            diff ~= 0.5: //freq(a) ~= freq(b)
               b_goal = a_goal

            df ~= 1:     //freq(a) > freq(b)
               b_goal = a_goal lerp to 1 proportionally to how close df is to 1

        Attenuate confidence by ~Math.abs(df - 0.5) to propagate uncertainty
    */
    public static Truth diffGoal(Truth a_goal, Truth diff, double confMin) {
        var c = weak(TruthFunctions.confCompose(a_goal, diff));

        double df = diff.freq(), gf = a_goal.freq();

        var b_goal = lerpSafe(Math.abs(df - 0.5) * 2, gf, df < 0.5 ? 0 : 1);
        var cFactor = 1 - Math.abs(df - 0.5);
        
        return c < confMin ? null :
                tt(b_goal, c * cFactor);
    }

    public static double diffCommutative(float a, float b) {
        return unpolarize(a-b);
//        if (a >= b) {
//            return 0.5 + (a - b) / 2;
//        } else {
//            return 0.5 - (b - a) / 2;
//        }
    }


    /** result confidence (and optionally frequency)
     *  is proportional to input freq alignments */
    public static Truth sameness(Truth a, Truth b, boolean flat, boolean weak, double confMin) {
        var c = TruthFunctions.confCompose(a, b);
        if (c < confMin) return null;

        var f = alignment(a.freq(), b.freq());
        c *= f;

        if (weak) c = weak(c);

        if (c < confMin) return null;

        return tt(flat ? 1 : f, c);
    }

    static boolean symNegates(Truth T, Truth B) {
        return symNegates(T.freq(), B.freq());
    }

    private static boolean symNegates(float tf, float bf) {
        return tf*bf < (1-tf)*(1-bf);
    }

    /** TODO test */
    @Nullable static Truth xor(Truth T, Truth B, float confMin) {
        var c = TruthFunctions.confCompose(T, B);
        if (c < confMin) return null;

        float tf = T.freq(), bf = B.freq();
        //                f,c
        //tf+ & bf+ => bf 1,0
        //tf+ & bf- => bf 1,1
        //tf- & bf+ => bf 0,1
        //tf- & bf- => bf 0,0

        var diff =
            Fuzzy.xor(tf, bf);
        //1 - Fuzzy.xnrNorm(tf, bf);

        c *= diff;
        if (c < confMin) return null;

        double f = tf;
        //1;
        //Util.lerpSafe(diff, 0.5, tf);
        //Util.lerpSafe(polarity(tf)*polarity(bf), 0.5, tf);
        //Util.lerpSafe(polarity(bf), 0.5, tf);

        return tt(f, c);
    }

    public static float delta(float Sfreq, float Efreq) {
        return unpolarize(Efreq - Sfreq);
    }

    /** 'positive' quadrant intersection
     * @param x (sub-)condition
     * @param conj (super-)conditions
     * */
    @Nullable public static Truth must(Truth x, boolean xHalf, Truth conj, boolean conjHalf, boolean weak, float confMin) {
        var xf = x.freq();
        double XF = xHalf ? polarize(xf) : xf;
        if (XF < FREQ_EPSILON_half) return null;

        var cf = conj.freq();
        double CF = conjHalf ? polarize(cf) : cf;
        if (CF < FREQ_EPSILON_half) return null;

        double F, C;

        //mode hard strong: (exemplification-like)
        F = 1;  C = xf * cf;

        //mode soft:
        //F= unpolarize(XF * CF); C = F;

        //mode medium: pretends X component is 1, and instead it affects conf
        //F = unpolarize( CF); C = XF * F;

        //mode hard: (semi-exemplification-like)
        //F = 1;  C = XF * CF;

        var c = confCompose(x, conj, C, weak); if (c < confMin) return null;
        return tt(F, c);
    }

    @Nullable public static Truth subtract(Truth a, Truth b, float confMin) {
        float af = a.freq(), bf = b.freq();
        var df = af - bf;
        if (df <= 0) return null;

        var c =
            confComposeFactor(a, b, df);
        return c < confMin ? null :
            tt(1, c);
    }
}

//    /**
//     * freq symmetric intersection
//     * to the degree the freq is the same, the evidence is additive
//     * to the degree the freq is different, the evidence is multiplicative
//     * resulting freq is weighted combination of inputs
//     */
//    public static Truth intersectionX(Truth a, Truth b, float confMin) {
//        float diff = Math.abs(a.freq() - b.freq());
//        float ac = a.conf(), bc = b.conf();
//        float conf = Util.lerp(diff, w2cSafe(c2wSafe(ac) + c2wSafe(bc)), (ac * bc));
//        float freq = ((a.freq() * ac) + (b.freq() * bc)) / (ac + bc);
//        return conf >= confMin ? $.t(freq, conf) : null;
//    }

//    /**
//     * freq symmetric difference
//     * to the degree the freq differs or is similar, the evidence is additive
//     * to the degree the freq is not different nor similar, the evidence is multiplicative
//     * resulting freq is weighted difference of inputs
//     */
//    public static Truth differenceX(Truth a, Truth b, float confMin) {
//        float extreme = 2f * Math.abs(0.5f - Math.abs(a.freq() - b.freq()));
//        float ac = a.conf(), bc = b.conf();
//        float conf = Util.lerp(extreme, (ac * bc), w2cSafe(c2wSafe(ac) + c2wSafe(bc)));
//
//        float freq = a.freq() * (1f - b.freq());
//        return conf >= confMin ? $.t(freq, conf) : null;
//    }

//    public static Truth unionX(Truth a, Truth b, float confMin) {
//        Truth z = intersectionX(a.neg(), b.neg(), confMin);
//        return z != null ? z.neg() : null;
//    }

//    @Nullable
//    public static Truth deduction(Truth a, float bF, float bC, float confMin) {
//
//
//        float f = and(a.freq(), bF);
//
//
//        float aC = a.conf();
//
//        float c = Util.lerp(bC/(bC + aC), 0 ,aC);
//
//        return c >= confMin ? tt(f, c) : null;
//    }

//    @Nullable
//    public static Truth deduction(Truth a, float bF, float bC, float confMin) {
//
//        float f;
//        float aF = a.freq();
////        if (bF >= 0.5f) {
////            f = Util.lerp(bF, 0.5f, aF);
////        } else {
////            f = Util.lerp(bF, 0.5f, 1- aF);
////        }
//        f = Util.lerp(bF, 1-aF, aF);
//
//        float p = Math.abs(f - 0.5f)*2f; //polarization
//
//        float c = //and(/*f,*/ /*p,*/ a.conf(), bC);
//                    TruthFunctions.confCompose(a.conf(), bC);
//
//        return c >= confMin ? tt(f, c) : null;
//    }
//
//    /**
//     * {<S ==> M>, <P ==> M>} |- <S ==> P>
//     *
//     * @param a Truth value of the first premise
//     * @param b Truth value of the second premise
//     * @return Truth value of the conclusion, or null if either truth is analytic already
//     */
//    public static Truth induction(Truth a, Truth b, float confMin) {
//        float c = w2cSafe(a.conf() * b.freqTimesConf());
//        return c >= confMin ? $.t(a.freq(), c) : null;
//    }

//
//    /**
//     * frequency determined entirely by the desire component.
//     */
//    @Nullable
//    public static Truth desireNew(Truth goal, Truth belief, float confMin, boolean strong) {
//
//        float c = and(goal.conf(), belief.conf(), belief.freq());
//
//        if (!strong) {
//            //c *= TruthFunctions.w2cSafe(1.0f);
//            c = w2cSafe(c);
//        }
//
//        if (c >= confMin) {
//
//
//            float f = goal.freq();
//
//            return $.t(f, c);
//
//        } else {
//            return null;
//        }
//    }

//    /**
//     * experimental
//     */
//    public static @Nullable Truth educt(Truth goal, Truth belief, boolean strong, float confMin) {
//
//        double cc = confCompose(belief, goal);
//        if (cc < confMin) return null;
//
//
//        double bF = belief.freq();
//        float gF = goal.freq();
//
//        cc *= (bF);
//        if (cc < confMin) return null;
//        if (!strong) {
//            cc = weak(cc);
//            if (cc < confMin) return null;
//        }
//
//        double f =
//                lerpSafe(bF, 1 - gF, gF);
////                lerpSafe(bF, 0.5, gF);
//
//        return tt((float) f, cc);
//
//    }

//    public static @Nullable Truth demand(Truth goal, Truth condition, float confMin, boolean strong) {
//
//        double cc = confCompose(goal, condition);
//        if (cc < confMin) return null;
//        if (!strong) {
//            cc = weak(cc);
//            if (cc < confMin) return null;
//        }
//
//        double cF = condition.freq();
//        float gF = goal.freq();
//
//        float f =
//                gF;
////                cF >= 0.5f ? gF : 1-gF;
//        //gF >= 0.5f ?  +1 : 0;
//        //(float) cF;
//        //(float) a;
//        //1;
//
//        double a =
////                polarity(cF) * gF;
//                cF;
//        //cF * gF;
//        //cF * polarity(gF);
//        //gF;
//        //Util.lerpSafe((float) (cF * gF), 0.5f, 1);
//        //Math.sqrt(cF * gF);
//        //1;
//
//        cc *= a;
//        if (cc < confMin) return null;
//
//
//        return tt(f, cc);
//    }