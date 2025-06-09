package nars.truth.func;

import jcog.Is;
import jcog.util.Reflect;
import nars.$;
import nars.NAL;
import nars.Term;
import nars.truth.PreciseTruth;
import nars.truth.Truth;
import nars.truth.func.annotation.AllowOverlap;
import nars.truth.func.annotation.SinglePremise;
import org.jetbrains.annotations.Nullable;

import java.lang.reflect.Field;

import static nars.truth.func.TruthFunctions.*;
import static nars.truth.func.TruthFunctions2.*;

/**
 * NAL Truth Model
 * derivative of the original set of NAL truth functions, preserved as much as possible
 */
public enum NALTruth implements TruthFunction {

    @AllowOverlap @SinglePremise Dynamic() {
        @Override
        public @Nullable Truth apply(@Nullable Truth task, @Nullable Truth belief, float confMin) {
            throw new UnsupportedOperationException();
        }

    },

    Deduction() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return deduction(T, B, true, confMin);
        }

//        @Override
//        public boolean apply(MutableTruth y, @Nullable Truth task, @Nullable Truth belief, float confMin) {
//            throw new TODO();
//            deduction(y, task, belief, true, confMin);
//            return y.is();
//        }

    },
//    DeductionSym() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return deductionSym(T,B,true,confMin);
//        }
//    },
//    DeductionSymWeak() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return deductionSym(T,B,false, confMin);
//        }
//    },

    Conduct() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return conduct(T, B, false, confMin);
        }
    },

    ConductWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return conduct(T, B, true, confMin);
        }
    },

    SuperConduct() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return superConduct(T, B, true, false, confMin);
        }
    },
    SuperConductWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return superConduct(T, B, true, true, confMin);
        }
    },
    SemiConduct() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return superConduct(T, B, false, true, confMin);
        }
    },

//    /** HACK for occurrence time calculation */
//    @Deprecated NegConductWeak() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return neg(ConductWeak.apply(T, B, confMin));
//        }
//    },
//
//    SemiConduct() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return semiconduct(T, B, confMin);
//        }
//    },
//    SemiConductWeak() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return weak(SemiConduct.apply(T, B, confMin), confMin);
//        }
//    },
//
    /** TODO reverse args TODO rename */
    @Deprecated Biduct() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return biduct(B, T, false, confMin);
            //return SemiConduct.apply(B, T, confMin);
        }
    },
    @Deprecated BiductWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return biduct(B, T, true, confMin);
        }
    },
//
//    /** TODO reverse args TODO rename */
//    @Deprecated BiductWeaker() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//
//            return weak(BiductWeak.apply(T, B, confMin), confMin);
//        }
//    },

    DeductionWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return deduction(T, B, false, confMin);
        }

    },

//    @AllowOverlap DeductionRecursive() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return deduction(T, B, true, confMin);
//        }
//
//    },
//    @AllowOverlap DeductionWeakRecursive() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return TruthFunctions.deduction(T, B, false, confMin);
//        }
//
//        @Override
//        public boolean taskTruthValid(@Nullable Truth t) {
//            return nonZeroFreq(t);
//        }
//    },

    /** this is like sqrt(truth) .... try actual: f = sqrt(f0) */
    @AllowOverlap @SinglePremise StructuralDeduction() {
        @Override
        public Truth apply(Truth T, Truth ignored, float confMin) {
            var B = structuralComponent(T); if (B==null) return null;
            return Deduction.apply(T, B, confMin);

//            final double c = T!=null ? confReduce(T) : 0;
//            return c > confMin ? Deduction.apply(T, $.t(1, c), confMin) : null;
        }

        @Override
        public boolean beliefTruthSignificant() {
            return false;
        }
    },

    @AllowOverlap @SinglePremise StructuralIntersection() {
        @Override
        public Truth apply(Truth T, Truth ignored, float confMin) {
            var B = structuralComponent(T); if (B==null) return null;
            return Intersection.apply(T, B, confMin);
        }

        @Override
        public boolean beliefTruthSignificant() {
            return false;
        }
    },
    @AllowOverlap @SinglePremise StructuralExemplification/*Strong*/() {
        @Override
        public Truth apply(Truth T, Truth ignored, float confMin) {
            var B = structuralComponent(T); if (B==null) return null;
            return ExemplificationStrong.apply(T, B, confMin);
        }
        @Override
        public boolean beliefTruthSignificant() {
            return false;
        }
    },

    @AllowOverlap @SinglePremise StructuralExemplificationWeak() {
        @Override
        public Truth apply(Truth T, Truth ignored, float confMin) {
            var B = structuralComponent(T); if (B==null) return null;
            return Exemplification.apply(T, B, confMin);
        }
        @Override
        public boolean beliefTruthSignificant() {
            return false;
        }
    },

    /**
     * similar to structural deduction but keeps the same input frequency, only reducing confidence
     * TODO check if this is equivalent to a 'structural induction' if so rename it
     */
    @SinglePremise StructuralReduction() {
        @Override
        public Truth apply(Truth T, Truth Bignored, float confMin) {
            return structuralReduction(T, confMin);
        }

        @Override
        public boolean beliefTruthSignificant() {
            return false;
        }
    },

//    @SinglePremise SymmetricReduction() {
//        @Override
//        public Truth apply(Truth T, Truth Bignored, float confMin) {
//            double c = confReduce(T);
//            return c >= confMin ? $.tt((float) Math.sqrt(T.freq()), c) : null;
//        }
//
//        @Override
//        public boolean beliefTruthSignificant() {
//            return false;
//        }
//    },

    Induction() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return induction(T, B, confMin);
        }
        //TODO belief freq  non-zero
    },


    @Is("Sherlock_Holmes#Holmesian_deduction") Abduction() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return induction(B, T, confMin);
        }

    },


    /**
     * polarizes according to an implication belief.
     * here if the belief is negated, then both task and belief truths are
     * applied to the truth function negated.  but the resulting truth
     * is unaffected as it derives the subject of the implication.
     */
    AbductionSym() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            if (symNegates(T, B)) {
                T = T.neg();
                B = B.neg();
            }
            return Abduction.apply(T, B, confMin);
        }
    },
    InductionSym() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            if (symNegates(T, B)) {
                T = T.neg();
                B = B.neg();
            }
            return Induction.apply(T, B, confMin);
        }
    },


//    AbductionXOR() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return B.NEGATIVE() ? Abduction.apply(T, B.neg(), confMin) : Abduction.apply(T.neg(), B, confMin);
//        }
//    },

    Comparison() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return comparison(T, B, confMin);
        }
    },
//    ComparisonSymmetric() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return TruthFunctions2.comparisonSymmetric(T, B, confMin);
//        }
//    },

    Conversion() {
        @Override
        public boolean taskTruthSignificant() {
            return false;
        }

        @Override
        public Truth apply(@Nullable Truth T, Truth B, float confMin) {
            return conversion(B, confMin);
        }
    },

    Resemblance() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return resemblance(T, B, confMin);
            //return TruthFunctions2.resemblance(T, B, confMin);
        }
    },

    Contraposition() {
        @Override
        public boolean taskTruthSignificant() {
            return false;
        }

        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return contraposition(B, confMin);
            //return TruthFunctions2.contraposition2(B, confMin);
        }
    },

    Intersection() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return intersect(T, false, B, false, confMin);
        }
    },

    Union() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return intersect(T, true, B, true, confMin);
        }
    },

    Subtract() {
        @Override public Truth apply(Truth T, Truth B, float confMin) { return TruthFunctions2.subtract(T, B, confMin); }
    },

    Diff() {
        @Override public Truth apply(Truth T, Truth B, float confMin) { return diff(T, B, confMin); }
    },

    DiffGoal() {
        @Override public Truth apply(Truth T, Truth B, float confMin) { return diffGoal(T, B, confMin); }
    },

    Sameness() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return sameness(T, B, true, true, confMin);
        }
    },

    Polarduct() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return polarduct(T, B, false, confMin);
        }
    },

    PolarductWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return polarduct(T, B, true, confMin);
        }
    },

    //    IntersectionSym() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return B.NEGATIVE() ?
//                    Intersection.apply(T.neg(), B.neg(), confMin)
//                    :
//                    Intersection.apply(T, B, confMin);
//        }
//    },
    Pre() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return pre(T, B, true, confMin);
        }

    },
    PreWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return pre(T, B, false, confMin);
        }
    },

    @AllowOverlap PreRecursive() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return pre(T, B, true, confMin);
        }

    },

    Post() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return post(T, B, true, true, confMin);
        }
    },

    Need() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return post(T, B, false, true, confMin);
        }
    },

    PostWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return post(T, B, true, false, confMin);
        }
    },

//    Want() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return pre(T, B, false, true, confMin);
//        }
//
//    },
//    WantWeak() {
//        @Override
//        public @Nullable Truth apply(@Nullable Truth task, @Nullable Truth belief, float confMin) {
//            return weak(Want.apply(task, belief, confMin), confMin);
//        }
//    },

//    Need() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return post(T, B, true, true, confMin);
//        }
//    },
//    NeedWeak() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return weak(Need.apply(T,B,confMin), confMin);
//        }
//    },
//    NeedPolar() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            boolean neg;
//            if (T.POSITIVE() && !B.POSITIVE()) {
//                neg = true;
//                B = B.neg();
//            } else if (!T.POSITIVE() && B.POSITIVE()) {
//                neg = true;
//                T = T.neg();
//            } else
//                neg = false;
//            return negIf(TruthFunctions2.post(T, B, true, true, confMin), neg);
//        }
//    },

    Mix() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return mix2(T, B, false, confMin);
        }
    },
    MixWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return mix2(T, B, true, confMin);
        }
    },

//    MixDisj() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return TruthFunctions2.mix(T, B, false, confMin);
//        }
//    },

//    IntersectionPB() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return B.isNegative() ?
//                    negIfNonNull(Intersection.apply(T.neg(), B.neg(), confMin))
//                    :
//                    Intersection.apply(T, B, confMin);
//        }
//    },
//    UnionPB() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return B.isNegative() ?
//                    negIfNonNull(Union.apply(T.neg(), B.neg(), confMin))
//                    :
//                    Union.apply(T, B, confMin);
//        }
//    },


    Analogy() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return analogy(T, B.freq(), B.conf(), confMin);
            //return weak(TruthFunctions2.analogyBipolar(T, B, confMin), confMin);
        }
    },

//    ReduceConjunction() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return reduceConj(T, B, confMin);
//        }
//    },

//
//    AnonymousAnalogy() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return TruthFunctions.anonymousAnalogy(T, B, confMin);
//        }
//    },


//    BixemplificationStrong() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            boolean tNeg = T.NEGATIVE();
//            return negIf(TruthFunctions2.exemplification(T.negIf(tNeg), B, confMin), tNeg);
//        }
//
//    },

//    /** weak exemplification */
//    Could() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return weak(Conduct.apply(T, B, confMin), confMin);
//        }
//
//    },

    /** TODO rename to 'Exemplification' */
    ExemplificationStrong() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return exemplification(T, B, false, confMin);
        }
    },
    /** TODO rename to 'ExemplificationWeak' */
    Exemplification() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return exemplification(T, B, true, confMin);
        }
    },
//    /** TODO rename to 'ExemplificationWeaker' */
//    ExemplificationWeak() {
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return weak(Exemplification.apply(T, B, confMin), confMin);
//        }
//    },


    Presume() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return must(T, false, B, false, false, confMin);
        }
    },
    PresumeWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return must(T, false, B, false, true, confMin);
        }
    },

//    Must() {
//        @Override
//        public @Nullable Truth apply(@Nullable Truth T, @Nullable Truth conj, float confMin) {
//            return must(T, true, conj, true, false, confMin);
//        }
//    },
//    MustWeak() {
//        @Override
//        public @Nullable Truth apply(@Nullable Truth T, @Nullable Truth conj, float confMin) {
//            return must(T, true, conj, true, true, confMin);
//        }
//    },

    /** TODO rename to 'Suppose' */
    SupposeStrong() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return suppose(T, true, B, false, false, confMin);
        }
    },
    /** TODO rename to 'SupposeWeak' */
    Suppose() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return suppose(T, true, B, false, true, confMin);
        }
    },
//    DecomposeDiff() {
//        @Override
//        public Truth apply(final Truth T, final Truth B, float minConf, NAL n) {
//            return TruthFunctions2.decomposeDiff(T, B, minConf);
//        }
//    },

    Divide() {
        @Override
        public Truth apply(Truth XY, Truth X, float confMin) {
            return NAL.derive.DIVIDE_OR_DECOMPOSE ?
                divide(XY, X, confMin) :
                decompose(XY, X, confMin);
        }
    },

    DivideWeak() {
        @Override
        public Truth apply(Truth XY, Truth X, float confMin) {
            return weak(Divide.apply(XY, X, confMin), confMin);
        }
    },

//    Undivide() {
//        @Override
//        public Truth apply(Truth XY, Truth X, float confMin) {
//            return TruthFunctions2.undivide(XY, X, confMin);
//        }
//    },

    @SinglePremise
    Identity() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return TruthFunction.identity(T, confMin);
        }

    },


    BeliefIdentity() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return TruthFunction.identity(B, confMin);
        }

        @Override
        public boolean taskTruthSignificant() {
            return false;
        }

    },


    /**
     * maintains input frequency but reduces confidence
     */
    BeliefStructuralReduction() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return StructuralReduction.apply(B, null, confMin);
        }
        @Override
        public boolean taskTruthSignificant() {
            return false;
        }
    },
//    BeliefSymmetricReduction() {
//        @Override
//        public boolean taskTruthSignificant() {
//            return false;
//        }
//
//        @Override
//        public Truth apply(Truth T, Truth B, float confMin) {
//            return B == null ? null : SymmetricReduction.apply(B, null, confMin);
//        }
//    },

    BeliefStructuralDeduction() {
        @Override
        public boolean taskTruthSignificant() {
            return false;
        }


        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return B == null ? null : StructuralDeduction.apply(B, null, confMin);
        }
    },

    Xor() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return xor(T, B, confMin);
        }

    },
    XorWeak() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return weak(Xor.apply(T, B, confMin), confMin);
        }
    },

    XorWeaker() {
        @Override
        public Truth apply(Truth T, Truth B, float confMin) {
            return weak(XorWeak.apply(T, B, confMin), confMin);
        }
    }

    ;

    @Nullable private static PreciseTruth structuralComponent(Truth T) {
        return $.t(1, T.conf());
    }

    public static final TruthModel the = new TruthModel(values());

    private final boolean single;
    private final boolean overlap;
    private final Term term;


    NALTruth() {
        this.term = TruthFunction.super.term();

        Field f = Reflect.on(NALTruth.class).field(name()).get();
        this.single = f.isAnnotationPresent(SinglePremise.class);
        this.overlap = f.isAnnotationPresent(AllowOverlap.class);
    }

    public static NALTruth implStrong(boolean fwd, boolean beliefOrGoal) {
        return beliefOrGoal ?
           (fwd ? Pre : PostWeak) :
           (fwd ? PreWeak : Need);
    }

    @Override
    public final Term term() {
        return term;
    }

    @Override
    public final boolean single() {
        return single;
    }

    @Override
    public final boolean allowOverlap() {
        return overlap;
    }

}