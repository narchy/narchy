package nars.pri;

import jcog.Util;

import static nars.truth.func.TruthFunctions.e2c;

/** confidence loss functions.  returns a multiplier proportional to
 *  the amount of evidence retained from a parent in a derivation. */
public enum ConfidenceRetention {
    EviLinear {
        @Override
        public double apply(double eParent, double eDerived) {
            return eDerived / eParent;
        }
    },
    ConfLinear {
        @Override
        public double apply(double eParent, double eDerived) {
            return e2c(eDerived) / e2c(eParent);
        }
    },
    EviSqrt {
        @Override
        public double apply(double eParent, double eDerived) {
            return Math.sqrt(eDerived / eParent);
        }
    },
    /** more lenient than linear */
    EviLog {
        @Override
        public double apply(double eParent, double eDerived) {
            return Util.log1p(eDerived) / Util.log1p(eParent);
        }
    };

    abstract public double apply(double eParent, double eDerived);
}