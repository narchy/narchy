package jcog.math.normalize;

import jcog.Util;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;

import static java.lang.Math.abs;
import static jcog.Util.lerp;

public class FloatNormalizer implements FloatToFloatFunction {

    public double min, max;
    double minMin = Double.NEGATIVE_INFINITY, minMax = Double.POSITIVE_INFINITY;
    double maxMin = Double.NEGATIVE_INFINITY, maxMax = Double.POSITIVE_INFINITY;

    /**
     * contraction rate: how quickly the normalizatoin limits shrink to zero
     * TODO determine according to # of iterations period count
     */
    private double contracting;

    /**
     * expansion rate: how quickly the normalization limits can grow
     * TODO determine according to # of iterations period count
     */
    private double expanding;

    /** whether to reflect the min/max across zero */
    private boolean polar = false;

    public FloatNormalizer(int expandIterations, int contractIterations) {
        period(expandIterations, contractIterations);
        clear();
    }

    public final FloatNormalizer minLimit(double minLimit, double maxLimit) {
        this.minMin = minLimit; this.minMax = maxLimit;
        return this;
    }
    public final FloatNormalizer maxLimit(double minLimit, double maxLimit) {
        this.maxMin = minLimit; this.maxMax = maxLimit;
        return this;
    }

    public final FloatNormalizer period(int expandIterations, int contractIterations) {
        expanding = Util.halflifeRate(expandIterations);
        contracting = Util.halflifeRate(contractIterations);
        if (Util.equals(contracting,1))
            throw new UnsupportedOperationException("instantaneous contraction");
        if (!Double.isFinite(expanding))
            throw new UnsupportedOperationException("expanding too large");
        if (!Double.isFinite(contracting))
            throw new UnsupportedOperationException("contracting too large");

        return this;
    }

    public final FloatNormalizer polar() {
        this.polar = true;
        return this;
    }

    @Override
    public String toString() {
        return "FloatNormalizer{" + min + ".." + max + '}';
    }

    public void clear() {
        this.min = max = Float.NaN;
    }

    public double min() {
        return min;
    }

    public double max() {
        return max;
    }

    public float valueOf(float x) {
        if (x != x)
            return Float.NaN;
        if (!Float.isFinite(x))
            throw new UnsupportedOperationException();

        updateRange(x);

        double y = ((x - min) / Math.max(Double.MIN_NORMAL, max - min));
        if (!Double.isFinite(y))
            throw new UnsupportedOperationException();

        y = Util.unitize(y);
        //assertUnitized(y);
//        if (y < 0)
//            Util.nop();
//        if (y > 1)
//            Util.nop();

        return (float) y;
    }

    FloatNormalizer updateRange(double x) {
        double min = this.min, max = this.max;
        //noinspection ConstantConditions
        if (min != min) {
            min = max = x; //initialize
        } else {
            double contractTarget =
                (min + max) / 2 //mid
                //x;
            ;

            if (x > min) {
                //contract bottom towards mid
                min = _min(lerp(contracting, min, contractTarget));
            }

            if (x < max) {
                //contract top towards mid
                max = _max(lerp(contracting, max, contractTarget));
            }

            if (x < min) {
                //stretch below min
                min = _min(lerp(expanding, min, x));
            }

            if (x > max) {
                //stretch above max
                max = _max(lerp(expanding, max, x));
            }
        }

        range(min, max);
        return this;
    }

    public final FloatNormalizer range(double min, double max) {
        if (polar) {
            double amp = Math.max(abs(min), abs(max));
            _range(-amp, +amp);
        } else {
            _range(Math.min(max, min), Math.max(min, max));
        }
        return this;
    }

    private void _range(double min, double max) {
        this.min = _min(min);
        this.max = _max(max);
//        if (min > max) {
//            min = max = (min+max)/2; //HACK mean if out of order
//        }
//        this.min = min; this.max = max;
    }

    private double _max(double max) {
        return Util.clampSafe(max, maxMin, maxMax);
    }

    private double _min(double min) {
        return Util.clampSafe(min, minMin, minMax);
    }

    //    public static class FloatBiasedNormalizer extends FloatNormalizer {
//        public final FloatRange bias;
//
//        public FloatBiasedNormalizer(FloatRange bias) {
//            this.bias = bias;
//        }
//
//        @Override
//        protected float normalize(float x, float min, float max) {
//            float y = super.normalize(x, min, max);
//            if (y == y) {
//                float balance = Util.unitize(bias.floatValue());
//                if (y >= 0.5f) {
//                    return Util.lerp(2f * (y - 0.5f), balance, 1f);
//                } else {
//                    return Util.lerp(2f * (0.5f - y), balance, 0f);
//                }
//            } else
//                return Float.NaN;
//
//            //return Util.unitize(y + (b - 0.5f));
//        }
//    }
}