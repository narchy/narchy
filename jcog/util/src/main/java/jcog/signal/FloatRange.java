package jcog.signal;

import jcog.Util;
import jcog.math.FloatSupplier;
import jcog.math.NumberException;
import org.eclipse.collections.api.block.function.primitive.DoubleToDoubleFunction;


public class FloatRange extends MutableFloat {

    public final float min, max;

    public FloatRange(float value, float min, float max) {
        if (value > max || value < min)
            throw new NumberException("out of expected range", value);

        this.min = min;
        this.max = max;
        set(value);
    }

    public void setMomentum(float next, float momentum) {
        setLerp(momentum, next, asFloat()); //TODO atomic?
    }

    /** includes a function which specifies how a control surface should map the values (ex: logarithmically) */
    public static class FloatRangeWithMapping extends FloatRange {

        /** x = physical space
         *  y = target space
         */
        private final DoubleToDoubleFunction control;

        public FloatRangeWithMapping(float value, float min, float max, DoubleToDoubleFunction control) {
            super(value, min, max);
            this.control = control;
        }
    }

    /** logarithmic scale */
    public static FloatRangeWithMapping unitLog(float value, float base) {
        return new FloatRangeWithMapping(value, 0, 1, (x)-> Math.log(-x/base));
    }

    @Override
    protected float post(float x) {
        return Util.clampSafe(x, min, max);
    }

    public final void set(double value) {
        set((float)value);
    }

    public static FloatRange unit(float initialValue) {
        return new FloatRange(initialValue, 0, 1);
    }

    public static FloatRange unit(FloatSupplier initialValue) {
        return unit(initialValue.asFloat());
    }

    public final void setLerp(float x) {
        setLerp(x, min, max);
    }

    public final void setSafe(float x) {
        if (x > max) throw new NumberException("above max", x);
        if (x < min) throw new NumberException("below min", x);
        set(x);
    }

//    public static FloatRange mapRange(float mapMin, float mapMax) {
//        throw new TODO();
//    }


}