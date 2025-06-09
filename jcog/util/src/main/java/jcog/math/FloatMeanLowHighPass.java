package jcog.math;

import jcog.Util;
import jcog.signal.FloatRange;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;

/**
 * exponential moving average of a Float source.
 * can operate in either low-pass (exponential moving average of a signal) or
 * high-pass modes (signal minus its exponential moving average).
 * <p>
 * https://dsp.stackexchange.com/a/20336
 * https://en.wikipedia.org/wiki/Exponential_smoothing
 * <p>
 * warning this can converge/stall.  best to use FloatAveragedWindow instead
 */
public class FloatMeanLowHighPass implements FloatToFloatFunction {
    protected final FloatRange alpha;
    private final boolean lowOrHighPass;
    private float value;

    public FloatMeanLowHighPass(float alpha) {
        this(alpha, true);
    }

    public FloatMeanLowHighPass(float alpha, boolean lowOrHighPass) {
        this(FloatRange.unit(alpha), lowOrHighPass);
    }

    public FloatMeanLowHighPass(FloatRange alpha, boolean lowOrHighPass) {
        this.alpha = alpha;
        this.lowOrHighPass = lowOrHighPass;
    }

    @Override
    public float valueOf(float x) {
//            if (x != x)
//                return this.value;

        float p = value, next;
        if (x!=x) {
            x = next = p;
        } else if (p!=p) {
            this.value = next = x;
        } else {
            float alpha = alpha(x, p);
            this.value = next = Util.lerpSafe(alpha, p, x);
        }

        return lowOrHighPass ? next : x - next;
    }

    protected float alpha(float next, float prev) {
        return this.alpha.asFloat();
    }

    /**
     * previous value computed by valueOf
     */
    public float floatValue() {
        return value;
    }

}