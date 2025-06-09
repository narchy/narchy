package jcog.signal;

import jcog.Util;

public class FloatRangeDivisible extends FloatRange {

    final int resMax, resMin;

    public FloatRangeDivisible(float value, float min, float max, int resMax, int resMin) {
        super(value, min, max);

        if (resMin < resMax) throw new UnsupportedOperationException();

        this.resMax = resMax;
        this.resMin = resMin;
        set(value); //HACK update again, now that resMin,resMax are set
    }

    /**
     * rounds to nearest integer divisor of 1/freqEpsilon
     */
    @Override
    protected float post(float x) {
        int xStep = Util.clampSafe((int) (1 / x), resMax, resMin);
        return 1f / xStep;
    }
}
