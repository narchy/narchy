package jcog.math;

import java.util.function.DoubleSupplier;

public class CumulativeSum implements DoubleSupplier {
    private final DoubleSupplier o;
    private final int interval;
    double sum;
    int i;

    public CumulativeSum(DoubleSupplier o, int interval) {
        this.o = o;
        this.interval = interval;
        sum = 0;
        i = 0;
    }

    @Override
    public double getAsDouble() {
        var sumNext = this.sum + o.getAsDouble();
        var interval = this.interval;
        if (++i % interval == 0) {
            this.sum = 0;
            return sumNext / interval; //mean
        } else {
            this.sum = sumNext;
            return Double.NaN;
        }
    }
}