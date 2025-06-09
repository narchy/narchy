package jcog.util;

import jcog.Util;

/**
 * Summing algorithm that preserves more floating point precision than Naive
 */
public final class KahanSum {
    private double value, c;

    public static double sum(float[] x, int from, int to) {
        return sum(Util.toDouble(x), from, to);
    }

    public static double sum(double[] x, int from, int to) {
        return switch (to - from) {
            case 0 -> 0;
            case 1 -> x[from];
            case 2 -> Util.sumPrecise(x[from], x[from + 1]);
            default -> sumN(x, from, to);
        };
    }

    private static double sumN(double[] x, int from, int to) {
        double sum = 0;
        double c = 0; //a running compensation for lost low-order bits

        for (int i = from; i < to; i++) {
            double d = x[i] - c;
            double sumNext = sum + d;
            c = (sumNext - sum) - d;
            sum = sumNext;
        }
        return sum;
    }

    public KahanSum clear() {
        value = c = 0;
        return this;
    }

    public KahanSum add(double[] x) {
        //return add(Util.sum(x));
        for (double xx : x)
            add(xx);
        return this;
    }

    public KahanSum addAbs(float[] x) {
        for (float xx : x)
            add(Math.abs(xx));
        return this;
    }

    public KahanSum addAbs(double[] x) {
        for (double xx : x)
            add(Math.abs(xx));
        return this;
    }

    public KahanSum add(double x) {
        _add(x - c);
        return this;
    }

    private void _add(double y) {
        if (y == y) {
            double v = this.value;
            this.c = ((this.value = v + y) - v) - y;
        }
    }

    /** fused product summation with FMA */
    public KahanSum addXY(double x, double y) {
        _add(Util.fma(x, y, -c));
        return this;
    }

    public double value() {
        return value;
    }

    public double valueClear() {
        double s = value();
        clear();
        return s;
    }

    public KahanSum add(double[][] xx) {
        for (double[] x : xx)
            add(x);
        return this;
    }

    public void addSqr(double f) {
        add(f * f); // y+=f*f
    }
}
