package jcog.pri.distribution;

import jcog.Util;
import jcog.signal.ITensor;

import java.util.Random;

import static jcog.Util.lerpSafe;

public abstract class DistributionApproximator {

    /**
     * 2-ary linear interpolation
     */
    public static float interpolate2(ITensor d, float offsetPlusFraction) {
        int offset = (int) offsetPlusFraction;
        return lerpSafe(offsetPlusFraction - offset, d.getAt(offset), d.getAt(offset + 1));
    }
    /**
     * 2-ary linear interpolation
     */
    public static float interpolate2(double[] d, float offsetPlusFraction) {
        int offset = (int) offsetPlusFraction;
        return (float) lerpSafe(offsetPlusFraction - offset, d[offset], d[offset + 1]);
    }

    public static void bin(float x, float weight, float[] bins, int binCount) {
        bin2(x, weight, binCount, bins);
        //addRaw1(pri, w, offset, bins, pdf);
    }

    public static int binFloor(float x, int bins) {
        return (int) bin(x, bins);
    }

    public static float bin(float x, int bins) {
    return x * (bins - 1);
}

    public static float unbin(int b, int bins) {
        return b / (bins - 1f);
    }


    /**
     * 2-ary supersample
     */
    public static void bin2(float x, float weight, int binCount, float[] bins) {
        float binned = x * (binCount - 1);
        int yMin = Math.min((int)binned, binCount - 2);
        float pMax = binned - yMin;
        bins[yMin]     += weight * (1 - pMax);
        bins[yMin + 1] += weight * pMax;
    }

    /** get a value from the distribution.
     *  the value returned is an index for
     *  the callee to lookup into its dataset */
    public abstract float sample(float q);

    public abstract void start(int bins, int values);

    /** record a value, v >= 0 */
    public abstract void accept(float v);

    /** finalize after accept(values) */
    public abstract void commit(float lo, float hi, int outBins);

    /** finalize for special case where the distribution is 2-ary */
    abstract public void commit2(float lo, float hi);

    public final int sampleInt(Random r) {
        return sampleInt(r.nextFloat());
    }

    public final int sampleInt(float uniform) {
        return Math.round(sample(uniform));
        //return (int) sample(uniform);
    }

    static double[] applySoftMax(double[] pmf) {
        double[] softMaxPmf = new double[pmf.length];
        double max = Util.max(pmf);
        double sum = 0;

        // Subtract max for numerical stability
        for (int i = 0; i < pmf.length; i++) {
            softMaxPmf[i] = Math.exp(pmf[i] - max);
            sum += softMaxPmf[i];
        }

        // Normalize to get probabilities
        for (int i = 0; i < pmf.length; i++)
            softMaxPmf[i] /= sum;

        return softMaxPmf;
    }

    public final static DistributionApproximator Empty = new DistributionApproximator() {
        @Override
        public float sample(float q) {
            return Float.NaN;
        }

        @Override
        public void start(int bins, int values) {

        }

        @Override
        public void accept(float v) {

        }

        @Override
        public void commit(float lo, float hi, int outBins) {

        }

        @Override
        public void commit2(float lo, float hi) {

        }
    };


}