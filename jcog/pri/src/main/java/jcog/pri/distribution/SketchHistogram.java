package jcog.pri.distribution;

import jcog.Util;
import jcog.util.ArrayUtil;
import org.apache.datasketches.quantiles.DoublesSketch;
import org.apache.datasketches.quantiles.UpdateDoublesSketch;

public class SketchHistogram extends DistributionApproximator {

    private final static int K_MAX = 128;

    /** TODO these fields should be moved to a
     * double-buffered parameter class since
     * commit asnchronously might occurr, interrupting operations
     * involving them and cause problems */
    private UpdateDoublesSketch sketch;

    /** capacity parameter for DoubleSketch */
    private int k;
    
    private double[] index = { 0, 1 }; //HACK

    @Override
    public float sample(float q) {
        var index = this.index;
        return interpolate2(index, bin(q, index.length));
        //return index!=null ? interpolate2(index, bin(q, index.length)) : 0/*?*/;
    }

    @Override
    public void start(int bins, int values) {
        if (bins < 2) {
            sketch = null;
        } else {
            var nextK =
                    //K_MAX;
                    Util.clampSafe(Util.largestPowerOf2NoGreaterThan(bins * 2), 2, K_MAX);
            if (sketch == null || this.k != nextK)
                sketch = DoublesSketch.builder().setK(this.k = nextK).build();
            else
                sketch.reset();
        }
    }

    @Override
    public void accept(float v) {
        sketch.update(v);
    }

    @Override
    public void commit(float lo, float hi, int bins) {
        var s = sketch;
        if (s ==null || s.isEmpty())
            commit2(lo, hi); //wtf?
        else
            commitN(lo, hi, bins, s);
    }

    private void commitN(float lo, float hi, int bins, UpdateDoublesSketch sketch) {
        this.index =  pmfToIndex(lo, sketch.getPMF(splitPoints(bins)), hi);
    }

    private static double[] pmfToIndex(float lo, double[] pmf, float hi) {
        var n = pmf.length;
        var cdf = new double[1 + n];
        double sum = 0;
        for (var i = 0; i < n; i++) {
            sum += pmf[i] * (i + 1); // Weight by bin index to prioritize higher values
            //sum += pmf[i] * ((n - 1) - i + 1); // Weight by bin index to prioritize lower
            //sum += pmf[i]; // Unweighted
            cdf[i+1] = sum;
        }

        return normalize(lo, hi, n, cdf, sum);
    }

    /** Normalize and scale to [lo, hi] range */
    private static double[] normalize(float lo, float hi, int n, double[] cdf, double sum) {
        var hilo = hi - lo;
        n++;
        double hiloDivSum = hilo/sum;
        for (var i = 0; i < n; i++)
            cdf[i] = (cdf[i] * hiloDivSum) + lo;

        return cdf;
    }

    private static double[] splitPoints(int bins) {
        if (bins < 2)
            throw new UnsupportedOperationException();
        else if (bins == 2)
            return ArrayUtil.EMPTY_DOUBLE_ARRAY;
        else
            return splitPointsN(bins);
    }

    private static double[] splitPointsN(int bins) {
        var p = new double[bins - 1];
        double sp = 0, binWidth = 1.0 / (bins);
        for (var i = 0; i < bins - 1; i++)
            p[i] = (sp += binWidth);
        return p;
    }

    @Override
    public void commit2(float lo, float hi) {
        this.index = new double[] { lo, hi };
    }

//    //TODO
//    public static class SoftMaxSketchHistogram extends SketchHistogram {
//
//        @Override
//        protected double[] commitN(float lo, float hi, int bins, UpdateDoublesSketch sketch) {
//            double[] pmf = sketch.getPMF(splitPoints(bins));
//            double[] softMaxPmf = applySoftMax(pmf);
//            return pmfToIndex(lo, softMaxPmf, hi);
//        }
//
//    }


}
