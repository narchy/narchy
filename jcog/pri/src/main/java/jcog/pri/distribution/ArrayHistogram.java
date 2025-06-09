package jcog.pri.distribution;

import jcog.Is;
import jcog.Util;
import jcog.pri.Prioritized;
import jcog.signal.tensor.ArrayTensor;
import jcog.signal.tensor.AtomicArrayTensor;
import jcog.signal.tensor.WritableTensor;

import java.util.Arrays;

import static jcog.Str.n2;

/**
 * DEPRECATED: SketchHistogram is more accurate
 *
 * dead-simple fixed range continuous histogram with fixed # and size of bins. supports PDF sampling
 * https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-mathematical-foundations/inverse-transform-sampling-method
 * <p>
 * TODO pluggable function-approximation back-end
 *   either:
 *      empirical distribution (current)
 *      polynomial fit
 *      ...
 */
@Is("Inverse_transform_sampling")
@Deprecated public class ArrayHistogram extends DistributionApproximator {

    /**
     * for single writer
     */
    float[] dataPre;

    /**
     * probabality density function
     */
    @Is("Probability_density_function")
    private transient float[] pdf;

    /**
     * for readers
     */
    private volatile WritableTensor dataOut;

//    private static int binRound(float x, int bins) {
//        return Math.round(bin(x, bins));
//        //return Math.min(bins - 1, (int) (x * bins));
//        //return (int) (x * (bins - 0.5f));
//    }


    //    /**
//     * 1-point raw
//     */
//    private static void addRaw1(float x, float weight, int offset, int binCount, float[] bins) {
//        bins[binRound(x, binCount) + offset] += weight;
//    }

    @Override
    public String toString() {
        return n2(dataOut.floatArray());
    }

    private void resize(int nextBins) {
        WritableTensor prevDataOut = this.dataOut;
        if (dataPre == null || (prevDataOut != null ? prevDataOut.volume() : -1) != nextBins) {
            this.dataPre = new float[nextBins];
            this.dataOut = dataAtomic(nextBins);
        } else {
            Arrays.fill(dataPre, 0);
        }
    }

    private static ArrayTensor dataPlain(int bins) {
        return bins == 0 ? ArrayTensor.Zero : new ArrayTensor(bins);
    }

    private static AtomicArrayTensor dataAtomic(int bins) {
        return bins == 0 ? AtomicArrayTensor.Empty : new AtomicArrayTensor(bins);
    }

    @Override
    public void start(int inBins, int values) {
        float[] pdf = this.pdf;
        if (pdf != null && pdf.length == inBins)
            Arrays.fill(pdf, 0);
        else {
            //assert (inBins > 0);
            this.pdf = new float[inBins];
        }
    }


    @Override
    public void accept(float v) {
        bin(Util.unitize(v), 1, pdf, pdf.length);
    }


    /**
     * inverse transform sampling
     */
    @Override
    public void commit(float lo, float hi, int outBins) {
        if (outBins <= 1)
            throw new UnsupportedOperationException();

        cdf(pdfToCdf(), lo, hi, outBins);
    }

    private float[] pdfToCdf() {
        float[] pdf = this.pdf;
        int inBins = pdf.length;

        //1. convert probabality density function (pdf) to cumulative density function (cdf)
        for (int i = 1 /* skip first */; i < inBins; i++)
            pdf[i] += pdf[i - 1];
        return pdf;
    }

    private void cdf(float[] cdf, float lo, float hi, int outBins) {
        int inBins = cdf.length;

        //2. rotate
        float min = cdf[0], max = cdf[inBins - 1];
        float range = max - min;
        if (range < outBins * Prioritized.EPSILON)
            commit2(lo, hi);
        else
            commitCurve(lo, hi, outBins, cdf, 0, inBins, min, range);
    }

    private void commitCurve(float lo, float hi, int outBins, float[] cdf, int bs, int be, float cdfMin, float cdfRange) {
        resize(outBins);

        int bw = be - bs;

        var data = dataPre;
        double sum = 0;
        for (int i = bs; i < be; i++) {
            double w = ((i + 0.5) - bs) / bw;
            sum += w;
            float cx = (cdf[i] - cdfMin) / cdfRange;
            bin(cx, (float)w, data, outBins);
        }

        //cumulative
        for (int i = 1; i < outBins - 1; i++)
            data[i] += data[i - 1];

        normalizeIndices(lo, hi, outBins, data, sum);

        dataOut.setAll(data);
    }

    private static void normalizeIndices(float lo, float hi, int outBins, float[] data, double sum) {
        data[0] = lo;

        float hilo = hi - lo;
        for (int i = 1; i < outBins - 1; i++)
            data[i] = (float) ((data[i] / sum) * hilo + lo);

        data[outBins - 1] = hi;
    }

    @Override
    public void commit2(float lo, float hi) {
        WritableTensor o = this.dataOut;
        if (o == null)
            this.dataOut = o = dataAtomic(2);

        o.setAt(0, lo);
        o.setAt(1, hi);
    }

    /**
     * TODO use the rng to generate one 64-bit or one 32-bit integer and use half of its bits for each random value needed
     */
    @Override
    public float sample(float q) {
        WritableTensor data = this.dataOut;
        return data == null ? q : interpolate2(data, bin(q, data.volume()));
    }


//    public String chart() {
//        return SparkLine.renderFloats(new FloatArrayList(dataOut.floatArrayShared()));
//    }
}