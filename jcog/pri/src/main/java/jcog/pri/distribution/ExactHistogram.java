package jcog.pri.distribution;

import jcog.Util;
import jcog.decide.Roulette;

/**
 * stores all inputs and applies roulette sampling
 */
public class ExactHistogram extends DistributionApproximator {

    private float[] data;
    int i;
    private double sum;

    @Override
    public float sample(float q) {
        return Roulette.select(data, i, sum, q);
    }

    @Override
    public void start(int bins, int values) {
        if (data == null || data.length < values)
            data = new float[values];
        i = 0;
    }

    @Override
    public void accept(float v) {
        data[i++] = v;
    }

    private void commit() {
        this.sum = Util.sum(data, 0, i);
    }

    @Override
    public void commit(float lo, float hi, int outBins) {
        commit();
    }

    @Override
    public void commit2(float lo, float hi) {
        commit();
    }
}
