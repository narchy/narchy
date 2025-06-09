package jcog.math;

import jcog.signal.buffer.CircularFloatBuffer;

/** implements simple moving average, TODO merge with EWMA */
public class FloatMeanWindow {

    private int minSize = 1;

    private final CircularFloatBuffer data;

    public FloatMeanWindow(int capacity) {
        this.data = new CircularFloatBuffer(capacity);
    }

    public int capacity() {
        return data.capacityInternal();
    }

    public FloatMeanWindow minSize(int ms) {
        this.minSize = ms;
        return this;
    }

    public final double acceptAndGetMean(float x) {
        accept(x);
        return mean();
    }
    public double[] acceptAndGetMeanAndVariance(float x) {
        accept(x);
        return data.meanAndVariance();
    }


    public double mean() {
        //TODO optional cache
        int s = size();
        if (/*s == 0 || */s < minSize) return Double.NaN;

        return data.sum(0, s) / s;
    }

    public void accept(float v) {
        if (v == v) {
            if (data.available()==0)
                data.read(new float[1], 1, false);
            data.write(v);
        }
    }

    public void clear() {
        data.clear();
    }

    public int size() {
        return data.size();
    }

}
