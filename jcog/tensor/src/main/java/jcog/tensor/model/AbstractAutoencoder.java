package jcog.tensor.model;

import jcog.Fuzzy;
import jcog.random.RandomBits;

import java.util.Random;

import static jcog.Util.toDouble;

public abstract class AbstractAutoencoder {

    public RandomBits rng;

    public abstract void clear(Random rng);

    protected void clear(double[] wi, float a) {
        var n = wi.length;
        for (var j = 0; j < n; j++)
            wi[j] = rng.nextGaussian() * a;
    }

    protected void clear(double[][] wi, float a) {
        for (var x : wi)
            clear(x, a);
    }

    private float uniform(float min, float max) {
        return rng.nextFloat() * (max - min) + min;
    }

    public abstract double[] get(double[] x);

    public final void put(float[] x) {
        put(toDouble(x)); //HACK
    }

    /**
     *
     */
    abstract public void put(double[] x);

    /** callback after encoding */
    @FunctionalInterface public interface LatentCallback {
        void latent(double[] input, double[] encoded, double[] decoded);
    }
    public LatentCallback latent = null;

    protected void preprocess(double[] x, double noiseLevel, float corruptionRate) {
        var r = this.rng;
        var n = x.length;

        if (corruptionRate > 0) {
            var nDropped = r.nextFloor(corruptionRate * n);
            for (var i = 0; i < nDropped; i++)
                x[r.nextInt(n)] = 0;
        }

        if (noiseLevel > 0) {
            for (var i = 0; i < n; i++)
                x[i] += Fuzzy.polarize(r.nextFloatFast8()) * noiseLevel;
        }
    }
}