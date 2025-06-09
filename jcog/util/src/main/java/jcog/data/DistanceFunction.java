package jcog.data;

import jcog.Util;

import static java.lang.Math.sqrt;

@FunctionalInterface
public interface DistanceFunction {

    //TODO fail fast version:
    // double distance(double[] a, double[] b, double distMin, double distMax);

    double distance(double[] a, double[] b);

    /** aka Euclidean */
    static double distanceCartesianSq(double[] x, double[] y) {
        int l = y.length;
        double sum = 0;
        for (int i = 0; i < l; i++)
            sum += Util.sqr(y[i] - x[i]);
        return sum;
    }

    /** aka Euclidean */
    static double distanceCartesian(double[] x, double[] y) {
        return sqrt(distanceCartesianSq(x,y));
    }

    static double distanceCartesianSq(float[] x, float[] y) {
        int l = y.length;
        double sum = 0.0;
        for (int i = 0; i < l; i++)
            sum += Util.sqr(y[i] - x[i]);
        return sum;
    }

    static double distanceCartesian(double[] x, double[] y, int from, int to) {
        return sqrt(distanceCartesianSq(x, y, from, to));
    }

    static double distanceCartesianSq(double[] x, double[] y, int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++)
            sum += Util.sqr(y[i] - x[i]);
        return sum;
    }
    /* L1 distance */
    static double distanceManhattan(double[] x, double[] y) {
        return distanceManhattan(x, y, 0, x.length);
    }

    static double distanceManhattan(double[] x, double[] y, int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++)
            sum += Math.abs(y[i] - x[i]);
        return sum;
    }

    static double normalizedCartesian(double[] a, double[] b, int from, int to) {
        return sqrt(normalizedCartesianSq(a, b, from, to));
    }

    static double normalizedCartesianSq(double[] a, double[] b, int from, int to) {
        double sumSqr = 0;
        for (int i = from; i < to; i++) {
            double ai = a[i], bi = b[i];
            if (ai != bi) {
                double absA = Math.abs(ai), absB = Math.abs(bi);
                var normFactor = Math.min(absA, absB);
                if (normFactor > 0)
                    sumSqr += Util.sqr((ai - bi) / normFactor);
            }
        }
        return sumSqr;
    }

}