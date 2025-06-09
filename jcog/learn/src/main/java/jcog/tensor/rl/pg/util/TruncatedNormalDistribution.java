package jcog.tensor.rl.pg.util;

import jcog.Util;
import jcog.tensor.Tensor;
import org.hipparchus.special.Erf;

import java.util.random.RandomGenerator;

import static java.lang.Math.sqrt;

public class TruncatedNormalDistribution {
    public double mean;
    public double stdev;
    public double min;
    public double max;
    private final RandomGenerator random;

    private double alpha, beta, alpha_cdf, beta_cdf;

    public TruncatedNormalDistribution(RandomGenerator rng) {
        this.random = rng;
    }

    public TruncatedNormalDistribution set(double mean, double stdev, double min, double max) {
        this.min = min;
        this.max = max;
        this.mean = mean = Util.clampSafe(mean, min, max);
        this.stdev = stdev;
        alpha = (min - mean) / stdev;
        beta = (max - mean) / stdev;
        alpha_cdf = normalCDF(alpha);
        beta_cdf = normalCDF(beta);
        return this;
    }

    public double sample() {
        double r = random.nextDouble();
        double x = mean + stdev * inverseNormalCDF(alpha_cdf + r * (beta_cdf - alpha_cdf));
        return Util.clampSafe(x, min, max);
    }

    public double pdf(double x) {
        return ((x < min) || (x > max)) ? 0 : _pdf(x);
    }

    private double _pdf(double x) {
        return normalPDF((x - mean) / stdev) / (stdev * (beta_cdf - alpha_cdf));
    }

    private static double normalPDF(double x) {
        return Math.exp(x * x / -2) / sqrt(2 * Math.PI);
    }

    private static double normalCDF(double x) {
        return (1 + Erf.erf(x / sqrt(2)))/2;
    }

    private static final double c0 = 2.515517;
    private static final double c1 = 0.802853;
    private static final double c2 = 0.010328;
    private static final double d1 = 1.432788;
    private static final double d2 = 0.189269;
    private static final double d3 = 0.001308;

    private static double inverseNormalCDF(double p) {
        if (p <= 0 || p >= 1)
            throw new IllegalArgumentException("The probability must be between 0 and 1");

        var t = sqrt(-2 * Math.log(Math.min(p, 1 - p)));
        var tt = t * t;
        return ((p < 0.5) ? -1 : +1) *
            (t - (c0 + c1 * t + c2 * tt) / (1 + d1 * t + d2 * tt + d3 * tt * t));
    }

    public double sample(double mean, double stddev, double min, double max) {
        return set(mean, stddev, min, max).sample();
    }

    /**
     * Warning: requires modifying the log-probability formula to account for the truncated range
     * which is not yet implemented
     */
    public void sample(Tensor mean, Tensor sigma, double[] a) {
        for (var i = 0; i < a.length; i++)
            a[i] = sample(mean.data(i), sigma.data(i), -1, +1);
    }
}
