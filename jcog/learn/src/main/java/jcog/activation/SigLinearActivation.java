package jcog.activation;

import static jcog.Util.lerpSafe;
import static jcog.Util.unitizeSafe;

/**
 * linear sigmoid-like variant
 * TODO abstract to generic list of piecewise-linear segments for different shapes
 */
public class SigLinearActivation implements DiffableFunction {

    /**
     * "bandwidth".  in sigmoid, it is effectively infinity
     */
    private final double xMin, xMax, xRange;
    private final double yMin, yMax;
    private final double slope;

    public static final SigLinearActivation the = new SigLinearActivation(4, 0, 1);

    public SigLinearActivation(float xRadius, float yMin, float yMax) {
        this(-xRadius, +xRadius, yMin, yMax);
    }

    public SigLinearActivation(float xMin, float xMax, float yMin, float yMax) {
        this.xMin = xMin; this.xMax = xMax; this.yMin = yMin; this.yMax = yMax;
        this.xRange = xMax - xMin;
        double yRange = yMax - yMin;
        this.slope = yRange / xRange;
    }

    @Override
    public double valueOf(double x) {
        return lerpSafe(unitizeSafe((x - xMin) / xRange), yMin, yMax);
    }

    @Override
    public double derivative(double x) {
        return
            x <  xMin || x >  xMax ? //exclusive
            //x <= xMin || x >= xMax ? //inclusive
                    0 : slope;
    }

}