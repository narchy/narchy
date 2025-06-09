package jcog.math.normalize;

import jcog.Util;
import org.HdrHistogram.AtomicHistogram;
import org.HdrHistogram.Histogram;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;

/** normalizes a stream of numbers using histogram percentile */
public class Percentilizer implements FloatToFloatFunction {

    public static final int DIGITS =
        2;
        //3;
        //4;

    private final Histogram h;
    private final int discreteness;

    public Percentilizer(int discreteness, boolean atomic) {
        this.discreteness = discreteness;

        int highestTrackableValue = discreteness + 1;
        h = atomic ?
            new AtomicHistogram(1, highestTrackableValue, DIGITS)
            :
            new Histogram(1, highestTrackableValue, DIGITS);

        seed(3);
    }

    /** seed with flat distribution */
    private void seed(int steps) {
        if (steps < 2) throw new UnsupportedOperationException();
        steps--;

        for (float i = 0; i <= steps; i++)
            h.recordValue(i(i/steps));
    }

    protected int i(float v) {
        return Util.toInt(v, discreteness) + 1;
    }

    @Override
    public float valueOf(float v) {
        int V = i(v);
        float p = (float) (h.getPercentileAtOrBelowValue(V-1) / 100.0);
        h.recordValue(V);
        return p;
    }
}
