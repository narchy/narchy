package jcog.optimize;

import jcog.Util;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.optim.PointValuePair;
import org.jetbrains.annotations.Nullable;

abstract public class MyAsyncCMAESOptimizer extends MyCMAESOptimizer {
    @Nullable
    private transient FitEval e;
    @Nullable
    public transient RealMatrix arx;
    @Nullable
    private transient RealMatrix arz;

    protected MyAsyncCMAESOptimizer(int popSize, double[] sigma) {
        super(Integer.MAX_VALUE, Double.NaN, popSize, sigma);
    }

    transient double[][] X = null;
    transient double[] penalty = null;

    public double[] best() {
        int best = Util.argmax(e.fitness);
        return X[best];
    }

    protected PointValuePair doOptimize(double[] startPoint) {
        return null;
    }

    @Override
    protected boolean iterateEval(RealMatrix arx, RealMatrix arz, FitEval e) {
        if (X == null)
            X = new double[capacity][arx.getColumnDimension()];

        if (e.isRepairMode && penalty == null)
            penalty = new double[capacity];

        for (int k = 0; k < capacity; k++) {
            var point = arx.getColumn(k);

            double penalty;
            if (e.isRepairMode) {
                double[] repaired = e.repair(point);
                penalty = e.penalty(point, repaired);
                point = repaired;
            } else
                penalty = 0;

            this.X[k] = point;
            this.penalty[k] = penalty;
        }

        this.e = e;
        this.arx = arx;
        this.arz = arz;

        if (apply(X)) {
            return true;
        } else {
            this.e = null; this.arx = this.arz = null; //clear refs
            return false;
        }
    }

    /**
     * afterwards, it asynchronously call commit(y) with the results later. returns whether to continue, and if false it wont expect any following asynch commit call
     */
    abstract protected boolean apply(double[][] X);

    /**
     * for callee to provide batch answers asynchronously
     */
    @Override public final synchronized void commit(double[] y) {
        for (int k = 0; k < capacity; k++)
            e.value[k] = valuePenalty(y[k], penalty!=null ? penalty[k] : null);

        e.iterateAfter(arx, arz);
    }

    @Override
    public double computeObjectiveValue(double[] params) {
        throw new UnsupportedOperationException();
    }
}
