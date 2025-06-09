package nars.nal.nal7;

import nars.$;
import nars.NAR;
import nars.NARS;
import nars.truth.Truth;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.eclipse.collections.api.block.function.primitive.DoubleToDoubleFunction;
import org.hipparchus.util.Precision;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NAL7ImplTruthProjectionTest {

    @Test
    void test1() {

        int implDT = 5;
        int dur = 1;

        NAR n = NARS.tmp();

        /* eventTime, relative to impl belief */
        int ts = 2;
        for (int implTime = 0; implTime < 3; implTime+= ts) {
            for (int eventTime = 0; eventTime < 3; eventTime+= ts) {

                n.complexMax.set(10);
                n.time.dur(dur);
                n.inputAt(eventTime, "x. |");
                n.inputAt(implTime, "(x ==>+" + implDT + " y). |");
                int end = Math.max(eventTime + implDT, implTime);
                n.run(end + 1);

                double[] max = new MyBrentOptimizer(0.001f, 0.01, 0, end, (t) -> {
                    Truth u = n.beliefTruth($.atomic("y"), Math.round(t));
                    return u == null ? -1f : (float) u.conf();
                }).max(0, end, end / 2.0);


                long yTimeEstimate = Math.round(max[0]);
                long yTimeActual = eventTime + implDT;
                assertTrue(Math.abs(yTimeEstimate - yTimeActual) <= ts, ()->yTimeEstimate + " estimated, " + yTimeActual + " actual");

                double yConfMax = max[1];
                long eventBeliefDelta = Math.abs(eventTime - implTime);
                System.out.println("+-" + eventBeliefDelta + " -> " + max[0] + '=' + max[1]);

                n.stop();
                n.clear();
            }
        }





        
        
    }

    static class MyBrentOptimizer /*extends UnivariateOptimizer*/ {
    /**
     * Golden section.
     */
    private static final double GOLDEN_SECTION = 0.5 * (3 - Math.sqrt(5));
    /**
     * Minimum relative tolerance.
     */
    private static final double MIN_RELATIVE_TOLERANCE = 2 * Math.ulp(1d);
    /**
     * Relative threshold.
     */
    private final double relativeThreshold;
    /**
     * Absolute threshold.
     */
    private final double absoluteThreshold;

    final double min;
        final double max;
    private final DoubleToDoubleFunction func;
























    /**
     * The arguments are used for implementing the original stopping criterion
     * of Brent's algorithm.
     * {@code abs} and {@code rel} define a tolerance
     * {@code tol = rel |x| + abs}. {@code rel} should be no smaller than
     * <em>2 macheps</em> and preferably not much less than <em>sqrt(macheps)</em>,
     * where <em>macheps</em> is the relative machine precision. {@code abs} must
     * be positive.
     *
     * @param rel Relative threshold.
     * @param abs Absolute threshold.
     * @param min
     * @param func
     * @throws NotStrictlyPositiveException if {@code abs <= 0}.
     * @throws NumberIsTooSmallException if {@code rel < 2 * Math.ulp(1d)}.
     */
    MyBrentOptimizer(double rel,
                     double abs, double min, double max, DoubleToDoubleFunction func) {


        this.min = min;
        this.max = max;
        this.func = func;
        if (rel < MIN_RELATIVE_TOLERANCE) {
            throw new RuntimeException();
            //throw new NumberIsTooSmallException(rel, MIN_RELATIVE_TOLERANCE, true);
        }
        if (abs <= 0) {
            throw new RuntimeException();
            //throw new NotStrictlyPositiveException(abs);
        }

        relativeThreshold = rel;
        absoluteThreshold = abs;
    }


    double[] max(double... xStart) {
        double[] previous = { Double.NaN, Double.NEGATIVE_INFINITY };
        for(double x : xStart) {
            previous = max(x, previous);
        }
        return previous;
    }

    double[] max(double xStart, double[] previous) {
        double lo = min;
        
        double hi = max;

        



        double a;
        double b;
        if (lo < hi) {
            a = lo;
            b = hi;
        } else {
            a = hi;
            b = lo;
        }

        double X = xStart;
        double v = X;
        double w = X;
        double Y = func.valueOf(X);
        final boolean isMinim = false;
        if (true) {
            Y = -Y;
        }
        double fv = Y;
        double fw = Y;

        double[] current
            = { X, isMinim ? Y : -Y };
        
        double[] best = current;

        double e = 0;
        double d = 0;
        while (true) {
            

            double m = 0.5 * (a + b);
            double tol1 = relativeThreshold * Math.abs(X) + absoluteThreshold;
            double tol2 = 2 * tol1;

            
            boolean stop = Math.abs(X - m) <= tol2 - 0.5 * (b - a);
            if (!stop) {
                double u = 0;

                if (Math.abs(e) > tol1) {
                    double r = (X - w) * (Y - fv);
                    double q = (X - v) * (Y - fw);
                    double p = (X - v) * q - (X - w) * r;
                    q = 2 * (q - r);

                    if (q > 0) {
                        p = -p;
                    } else {
                        q = -q;
                    }

                    r = e;
                    e = d;

                    if (p > q * (a - X) &&
                        p < q * (b - X) &&
                        Math.abs(p) < Math.abs(0.5 * q * r)) {
                        
                        d = p / q;
                        u = X + d;

                        
                        if (u - a < tol2 || b - u < tol2) {
                            if (X <= m) {
                                d = tol1;
                            } else {
                                d = -tol1;
                            }
                        }
                    } else {
                        
                        if (X < m) {
                            e = b - X;
                        } else {
                            e = a - X;
                        }
                        d = GOLDEN_SECTION * e;
                    }
                } else {
                    
                    e = ((X < m ? b : a) - X);
                    d = GOLDEN_SECTION * e;
                }

                
                u = (Math.abs(d) < tol1) ? ((d >= 0) ? (X + tol1) : (X - tol1)) : (X + d);

                double fu = func.valueOf(u);
                if (true) {
                    fu = -fu;
                }

                
                previous = best(previous, current, isMinim);
                current = new double[] {u, isMinim ? fu : -fu };
                best = best(best,
                            previous,
                            isMinim);





                
                if (fu <= Y) {
                    if (u < X) {
                        b = X;
                    } else {
                        a = X;
                    }
                    v = w;
                    fv = fw;
                    w = X;
                    fw = Y;
                    X = u;
                    Y = fu;
                } else {
                    if (u < X) {
                        a = u;
                    } else {
                        b = u;
                    }
                    if (fu <= fw ||
                        Precision.equals(w, X)) {
                        v = w;
                        fv = fw;
                        w = u;
                        fw = fu;
                    } else if (fu <= fv ||
                               Precision.equals(v, X) ||
                               Precision.equals(v, w)) {
                        v = u;
                        fv = fu;
                    }
                }
            } else { 
                return best(best,
                        best(previous,
                                current,
                                isMinim),
                        isMinim);
            }

            
        }
    }

    /**
     * Selects the best of two points.
     *
     * @param a Point and value.
     * @param b Point and value.
     * @param isMinim {@code true} if the selected point must be the one with
     * the lowest value.
     * @return the best point, or {@code null} if {@code a} and {@code b} are
     * both {@code null}. When {@code a} and {@code b} have the same function
     * value, {@code a} is returned.
     */
    private static double[] best(double[] a, double[] b, boolean isMinim) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }

        if (isMinim) {
            return a[1] <= b[1] ? a : b;
        } else {
            return a[1] >= b[1] ? a : b;
        }
    }
}

}