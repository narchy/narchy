package jcog.tensor.deprtensor;

import jcog.Util;
import jcog.tensor.Tensor;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

import static java.lang.Double.isFinite;
import static jcog.Str.n4;
import static jcog.Util.clampSafe;
import static jcog.Util.lerpSafe;
import static jcog.util.ArrayUtil.copy;

@Deprecated public class Tens0r {
    public double[][] data, grad;

    protected Tens0r() { }

    public Tens0r(int col) {
        this(1, col);
    }

    public Tens0r(int col, int row) {
        this(new double[col][row]);
    }

    public Tens0r(double[] data) {
        this(new double[][]{data});
    }

    public Tens0r(double[][] data) {
        setData(data);
    }
    @Override
    public String toString() {
        return (data != null ? "(" + data[0].length + "," + data.length +
                        "){d=" + n4(data[0]) + ", g=" + (this.grad != null ? n4(grad[0]) : null) + '}' : "d=null");
    }

    public static void clip(double[] x, double threshold) {
        if (threshold != Double.POSITIVE_INFINITY) {
            for (int i = 0, length = x.length; i < length; i++)
                x[i] = Util.clamp(x[i], -threshold, +threshold);
        }
    }
    public static void clip(double[][] x, double threshold) {
        if (threshold != Double.POSITIVE_INFINITY)
            Util.clamp(x, -threshold, +threshold);
    }

    public static void scaleNorm(Iterable<Tens0r> grad, double threshold) {

//        double norm =
//            //l1Orl2 ? l1normGrad(grad) : l2normGrad(grad);


//        //MAX
//        double norm = stream(grad.spliterator(), false)
//                .mapToDouble(Tensor::gradMaxAbs).max().getAsDouble();

        //Manhattan
//        double sum = 0;
//        for (var t : grad) {
//            for (double[] xy : t.grad)
//                for (double x : xy)
//                    sum += x;
//        }
//        double norm = sum;

        //L2
        double sumSqr = 0;
        for (var t : grad)
            sumSqr += Util.sumSqr(t.grad);

        double norm = Math.sqrt(sumSqr);
        if (norm > threshold)
            scaleGrad(grad, threshold / norm);
    }

    private static double l1normGrad(Iterable<Tens0r> p) {
        double s = 0;
        for (Tens0r X : p) {
            double[][] x = X.grad;
            int J = x[0].length;
            for (double[] xx : x)
                for (int j = 0; j < J; j++)
                    s += Math.abs(xx[j]);
        }
        return s;
    }

    private static double l2normGrad(Iterable<Tens0r> p) {
        double s = 0;
        for (Tens0r X : p) {
            double[][] x = X.grad;
            int J = x[0].length;
            for (double[] xx : x)
                for (int j = 0; j < J; j++)
                    s += Util.sqr(xx[j]);
        }
        return Math.sqrt(s);
    }

    static void scaleGrad(Iterable<Tens0r> x, double y) {
        for (Tens0r t : x)
            scale(t.grad, y);
    }

    static void scale(double[][] x, double y) {
        if (y == 1) return;
        int I = x.length, J = x[0].length;
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
                x[i][j] *= y;
    }


    static double[][] newArraySizeOf(Tens0r a) {
        double[][] A = a.data;
        return new double[A.length][A[0].length];
    }

    /**
     * if the value is not finite, impute a random number
     */
    private static double clampSafeRNG(double x, double min, double max, Random rng) {
        return isFinite(x) ? clampSafe(x, min, max) : rng.nextDouble(min, max);
    }

    //    public static void mult3(double[][][] x, double b, double[][][] y) {
//        for (int i = 0; i < x.length; i++)
//            for (int j = 0; j < x[0].length; j++)
//                for (int k = 0; k < x[0][0].length; k++)
//                    y[i][j][k] = x[i][j][k] * b;
//    }


    public void randomNormal(float min, float max) {
        Random rng = ThreadLocalRandom.current();
        int h = data.length, w = data[0].length;
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                data[i][j] = rng.nextFloat(min, max);
    }

    /**
     * fast signum
     */
    protected static double signum(double x) {
        if (x > 0) return +1;
        else if (x < 0) return -1;
        else return 0;
    }

    static Tens0r newTensorSizeOf(Tens0r t) {
        return new Tens0r(newArraySizeOf(t));
    }

    public static void clamp3(double[][][] x, double[][][] y, double min, double max) {
        if (min == Double.NEGATIVE_INFINITY && max == Double.POSITIVE_INFINITY) return;
        int I = y.length, J = y[0].length, K = y[0][0].length;
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
                for (int k = 0; k < K; k++)
                    y[i][j][k] = Util.clamp/*SafeRNG*/(x[i][j][k], min, max);
    }

    public static Tens0r scalar(boolean b) {
        return scalar(b ? 1 : 0);
    }

    public static Tens0r scalar(double scalar) {
        return new Tens0r(new double[]{scalar});
    }

    public void backward() { /* nop */ }

    public double scalar() {
        assertScalar();
        return data[0][0];
    }

    public double scalarGrad() {
        assertScalar();
        return grad[0][0];
    }

    public void scalarGrad(double g) {
        assertScalar();
        grad[0][0] = g;
    }

    private void assertScalar() {
        if (!isScalar())
            throw new UnsupportedOperationException("not scalar");
    }

    public boolean isScalar() {
        return data.length == 1 && data[0].length == 1;
    }

    /**
     * Gradient clipping is a common technique used to prevent the gradients
     * from exploding during training, which can lead to unstable learning.
     *
     * @param gradientClip use Double.POSITIVE_INFINITY to disable
     */
    public void clipGradients(double threshold) {
        clip(grad, threshold);
    }

    public void clipData(double threshold) {
        clip(data, threshold);
    }

    public void scaleGrad(double v) {
        scale(grad, v);
    }

    public void zeroGrad() {
        Tensor.zero(grad);
    }

    protected void setGaussian(double a, RandomGenerator rng) {
        Util.randomGaussian(data, a, rng);
    }

    /**
     * fan out
     */
    public final int w() {
        return data[0].length;
    }

    /**
     * fan in
     */
    public final int h() {
        return data.length;
    }

    public Tens0r setData(boolean b) {
        return setData(b ? 1 : 0);
    }

    public Tens0r setData(double x) {
        return setData(new double[]{x});
    }

    public Tens0r setData(double[] x) {
        return setData(new double[][]{x});
    }

    public Tens0r setData(double[][] x) {
        if (this.data == null) {
            this.data = x.clone();
            this.grad = new double[x.length][x[0].length];
        } else
            copy(x, this.data);

        return this;
    }

    public Tens0r setGrad(double[][] x) {
        copy(x, grad);
        return this;
    }

    public double gradSumAbs() {
        return Util.sumAbs(this.grad);
    }

    public double gradMaxAbs() {
        return Util.maxAbs(this.grad);
    }
    public double dataMaxAbs() {
        return Util.maxAbs(this.data);
    }

    /**
     * LERPs parameters
     */
    public void setDataSoft(Tens0r src, double tau) {
        if (tau == 1) {
            this.setData(src.data);
        } else {
            double[][] sd = src.data, td = this.data;
            for (int j = 0; j < td.length; j++) {
                for (int k = 0; k < td[0].length; k++) {
                    td[j][k] = lerpSafe(tau, td[j][k], sd[j][k]);
                    //td[j][k] = td[j][k] * (1 - tau) + sd[j][k] * tau; //maybe this was reversed..
                }
            }
        }
    }

    public boolean sameShape(Tens0r t) {
        return this == t ||
                (data.length == t.data.length && data[0].length == t.data[0].length);
    }

    public final Tens0r fillData(double x) {
        Util.fill(data, x);
        return this;
    }

    public double[] row() {
        if (h() != 1) throw new UnsupportedOperationException();
        return data[0];
    }

    public double minimize() {
        zeroGrad();
        double loss = scalar();
        scalarGrad(1);
        backward();
        return loss;
    }

    public Tens0r detach() {
        /* nop */
        return this;
    }

    public final int volume() {
        return w() * h();
    }
}
