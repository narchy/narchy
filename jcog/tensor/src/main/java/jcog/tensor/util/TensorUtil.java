package jcog.tensor.util;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.util.KahanSum;
import org.ejml.concurrency.EjmlConcurrency;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_MT_DDRM;
import org.ejml.simple.SimpleMatrix;
import pabeles.concurrency.IntRangeConsumer;
import pabeles.concurrency.IntRangeTask;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

public class TensorUtil {
    /**
     * s += m * x
     */
    static SimpleMatrix addTo(SimpleMatrix s, SimpleMatrix x, double m) {
        int r = s.getNumRows(), c = s.getNumCols();
        if (r != x.getNumRows() || c != x.getNumCols())
            throw new UnsupportedOperationException("size mismatch");

        if (m == 1)
            _addTo(s, x);
        else
            Util.addTo(Tensor.array(s), Tensor.array(x), m);

        return s;
    }

    /**
     * s += x
     */
    static SimpleMatrix addTo(SimpleMatrix s, SimpleMatrix x) {
        return addTo(s, x, 1);
    }

    /**
     * x += y
     */
    public static SimpleMatrix _addTo(SimpleMatrix x, SimpleMatrix y) {
        Util.addTo(Tensor.array(x), Tensor.array(y));
        return x;
    }

    public static SimpleMatrix newMatrix(SimpleMatrix d, DoubleUnaryOperator f) {
        return d.elementOp((int r, int c, double x) -> f.applyAsDouble(x));
    }

    public static void assertSameShape(Tensor x, Tensor y) {
        if (!x.sameShape(y))
            throw new IllegalArgumentException("Tensors must have the same shape for binary operations: " + x.shapeStr() + " != " + y.shapeStr());
    }

    public static double sumAbs(SimpleMatrix m) {
        return Util.sumAbs(Tensor.array(m));
    }

    public static double sumSqr(SimpleMatrix m) {
        return Util.sumSqr(Tensor.array(m));
    }

    public static void eleMul(SimpleMatrix X, double y) {
        Util.mul(Tensor.array(X), y);
    }

    public static void eleMul(SimpleMatrix O, SimpleMatrix X, double y) {
        double[] o = Tensor.array(O), x = Tensor.array(X);
        var n = o.length;
        if (n!=x.length)
            throw new UnsupportedOperationException();
        for (var i = 0; i < n; i++)
            o[i] = x[i] * y;
    }

    public static void eleMul(SimpleMatrix O, SimpleMatrix X, SimpleMatrix Y) {
        double[] o = Tensor.array(O), x = Tensor.array(X), y = Tensor.array(Y);
        if (o.length!=x.length || o.length!=y.length)
            throw new UnsupportedOperationException();
        var n = o.length;
        for (var i = 0; i < n; i++)
            o[i] = x[i] * y[i];
    }

    public static double eleSum(SimpleMatrix d) {
        return Util.sum(Tensor.array(d));
    }

    private static SimpleMatrix signum(SimpleMatrix matrix) {
        var x = Tensor.array(matrix);
        var v = x.length;
        for (var i = 0; i < v; i++) {
            var xi = x[i];
            double y;
            if (xi > 0) y = +1;
            else if (xi < 0) y = -1;
            else y = 0;
            x[i] = y;
        }
        return matrix;
    }

    /** elementSum(g.elementMult(data)) */
    public static double sumEleMult(SimpleMatrix D, SimpleMatrix G) {
        var d = Tensor.array(D);
        var g = Tensor.array(G);
        var n = d.length;
        if (n!=g.length) throw new UnsupportedOperationException();
        var s = new KahanSum();
        for (var i = 0; i < n; i++)
            s.add(d[i] * g[i]);
        return s.value();
    }

    public static void assertClippable(double min, double max) {
        if (max <= min)
            throw new IllegalArgumentException();
    }

    public static boolean clipDisabled(double min, double max) {
        return min == Double.NEGATIVE_INFINITY && max == Double.POSITIVE_INFINITY;
    }

    static boolean isZero(SimpleMatrix x) {
        return isZero(Tensor.array(x));
    }

    static boolean isZero(double[] x) {
        for (var xx : x) if (xx!=0) return false;
        return true;
    }

    public static void matmul(DMatrixRMaj A, DMatrixRMaj B, DMatrixRMaj AB) {
        if (EjmlConcurrency.useConcurrent(AB))
            CommonOps_MT_DDRM.mult(A, B, AB);
        else if (Math.max(A.getNumElements(), B.getNumElements()) > MatUtil_OpenCL.ELEMENT_THRESHOLD)
            MatUtil_OpenCL.get().mult(A, B, AB);
        else
            MatUtil_CPU.mult(A, B, AB);
    }

    public static void matmulTransA(DMatrixRMaj A, DMatrixRMaj B, DMatrixRMaj AtB) {
        if (EjmlConcurrency.useConcurrent(AtB))
            CommonOps_MT_DDRM.multTransA(A, B, AtB);
        else
            MatUtil_CPU.multTransA(A, B, AtB); //CommonOps_DDRM.multTransA(dd, gg, gg1);
    }

    public static void matmulTransB(DMatrixRMaj A, DMatrixRMaj B, DMatrixRMaj ABt) {
        if (EjmlConcurrency.useConcurrent(ABt))
            CommonOps_MT_DDRM.multTransB(A, B, ABt);
        else
            MatUtil_CPU.multTransB(A, B, ABt);
    }

    protected static final ForkJoinPool FORK_JOIN_POOL = new ForkJoinPool(/*TODO parallelism, async, etc */);
    protected static final int parallelism = Math.max(1, Runtime.getRuntime().availableProcessors()-1);

    public static double sumParallel(int size, ToDoubleFunction<Integer> computeFunction) {
        var threshold = Math.max(1, size / parallelism);
        return FORK_JOIN_POOL.invoke(new ParallelSumTask(0, size, threshold, computeFunction));
    }

    public static void runParallel(int s, int e, IntRangeConsumer o) {
        FORK_JOIN_POOL.invoke(new IntRangeTask(-1, s, e, 1, o));
    }

    public static class ParallelSumTask extends RecursiveTask<Double> {

        private final int start, end, threshold;
        private final ToDoubleFunction<Integer> f;

        ParallelSumTask(int start, int end, int threshold, ToDoubleFunction<Integer> f) {
            this.start = start;
            this.end = end;
            this.threshold = threshold;
            if (end<=start)
                throw new UnsupportedOperationException();
            this.f = f;
        }

        @Override
        protected Double compute() {
            return compute(start,end);
        }

        private double compute(int start, int end) {
            var len = end - start;
            if (len == 1) {
                return compute1();
            } else if (len <= threshold) {
                return computeN();
            } else {
                return computeFork(len);
            }
        }

        private double compute1() {
            return f.applyAsDouble(start);
        }

        private double computeN() {
            var k = new KahanSum();
            for (var i = start; i < end; i++)
                k.add(f.applyAsDouble(i));
            return k.value();
        }

        private double computeFork(int length) {
            var mid = start + length / 2;
            var left = new ParallelSumTask(start, mid, threshold, f);
            var right = end-mid <= threshold ? null : new ParallelSumTask(mid, end, threshold, f);
            left.fork();
            var rightResult = right==null ? compute(mid, end) : right.compute();
            double leftResult = left.join();
            return Util.sum(rightResult, leftResult);
        }
    }
}
