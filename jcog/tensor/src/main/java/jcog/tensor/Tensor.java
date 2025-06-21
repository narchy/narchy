package jcog.tensor;

import com.google.common.base.Charsets;
import jcog.Str;
import jcog.Util;
import jcog.data.bit.MetalBitSet;
import jcog.data.list.Lst;
import jcog.func.TriConsumer;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.util.ParallelBackward;
import jcog.tensor.util.TensorUtil;
import jcog.util.Reflect;
import org.eclipse.collections.api.block.function.primitive.DoubleDoubleToDoubleFunction;
import org.ejml.concurrency.EjmlConcurrency;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.MatrixType;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.ConstMatrix;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.*;
import java.util.stream.Stream;

import static jcog.Util.*;

public class Tensor {

    public static final Tensor[] EMPTY_ARRAY = new Tensor[0];
    public static final UnaryOperator<Tensor> TANH = Tensor::tanh;
    public static final UnaryOperator<Tensor> RELU = Tensor::relu;
    public static final UnaryOperator<Tensor> RELU_LEAKY = Tensor::reluLeaky;
    public static final UnaryOperator<Tensor> SIGMOID = Tensor::sigmoid;
    private static final Comparator<Tensor> generationComparator = Comparator.comparingInt(t -> -t.generation());

    static {
        EjmlConcurrency.USE_CONCURRENT = false;
        EjmlConcurrency.ELEMENT_THRESHOLD = 5_000;
    }

    public SimpleMatrix data;
    @Nullable public Tensor grad;
    public boolean parameter;
    public TensorOp op;

    public Tensor(double data, boolean requiresGrad) {
        this(new double[]{data}, requiresGrad);
    }

    public Tensor(double[] data, boolean requiresGrad) {
        this(data, 1, data.length, requiresGrad);
    }

    /** clones the data array */
    public Tensor(double[] data, int rows, int cols, boolean requiresGrad) {
        this(new SimpleMatrix(rows, cols, true, data), requiresGrad);
    }

    /** clones the data array */
    public Tensor(double[][] data, boolean requiresGrad) {
        this(new SimpleMatrix(data), requiresGrad);
    }

    public Tensor(int rows, int cols, boolean requiresGrad) {
        this(new SimpleMatrix(rows, cols), requiresGrad);
    }

    public Tensor(SimpleMatrix data, boolean requiresGrad) {
        this.op = null;
        setData(data);
        grad(requiresGrad);
    }

    public static Tensor scalar(DoubleSupplier x) {
        return row(x.getAsDouble());
    }

    public static Tensor scalar(double x) {
        return row(x);
    }

    public static Tensor row(double... data) {
        return new Tensor(data, false);
    }

    public static Tensor matrix(double[][] data) {
        return new Tensor(data, false);
    }

    public static Tensor zeros(int rows, int cols) {
        return new Tensor(new SimpleMatrix(rows, cols), false);
    }

    public static Tensor ones(int rows, int cols) {
        return new Tensor(SimpleMatrix.ones(rows, cols), false);
    }

    @Deprecated public static Tensor ones(int[] rowsCols) {
        if (rowsCols.length!=2) throw new UnsupportedOperationException();
        return ones(rowsCols[0], rowsCols[1]);
    }

    @Deprecated public static Tensor zeros(int[] rowsCols) {
        if (rowsCols.length!=2) throw new UnsupportedOperationException();
        return zeros(rowsCols[0], rowsCols[1]);
    }

    public static Tensor ones(int cols) {
        return ones(1, cols);
    }

    public static Tensor zerosShaped(Tensor x) {
        return zeros(x.rows(), x.cols());
    }

    public static UnaryOperator<Tensor> compose(UnaryOperator<Tensor> a, @Nullable UnaryOperator<Tensor> b) {
        return b!=null ? x -> b.apply(a.apply(x)) : a;
    }

    public static void zero(double[][] x) {
        for (var row : x)
            zero(row);
    }

    public static void zero(double[][][] x) {
        for (var col : x)
            for (var row : col)
                zero(row);
    }

    public static void zero(double[] x) {
        Arrays.fill(x, 0);
    }

    public static void mult(double[][] x, double b) {
        mult(x, b, x);
    }

    public static void mult(double[][] x, double b, double[][] y) {
        var n = x.length;
        for (var i = 0; i < n; i++)
            for (var j = 0; j < x[0].length; j++)
                y[i][j] = x[i][j] * b;
    }

    public static Tensor randGaussian(int... shape) {
        if (shape.length!=2) throw new UnsupportedOperationException();
        return randGaussian(shape[0], shape[1]);
    }

    public static Tensor randGaussian(int rows, int cols) {
        return randGaussian(rows, cols, 1);
    }

    public static Tensor randGaussian(int rows, int cols, double stddev) {
        var rng = new XoRoShiRo128PlusRandom();
        var n = rows * cols;
        var data = new double[n];
        for (var i = 0; i < n; i++)
            data[i] = rng.nextGaussian() * stddev;
        return new Tensor(data, rows, cols, false);
    }

    /** He Initialization: For layers with ReLU activations */
    public static Tensor randHe(int inFeatures, int outFeatures) {
        return randGaussian(inFeatures, outFeatures, Math.sqrt(2.0 / outFeatures));
    }

    /** Xavier initialization: for tanh or linear */
    public static Tensor randXavier(int rows, int cols) {
        var stddev = Math.sqrt(2.0 / (rows + cols));
        return randGaussian(rows, cols, stddev).parameter();
    }

    public static Tensor max(Tensor a, Tensor b) {
        return min(a.neg(), b.neg()).neg();
    }

    public static Tensor min(Tensor a, Tensor b) {
        TensorUtil.assertSameShape(a, b);

        final var epsilon =
            Float.MIN_NORMAL;
            //1e-8;
            //Double.MIN_NORMAL;

        int R = a.rows(), C = a.cols();
        var V = R * C;
        double[] aa = a.array(), bb = b.array();

        var y = new SimpleMatrix(R, C);
        var yy = array(y);
        for (var i = 0; i < V; i++)
            yy[i] = Math.min(aa[i], bb[i]);

        var x = new Tensor(y, a.hasGrad() || b.hasGrad());
        if (x.hasGrad()) {
            x.op = new TensorOp(a, b) {
                @Override public void backward(SimpleMatrix g, SimpleMatrix[] gradOut /* preallocated */) {
                    var gg = array(g);
                    double[] GA = array(gradOut[0]), GB = array(gradOut[1]);
                    for (var i = 0; i < V; i++) {
                        var gi = gg[i];

                        var diff = aa[i] - bb[i];
                        if (diff < -epsilon)
                            GA[i] = gi; //ai < bi
                        else if (diff > +epsilon)
                            GB[i] = gi; //ai > bi
                        else
                            GA[i] = GB[i] = gi/2; // split gradient when values equal
                    }
                }
            };
        }
        return x;
    }

    /** assumes the list of tensors are all scalar */
    public static Tensor minEle(List<Tensor> tensors) {
        if (tensors.isEmpty())
            throw new IllegalArgumentException("List of tensors cannot be empty.");
        var T = tensors.size();
        if (T == 1) return tensors.getFirst();

        // Find the minimum value and its index
        var minValue = Double.POSITIVE_INFINITY;
        var minIndex = -1;
        for (var i = 0; i < T; i++) {
            var value = tensors.get(i).scalar();
            if (value < minValue) {
                minValue = value;
                minIndex = i;
            }
        }

        // Create the output tensor (a scalar)
        var x = new Tensor(minValue, tensors.stream().anyMatch(Tensor::hasGrad));

        // Set up gradient dependencies
        if (x.hasGrad()) {
            var finalMinIndex = minIndex;
            x.op = new TensorOp(tensors.toArray(EMPTY_ARRAY)) {
                @Override
                public void backward(SimpleMatrix gradIn, SimpleMatrix[] gradOut /* preallocated */) {
                    gradOut[finalMinIndex].set(0, gradIn.get(0));
                }
            };
        }
        return x;
    }

    @Nullable public static double[] array(@Nullable SimpleMatrix m) {
        if (m == null) return null;
        if (m.getType()!=MatrixType.DDRM)
            throw new UnsupportedOperationException();
        var d = m.getDDRM().data;
        if (d.length!=m.getNumElements())
            throw new UnsupportedOperationException();
        return d;
    }

    /** ordered backprop */
    private static IdentityHashMap<Tensor, SimpleMatrix> gradients(PriorityQueue<Tensor> q, IdentityHashMap<Tensor, SimpleMatrix> g) {
        Tensor next;
        while ((next = q.poll()) != null) {
            next.op.gradients(next, g, q);
        }
        return g;
    }

    // Helper method to create an identity matrix
    public static Tensor eye(int n) {
        var identity = new Tensor(n, n, false);
        for (var i = 0; i < n; i++)
            identity.data.set(i, i, 1.0);
        return identity;
    }

    public static int volume(Iterable<Tensor> t) {
        var sum = 0;
        for (var z : t)
            sum += z.volume();
        return sum;
    }

    public static Tensor concat(Tensor... others) {
        var x = others[0];
        for (var i = 1; i < others.length; i++)
            x = x.concat(others[i]);
        return x;
    }

    public static Stream<Tensor> parameters(Object x) {
        if (x instanceof Models.Layers l) {
            return Stream.concat(l.layer.stream().flatMap(Tensor::parameters), parametersReflect(x));
        }

        return parametersReflect(x);
    }

    private static @NotNull Stream<Tensor> parametersReflect(Object x) {
        return Reflect.on(x).fieldsRecursive(true, false, false,
                        (fieldName, obj, parent) -> obj instanceof Tensor t && t.parameter)
                .stream().map(y -> (Tensor) ((Reflect) y).object);
    }

    public static Tensor stack(Tensor[] tensors, int dim) {
        if (dim != 1)
            throw new IllegalArgumentException("Only stacking along dimension 1 is supported for now.");

        // Check that all tensors have the same number of rows
        var rows = tensors[0].rows();
        for (var tensor : tensors)
            if (tensor.rows() != rows)
                throw new IllegalArgumentException("All tensors must have the same number of rows for stacking.");

        // Calculate the total number of columns
        var totalCols = 0;
        for (var tensor : tensors)
            totalCols += tensor.cols();

        // Create the output tensor
        var stacked = new Tensor(rows, totalCols, false);

        // Copy data from each tensor into the stacked tensor
        var currentCol = 0;
        for (var tensor : tensors) {
            var r = tensor.rows();
            var c = tensor.cols();
            for (var i = 0; i < r; i++)
                for (var j = 0; j < c; j++)
                    stacked.data.set(i, currentCol + j, tensor.data(i, j));
            currentCol += c;
        }

        // Handle gradients if required
        if (stacked.hasGrad()) {
            stacked.op = new TensorOp(tensors) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var colOffset = 0;
                    for (var i = 0; i < tensors.length; i++) {
                        var tensor = tensors[i];
                        if (gradOut[i] != null) {
                            for (var r = 0; r < tensor.rows(); r++) {
                                for (var c = 0; c < tensor.cols(); c++) {
                                    gradOut[i].set(r, c, grad.get(r, colOffset + c));
                                }
                            }
                        }
                        colOffset += tensor.cols();
                    }
                }
            };
        }

        return stacked;
    }

    /**
     * Generates a tensor with random values from uniform distribution [0,1)
     *
     * @param rows Number of rows
     * @param cols Number of columns
     * @return New tensor with random values
     */
    public static Tensor randUniform(int rows, int cols) {
        var rng = new XoRoShiRo128PlusRandom();
        var data = new double[rows * cols];
        for (var i = 0; i < data.length; i++)
            data[i] = rng.nextDouble();
        return new Tensor(data, rows, cols, false);
    }

    public static Tensor fill(int rows, int cols, double val) {
        return zeros(rows, cols).fill(val);
    }

    public static Tensor concat(Stream<Tensor> stream) {
        return concat(stream.toArray(Tensor[]::new));
    }

    public static NoGrad noGrad() {
        return new NoGrad();
    }

    /** TODO optional Kahan summing? */
    private void addToThis(Queue<double[]> v) {
        var x = array();
        var length = x.length;
        var yy = v.iterator();
        while (yy.hasNext()) {
            var y = yy.next();
            yy.remove();
            addTo(x, y);
        }
    }

    public Tensor binaryOp(Tensor other, DoubleDoubleToDoubleFunction op, DoubleDoubleToDoubleFunction gradOp) {
        TensorUtil.assertSameShape(this, other);

        var y = new Tensor(new SimpleMatrix(rows(), cols()), this.hasGrad() || other.hasGrad());

        double[] yy = y.array(), aa = array(), bb = array(other.data);

        var n = yy.length;
        for (var i = 0; i < n; i++)
            yy[i] = op.valueOf(aa[i], bb[i]);

        if (y.hasGrad()) {
            y.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var g = array(grad);
                    binaryBackward(gradOut[0], bb, g);
                    binaryBackward(gradOut[1], aa, g);
                }

                private void binaryBackward(SimpleMatrix gradOut, double[] d, double[] g) {
                    if (gradOut != null) {
                        var go = array(gradOut);
                        var n = go.length;
                        for (var i = 0; i < n; i++)
                            go[i] = gradOp.valueOf(g[i], d[i]);
                    }
                }
            };
        }

        return y;
    }

    public String shapeStr() {
        return rows() + "x" + cols();
    }

    /** TODO optimize */
    public final Tensor minValue() {
        return neg().maxValue().neg();
    }

    public Tensor maxValue() {
        if (isScalar())
            return this;

        var data = array();
        var n = data.length;

        var maxIndices = MetalBitSet.bits(volume());

        var maxVal = Double.NEGATIVE_INFINITY;
        for (var i = 0; i<n; i++) {
            var val = data[i];
            if (val > maxVal) {
                maxVal = val;
                maxIndices.clear();
                maxIndices.set(i);
            } else if (val == maxVal) {
                maxIndices.set(i);
            }
        }

        var y = new Tensor(maxVal, this.hasGrad());

        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null) {
                        var gradShare = grad.get(0) / maxIndices.cardinality();
                        var gg = array(gradOut[0]);
                        for (var i = maxIndices.next(true, 0, n); i >= 0; i = maxIndices.next(true, i+1, n))
                            gg[i] = gradShare;
                    }
                }
            };
        }

        return y;
    }

    public final double[] array() {
        return array(data);
    }

    public boolean hasGrad() {
        return parameter || grad!=null;
        //return grad != null || op != null; //necessary, or will grad always be non-null if op is?
    }

    public boolean isScalar() {
        return volume() == 1;
    }

    public double scalar() {
        if (!isScalar()) throw new UnsupportedOperationException("not a scalar");
        return array()[0];
        //return data.get(0);
    }

    public void zero() {
        Arrays.fill(array(), 0);
    }

    public final int rows() {
        return data.getNumRows();
    }

    public final int cols() {
        return data.getNumCols();
    }

    public Tensor transpose() {
        if (isScalar())
            return this;

        var y = new Tensor(data.transpose(), this.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    CommonOps_DDRM.transpose(grad.getDDRM(), gradOut[0].getDDRM()); //gradOut[0].setTo(grad[0].transpose());
                }
            };
        }

        return y;
    }

    public Tensor transpose(int dim1, int dim2) {
        if (dim1 == -2 && dim2 == -1)
            return transposeLastTwo(); // Special case for transposing last two dimensions

        throw new UnsupportedOperationException("Only transpose(-2, -1) is supported for now.");
    }

    private Tensor transposeLastTwo() {
        var rows = rows();
        var cols = cols();
        var lastDim = cols / rows;
        var secondLastDim = rows;

        var transposed = new SimpleMatrix(secondLastDim * lastDim, rows);

        for (var i = 0; i < secondLastDim; i++) {
            for (var j = 0; j < lastDim; j++) {
                for (var k = 0; k < rows; k++) {
                    transposed.set(j * secondLastDim + i, k, this.data.get(k, i * lastDim + j));
                }
            }
        }

        var y = new Tensor(transposed, this.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var transposedGrad = new SimpleMatrix(rows, cols);
                    for (var i = 0; i < secondLastDim; i++) {
                        for (var j = 0; j < lastDim; j++) {
                            for (var k = 0; k < rows; k++) {
                                transposedGrad.set(k, i * lastDim + j, grad.get(j * secondLastDim + i, k));
                            }
                        }
                    }
                    gradOut[0].setTo(transposedGrad);
                }
            };
        }
        return y;
    }

    public Tensor reshape(int newRows, int newCols) {
        if (volume() != newRows * newCols)
            throw new IllegalArgumentException("New shape must have the same number of elements as the original tensor.");

        if (rows()==newRows && cols()==newCols)
            return this;

//        var reshaped = new SimpleMatrix(newRows, newCols);
//        System.arraycopy(array(), 0, array(reshaped), 0, this.data.getNumElements());
        var reshaped = new SimpleMatrix(newRows, newCols, false, array());

        var y = new Tensor(reshaped, this.hasGrad());

        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    System.arraycopy(array(g), 0, array(gradOut[0]), 0, g.getNumElements());
                }
            };
        }

        return y;
    }

    public double data(int index) {
        return array()[index];
    }

    public double data(int r, int c) {
        return data.getDDRM().unsafe_get(r, c);
    }

    public Tensor setData(boolean b) {
        return setData(b ? 1 : 0);
    }

    public Tensor setData(double x) {
        if (!isScalar())
            setData(new SimpleMatrix(1, 1, true, x));
        else
            data.set(0, x);
        return this;
    }

    public Tensor setData(double[] x) {
        if (rows()!=1 || cols()!=x.length)
            setData(new SimpleMatrix(1, x.length, true, x));
        else
            System.arraycopy(x, 0, array(), 0, x.length);
        return this;
    }

    public Tensor setData(double[][] x) {
        setData(new SimpleMatrix(x));
        return this;
    }

    @Deprecated private Tensor unaryOp(Function<SimpleMatrix, SimpleMatrix> forward,
                           Function<SimpleMatrix, SimpleMatrix> backward) {
        var y = new Tensor(forward.apply(this.data), this.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    gradOut[0].setTo(backward.apply(grad));
                }
            };
        }
        return y;
    }

    private Tensor unaryOp(Function<SimpleMatrix, SimpleMatrix> forward,
                           TriConsumer<Tensor, SimpleMatrix, double[]> backward) {
        var g = hasGrad();
        var y = new Tensor(forward.apply(this.data), g);
        if (g) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    backward.accept(y, grad, array(gradOut[0]));
                }
            };
        }
        return y;
    }

    private Tensor unaryOp(Function<SimpleMatrix, SimpleMatrix> forward,
                           BiConsumer<SimpleMatrix, SimpleMatrix> backward) {
        var y = new Tensor(forward.apply(this.data), this.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    backward.accept(grad, gradOut[0]);
                }
            };
        }
        return y;
    }

    public boolean sameShape(Tensor x) {
        return rows() == x.rows() && cols() == x.cols();
    }

    public boolean sameShape(ConstMatrix<SimpleMatrix> x) {
        return rows() == x.getNumRows() && cols() == x.getNumCols();
    }

    public Tensor add(double x) {
        return x==0 ? this : addScalar(scalar(x));
    }

    public Tensor add(Tensor other) {
        if (other.isScalar())
            return addScalar(other);

        if (this.rows() != other.rows() && (this.rows() == 1 || other.rows() == 1))
            return broadcastAdd(other);

//        return binaryOp(other,
//            (a, b) -> a + b,
//            (grad, _) -> grad
//        );

        if (!sameShape(other))
            throw new IllegalArgumentException("Tensors dimensions are incompatible for addition.");

        double[] d = array(), o = other.array();
        var yy = new double[d.length];
        for (var i = 0; i < d.length; i++)
            yy[i] = d[i] + o[i];

        // Create the resulting tensor with gradient tracking if necessary.
        var y = new Tensor(yy, rows(), cols(), this.hasGrad() || other.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    var G = array(g);
                    if (gradOut[0] != null)
                        System.arraycopy(G, 0, array(gradOut[0]), 0, G.length);
                    if (gradOut[1] != null)
                        System.arraycopy(G, 0, array(gradOut[1]), 0, G.length);
                }
            };
        }

        return y;
    }

    /** = add(x.mul(t)) */
    public final Tensor addMul(Tensor x, double t) {

        if (t == 0)
            return this; // Adding 0 * x has no effect, return this tensor.
        else if (t == 1)
            return add(x); // If t == 1, just add x directly.
        else if (x.isScalar())
            return addScalar(x.mul(t));

        double[] d = array(), xx = x.array();
        var n = d.length;
        var yy = new double[n];
        for (var i = 0; i < n; i++)
            yy[i] = fma(xx[i], t, d[i]);

        var y = new Tensor(yy, rows(), cols(), this.hasGrad() || x.hasGrad());


        if (y.hasGrad()) {
            y.op = new TensorOp(this, x) {
                @Override
                public void backward(SimpleMatrix G, SimpleMatrix[] gradOut) {
                    var g = array(G);
                    if (gradOut[0] != null)
                        System.arraycopy(g, 0, array(gradOut[0]), 0, g.length);
                    if (gradOut[1] != null) {
                        var go = array(gradOut[1]);
                        for (var i = 0; i < go.length; i++)
                            go[i] = g[i] * t;
                    }
                }
            };
        }

        return y;
    }

    private Tensor broadcastAdd(Tensor other) {
        var Rt = this.rows();
        var Ro = other.rows();
        var larger = Rt > Ro ? this : other;
        var smaller = Rt > Ro ? other : this;

        var lr = larger.rows();
        var lc = larger.cols();
        if (smaller.rows() != 1 || lc != smaller.cols())
            throw new IllegalArgumentException("Invalid shapes for broadcast addition");

        var y = new Tensor(lr, lc, this.hasGrad() || other.hasGrad());

        for (var i = 0; i < lr; i++)
            for (var j = 0; j < lc; j++)
                y.data.set(i, j, larger.data(i, j) + smaller.data(0, j));

        if (y.hasGrad()) {
            y.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var g0 = gradOut[0];
                    if (g0 != null)
                        g0.setTo(grad);

                    var g1 = gradOut[1];
                    if (g1 != null) {
                        int gc = grad.numCols(), gr = grad.numRows();
                        for (var j = 0; j < gc; j++) {
                            double sum = 0;
                            for (var i = 0; i < gr; i++)
                                sum += grad.get(i, j);
                            g1.set(0, j, sum);
                        }
                    }
                }
            };
        }

        return y;
    }
    public Tensor sub(Tensor other) {
        if (other.isScalar())
            return addScalar(other.neg());

        return binaryOp(other,
                (a, b) -> a - b,
                (grad, _) -> grad
        );
    }
//    public Tensor sub(Tensor other) {
//        if (other.isScalar())
//            return addScalar(other.neg()); // neg() correctly handles gradients
//
//        // Check for broadcasting conditions similar to add()
//        if (this.rows() != other.rows() && (this.rows() == 1 || other.rows() == 1)) {
//            return broadcastSub(other);
//        }
//
//        if (!sameShape(other))
//            throw new IllegalArgumentException("Tensors dimensions are incompatible for subtraction.");
//
//        double[] d = array(), o = other.array();
//        var yy = new double[d.length];
//        for (var i = 0; i < d.length; i++)
//            yy[i] = d[i] - o[i];
//
//        var y = new Tensor(yy, rows(), cols(), this.hasGrad() || other.hasGrad());
//        if (y.hasGrad()) {
//            y.op = new TensorOp(this, other) {
//                @Override
//                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
//                    var G = array(g);
//                    if (gradOut[0] != null) { // Gradient for 'this'
//                        System.arraycopy(G, 0, array(gradOut[0]), 0, G.length);
//                    }
//                    if (gradOut[1] != null) { // Gradient for 'other'
//                        var go1 = array(gradOut[1]);
//                        for (int i = 0; i < G.length; i++) {
//                            go1[i] = -G[i];
//                        }
//                    }
//                }
//            };
//        }
//        return y;
//    }

    private Tensor broadcastSub(Tensor other) {
        var Rt = this.rows();
        var Ct = this.cols();
        var Ro = other.rows();
        var Co = other.cols();

        var thisLargerRows = Rt > Ro;
        var thisLargerCols = Ct > Co; // Though current broadcastAdd only handles row broadcasting

        Tensor larger, smaller;
        boolean largerIsThis;

        if (Rt != Ro) { // Row broadcasting: one has 1 row, the other has R rows
            if (! ( (Rt == 1 && Ro > 1 && Ct == Co) || (Ro == 1 && Rt > 1 && Ct == Co) ) ) {
                 throw new IllegalArgumentException("Invalid shapes for broadcast subtraction: " + this.shapeStr() + ", " + other.shapeStr());
            }
            larger = Rt > Ro ? this : other;
            smaller = Rt > Ro ? other : this;
            largerIsThis = Rt > Ro;
        } else if (Ct != Co) { // Column broadcasting: one has 1 col, the other has C cols
             if (! ( (Ct == 1 && Co > 1 && Rt == Ro) || (Co == 1 && Ct > 1 && Rt == Ro) ) ) {
                 throw new IllegalArgumentException("Invalid shapes for broadcast subtraction: " + this.shapeStr() + ", " + other.shapeStr());
            }
            larger = Ct > Co ? this : other; // e.g. (R, C) and (R, 1)
            smaller = Ct > Co ? other : this;
            largerIsThis = Ct > Co;
        } else {
            throw new IllegalArgumentException("Broadcast subtraction called with incompatible shapes or same shapes: " + this.shapeStr() + ", " + other.shapeStr());
        }


        var lr = larger.rows();
        var lc = larger.cols();

        var y = new Tensor(lr, lc, this.hasGrad() || other.hasGrad());

        if (Rt != Ro) { // Row broadcasting (e.g. (R,C) and (1,C) )
            var smallerRow = smaller.array();
            for (var i = 0; i < lr; i++) { // Iterate through rows of larger tensor
                for (var j = 0; j < lc; j++) { // Iterate through columns
                    y.data.set(i, j, larger.data(i, j) - smallerRow[j]);
                }
            }
        } else { // Column broadcasting (e.g. (R,C) and (R,1) )
            var smallerCol = smaller.array();
             for (var i = 0; i < lr; i++) { // Iterate through rows
                for (var j = 0; j < lc; j++) { // Iterate through columns of larger tensor
                    y.data.set(i, j, larger.data(i, j) - smallerCol[i]);
                }
            }
        }
        // If !largerIsThis, it means y = smaller_broadcasted - larger_actual, so we negate y.
        if (!largerIsThis) {
            y.mulThis(-1.0); // In-place multiplication
        }


        if (y.hasGrad()) {
            y.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var gThis = gradOut[0];
                    var gOther = gradOut[1];

                    if (Rt != Ro) { // Row broadcasting
                        if (largerIsThis) { // this(larger) - other(smaller_broadcasted)
                            if (gThis != null) gThis.setTo(grad);
                            if (gOther != null) { // sum grad along rows for other
                                for (var j = 0; j < lc; j++) {
                                    double sum = 0;
                                    for (var i = 0; i < lr; i++) sum += grad.get(i, j);
                                    gOther.set(0, j, -sum);
                                }
                            }
                        } else { // this(smaller_broadcasted) - other(larger)
                            if (gThis != null) { // sum grad along rows for this
                                for (var j = 0; j < lc; j++) {
                                    double sum = 0;
                                    for (var i = 0; i < lr; i++) sum += grad.get(i, j);
                                    gThis.set(0, j, sum);
                                }
                            }
                            if (gOther != null) {
                                TensorUtil.eleMul(gOther, grad, -1.0); // gOther.setTo(grad.scale(-1));
                            }
                        }
                    } else { // Column broadcasting
                        if (largerIsThis) { // this(larger) - other(smaller_broadcasted)
                            if (gThis != null) gThis.setTo(grad);
                            if (gOther != null) { // sum grad along columns for other
                                for (var i = 0; i < lr; i++) {
                                    double sum = 0;
                                    for (var j = 0; j < lc; j++) sum += grad.get(i, j);
                                    gOther.set(i, 0, -sum);
                                }
                            }
                        } else { // this(smaller_broadcasted) - other(larger)
                             if (gThis != null) { // sum grad along columns for this
                                for (var i = 0; i < lr; i++) {
                                    double sum = 0;
                                    for (var j = 0; j < lc; j++) sum += grad.get(i, j);
                                    gThis.set(i, 0, sum);
                                }
                            }
                            if (gOther != null) {
                                TensorUtil.eleMul(gOther, grad, -1.0);
                            }
                        }
                    }
                }
            };
        }
        return y;
    }

    public Tensor mul(Tensor other) {
        if (other.isScalar())
            return mulScalar(other);

        return binaryOp(other,
            (a, b) -> a * b,
            (grad, b) -> grad * b
        );
    }

    public final Tensor subAbs(double x) {
        return subAbs(scalar(x));
    }

    public Tensor subAbs(Tensor x) {
        TensorUtil.assertSameShape(this, x);

        boolean thg = this.hasGrad(), ohg = x.hasGrad();

        var t = new Tensor(Util.abs(array(data.minus(x.data))), rows(), cols(), thg || ohg);
        if (t.hasGrad()) {
            t.op = new TensorOp(this, x) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    double[] g0 = array(gradOut[0]), g1 = array(gradOut[1]);
                    var gg = array(g);
                    var v = volume();
                    double[] aa = array(), bb = x.array();
                    for (var i = 0; i < v; i++) {
                        var diff = aa[i] - bb[i];
                        var g01 = Util.signum(diff) * gg[i];
                        if (g0!=null) g0[i] = +g01;
                        if (g1!=null) g1[i] = -g01;
                    }
                }
            };
        }
        return t;
    }

    private Tensor addScalar(Tensor other) {
        if (!other.isScalar())
            throw new UnsupportedOperationException();

        var scalar = other.scalar();
        if (scalar == 0)
            return this;

        var t = new Tensor(this.data.plus(scalar), this.hasGrad() || other.hasGrad());
        if (t.hasGrad()) {
            t.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    if (gradOut[0]!=null) gradOut[0].setTo(g);
                    if (gradOut[1]!=null) gradOut[1].set(0, TensorUtil.eleSum(g));
                }
            };
        }
        return t;
    }

    public Tensor mulScalar(Tensor other) {
        var scalar = other.scalar();
        if (scalar == 1)
            return this;

        var t = new Tensor(this.data.scale(scalar), this.hasGrad() || other.hasGrad());
        if (t.hasGrad()) {
            t.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    if (gradOut[0]!=null)
                        TensorUtil.eleMul(gradOut[0], g, scalar);
                    if (gradOut[1]!=null)
                        gradOut[1].set(0, TensorUtil.sumEleMult(data, g)); //TODO optimize
                }
            };
        }
        return t;
    }

    public Tensor divScalar(Tensor other) {
        if (!other.isScalar()) throw new UnsupportedOperationException();

        var scalar = other.scalar();
        if (scalar == 1) return this;

        var t = new Tensor(data.divide(scalar), this.hasGrad() || other.hasGrad());

        if (t.hasGrad()) {
            t.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    if (gradOut[0]!=null)
                        TensorUtil.eleMul(gradOut[0], g, 1/scalar); //gradOut[0].setTo(grad[0].divide(scalar));
                    if (gradOut[1]!=null)
                        gradOut[1].set(0, TensorUtil.sumEleMult(data, g) / (-scalar * scalar));
                }
            };
        }
        return t;
    }

    public void zeroGrad() {
        if (grad != null)
            grad.zero();
    }

    public final Tensor sub(double x) {
        return add(-x);
    }

    public final Tensor mul(double x) {
        return x==1 ? this : mulScalar(scalar(x));
    }
    public final Tensor mul(double[] x) {
        var y = array().clone();
        for (var i = 0; i < y.length; i++)
            y[i] *= x[i];
        return new Tensor(y, rows(), cols(), hasGrad());
    }

    public Tensor neg() {
        return mul(-1);
    }

    /** fused matmul + bias: matmul(weight).add(bias) */
    public Tensor matmulAdd(Tensor weight, Tensor bias) {
        if (isScalar())
            return weight.mul(this).add(bias);
        if (weight.isScalar())
            return this.mul(weight).add(bias);

        DMatrixRMaj A = data.getDDRM(), B = weight.data.getDDRM();
        if (A.getNumCols() != B.getNumRows())
            throw new IllegalArgumentException("'A' and 'B' matrices need A.cols==B.rows: " + Arrays.toString(shape()) + " | " + Arrays.toString(weight.shape()));

        // Perform fused matmul and bias addition
        var yy = new SimpleMatrix(rows(), weight.cols());

        TensorUtil.matmul(A, B, yy.getDDRM());

        // Add bias to each row of the result
        var b = bias.array();
        var yyy = yy.getDDRM().data;
        int r = yy.numRows(), c = yy.numCols();
        for (var i = 0; i < r; i++) {
            var ic = i * c;
            for (var j = 0; j < c; j++)
                yyy[ic + j] += b[j];
        }

        var y = new Tensor(yy, this.hasGrad() || weight.hasGrad() || bias.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this, weight, bias) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    var G = g.getDDRM();
                    // Gradient for input
                    var g0 = gradOut[0];
                    if (g0 != null)
                        TensorUtil.matmulTransB(G, B, g0.getDDRM());
                    // Gradient for weight
                    var g1 = gradOut[1];
                    if (g1 != null)
                        TensorUtil.matmulTransA(A, G, g1.getDDRM());
                    // Gradient for bias
                    var g2 = gradOut[2];
                    if (g2 != null) {
                        var biasGrad = g2.getDDRM().data;
                        var gg = G.data;
                        int gr = G.numRows, gc = G.numCols;
                        for (var i = 0; i < gr; i++) {
                            var ic = i * gc;
                            for (var j = 0; j < gc; j++)
                                biasGrad[j] += gg[ic + j];//G.get(i, j);
                        }
                    }
                }
            };
        }
        return y;
    }

    public Tensor matmul(Tensor other) {
        if (isScalar())
            return other.mul(this);
        if (other.isScalar())
            return this.mul(other);

        DMatrixRMaj A = data.getDDRM(), B = other.data.getDDRM();
        if (A.getNumCols()!=B.getNumRows())
            throw new IllegalArgumentException("The 'A' and 'B' matrices do not have compatible dimensions");

        var yy = new SimpleMatrix(rows(), other.cols());
        TensorUtil.matmul(A, B, yy.getDDRM());

        var y = new Tensor(yy, this.hasGrad() || other.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    var G = g.getDDRM();
                    var g0 = gradOut[0];
                    if (g0 != null)
                        TensorUtil.matmulTransB(G, B, g0.getDDRM());
                    var g1 = gradOut[1];
                    if (g1 != null)
                        TensorUtil.matmulTransA(A, G, g1.getDDRM());
                }
            };
        }
        return y;
    }

    public Tensor exp(double clipMin, double clipMax) {
        return clip(Math.log(clipMin), Math.log(clipMax)).exp();
    }

    public Tensor exp() {
        var y = new Tensor(this.data.elementExp(), this.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    TensorUtil.eleMul(gradOut[0], grad, y.data);
                }
            };
        }
        return y;
    }

    public final Tensor maximize() {
        return neg().minimize();
    }

    public final Tensor minimize() {
        return minimize(null, null);
    }

    public final Tensor maximize(@Nullable GradQueue context) {
        return neg().minimize(context);
    }

    public final Tensor minimize(@Nullable GradQueue context) {
        return minimize(null, context);
    }

    public Tensor minimize(@Nullable Optimizer opt, @Nullable GradQueue context) {
        return minimize(opt, context, 1);
    }

    public Tensor minimize(@Nullable Optimizer opt, @Nullable GradQueue context, double value) {
        if (!this.hasGrad())
            throw new IllegalStateException("minimize() can only be called on a tensor that requires gradient");

        var g = gradients(value);
        var tensors = context==null ? backwardSerial(g) : backwardSerialBuffered(g, context);
        //var tensors = backwardParallel();

        if (opt != null)
            opt.run(tensors);

        return this;
    }

    private Stream<Tensor> backwardSerial(Map<Tensor, SimpleMatrix> grads) {
        grads.forEach(Tensor::addGrad);
        return grads.keySet().stream();
    }

    private Stream<Tensor> backwardSerialBuffered(Map<Tensor, SimpleMatrix> grads, GradQueue context) {
        for (var e : grads.entrySet()) {
            var t = e.getKey();
            var g = e.getValue();
            if (t.parameter)
                context.addGrad(t, array(g));
            else
                t.addGrad(g);
        }
        return grads.keySet().stream();
    }

    /** untested */
    private Iterable<Tensor> backwardParallel() {
        return ParallelBackward.backward(this);
    }

    private Map<Tensor, SimpleMatrix> gradients(double grad) {
        return gradients(gradientQueue(), gradientMap(grad));
    }

    private PriorityQueue<Tensor> gradientQueue() {
        var q = new PriorityQueue<>(generationComparator);
        q.add(this); //initialize
        return q;
    }

    private IdentityHashMap<Tensor, SimpleMatrix> gradientMap(double grad) {
        var g = new IdentityHashMap<Tensor, SimpleMatrix>();
        g.put(this, gradRoot(grad));
        return g;
    }

    private SimpleMatrix gradRoot(double grad) {
        var root = new SimpleMatrix(rows(), cols());
        Arrays.fill(array(root), grad);
        return root;
    }

    public Tensor grad(boolean g) {
        if (g) {
            if (this.grad == null)
                this.grad = zeros(rows(), cols());
        } else {
            this.grad = null;
        }
        return this;
    }

    @Override
    public String toString() {
        var stream = new ByteArrayOutputStream();
        var dd = (DMatrix) (data.getMatrix());
        MatrixIO.print(new PrintStream(stream), dd, "%.4f");
        return stream.toString(Charsets.UTF_8);
        //return "Tensor(data=" + this.data + ", requiresGrad=" + this.hasGrad() + ")";
    }

    public final Tensor siglinear() {
        return siglinear(-4, +4, 0, +1); //-4..+4 resembles sigmoid's range
    }

    /** linear-approximated Sigmoid, inspired by Relu */
    public Tensor siglinear(float xMin, float xMax, float yMin, float yMax) {
        var xRange = xMax - xMin;
        var yRange = yMax - yMin;
        var slope = yRange / xRange;
        return op(
            x -> lerpSafe(unitizeSafe((x - xMin) / xRange), yMin, yMax),
            //x -> x < xMin || x > xMax ? 0 : slope //exclusive
             x -> x <= xMin || x >= xMax ? 0 : slope //inclusive
        );
    }

    public Tensor relu() {
        return op(
                x -> Math.max(x, 0),
                x ->
                    //x > 0 ? 1 : 0 //exclusive
                    x >= 0 ? 1 : 0 //inclusive
        );
    }

    public Tensor reluLeaky(float alpha) {
        return op(
            x -> x > 0 ? x : alpha * x,
            x ->
                //x > 0 ? 1 : alpha //exclusive
                x >= 0 ? 1 : alpha //inclusive
        );
    }

    public final Tensor reluLeaky() {  return reluLeaky(0.01f); }

    public Tensor gelu() {
        return this.mul(0.5).mul(this.add(1).tanh().add(1));
    }

    public final Tensor elu() {
        return elu(1);
    }

    public Tensor elu(double alpha) {
        return op(
            x -> x > 0 ? x : alpha * Math.expm1(x),
            x -> x > 0 ? 1 : alpha * Math.exp(x)
        );
    }

    public Tensor selu() {
        var alpha = 1.6732632423543772848170429916717;
        var scale = 1.0507009873554804934193349852946;
        return op(
                x -> scale * (x > 0 ? x : alpha * Math.expm1(x)),
                x -> scale * (x > 0 ? 1 : alpha * Math.exp(x))
        );
    }

    public Tensor swish() {
        return mul(sigmoid());
    }

    public Tensor mish() {
        return mul(tanh().softplus());
    }

    private Tensor op(DoubleUnaryOperator f, DoubleUnaryOperator fDeriv) {
        return unaryOp(
            x -> TensorUtil.newMatrix(x, f),
            (g,y) -> {
                double[] gg = array(g), yy = array(y), dd = array();
                var n = gg.length;
                for (var i = 0; i < n; i++)
                    yy[i] = gg[i] * fDeriv.applyAsDouble(dd[i]);
            }
        );
    }

    /** L1 loss, mean-absolute-error (MAE)
     *    for continuous data, more robust to outliers than MSE */
    public Tensor mae(Tensor target) {
        return subAbs(target).mean();
    }

    /** L2 loss, mean-squared-error (MSE)
     *     for continuous data, especially when Gaussian noise is assumed. */
    public Tensor mse(Tensor target) {
        return sub(target).sqr().mean();
    }

    public Tensor lenL1() {
        return abs().sum();
    }

    public Tensor lenL2() {
        return sqr().sum().sqrt();
    }

    public final Tensor huber(Tensor target) {
        return huber(target, 1);
    }

    /** δ is a hyperparameter that controls the point where the function transitions from quadratic to linear. */
    public Tensor huber(Tensor target, double delta) {
        var error = this.sub(target);
        var errorAbs = error.abs();

        // For |error| > delta: delta * |error| - 0.5 * delta^2
        var linear = errorAbs.mul(delta).sub(delta * delta / 2);

        // For |error| <= delta: 0.5 * error^2
        var quadratic = error.sqr().div(2);

        // Use quadratic when |error| <= delta, linear when |error| > delta
        return errorAbs.lt(delta).where(quadratic, linear).mean();
    }

    public Tensor abs() {
        var y = new Tensor(data.elementOp((SimpleOperations.ElementOpReal) (r, c, x) -> Math.abs(x)), this.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    double[] g = array(grad), d = array(), out = array(gradOut[0]);
                    var n = g.length;
                    for (var i = 0; i < n; i++)
                        out[i] = g[i] * Util.signum(d[i]);
                }
            };
        }
        return y;
    }

    /** doesn't need gradient handling as it produces a boolean tensor */
    public Tensor lt(double value) {
        var y = new Tensor(new SimpleMatrix(rows(), cols()), false);
        double[] d = array(), out = y.array();
        var n = d.length;
        for (var i = 0; i < n; i++)
            out[i] = d[i] < value ? 1 : 0;
        return y;
    }

    public Tensor where(Tensor trueValues, Tensor falseValues) {
        TensorUtil.assertSameShape(this, trueValues);
        TensorUtil.assertSameShape(this, falseValues);

        var y = new Tensor(new SimpleMatrix(rows(), cols()),
                this.hasGrad() || trueValues.hasGrad() || falseValues.hasGrad());

        double[] condition = array(), trueVals = array(trueValues.data), falseVals = array(falseValues.data);
        var n = condition.length;
        var yy = y.array();
        for (var i = 0; i < n; i++)
            yy[i] = (condition[i] != 0 ? trueVals : falseVals)[i];

        if (y.hasGrad()) {
            y.op = new TensorOp(this, trueValues, falseValues) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var g = array(grad);
                    var tg = gradOut[1] != null ? array(gradOut[1]) : null;
                    var fg = gradOut[2] != null ? array(gradOut[2]) : null;

                    for (var i = 0; i < n; i++) {
                        if (condition[i] != 0) {
                            if (tg != null) tg[i] = g[i];
                        } else {
                            if (fg != null) fg[i] = g[i];
                        }
                        // Note: The condition tensor doesn't receive gradients in this operation
                    }
                }
            };
        }

        return y;
    }

    public final int volume() {
        return data.getNumElements();
    }

    /** Concat operation across columns for tensors with the same number of rows */
    public Tensor concat(Tensor other) {
        var R = rows();
        if (R != other.rows())
            throw new IllegalArgumentException("Tensors must have the same number of rows for concatenation.");

        var X = data.concatColumns(other.data);

        var x = new Tensor(X, hasGrad() || other.hasGrad());
        if (x.hasGrad()) {
            x.op = new TensorOp(this, other) {

                final int C1 = cols(), C2 = other.cols();

                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    backward(g, gradOut[0],0, C1);
                    backward(g, gradOut[1], C1, C2);
                }

                private void backward(SimpleMatrix g, SimpleMatrix go, int offset, int n) {
                    if (go !=null) {
                        //TODO optimize matrix get/set
                        for (var r = 0; r < R; r++)
                            for (var j = 0; j < n; j++)
                                go.set(r, j, g.get(r, j + offset));
                    }
                }
            };
        }
        return x;
    }

    /**
     * Stacks a list of tensors vertically (concatenates along rows).
     * All tensors must have the same number of columns.
     *
     * @param tensors A list of Tensors to stack.
     * @return A new Tensor containing the vertically stacked data.
     */
    public static Tensor concatRows(List<Tensor> tensors) {
        if (tensors == null || tensors.isEmpty()) {
            throw new IllegalArgumentException("Input tensor list cannot be null or empty.");
        }

        var first = tensors.getFirst();
        var cols = first.cols();
        var requiresGrad = tensors.stream().anyMatch(Tensor::hasGrad);

        // Validate that all tensors have the same number of columns.
        for (var i = 1; i < tensors.size(); i++) {
            if (tensors.get(i).cols() != cols) {
                throw new IllegalArgumentException("All tensors must have the same number of columns for row concatenation. " +
                        "Tensor 0 has " + cols + ", but tensor " + i + " has " + tensors.get(i).cols());
            }
        }

        // Calculate total rows and create the new matrix
        var totalRows = tensors.stream().mapToInt(Tensor::rows).sum();
        var resultMatrix = new SimpleMatrix(totalRows, cols);

        // Forward pass: copy data from each tensor into the result matrix
        var currentRow = 0;
        for (var t : tensors) {
            resultMatrix.insertIntoThis(currentRow, 0, t.data);
            currentRow += t.rows();
        }

        var y = new Tensor(resultMatrix, requiresGrad);
        if (requiresGrad) {
            y.op = new TensorOp(tensors.toArray(Tensor.EMPTY_ARRAY)) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    var runningRow = 0;
                    for (var i = 0; i < parents.length; i++) {
                        var parent = parents[i];
                        if (gradOut[i] != null) {
                            // Extract the corresponding slice of the gradient for this parent
                            var gradSlice = g.extractMatrix(runningRow, runningRow + parent.rows(), 0, cols);
                            gradOut[i].plus(gradSlice);
                        }
                        runningRow += parent.rows();
                    }
                }
            };
        }
        return y;
    }

    public Tensor sum() {
        if (isScalar()) return this;

        var y = new Tensor(TensorUtil.eleSum(data), hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    gradOut[0].fill(grad.get(0));
                }
            };
        }
        return y;
    }

    /**
     * Computes the sum of tensor elements along a specified axis.
     *
     * @param axis Axis along which to compute the sum (0 for columns, 1 for rows).
     *             If axis is 0, sums along columns, output is a row vector (1, C).
     *             If axis is 1, sums along rows, output is a column vector (R, 1).
     * @return Tensor containing the summed values.
     */
    public Tensor sum(int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("Axis must be 0 (columns) or 1 (rows).");
        }

        Tensor result;
        if (axis == 0) { // Sum along columns
            result = this.sumCols();
        } else { // Sum along rows
            result = this.sumRows();
        }

        // Replace the TensorOp from sumCols/sumRows with one specific to sum(axis)
        // to ensure the backward pass logic matches the sum(axis) definition precisely.
        if (result.hasGrad()) { // Check if original operation requires grad
             // Keep original parents if any, or just 'this' if sumCols/sumRows was the first op
            var parents = result.op != null ? result.op.parents : new Tensor[]{this};
            if (parents.length == 0 || parents[0] != this) { // Ensure 'this' is the parent
                parents = new Tensor[]{this};
            }

            result.op = new TensorOp(parents) { // parents should be just 'this'
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null) {
                        var parentGrad = gradOut[0]; // Gradient for 'this' tensor
                        var R_parent = Tensor.this.rows(); // Rows of the original tensor
                        var C_parent = Tensor.this.cols(); // Cols of the original tensor

                        parentGrad.reshape(R_parent, C_parent); // Ensure correct shape

                        if (axis == 0) { // Summed over columns (dimension 0), grad is (1, C_parent), parentGrad is (R_parent, C_parent)
                            if (grad.getNumRows() != 1 || grad.getNumCols() != C_parent) {
                                throw new IllegalArgumentException("Gradient shape mismatch for axis 0 sum. Expected (1, " + C_parent + "), got (" + grad.getNumRows() + ", " + grad.getNumCols() + ")");
                            }
                            for (var c = 0; c < C_parent; c++) {
                                var g_c = grad.get(0, c);
                                for (var r = 0; r < R_parent; r++) {
                                    parentGrad.set(r, c, g_c);
                                }
                            }
                        } else { // axis == 1, summed over rows (dimension 1), grad is (R_parent, 1), parentGrad is (R_parent, C_parent)
                            if (grad.getNumRows() != R_parent || grad.getNumCols() != 1) {
                                throw new IllegalArgumentException("Gradient shape mismatch for axis 1 sum. Expected (" + R_parent + ", 1), got (" + grad.getNumRows() + ", " + grad.getNumCols() + ")");
                            }
                            for (var r = 0; r < R_parent; r++) {
                                var g_r = grad.get(r, 0);
                                for (var c = 0; c < C_parent; c++) {
                                    parentGrad.set(r, c, g_r);
                                }
                            }
                        }
                    }
                }
            };
        }
        return result;
    }

    /**
     * @deprecated Prefer {@link #sum(int axis)} where axis=0 for sum over columns (like sumRows in old logic, which was confusing)
     * or axis=1 for sum over rows (like sumCols in old logic).
     * Note: The boolean logic was (true=rowsOrCols -> sumCols -> sum over rows -> output R,1),
     * (false=rowsOrCols -> sumRows -> sum over columns -> output 1,C)
     * sum(0) = sum over columns (axis 0 / R dimension), output (1,C) -> like old sumRows() or sum(false)
     * sum(1) = sum over rows (axis 1 / C dimension), output (R,1) -> like old sumCols() or sum(true)
     */
    @Deprecated
    public final Tensor sum(boolean rowsOrCols) {
        // True for sumCols (sum over rows, R,1), False for sumRows (sum over columns, 1,C)
        // sum(1) -> sum over rows -> sumCols()
        // sum(0) -> sum over columns -> sumRows()
        return rowsOrCols ? sumCols() : sumRows();
    }

    /** Sum along columns, resulting in a row vector */
    public Tensor sumCols() {
        var R = rows();
        if (R==1) return this; //includes scalar
        var C = cols();

        var summed = new SimpleMatrix(1, C);
        for (var j = 0; j < C; j++) {
            double s = 0;
            for (var i = 0; i < R; i++)
                s += data(i, j);
            summed.set(0, j, s);
        }

        var y = new Tensor(summed, hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null)
                        for (var j = 0; j < C; j++) {
                            var gj = grad.get(j);
                            for (var i = 0; i < R; i++)
                                gradOut[0].set(i, j, gj);
                        }
                }
            };
        }
        return y;
    }

    /** Sum along rows, resulting in a column vector */
    public Tensor sumRows() {
        if (isScalar()) return this;
        int R = rows(), C = cols();
        if (C==1) return this;

        var summed = new SimpleMatrix(R, 1);
        for (var i = 0; i < R; i++) {
            var s = 0.0;
            for (var j = 0; j < C; j++)
                s += data(i, j);
            summed.set(i, s);
        }

        var y = new Tensor(summed, hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null)
                        for (var i = 0; i < R; i++) {
                            var gi = grad.get(i);
                            for (var j = 0; j < C; j++)
                                gradOut[0].set(i, j, gi);
                        }
                }
            };
        }
        return y;
    }

    public Tensor pow(double power) {
        //if (p < 0) throw new UnsupportedOperationException();
        if (power == 1) return this;

        var d = array();
        var yy = new SimpleMatrix(rows(), cols());
        var y = array(yy);
        if (power == 0) {
            Arrays.fill(y, 1);
        } else if (power == 2) {
            for (var i = 0; i < y.length; i++)
                y[i] = Util.sqr(d[i]);
        } else {
            for (var i = 0; i < y.length; i++)
                y[i] = Math.pow(d[i], power);
        }
        var t = new Tensor(yy, hasGrad());
        if (t.hasGrad()) {
            t.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    //gradOut[0].setTo(grad.elementMult(data.elementPower(p - 1)).scale(p));
                    var gg = array(grad);
                    var ga = array(gradOut[0]);
                    var n = ga.length;
                    if (power==2) {
                        //optimized quadratic case
                        for (var i = 0; i < n; i++)
                            ga[i] = gg[i] * d[i] * power;
                    } else {
                        for (var i = 0; i < n; i++)
                            ga[i] = gg[i] * Math.pow(d[i], power - 1) * power;
                    }
                }
            };
        }
        return t;
    }

    /**
     * TODO optimize?
     */
    public Tensor sqrt() {
        return pow(0.5);
    }

    public Tensor pow(Tensor exp) {
        if (!exp.isScalar())
            throw new IllegalArgumentException("Exponent must be a scalar Tensor");
        if (!exp.hasGrad())
            return pow(exp.scalar());

        var p = exp.scalar();

        if (p == 1) return this;
        if (p == 0) return ones(rows(), cols());

        var y = data.elementPower(p);
        var t = new Tensor(y, hasGrad() || exp.hasGrad());

        if (t.hasGrad()) {
            t.op = new TensorOp(this, exp) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    double[] dd = array(), gg = array(grad);
                    if (gradOut[0] != null) {
                        // Gradient for base: p * x^(p-1) * grad
                        //gradOut[0].setTo(grad.elementMult(data.elementPower(p - 1)).scale(p));
                        var g0 = array(gradOut[0]);
                        for (var i = 0; i < g0.length; i++)
                            g0[i] = p * Math.pow(dd[i], p-1) * gg[i];
                    }
                    if (gradOut[1] != null) {
//                        // Gradient for exponent: x^p * ln(x) * grad
//                        var logData = data.copy();  var logDataArray = array(logData);
//                        for (var i = 0; i < logDataArray.length; i++) {
//                            var v = logDataArray[i]; logDataArray[i] = v > 0 ? Math.log(v) : 0; // Handle non-positive values
//                        }
//                        gradOut[1].set(0, elementSum(grad.elementMult(y).elementMult(logData)));
                        var g1 = array(gradOut[1]);
                        //TODO kahan sum?
                        double s = 0;
                        var yy = array(y);
                        for (var i = 0; i < g1.length; i++) {
                            var di = dd[i];
                            var li = di > 0 ? Math.log(di) : 0;
                            s += gg[i] * yy[i] * li;
                        }
                        g1[0] = s;
                    }
                }
            };
        }
        return t;
    }

    public Tensor softmax() {
        int rows = rows(), cols = cols();
        var expValues = new SimpleMatrix(rows, cols);

        for (var i = 0; i < rows; i++) {
            var maxVal = Double.NEGATIVE_INFINITY;

            for (var j = 0; j < cols; j++)
                maxVal = Math.max(maxVal, data.get(i, j));

            var sumExp = 0.0;
            for (var j = 0; j < cols; j++) {
                var expVal = Math.exp(data.get(i, j) - maxVal);
                expValues.set(i, j, expVal);
                sumExp += expVal;
            }

            for (var j = 0; j < cols; j++)
                expValues.set(i, j, expValues.get(i, j) / sumExp);
        }

        var y = new Tensor(expValues, this.hasGrad());

        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    var softmaxGrad = gradOut[0];
                    var yData = y.data;
                    for (var i = 0; i < rows; i++) {
                        double dotProduct = 0;
                        for (var j = 0; j < cols; j++)
                            dotProduct += grad.get(i, j) * yData.get(i, j);

                        for (var j = 0; j < cols; j++)
                            softmaxGrad.set(i, j, yData.get(i, j) * (grad.get(i, j) - dotProduct));
                    }
                }
            };
        }

        return y;
    }


    public Tensor sigmoid() {
        return unaryOp(
        x -> TensorUtil.newMatrix(x, Util::sigmoid),
                (t, grad, out) -> {
                    var s = t.array();
                    var n = s.length;
                    var g = array(grad);
                    for (var i = 0; i < n; i++) {
                        var si = s[i];
                        out[i] = si * (1 - si) * g[i];
                    }
            }
        );
    }

    public Tensor tanh() {
        return unaryOp(
                x -> TensorUtil.newMatrix(x, Math::tanh),
                g -> TensorUtil.newMatrix(data, Math::tanh).elementPower(2).scale(-1).plus(1).elementMult(g)
        );
    }

    public Tensor log() {
        var t = new Tensor(data.elementLog(), hasGrad());

        if (t.hasGrad()) {
            t.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    //gradOut[0].setTo(grad.elementDiv(data));
                    double[] gi = array(grad), go = array(gradOut[0]), d = array(data);
                    var n = d.length;
                    for (var i = 0; i < n; i++)
                        go[i] = gi[i] / d[i];
                }
            };
        }
        return t;
    }

    public Tensor div(Tensor other) {
        if (other.isScalar())
            return divScalar(other);

        boolean tg = this.hasGrad(), og = other.hasGrad();
        var t = new Tensor(this.data.elementDiv(other.data), tg || og);
        if (t.hasGrad()) {
            t.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix g, SimpleMatrix[] gradOut) {
                    if (gradOut[0]!=null)
                        gradOut[0].setTo(g.elementDiv(other.data));
                    if (gradOut[1]!=null)
                        gradOut[1].setTo(g.elementMult(data)
                            .elementDiv(other.data.elementMult(other.data))
                            .scale(-1));
                }
            };
        }
        return t;
    }

    public Tensor div(double other) {
        return mul(1.0 / other);
    }

    public Tensor log1p() {
        return unaryOp(
            x -> TensorUtil.newMatrix(x, Util::log1p),
            (Y,G,gradOut)-> {
                double[] d = array(), gradIn = array(G);
                for (var i = 0; i < gradOut.length; i++)
                    gradOut[i] = gradIn[i] / (d[i] + 1);
            }
        );
    }

    /** SoftPlus(x) = log(1 + exp(x))
     *  TODO optimize
     * */
    public Tensor softplus() {
        return exp().log1p();

        //TODO optimize
//        return unaryOp(
//            x -> x.elementExp().plus(1).elementLog(),
//            g -> {
//                var e = data.elementExp();
//                return g.elementMult(e.elementDiv(e.plus(1)));
//            }
//        );
    }

    /** TODO optimize */
    public Tensor clip(double min, double max) {
        TensorUtil.assertClippable(min, max); if (TensorUtil.clipDisabled(min, max)) return this;

        return unaryOp(
            d -> TensorUtil.newMatrix(d, x -> Util.clampSafe(x, min, max)),
            g -> {
                var y = new SimpleMatrix(g.getNumRows(), g.getNumCols());
                double[] cc = array(y), dd = array(), gg = array(g);
                var v = cc.length;
                for (var i = 0; i < v; i++) {
                    var dij = dd[i];
                    //if (dij > min && dij < max) //exclusive
                    if (dij >= min && dij <= max) //inclusive
                        cc[i] = gg[i];
                }
                return y;
            }
        );
    }

    public Tensor clipGrad(double min, double max) {
        TensorUtil.assertClippable(min, max); if (TensorUtil.clipDisabled(min, max)) return this;

        return unaryOp(
                d -> d,
                G -> {
                    var y = new SimpleMatrix(G.getNumRows(), G.getNumCols());
                    double[] go = array(y), gi = array(G);
                    var v = go.length;
                    for (var j = 0; j < v; j++)
                        go[j] = clampSafe(gi[j], min, max);
                    return y;
                }
        );
    }

    public final Tensor clipSigmoid(double min, double max) {
        //return clipSigmoid(min, max, 1);
        return sigmoid().mul(max-min).add(min);
    }

//    /** TODO test */
//    public Tensor clipSigmoid(double min, double max, double sharpness) {
//        TensorUtil.assertClippable(min, max); if (TensorUtil.clipDisabled(min, max)) return this;
//        var range = max - min;
//        var center = (min + max) / 2;
//        var scale = sharpness / range;  // Adjust this for steepness of the sigmoid
//        return this.sub(center).mul(scale).sigmoid().mul(range).add(min);
//    }

    public final Tensor clipTanh() {
        return clipTanh(-1, +1);
    }

    public Tensor clipTanh(double yMin, double yMax) {
        return tanh().mul((yMax-yMin)/2).add(yMin);
    }

    public Tensor mean() {
        return sum().div(volume());
    }

    /**
     * Computes the mean of tensor elements along a specified axis.
     *
     * @param axis Axis along which to compute the mean (0 for columns, 1 for rows)
     * @return Tensor containing the mean values
     */
    public Tensor mean(int axis) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("Axis must be 0 (columns) or 1 (rows)");
        }

        var R = rows();
        var C = cols();
        Tensor summed;
        int N; // Number of elements summed over

        if (axis == 0) { // Mean along columns (output is row vector)
            if (R == 1) return this; // Already a row vector or scalar
            summed = sumCols();
            N = R;
        } else { // Mean along rows (output is column vector)
            if (C == 1) return this; // Already a column vector or scalar
            summed = sumRows();
            N = C;
        }

        if (N == 0) { // Avoid division by zero if a dimension is zero
            return zerosShaped(summed); // Or handle as an error
        }

        var result = summed.div(N); // div by scalar N

        // Adjust gradient propagation for the division by N
        // The 'div(double)' operation already handles this if N is treated as a constant (no grad for N)
        // The backward pass for 'summed.div(N)' will be:
        // grad_summed = incoming_grad / N
        // This grad_summed then flows back through the sum operation.

        return result;
    }

    public Tensor clipData(double min, double max) {
        clamp(array(), min, max);
        return this;
    }

    public Tensor setData(SimpleMatrix m) {
        if (data != null && sameShape(m))
            data.setTo(m); //copies
        else
            data = m;
        return this;
    }

    public final Tensor setGrad(Tensor gradient) {
        return setGrad(gradient.data);
    }

    public Tensor setGrad(SimpleMatrix gradient) {
        if (grad != null)
            grad.setData(gradient);
        else
            grad = new Tensor(gradient, false);
        return this;
    }

    public String stats() {
        return String.format("Min: %f, Max: %f", data.elementMin(), data.elementMax());
    }

    public Tensor normalizeL2() {
        var ll = lenL2().scalar();
        return mul(ll / volume() < Double.MIN_VALUE ? 0 : 1 / ll);
    }

    public Tensor rowNormsL2() {
        var R = rows();
        var C = cols();
        var y = zeros(R, 1); // Create a new Tensor of shape [R, 1] to hold the L2 norms for each row

        for (var r = 0; r < R; r++) {
            double sumSq = 0;
            for (var c = 0; c < C; c++)
                sumSq += Util.sqr(data(r, c));
            y.data.set(r, 0, Math.sqrt(sumSq));
        }

        return y;
    }

    public void setData(Tensor tensor) {
        setData(tensor.data);
    }

    public void mulThis(double x) {
        jcog.Util.mul(array(), x);
    }

    public void addThis(double[] x) {
        addTo(array(), x);
    }

    /** TODO optimize */
    public final Tensor sqr() {
        return pow(2);
    }

    public final void addGrad(Tensor x) {
        addTo(grad.array(), x.array());
    }

    public void addGrad(SimpleMatrix x) {
        addTo(grad.array(), array(x));
    }

    public final Tensor setGrad(double[][] x) {
        return setGrad(new SimpleMatrix(x));
    }

    public double sumAbs() {
        return TensorUtil.sumAbs(data);
    }

    public double sumSqr() {
        return TensorUtil.sumSqr(data);
    }

    public double maxAbs() {
        return data.elementMaxAbs();
    }

    public void setSoft(Tensor src, double tau) {
        setData(tau == 1 ? src.data : this.data.scale(1 - tau).plus(src.data.scale(tau)));
    }

    public Tensor detach() {
        grad = null;
        op = null;
        return this;
    }

    public Tensor slice(int col) {
        return slice(col, col+1);
    }

    public Tensor slice(int colStart, int colEnd) {
        return slice(0, rows(), colStart, colEnd);
    }

    public Tensor slice(int rowStart, int rowEnd, int colStart, int colEnd) {
        if (colStart < 0 || colEnd > this.cols() || colStart >= colEnd)
            throw new IllegalArgumentException("Invalid start or end column indices for slice");
        if (rowStart < 0 || rowEnd > this.rows() || rowStart >= rowEnd)
            throw new IllegalArgumentException("Invalid start or end row indices for slice");

        var t = new Tensor(this.data.extractMatrix(rowStart, rowEnd, colStart, colEnd), hasGrad());
        if (t.hasGrad())
            t.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    gradOut[0].insertIntoThis(rowStart, colStart, grad);
                }
            };
        return t;
    }

    public final Tensor clipUnitPolar() {
        return clip(-1, +1);
    }

    public Tensor fill(double x) {
        this.data.fill(x);
        return this;
    }

    /** sets as parameter */
    public final Tensor parameter() {
        this.parameter = true;
        return this;
    }

    public Tensor detachCopy() {
        return hasGrad() ? new Tensor(data, false) : this;
    }

    public Tensor klDivergence(Tensor prob) {
        return this.mul(this.div(prob).log());
//        var dLogProb = sub(prob);
//        return dLogProb.exp().sub(1).sub(dLogProb);
    }

    public void recurse(Consumer<Tensor> t) {
        t.accept(this);

        if (op != null)
            for (var p : op.parents)
                p.recurse(t);
    }

    /** TODO optimize with CommonOps_DDRM  */
    public Tensor matmulTranspose(Tensor x) {
        return matmul(x.transpose());
    }

    /** TODO optimize with CommonOps_DDRM */
    public Tensor transposeMatmul(Tensor x) {
        return transpose().matmul(x);
    }

    public int generation() {
        return op==null ? -1 : op.generation;
    }

    /**
     * Gather values from this tensor according to the indices provided
     */
    public Tensor gather(int dim, int[] indices) {
        if (dim != 1)
            throw new IllegalArgumentException("Currently only supports dim=1 and indices as a column vector");

        var R = indices.length;
        var y = new Tensor(R, 1, this.hasGrad());
        for (var i = 0; i < R; i++)
            y.data.set(i, 0, this.data(i, indices[i]));

        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null) {
                        for (var i = 0; i < R; i++) {
                            var index = indices[i];
                            gradOut[0].set(i, index, gradOut[0].get(i, index) + grad.get(i, 0));
                        }
                    }
                }
            };
        }

        return y;
    }

    /** TODO use broadcasting? */
    @Deprecated public Tensor expandToMatch(int rows, int colsPerK, int K) {
        // Create a new tensor with the desired shape [N, K, D_out/K]
        var expanded = new Tensor(rows, K * colsPerK, this.hasGrad());

        // Copy gate weights into the expanded tensor
        for (var i = 0; i < rows; i++) {
            for (var k = 0; k < K; k++) {
                for (var j = 0; j < colsPerK; j++) {
                    expanded.data.set(i, k * colsPerK + j, this.data(i, k));
                }
            }
        }

        // Handle gradients if required
        if (expanded.hasGrad()) {
            expanded.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    for (var i = 0; i < rows; i++) {
                        for (var k = 0; k < K; k++) {
                            double sum = 0;
                            for (var j = 0; j < colsPerK; j++) {
                                sum += grad.get(i, k * colsPerK + j);
                            }
                            gradOut[0].set(i, k, sum);
                        }
                    }
                }
            };
        }

        return expanded;
    }

    /**
     * Reshape the tensor to the specified dimensions
     * untested: TODO test
     */
    public Tensor reshape(int... dims) {
        if (dims.length == 2)
            return reshape(dims[0], dims[1]);  // Use the existing 2D reshape

        var totalElements = 1; for (var dim : dims) totalElements *= dim;
        var E = totalElements;

        if (E != this.volume())
            throw new IllegalArgumentException("New shape must have the same number of elements as the original tensor");

        var y = new Tensor(totalElements, 1, this.hasGrad());
        System.arraycopy(array(), 0, y.array(), 0, totalElements);

        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null)
                        System.arraycopy(array(grad), 0, array(gradOut[0]), 0, E);
                }
            };
        }

        return y.reshape(dims[0], totalElements / dims[0]);  // Reshape to 2D, then use existing reshape
    }

    public final Tensor loss(Tensor target, Loss loss) {
        return switch (loss) {
            case Huber -> huber(target);
            case MeanSquared -> mse(target);
            case SubAbs -> subAbs(target);
        };
    }

    /**
     * Computes the variance of tensor elements.
     * For 2D tensors, this computes variance across all elements.
     * Uses a numerically stable single-pass algorithm.
     *
     * @return Tensor containing the variance (scalar)
     */
    public Tensor variance() {
        if (isScalar())
            return scalar(0.0);

        var n = volume();
        var d = array();

        // Use Welford's online algorithm for numerical stability
        var mean = 0.0;
        var m2 = 0.0;

        for (var i = 0; i < n; i++) {
            var x = d[i];
            var delta = x - mean;
            mean += delta / (i + 1);
            var delta2 = x - mean;
            m2 += delta * delta2;
        }

        var variance = m2 / (n - 1); // Use unbiased estimator (n-1)
        var y = new Tensor(variance, this.hasGrad());

        if (y.hasGrad()) {
            var MEAN = mean;
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null) {
                        var g = grad.get(0);
                        var scale = 2.0 / (n - 1);
                        // Gradient of variance with respect to inputs
                        for (var i = 0; i < n; i++)
                            gradOut[0].set(i, (d[i] - MEAN) * scale * g);
                    }
                }
            };
        }

        return y;
    }

    /**
     * Computes the variance of tensor elements along a specified axis.
     * Uses an unbiased estimator (divides by N-1).
     *
     * @param axis Axis along which to compute the variance (0 for columns, 1 for rows)
     * @return Tensor containing the variance values
     */
    public Tensor variance(int axis) {
        return variance(axis, true);
    }

    /**
     * Computes the variance of tensor elements along a specified axis.
     *
     * @param axis Axis along which to compute the variance (0 for columns, 1 for rows)
     * @param unbiased If true, uses unbiased estimator (N-1 divisor), otherwise N.
     * @return Tensor containing the variance values
     */
    public Tensor variance(int axis, boolean unbiased) {
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("Axis must be 0 (columns) or 1 (rows)");
        }

        var R = rows();
        var C = cols();
        int N;
        Tensor expectedShapeZeros; // Used for returning early with zeros

        if (axis == 0) { // Variance along columns (operates on each column), output is (1,C)
            N = R; // Number of elements in each column
            expectedShapeZeros = zeros(1, C);
        } else { // Variance along rows (operates on each row), output is (R,1)
            N = C; // Number of elements in each row
            expectedShapeZeros = zeros(R, 1);
        }

        if (N == 0) {
            return expectedShapeZeros.fill(Double.NaN); // Variance of zero elements is undefined
        }

        if (unbiased) {
            if (N == 1) {
                // Unbiased variance for N=1 (X-X_mean)^2 / (1-1) is undefined (div by 0)
                // Common practice is to return 0 or NaN. NaN is more informative.
                return expectedShapeZeros.fill(Double.NaN);
            }
        } else { // Biased variance
            if (N == 1) {
                // Biased variance for N=1 (X-X_mean)^2 / 1 is 0, since X_mean = X
                return expectedShapeZeros.fill(0.0);
            }
        }
        // At this point, N > 1 for unbiased, or N >= 1 for biased (N=1 case for biased handled)
        // N cannot be 0.

        var mean_ax = this.mean(axis); // (1,C) if axis=0, or (R,1) if axis=1
                                         // 'this' is (R,C)
                                         // mean_ax will be broadcast correctly by sub

        var diff = this.sub(mean_ax);  // Broadcasting sub: (R,C) - (1,C) -> (R,C)
                                         // or (R,C) - (R,1) -> (R,C)
        var diff_sq = diff.sqr();      // (R,C), element-wise square

        Tensor sum_sq_diff;
        if (axis == 0) { // sum down columns
            sum_sq_diff = diff_sq.sumCols(); // Result is (1,C)
        } else { // sum across rows
            sum_sq_diff = diff_sq.sumRows(); // Result is (R,1)
        }

        var divisor = unbiased ? (double)(N - 1) : (double)N;
        // This divisor should not be zero based on the checks above (N > 1 for unbiased, N >=1 for biased)

        return sum_sq_diff.div(divisor);
    }

    /**
     * Computes the standard deviation of tensor elements.
     * This computes standard deviation across all elements, using unbiased variance.
     *
     * @return Tensor containing the standard deviation (scalar)
     */
    public Tensor std() {
        // variance() already uses (n-1) for all elements
        return variance().sqrt();
    }

    /**
     * Computes the standard deviation of tensor elements along a specified axis.
     * Uses an unbiased estimator for variance (N-1).
     *
     * @param axis Axis along which to compute the standard deviation (0 for columns, 1 for rows)
     * @return Tensor containing the standard deviation values
     */
    public Tensor std(int axis) {
        // variance(axis, true) uses unbiased estimator
        return variance(axis, true).sqrt();
    }


    /**
     * Finds the indices of the maximum values along a specified axis.
     * This operation is not differentiable and will throw an error if grads are required.
     *
     * @param axis Axis along which to find the maximum values (0 for columns, 1 for rows).
     * @return Tensor containing the indices of the maximum values. Shape will be (1, C) if axis=0, or (R, 1) if axis=1.
     */
    public Tensor argmax(int axis) {
        if (hasGrad()) {
            throw new UnsupportedOperationException("argmax is not differentiable and cannot be used if gradients are required.");
        }
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("Axis must be 0 (columns) or 1 (rows).");
        }

        var R = rows();
        var C = cols();

        if (R == 0 || C == 0) {
            throw new IllegalArgumentException();
//            // Handle empty tensor case: return an empty tensor of the expected shape.
//            if (axis == 0) return new Tensor(0, C, false); // Shape (0,C) or (1,C) with 0 rows? Let's do (1,C) filled with 0 or NaN if C > 0.
//            else return new Tensor(R, 0, false); // Shape (R,0) or (R,1) with 0 cols? Let's do (R,1) filled with 0 or NaN if R > 0.
//            // For simplicity, if tensor is empty (R=0 or C=0), the result is also "empty" in a sense.
//            // If R=0, axis=0 -> (1,C) is problematic. Let's return (1,C) if C>0, else (0,0).
//            // If C=0, axis=1 -> (R,1) is problematic. Let's return (R,1) if R>0, else (0,0).
//            // Simplest for now: if R or C is 0, output is zeros of target shape, which might be (0,C) or (R,0).
//            // Or more concretely, if axis = 0, output is (1,C). If R=0, values are undefined.
//            // If axis = 1, output is (R,1). If C=0, values are undefined.
//            // if (axis == 0) return zeros(1,C); // if R=0, this is still (1,C) - indices are effectively 0.
//            //else return zeros(R,1); // if C=0, this is still (R,1) - indices are effectively 0.
        }

        Tensor result;
        if (axis == 0) { // Along columns, output is (1, C)
            result = new Tensor(1, C, false);
            for (var j = 0; j < C; j++) { // Iterate through each column
                var maxVal = Double.NEGATIVE_INFINITY;
                var maxIdx = 0; // Default to 0 if all are -INF or column is empty (though caught by R=0)
                for (var i = 0; i < R; i++) { // Iterate through rows of current column
                    var val = data(i, j);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
                result.data.set(0, j, maxIdx);
            }
        } else { // Along rows, output is (R, 1)
            result = new Tensor(R, 1, false);
            for (var i = 0; i < R; i++) { // Iterate through each row
                var maxVal = Double.NEGATIVE_INFINITY;
                var maxIdx = 0; // Default to 0 if all are -INF or row is empty (though caught by C=0)
                for (var j = 0; j < C; j++) { // Iterate through columns of current row
                    var val = data(i, j);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
                result.data.set(i, 0, maxIdx);
            }
        }
        return result;
    }

    /**
     * Finds the indices of the minimum values along a specified axis.
     * This operation is not differentiable and will throw an error if grads are required.
     *
     * @param axis Axis along which to find the minimum values (0 for columns, 1 for rows).
     * @return Tensor containing the indices of the minimum values. Shape will be (1, C) if axis=0, or (R, 1) if axis=1.
     */
    public Tensor argmin(int axis) {
        if (hasGrad()) {
            throw new UnsupportedOperationException("argmin is not differentiable and cannot be used if gradients are required.");
        }
        if (axis != 0 && axis != 1) {
            throw new IllegalArgumentException("Axis must be 0 (columns) or 1 (rows).");
        }

        var R = rows();
        var C = cols();

        if (R == 0 || C == 0) {
            // Consistent with argmax: if R or C is 0, output is zeros of target shape.
             if (axis == 0) return zeros(1,C);
             else return zeros(R,1);
        }

        Tensor result;
        if (axis == 0) { // Along columns, output is (1, C)
            result = new Tensor(1, C, false);
            for (var j = 0; j < C; j++) { // Iterate through each column
                var minVal = Double.POSITIVE_INFINITY;
                var minIdx = 0;
                for (var i = 0; i < R; i++) { // Iterate through rows of current column
                    var val = data(i, j);
                    if (val < minVal) {
                        minVal = val;
                        minIdx = i;
                    }
                }
                result.data.set(0, j, minIdx);
            }
        } else { // Along rows, output is (R, 1)
            result = new Tensor(R, 1, false);
            for (var i = 0; i < R; i++) { // Iterate through each row
                var minVal = Double.POSITIVE_INFINITY;
                var minIdx = 0;
                for (var j = 0; j < C; j++) { // Iterate through columns of current row
                    var val = data(i, j);
                    if (val < minVal) {
                        minVal = val;
                        minIdx = j;
                    }
                }
                result.data.set(i, 0, minIdx);
            }
        }
        return result;
    }

    /**
     * Creates a boolean mask tensor where elements are 1 if greater than threshold,
     * 0 otherwise
     *
     * @param threshold Value to compare against
     * @return New tensor containing the boolean mask
     */
    public Tensor gt(double threshold) {
        var y = new Tensor(rows(), cols(), false);
        var x = array();
        var yy = y.array();
        for (var i = 0; i < x.length; i++)
            yy[i] = x[i] > threshold ? 1 : 0;
        return y;
    }

    public Tensor signum() {
        var d = array();
        var y = new Tensor(arrayOf(x -> Util.signum(d[x]), new double[volume()]), this.hasGrad());

        // If gradient tracking is enabled, define the backward operation
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    // The gradient of signum is zero almost everywhere, so the backward pass results in zero gradients
                    // It will already be zero.
                    //if (gradOut[0] != null) gradOut[0].zero();
                }
            };
        }

        return y;
    }

    public Tensor gradDebug() {
        var debug = true;

        var y = new Tensor(data, hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradParent) {
                    var g = gradParent[0];
                    var gg = array(g);
                    var nonFinite = 0;
                    double[] g0 = null;
                    for (var i = 0; i < gg.length; i++) {
                        if (!Float.isFinite(i)) {
                            if (g0 == null && debug)
                                g0 = gg.clone();
                            nonFinite++;
                            gg[i] = 0;
                        }
                    }
                    if (g0!=null) {
                        System.err.println(Tensor.this + " gradient nonFinite=" + nonFinite + "\n:\t" + Str.n4(g0));
                    }

                    grad.setTo(g);
                }
            };
        }
        return y;



    }

    public Tensor dot(Tensor x) {
//        if (x.hasGrad())
//            throw new TODO();

        sameShape(x);

        var a = array();
        var b = x.array();
        var n = a.length;
        double sum = 0;
        for (var i = 0; i < n; i++)
            sum += a[i] * b[i];

        return scalar(sum);
    }

    public Tensor setDataRow(int row, Tensor x) {
        data.setRow(row, x.data);
        return this;
    }

    /**
     * Flatten to a row: [R, C] -> [1, R*C]
     */
    public final Tensor flattenRow() {
        return reshape(1, volume());
    }

    /**
     * Flatten to a column: [R, C] -> [R*C, 1]
     */
    public final Tensor flattenCol() {
        return reshape(volume(), 1);
    }

    public int[] shape() {
        return new int[] { rows(), cols() };
    }

    public enum Loss {
        Huber, MeanSquared /* L2 */, SubAbs /* L1 */
    }

    public static class Optimizer {
        public final List<Optimizers.OptimizerStep> step = new Lst<>();

        public Optimizer(Optimizers.OptimizerStep... steps) {
            for (var s : steps) if (s!=null) step.add(s);
        }

        public final void run(Stream<Tensor> tt) {
            var p = tt
                    .filter(t -> t.parameter)
                    .sorted(GradQueue.TensorSorter)
                    .toList();

            if (!p.isEmpty()) {
                _run(p);
                p.forEach(Tensor::zeroGrad);
            }
        }

        protected void _run(List<Tensor> p) {
            for (var s : step)
                s.accept(p);
        }

    }

    public static class GradQueue implements Iterable<Tensor> {
        private static final Comparator<Tensor> TensorSorter =
            Comparator.comparingInt(System::identityHashCode);

        final ConcurrentHashMap<Tensor, Queue<double[]>> grads = new ConcurrentHashMap<>(512);

        private double factor = 1;

        public void clear() {
            grads.clear();
        }

        public void addGrad(Tensor x, double[] y) {
            grads.computeIfAbsent(x, _x -> new ConcurrentLinkedQueue<>()).add(y);
        }

        public void optimize(@Nullable Optimizer o) {
            synchronized(this) {
                grads.entrySet()
                    .forEach(e -> e.getKey().grad.addToThis(e.getValue()));
                    //.parallelStream().unordered() //NOT THREAD SAFE YET
                    //.stream()
                    //.sorted(Comparator.comparingInt(x -> x.getValue().stream().mapToInt(z -> z.length).sum()))
                    //.toList();

                if (factor != 1)
                    grads.keySet().forEach(g -> Util.mul(g.grad.array(), factor));

                if (o != null)
                    o.run(grads.keySet().stream());

                clear();
            }
        }

        @Override
        public Iterator<Tensor> iterator() {
            return grads.keySet().iterator();
        }

        public GradQueue mul(double m) {
            factor = m;
            return this;
        }

        public GradQueue div(double d) {
            return mul(1.0/d);
        }
    }

    abstract public static class TensorOp {
        private static final BiFunction<SimpleMatrix, SimpleMatrix, SimpleMatrix> tensorGradMerge = TensorUtil::_addTo;
        public final Tensor[] parents;
        public final int generation;
        public int color = -1;

        public TensorOp(Tensor... parents) {
            if (parents.length < 1)
                throw new UnsupportedOperationException();
            this.parents = parents;
            this.generation = ((int) Util.max(Tensor::generation, parents)) + 1;
        }

        public final SimpleMatrix[] allocateParentGrads() {
            return map(p -> p.hasGrad() ? new SimpleMatrix(p.rows(), p.cols()) : null, new SimpleMatrix[parents.length], parents);
        }

        public abstract void backward(SimpleMatrix grad, SimpleMatrix[] gradParent);

        public final void gradients(Tensor next, Map<Tensor, SimpleMatrix> grads, PriorityQueue<Tensor> q) {

            var parentGrads = allocateParentGrads();

            backward(grads.get(next), parentGrads);

            var n = parents.length;
            for (var i = 0; i < n; i++) {
                var pg = parentGrads[i];
                if (pg != null) {
//                    if (isZero(gi)) continue;
                    var p = parents[i];
                    if (p.op != null) {
                        if (!grads.containsKey(p))
                            q.add(p); //queue for analysis
                    }

                    grads.merge(p, pg, tensorGradMerge);
                    parentGrads[i] = null; //GC help
                }
            }
        }
    }


    /** TODO */
    public static class NoGrad implements AutoCloseable {
        @Override
        public void close() {
            //TODO
        }
    }

    // Add this method to your Tensor class, or a Tensor utility class.

    /**
     * Initializes the weight tensor with an orthogonal matrix, scaled by a gain factor.
     * <p>
     * Orthogonal initialization is a common technique for training deep neural networks,
     * as it helps prevent gradients from exploding or vanishing by preserving the norm
     * of activations and gradients during backpropagation.
     * <p>
     * This implementation uses the Modified Gram-Schmidt process on a random Gaussian
     * matrix to generate the orthogonal matrix, which is more numerically stable than
     * the classical version.
     *
     * @param W    The tensor (matrix) to initialize. Its data will be overwritten.
     * @param gain The scaling factor (gain) to apply to the orthogonal matrix. A gain of
     *             sqrt(2.0) is common for layers followed by a ReLU activation.
     */
    public static void orthoInit(Tensor W, double gain) {
        var R = W.rows();
        var C = W.cols();
        if (R * C == 0) {
            return; // Nothing to initialize
        }

        if (R < C) {
            // For wide matrices (rows < cols), it's more straightforward to initialize
            // the transpose and then copy the result back.
            var WT = new Tensor(C, R, false);
            _orthoInit(WT, gain);

            // Copy the transposed result back to W
            var wData = W.array();
            var wtData = WT.array();
            for (var r = 0; r < R; r++) {
                for (var c = 0; c < C; c++) {
                    wData[r * C + c] = wtData[c * R + r];
                }
            }
        } else {
            // For tall or square matrices (rows >= cols)
            _orthoInit(W, gain);
        }
    }

    /**
     * Internal implementation of orthogonal initialization for tall matrices (rows >= cols).
     * This method performs the Modified Gram-Schmidt process.
     *
     * @param Q    The tensor to fill with orthogonal columns.
     * @param gain The scaling factor.
     */
    private static void _orthoInit(Tensor Q, double gain) {
        var rows = Q.rows();
        var cols = Q.cols();

        // 1. Create a temporary matrix with random data from a standard normal distribution.
        // We will perform the orthogonalization in-place on this data.
        var temp = Tensor.randGaussian(rows, cols, 1.0);
        var qData = temp.array();

        // 2. Perform Modified Gram-Schmidt on the columns of the temporary matrix.
        for (var j = 0; j < cols; j++) {
            // a) Calculate the L2 norm of the j-th column vector.
            var normSq = 0.0;
            for (var i = 0; i < rows; i++) {
                var val = qData[i * cols + j];
                normSq += val * val;
            }
            var norm = Math.sqrt(normSq);

            // b) Normalize the j-th column vector (it is now an orthonormal vector q_j).
            // A small epsilon is used to prevent division by zero, although this is
            // extremely unlikely with random data.
            if (norm > 1e-12) {
                var invNorm = 1.0 / norm;
                for (var i = 0; i < rows; i++) {
                    qData[i * cols + j] *= invNorm;
                }
            }
            // If norm is zero, the column is already zero, and projections onto it will be zero.

            // c) Make all subsequent column vectors (v_k) orthogonal to q_j.
            for (var k = j + 1; k < cols; k++) {
                // Project v_k onto q_j: dot_product = q_j^T * v_k
                var dotProduct = 0.0;
                for (var i = 0; i < rows; i++) {
                    dotProduct += qData[i * cols + j] * qData[i * cols + k];
                }

                // Subtract the projection from v_k: v_k = v_k - dot_product * q_j
                for (var i = 0; i < rows; i++) {
                    qData[i * cols + k] -= dotProduct * qData[i * cols + j];
                }
            }
        }

        // 3. Copy the resulting orthogonal matrix data to the target tensor and apply the gain.
        var wData = Q.array();
        for (var i = 0; i < wData.length; i++)
            wData[i] = qData[i] * gain;
    }

}
