package jcog.tensor;

import com.google.common.base.Charsets;
//import jcog.Str;
//import jcog.Util; // Keep specific static imports if used after replacement
//import jcog.data.bit.MetalBitSet;
//import jcog.data.list.Lst;
//import jcog.func.TriConsumer; // Will be replaced by a local interface or commented out
import java.util.Random; // For XoRoShiRo128PlusRandom replacement
//import jcog.random.XoRoShiRo128PlusRandom; // No longer needed
//import jcog.util.Reflect;
import org.eclipse.collections.api.block.function.primitive.DoubleDoubleToDoubleFunction;
import org.ejml.concurrency.EjmlConcurrency;
import org.ejml.data.DMatrix;
import java.util.ArrayList; // For Lst replacement
import java.util.BitSet;    // For MetalBitSet replacement
import java.util.Arrays;    // For Arrays.fill, Arrays.stream etc.
import java.util.Objects;   // For Objects.requireNonNullElseGet
import java.lang.Math;      // For Math.max, Math.min, Math.abs, Math.signum, Math.sqrt, Math.pow, Math.exp, Math.log, Math.log1p, Math.tanh
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.MatrixType;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.ConstMatrix;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;
import org.jetbrains.annotations.Nullable;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.*;
import java.util.stream.Stream;

//import static jcog.Util.*; // Comment this out
import static jcog.tensor.TensorUtil.assertClippable;

public class Tensor {

    public static final Tensor[] EMPTY_ARRAY = new Tensor[0];
    public static final UnaryOperator<Tensor> RELU = Tensor::relu;
    public static final UnaryOperator<Tensor> RELU_LEAKY = Tensor::reluLeaky;
    public static final UnaryOperator<Tensor> SIGMOID = Tensor::sigmoid;
    private static final Comparator<Tensor> generationComparator = Comparator.comparingInt(t -> -t.generation());

    // Local functional interface to replace jcog.func.TriConsumer
    @FunctionalInterface
    interface MyTriConsumer<T, U, V> {
        void accept(T t, U u, V v);
    }

    static {
        EjmlConcurrency.USE_CONCURRENT = false;
        EjmlConcurrency.ELEMENT_THRESHOLD = 5_000;
    }

    public SimpleMatrix data;
    @Nullable
    public Tensor grad;
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

    public static Tensor randGaussian(int rows, int cols, double stddev) {
        var rng = new Random(); // Replaced XoRoShiRo128PlusRandom
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

    /*
    public static Stream<Tensor> parameters(Object x) {
        return Reflect.on(x).fieldsRecursive(true, false, false,
            (fieldName, obj, parent)-> obj instanceof Tensor t && t.parameter)
            .stream().map(y -> (Tensor)((Reflect)y).object);
    }
    */

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
        var rng = new Random(); // Replaced XoRoShiRo128PlusRandom
        var data = new double[rows * cols];
        for (var i = 0; i < data.length; i++)
            data[i] = rng.nextDouble();
        return new Tensor(data, rows, cols, false);
    }

    public static Tensor fill(int rows, int cols, double val) {
        return zeros(rows, cols).fill(val);
    }


    /** TODO optional Kahan summing? */
    private void addToThis(Queue<double[]> v) {
        var x = array();
        var length = x.length;
        var yy = v.iterator();
        while (yy.hasNext()) {
            var y = yy.next();
            yy.remove();
            for (int i = 0; i < x.length; i++) {
                 x[i] += y[i];
            }
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

    String shapeStr() {
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

        var maxIndices = new BitSet(volume());

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
                        for (int i = maxIndices.nextSetBit(0); i >= 0; i = maxIndices.nextSetBit(i + 1))
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
                           MyTriConsumer<Tensor, SimpleMatrix, double[]> backward) { // Replaced TriConsumer with MyTriConsumer
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
            yy[i] = xx[i] * t + d[i];

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

        if (other.isScalar())
            return addScalar(other.neg());

        // Z = A - B
        // dZ/dA = 1  => gradA = gradZ * 1
        // dZ/dB = -1 => gradB = gradZ * -1
        var y = new Tensor(this.data.minus(other.data), this.hasGrad() || other.hasGrad());
        if (y.hasGrad()) {
            y.op = new TensorOp(this, other) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    if (gradOut[0] != null) { // Gradient for 'this' (A)
                        // gradOut[0].setTo(grad) would overwrite, ensure accumulation if gradOut[0] can have pre-existing values
                        double[] g = array(grad);
                        double[] go = array(gradOut[0]);
                        for (int i = 0; i < g.length; i++) {
                            go[i] += g[i]; // Accumulate
                        }
                    }
                    if (gradOut[1] != null) { // Gradient for 'other' (B)
                        double[] g = array(grad);
                        double[] go = array(gradOut[1]);
                        for (int i = 0; i < g.length; i++) {
                            go[i] += -g[i]; // Accumulate
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

        SimpleMatrix diffMatrix = data.minus(x.data);
        double[] diffArray = array(diffMatrix);
        double[] absArray = new double[diffArray.length];
        for (int i = 0; i < diffArray.length; i++) {
            absArray[i] = Math.abs(diffArray[i]);
        }
        var t = new Tensor(new SimpleMatrix(rows(), cols(), true, absArray), thg || ohg);
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
                        var g01 = Math.signum(diff) * gg[i];
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
            throw new IllegalArgumentException("The 'A' and 'B' matrices do not have compatible dimensions");

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

    /*
    public final Tensor maximize() {
        return neg().minimize();
    }
    */

    /*
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
        //var tensors = context==null ? backwardSerial(g) : backwardSerialBuffered(g, context);
        var tensors = backwardSerial(g); // Removed GradQueue dependent path
        //var tensors = backwardParallel();

        if (opt != null)
            opt.run(tensors);

        return this;
    }

    private Stream<Tensor> backwardSerial(Map<Tensor, SimpleMatrix> grads) {
        grads.forEach(Tensor::addGrad);
        return grads.keySet().stream();
    }

    // private Stream<Tensor> backwardSerialBuffered(Map<Tensor, SimpleMatrix> grads, GradQueue context) {
    //     for (var e : grads.entrySet()) {
    //         var t = e.getKey();
    //         var g = e.getValue();
    //         if (t.parameter)
    //             context.addGrad(t, array(g));
    //         else
    //             t.addGrad(g);
    //     }
    //     return grads.keySet().stream();
    // }
    */

    /** untested */
    /*
    private Iterable<Tensor> backwardParallel() {
        return ParallelBackward.backward(this);
    }
    */

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

    /*
    public final Tensor siglinear() {
        return siglinear(-4, +4, 0, +1); //-4..+4 resembles sigmoid's range
    }

    */
    /** linear-approximated Sigmoid, inspired by Relu */
    /*
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
    */

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
                        out[i] = g[i] * Math.signum(d[i]);
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

    public final Tensor sum(boolean rowsOrCols) {
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
                y[i] = d[i] * d[i];
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
        x -> TensorUtil.newMatrix(x, val -> 1.0 / (1.0 + Math.exp(-val))),
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
            x -> TensorUtil.newMatrix(x, Math::log1p),
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
        assertClippable(min, max); if (TensorUtil.clipDisabled(min, max)) return this;

        return unaryOp(
            d -> TensorUtil.newMatrix(d, x_val -> {
                if (x_val < min) return min;
                if (x_val > max) return max;
                return x_val;
            }),
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
        assertClippable(min, max); if (TensorUtil.clipDisabled(min, max)) return this;

        return unaryOp(
                d -> d,
                G -> {
                    var y = new SimpleMatrix(G.getNumRows(), G.getNumCols());
                    double[] go = array(y), gi = array(G);
                    var v = go.length;
                    for (var j = 0; j < v; j++) {
                        double val_gi_j = gi[j];
                        if (val_gi_j < min) go[j] = min;
                        else if (val_gi_j > max) go[j] = max;
                        else go[j] = val_gi_j;
                    }
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

    public Tensor clipData(double min, double max) {
        double[] arr = array();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] < min) arr[i] = min;
            else if (arr[i] > max) arr[i] = max;
        }
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
                sumSq += data(r,c) * data(r,c);
            y.data.set(r, 0, Math.sqrt(sumSq));
        }

        return y;
    }

    public void setData(Tensor tensor) {
        setData(tensor.data);
    }

    public void mulThis(double x) {
        double[] arr = array();
        for (int i = 0; i < arr.length; i++) {
            arr[i] *= x;
        }
    }

    public void addThis(double[] x) {
        double[] thisArray = array();
        for (int i = 0; i < thisArray.length; i++) {
            thisArray[i] += x[i];
        }
    }

    /** TODO optimize */
    public final Tensor sqr() {
        return pow(2);
    }

    public final void addGrad(Tensor x) {
        double[] thisGradArray = grad.array();
        double[] xArray = x.array();
        for (int i = 0; i < thisGradArray.length; i++) {
            thisGradArray[i] += xArray[i];
        }
    }

    public void addGrad(SimpleMatrix x) {
        double[] thisGradArray = grad.array();
        double[] xArray = array(x);
        for (int i = 0; i < thisGradArray.length; i++) {
            thisGradArray[i] += xArray[i];
        }
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
        double[] resultArr = new double[volume()];
        for(int i=0; i<d.length; i++) {
            resultArr[i] = Math.signum(d[i]);
        }
        var y = new Tensor(resultArr, rows(), cols(), this.hasGrad());

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
                        //System.err.println(Tensor.this + " gradient nonFinite=" + nonFinite + "\n:\t" + Str.n4(g0));
                        System.err.println("Tensor gradient nonFinite=" + nonFinite);
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

    public enum Loss {
        Huber, MeanSquared /* L2 */, SubAbs /* L1 */
    }

    /*
    public static class Optimizer {
        // This class depends on Optimizers.OptimizerStep which is in an excluded file.
        // public final List<Optimizers.OptimizerStep> step = new ArrayList<>();

        // public Optimizer(Optimizers.OptimizerStep... steps) {
        //     for (var s : steps) if (s!=null) step.add(s);
        // }

        public final void run(Stream<Tensor> tt) {
            var p = tt
                    .filter(t -> t.parameter)
                    // .sorted(GradQueue.TensorSorter) // GradQueue is commented out
                    .toList();

            if (!p.isEmpty()) {
                _run(p);
                p.forEach(Tensor::zeroGrad);
            }
        }

        protected void _run(List<Tensor> p) {
            // for (var s : step)
            //     s.accept(p);
        }

    }
    */

    /*
    public static class GradQueue implements Iterable<Tensor> {
        private static final Comparator<Tensor> TensorSorter =
            Comparator.comparingInt(System::identityHashCode);

        final ConcurrentHashMap<Tensor, Queue<double[]>> grads = new ConcurrentHashMap<>(512);

        private double factor = 1;

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
                    grads.keySet().forEach(g -> {
                        double[] arr = g.grad.array();
                        for (int i = 0; i < arr.length; i++) {
                            arr[i] *= factor;
                        }
                    });

                if (o != null)
                    o.run(grads.keySet().stream());

                grads.clear();
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
    */

    abstract public static class TensorOp {
        private static final BiFunction<SimpleMatrix, SimpleMatrix, SimpleMatrix> tensorGradMerge = TensorUtil::_addTo;
        protected final Tensor[] parents;
        final int generation;
        int color = -1;

        public TensorOp(Tensor... parents) {
            if (parents.length < 1)
                throw new UnsupportedOperationException();
            this.parents = parents;
            int maxGen = -1;
            for (Tensor p : parents) {
                if (p != null) { // Add null check for safety, though parents shouldn't be null
                    maxGen = Math.max(maxGen, p.generation());
                }
            }
            this.generation = maxGen + 1;
        }

        final SimpleMatrix[] allocateParentGrads() {
            SimpleMatrix[] result = new SimpleMatrix[parents.length];
            for (int i = 0; i < parents.length; i++) {
                Tensor p = parents[i];
                if (p != null && p.hasGrad()) { // Add null check
                    result[i] = new SimpleMatrix(p.rows(), p.cols());
                } else {
                    result[i] = null;
                }
            }
            return result;
        }

        public abstract void backward(SimpleMatrix grad, SimpleMatrix[] gradParent);

        public final void gradients(Tensor next, Map<Tensor, SimpleMatrix> grads, PriorityQueue<Tensor> q) {

            var parentGrads = allocateParentGrads();

            backward(grads.get(next), parentGrads);

            var n = parents.length;
            for (var i = 0; i < n; i++) {
                var pg = parentGrads[i];
                if (pg != null) {
//                    if (isZero(gi))
//                        continue;
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
}
