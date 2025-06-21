package jcog.optimize;

import jcog.Util;
import jcog.WTF;
import jcog.data.list.Lst;
import jcog.math.MatrixDeterminant;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Optimizers;
import org.eclipse.collections.impl.list.mutable.primitive.DoubleArrayList;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import static java.lang.Math.sqrt;
import static jcog.data.DistanceFunction.distanceCartesianSq;

/**
 * A general-purpose Bayesian Optimization implementation using Gaussian Process regression
 * for surrogate modeling and Expected Improvement for acquisition function.
 *
 * This implementation follows the standard Bayesian Optimization approach:
 * 1. Maintain a probabilistic surrogate model (Gaussian Process)
 * 2. Use an acquisition function (Expected Improvement) to suggest next points
 * 3. Update the model with new observations
 */
public class BayesOptimizer {


    // Gaussian Process hyperparameters
    private static final double SIGNAL_VARIANCE = 1.0;
    private static final double NOISE_VARIANCE = 1e-6;
    private static final int SEEDS = 4;
    private static final double GRADIENT_LEARNING_RATE = 0.1;
    private static final int GRADIENT_STEPS = 2;
    /** specified as a proportion of the dimension range.
     *  controls gradient resolution  */
    private static final double GRADIENT_DX = 0.01;
    // Numerical stability parameters
    private static final double MIN_VARIANCE = 1e-10;
    private static final double CHOLESKY_EPS = 1e-10;
    public final Surrogate model;
    public final Memory memory;
    private final Optimizers.GradNormL2 gradNormalize = new Optimizers.GradNormL2();


    /**
     * "Use EI for smaller search spaces or when function evaluations are expensive.
     *  Use UCB for larger, noisier, or more complex spaces requiring broader exploration.
     */
    private final AcquisitionFunction acquisition;

    private final double[] lowerBounds, upperBounds, range;
    private final Random rng;
    private final int dim;
    public final AtomicBoolean retainBest = new AtomicBoolean(false);

    /**
     * @param lowerBounds Lower bounds for each dimension
     * @param upperBounds Upper bounds for each dimension
     */
    public BayesOptimizer(int capacity, double[] lowerBounds, double[] upperBounds) {
        validateDimensions(lowerBounds.length, lowerBounds, upperBounds);

        this.dim = lowerBounds.length;
        this.lowerBounds = lowerBounds.clone();
        this.upperBounds = upperBounds.clone();
        this.range = Util.sub(this.upperBounds, this.lowerBounds);

        this.rng = new XoRoShiRo128PlusRandom();
        memory = new Memory(capacity);

        model =
            new NeuralSurrogate(dim, dim*2);
            //new GaussianProcess(new RBFKernel());

        acquisition =
            new UpperConfidenceBound();
            //new ExpectedImprovement();

    }

    //private final SmartEvictionStrategy eviction = new SmartEvictionStrategy();

    private static void validateDimensions(int dimensions, double[] lowerBounds, double[] upperBounds) {
        if (dimensions != lowerBounds.length || dimensions != upperBounds.length)
            throw new IllegalArgumentException("Dimensions mismatch");
        if (IntStream.range(0, dimensions).anyMatch(i -> lowerBounds[i] > upperBounds[i]))
            throw new IllegalArgumentException("Lower bounds must be strictly <= upper bounds");
    }

    /**
     * Suggests the next point to evaluate based on the acquisition function.
     * @return The next point to evaluate
     */
    public double[] next() {
        return acquisition.next();
    }


    private double[] best(double[][] points) {
        var maxValue = Double.NEGATIVE_INFINITY;
        double[] maxPoint = null;

        for (var point : points) {
            var value = acquisition.apply(point);
            if (value > maxValue) {
                maxValue = value;
                maxPoint = point;
            }
        }

        return maxPoint;
    }

    /** fine-tunes a point */
    public void optimize(double[] p) {
        if (!memory.x.isEmpty()) {
            for (var i = 0; i < GRADIENT_STEPS; i++)
                updatePoint(p, gradient(p, GRADIENT_DX));
        }
    }

    private void updatePoint(double[] p, double[] grad) {
        for (var i = 0; i < dim; i++) {
            p[i] = Util.clamp/*Safe*/(
                     p[i] +
                       grad[i] * GRADIENT_LEARNING_RATE
                   , lowerBounds[i], upperBounds[i]);
        }
    }

    /** finite differences gradient estimation */
    private double[] gradient(double[] p, double delta) {
        var grad = new double[dim];

        var xx = p.clone();

        for (var i = 0; i < dim; i++) {
            var x = p[i]; //original compoenent

            var dx = this.range[i] * delta;

            var upper = acquire(xx, i, x, +dx);
            if (upper==upper) {
                var lower = acquire(xx, i, x, -dx);
                if (lower==lower)
                    grad[i] = (upper - lower) / delta;
            }

            xx[i] = x; //restore
        }

        Optimizers.GradNormL2.accept(grad);

        return grad;
    }

    private double acquire(double[] xx, int i, double x, double dx) {
        xx[i] = dx(x, dx, i);
        var upper = acquisition.apply(xx);
        return upper;
    }

    private double dx(double x, double dx, int i) {
        return Util.clampSafe(x + dx / 2, lowerBounds[i], upperBounds[i]);
    }


    /**
     * Updates the model with a new observation.
     * @param x The input point
     * @param y The observed value
     * @throws IllegalArgumentException if dimensions don't match
     */
    public void put(double[] x, double y) {
        if (x.length != dim)
            throw new IllegalArgumentException("Input dimensions mismatch");

        x = x.clone();

        memory.put(x, y);

        model.update(x, y);
    }


    /**
     * Gets the current best observed value.
     * @return The minimum observed value
     */
    public double bestValue() {
        return memory.bestValue();
    }

    /**
     * Gets the point corresponding to the best observed value.
     * @return The input point giving the best value
     */
    @Nullable public double[] best() {
        var b = memory.bestPoint();
        return b!=null ? b.clone() : null;
    }

    public int bestIndex(boolean bestOrWorst) {
        return memory.bestIndex(bestOrWorst);
    }

    public void put(ToDoubleFunction<double[]> f) {
        var x = next();
        var y = f.applyAsDouble(x);
        put(x, y);
    }

    private double[] rangeNorm(double[] x) {
        return Util.arrayOf(i -> Util.normalizeSafe(x[i], lowerBounds[i], upperBounds[i]), new double[x.length]);
    }

    public int capacity() {
        return memory.capacity;
    }

    interface Surrogate {
        /** Train the model on current observations */
        void update(double[] x, double y);

        /** Predict mean and variance for a point */
        double[] predict(double[] x);
    }

    public interface Kernel {
        double get(double[] a, double[] b);
        //double[] gradient(double[] a, double[] b);
    }

    /**
     * Interface for acquisition functions.
     */
    private abstract class AcquisitionFunction {
        public abstract double apply(double[] x);

        /** called at beginning of each iteration, prepares for apply() calls */
        public void update() { }

        public double[] next() {
            return memory.x.isEmpty() ?
                    random() :
                    best(seeds(SEEDS));
        }

        private double[][] seeds(int n) {
            var p = new double[n][];
            for (var i = 0; i < n; i++)
                optimize(p[i] = random());
            return p;
        }

        /** generate a random point to evalaute */
        private double[] random() {
            return Util.arrayOf(this::randomElement, new double[dim]);
        }

        private double randomElement(int i) {
            return lowerBounds[i] + rng.nextDouble() * range[i];
        }
    }

    /** radial basis function */
    public static class RBFKernel implements Kernel {

        private final double lengthScaleSqr, signalVariance;

        public RBFKernel() {
            this(1, 1);
        }

        public RBFKernel(double lengthScale, double signalVariance) {
            this.lengthScaleSqr = Util.sqr(lengthScale);
            this.signalVariance = signalVariance;
        }

        @Override
        public double get(double[] a, double[] b) {
            return signalVariance *
                    Math.exp(-0.5 * distanceCartesianSq(a, b)) / lengthScaleSqr;
        }
    }
    
    final class Memory {
        public final int capacity;
        public final List<double[]> x;
        public final DoubleArrayList y;

        Memory(int capacity) {
            if (capacity < 2) throw new IllegalArgumentException();
            this.capacity = capacity;
            x = new Lst<>(capacity);
            y = new DoubleArrayList(capacity);
        }

        /**
         * TODO smarter eviction heuristic than LRU
         */
        public void ensureCapacity(int n) {
            while (x.size() + n > capacity) {
                var e = evict();
                x.remove(e);
                y.removeAtIndex(e);
            }
        }

        /**
         * Gets the current best observed value.
         *
         * @return The minimum observed value
         */
        public double bestValue() {
            return y.minIfEmpty(Double.POSITIVE_INFINITY);
        }

        public int evict() {
            return
                evictRandom(retainBest.get(), false);
                //eviction.get();
                //evictMean();
        }

        public int evictRandom(boolean retainBest, boolean retainWorst) {
            var n = y.size();
            var best = retainBest ? bestIndex(true) : -1;
            var worst = retainWorst ? bestIndex(false) : -1;
            var which = rng.nextInt(n);
            if (which == best)
                which = (which + 1) % n; //HACK just choose the next one
            if (retainWorst && n > 2) {
                if (which == worst) {
                    which = (which + 1) % n; //HACK just choose the next one
                    if (which == best)
                        which = (which + 1) % n; //HACK just choose the next one
                }
                if (which == worst) throw new WTF();
            }
            if (retainBest && which == best) throw new WTF();

            return which;
        }

        /**
         * preserves extrema
         */
        public int evictMean() {
//        var worstValue = Double.NEGATIVE_INFINITY;
//        var bestValue = Double.POSITIVE_INFINITY;
            var meanestValue = Double.POSITIVE_INFINITY;
            int worst = -1, best = -1, meanest = -1;
            var n = y.size();
            var mean = y.average();
            for (var i = 0; i < n; i++) {
                var value = y.get(i);

                var dMean = Math.abs(value - mean);
                if (dMean < meanestValue) {
                    meanestValue = dMean;
                    meanest = i;
                }

//            if (value > worstValue) {
//                worstValue = value;
//                worst = i;
//            }
//
//            if (value < bestValue) {
//                bestValue = value;
//                best = i;
//            }
            }
            //return worst;
            return meanest;
        }

        public int evictUnimportant() {
            final double recencyBias = 0.05f;
            var importance = importances(recencyBias);
            var worstIdx = 0;
            for (var i = 1; i < importance.length; i++) {
                if (importance[i] < importance[worstIdx])
                    worstIdx = i;
            }
            return worstIdx;
        }

        /**
         * Gets the point corresponding to the best observed value.
         *
         * @return The input point giving the best value
         */
        @Nullable
        public double[] bestPoint() {
            var i = bestIndex(true);
            return i < 0 ? null : x.get(i);
        }

        public int bestIndex(boolean bestOrWorst) {
            var n = y.size();
            if (n == 0) return -1;

            var w = 0;
            for (var i = 1; i < n; i++) {
                var yw = y.get(w);
                var yi = y.get(i);
                w = (bestOrWorst ? yw <= yi : yw > yi) ? w : i;
            }
            return w;
        }

        /**
         * point importances
         */
        private double[] importances(double recencyBias) {
            var n = x.size();
            var importance = new double[n];

            // Compute information gain for each point
            for (var i = 0; i < n; i++)
                importance[i] = MatrixDeterminant.informationGain(
                    ((GaussianProcess) model).K, i); //TODO can this work using X for non-GaussianProcess ?

            // Add recency bias
            var decayFactor = 1 - recencyBias;
            for (var i = 0; i < n; i++)
                importance[i] *= Math.pow(decayFactor, n - i - 1);

            // Add value-based importance
            var bestVal = bestValue();
            for (var i = 0; i < n; i++) {
                var relativeValue = (bestVal - y.get(i)) / Math.abs(bestVal);
                importance[i] *= 1 + Math.exp(-relativeValue);
            }

            return importance;
        }

        public void put(double[] x, double y) {
            ensureCapacity(1);
            this.x.add(x);
            this.y.add(y);
        }
    }

    /**
     * Gaussian Process implementation for surrogate modeling.
     */
    public class GaussianProcess implements Surrogate {
        final Kernel kernel;
        public double[][] K;
        public double[] alpha;

        public GaussianProcess(Kernel kernel) {
            this.kernel = kernel;
        }

        private static double[] solveCholesky(double[][] A, double[] b) {
            var n = A.length;
            var L = new double[n][n];

            // Compute Cholesky decomposition with added stability
            for (var i = 0; i < n; i++) {
                for (var j = 0; j <= i; j++) {
                    var sum = 0.0;
                    for (var k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }

                    L[i][j] = i == j ? sqrt(Math.max(A[i][i] - sum, CHOLESKY_EPS)) : (A[i][j] - sum) / L[j][j];
                }
            }

            // Forward substitution
            var y = new double[n];
            for (var i = 0; i < n; i++) {
                var sum = 0.0;
                for (var j = 0; j < i; j++)
                    sum += L[i][j] * y[j];

                y[i] = (b[i] - sum) / L[i][i];
            }

            // Backward substitution
            var x = new double[n];
            for (var i = n - 1; i >= 0; i--) {
                var sum = 0.0;
                for (var j = i + 1; j < n; j++)
                    sum += L[j][i] * x[j];

                x[i] = (y[i] - sum) / L[i][i];
            }

            return x;
        }

        private double[] kStar(double[] x) {
            var X = memory.x;
            var n = X.size();
            var kStar = new double[n];
            for (var i = 0; i < n; i++)
                kStar[i] = kernel.get(x, X.get(i));
            return kStar;
        }

        @Override
        public void update(double[] x, double y) {
            var XX = memory.x;
            var YY = memory.y;
            var n = XX.size();
            var K = this.K!=null && this.K.length==n && this.K[0].length == n ? this.K : new double[n][n];

            // Compute kernel matrix
            for (var i = 0; i < n; i++) {
                for (var j = 0; j <= i; j++) {
                    K[i][j] = K[j][i] =
                            kernel.get(XX.get(i), XX.get(j));
                    if (i == j)
                        K[i][j] += NOISE_VARIANCE;
                }
            }

            alpha = solveCholesky(this.K = K, YY.toArray());
        }

        @Override
        public double[] predict(double[] x) {

            var kStar = kStar(x);

            var mean = Util.dot(kStar, alpha);

            var variance = kernel.get(x, x) - Util.dot(kStar, solveCholesky(K, kStar));

            return new double[] { mean, Math.max(variance, MIN_VARIANCE)};
        }

    }

    class UpperConfidenceBound extends AcquisitionFunction {
        /** exploration rate */
        private final double beta;

        public UpperConfidenceBound() {
            this(0.5f);
        }
        public UpperConfidenceBound(double beta) {
            this.beta = beta;
        }

        @Override
        public double apply(double[] x) {
            var meanVar = model.predict(x);
            var exploitability = meanVar[0];
            var explorability = sqrt(Math.max(0, meanVar[1]));
            return exploitability + beta * explorability;
        }
    }

    /**
     * Expected Improvement acquisition function implementation.
     */
    private class ExpectedImprovement extends AcquisitionFunction {
        private double best;

        // Error function approximation using Abramowitz and Stegun method
        private static double erf(double x) {
            var t = 1.0 / (1.0 + 0.47047 * Math.abs(x));
            var poly = t * (0.3480242 + t * (-0.0958798 + t * 0.7478556));
            var result = 1 - poly * Math.exp(-x * x);
            return x >= 0 ? +result : -result;
        }

        @Override
        public void update() {
            best = bestValue();
        }

        @Override
        public double apply(double[] x) {
            var prediction = model.predict(x);

            if (prediction[1] <= MIN_VARIANCE)
                return 0.0;

            var std = sqrt(prediction[1]);

            var mean = prediction[0];

            var gain = mean - best;

            var z = gain / std;

            // Handle numerical stability for very large z values
            if (z < -6)
                return 0;  // Negligible improvement possible
            else if (z > +6)
                return gain;  // Guaranteed improvement
            else {
                var normalPdf = Math.exp(-0.5 * z * z) / sqrt(2 * Math.PI);
                var normalCdf = 0.5 * (1 + erf(z / sqrt(2)));

                return gain * normalCdf +
                        std * normalPdf;
            }
        }
    }

    /**
     * Neural Network based surrogate model using simple feedforward architecture
     *  - Simple feedforward architecture with one hidden layer
     *  - Outputs both mean and variance predictions
     *  - Uses ReLU activation and Xavier initialization
     *  - Implements mini-batch SGD training
     *  - Exponential activation on variance output to ensure positivity
     */
    public class NeuralSurrogate implements Surrogate {

        public final NN net;
        final int epochs = 1;
        final float learningRate = 0.01f;

        /** true: train entire memory, false: only the latest update */
        boolean inputMemory = true;

        /** training error from last iteration */
        public double err;

        public NeuralSurrogate(int dim, int hidden) {
            this.net = new NN(dim, hidden, new XoRoShiRo128PlusRandom());
        }

        @Override
        public double[] predict(double[] x) {
            return net.get(x);
        }

        @Override
        public void update(double[] x, double y) {
            var XX = memory.x;
            var YY = memory.y;
            var n = inputMemory ? YY.size() : 1;
            var lr =
                    learningRate;
                    //inputMemory ? learningRate/n : learningRate;
            double err = 0;
            for (var epoch = 0; epoch < epochs; epoch++) {
                double delta;
                if (inputMemory) {
                    delta = 0;
                    for (var i = 0; i < n; i++)
                        delta += net.put(rangeNorm(XX.get(i)), YY.get(i), lr);
                } else {
                    delta = net.put(rangeNorm(x), y, lr);
                }
                err += Math.abs(delta);
            }
            this.err = err/(n*epochs);
            //System.out.println("err: " + this.err);
        }

    }

    /**
     * Minimal neural network surrogate model.
     * Single hidden layer with ReLU activation.
     * Outputs mean prediction and epistemic uncertainty.
     */
    public class NN {
        private final int inputDim, hiddenDim;

        public final double[] weightsIH;  // input->hidden
        public final double[] biasH;
        public final double[] weightsHO;  // hidden->output
        public final double[] biasO;
        private final double[] hidden;

        private final double beta = 2.0;  // log-variance regularization weight
        private float weightDecay = 1e-6f;

        // Constants for log variance bounds
        private final static double MAX_LOG_VARIANCE = Math.log(2.0);    // max variance of 2.0
        private final static double MIN_LOG_VARIANCE = Math.log(MIN_VARIANCE);   // min variance of 1e-6

        /**
         * Creates a neural network for heteroscedastic regression
         * @param inputDim Input dimension
         * @param hiddenDim Hidden layer dimension
         * @param rng Random number generator for weight initialization
         */
        NN(int inputDim, int hiddenDim, Random rng) {
            this.inputDim = inputDim;
            this.hiddenDim = hiddenDim;

            this.weightsIH = new double[inputDim * hiddenDim];
            this.weightsHO = new double[hiddenDim * 2];  // 2 outputs: mean and log_variance
            this.biasH = new double[hiddenDim];
            this.biasO = new double[2];
            this.hidden = new double[hiddenDim];

            // Initialize weights using He initialization
            var inputScale = Math.sqrt(2.0 / inputDim);
            for (var i = 0; i < weightsIH.length; i++)
                weightsIH[i] = rng.nextGaussian() * inputScale;

            var hiddenScale = Math.sqrt(2.0 / hiddenDim);
            for (var i = 0; i < weightsHO.length; i++)
                weightsHO[i] = rng.nextGaussian() * hiddenScale;
        }

        /**
         * Forward pass returning raw network outputs
         * @return raw predictions: double[] containing [mean, log_variance]
         */
        private double[] _forward(double[] x) {
            x = rangeNorm(x);

            // Hidden layer with ReLU activation
            for (var h = 0; h < hiddenDim; h++) {
                var sum = biasH[h];
                var offset = h * inputDim;
                for (var i = 0; i < inputDim; i++)
                    sum += x[i] * weightsIH[offset + i];
                hidden[h] = Math.max(0, sum);  // ReLU
            }

            // Output layer - linear activation for both mean and log_variance
            var output = new double[2];
            for (var o = 0; o < 2; o++) {
                var sum = biasO[o];
                var offset = o * hiddenDim;
                for (var h = 0; h < hiddenDim; h++)
                    sum += hidden[h] * weightsHO[offset + h];
                output[o] = sum;
            }

            // Clamp log variance to prevent numerical issues
            output[1] = Util.clamp(output[1], MIN_LOG_VARIANCE, MAX_LOG_VARIANCE);

            return output;
        }

        /**
         * Get predictions: mean and variance
         * @param x Input features
         * @return double[] containing [mean, variance]
         */
        public double[] get(double[] x) {
            var output = _forward(x);
            var mean = output[0];
            var variance = Math.exp(output[1]); // Convert log variance to variance for return value
            return new double[]{mean, variance};
        }

        /**
         * Train single observation using heteroscedastic loss
         * @param x Input features
         * @param actual Target value
         * @return absolute error of prediction
         */
        public double put(double[] x, double actual, float learningRate) {
            Util.replaceNaNwithRandom(x, rng);
            actual = Util.replaceNaNwithRandom(actual, rng);

            var y = _forward(x);
            var predicted = y[0];
            var g = grads(actual, predicted, y[1]);

            clampGradient(g);
            decayWeights();
            _put(x, g, learningRate);
            clampWeights();

            return Math.abs(predicted - actual);
        }

        /**
         * Compute gradients for heteroscedastic loss
         * @return double[] containing [mean_gradient, log_variance_gradient]
         */
        private double[] grads(double actual, double predicted, double logVar) {
            // Convert log variance to variance for loss computation
            var variance = Math.exp(logVar);
            var invVar = 1.0 / variance;
            var diff = predicted - actual;

            // Gradient for mean prediction
            var meanGrad = diff * invVar;

            // Gradient for log variance:
            // Start with gradient for variance: (d²/2v - 1/2)
            // Then multiply by v to get gradient for log(v) due to chain rule
            var varGrad = (diff * diff * invVar * invVar - invVar) / 2;
            varGrad *= variance;  // Chain rule through exp(log_var)

            // Add regularization in log space
            // Regularizing log(variance) towards 0 is equivalent to regularizing variance towards 1
            varGrad += beta * logVar;

            return new double[]{meanGrad, varGrad};
        }

        private static void clampGradient(double[] g) {
            Util.clampSafe(g, -1, +1);
        }

        private void _put(double[] x, double[] outputGrad, float learningRate) {
            // Compute hidden layer gradients
            var hiddenGrad = new double[hiddenDim];
            for (var h = 0; h < hiddenDim; h++) {
                if (hidden[h] > 0) {  // ReLU gradient
                    double grad = 0;
                    for (var o = 0; o < 2; o++)
                        grad += outputGrad[o] * weightsHO[o * hiddenDim + h];
                    hiddenGrad[h] = grad;
                }
            }

            // Update output layer weights
            for (var o = 0; o < 2; o++) {
                var offset = o * hiddenDim;
                var grad = outputGrad[o];
                for (var h = 0; h < hiddenDim; h++)
                    weightsHO[offset + h] -= learningRate * grad * hidden[h];
                biasO[o] -= learningRate * grad;
            }

            // Update hidden layer weights
            for (var h = 0; h < hiddenDim; h++) {
                var offset = h * inputDim;
                var grad = hiddenGrad[h];
                for (var i = 0; i < inputDim; i++)
                    weightsIH[offset + i] -= learningRate * grad * x[i];

                biasH[h] -= learningRate * grad;
            }
        }

        private void decayWeights() {
            if (weightDecay > 0) {
                var weightDecayRate = 1 - weightDecay;
                Util.mul(weightsHO, weightDecayRate);
                Util.mul(biasO, weightDecayRate);
                Util.mul(weightsIH, weightDecayRate);
                Util.mul(biasH, weightDecayRate);
            }
        }

        private void clampWeights() {
            Util.clampSafe(weightsHO, -1, +1);
            Util.clampSafe(biasO, -1, +1);
            Util.clampSafe(weightsIH, -1, +1);
            Util.clampSafe(biasH, -1, +1);
        }
    }
}