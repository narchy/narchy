package jcog.tensor.rl.streamx;

import jcog.Fuzzy;
import jcog.Util;
import jcog.math.FloatMeanWindow;
import jcog.math.FloatSupplier;
import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Optimizers.OptimizerStep;
import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.DoubleSupplier;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * Base components shared between StreamTD and StreamAC implementations
 */
public class StreamXUtil {

    public static final float layerNormRate =
        1e-3f; //PAPER
        //0.005f;
        //0.1f;
        //1e-4f;
        //1e-6f;
        //1;

    /**
     * Layer normalization with sparse linear layer and LeakyReLU activation
     */
    public static class LayerNormLinear implements UnaryOperator<Tensor> {
        public final Models.LayerNorm norm;
        public final Models.Linear linear;

        LayerNormLinear(int in, int out, float sparsity, float sparseNoise) {
            this.norm = new Models.LayerNorm(in, layerNormRate);
            this.linear = new Models.Linear(in, out, null, true);
        }
        
        @Override
        public Tensor apply(Tensor x) {
            return linear.apply(norm.apply(x));
        }
        
    }

    public static void initSparse(Tensor t, double sparsity, float noiseFloor, RandomGenerator rng) {
        var cols = t.cols();
        var w = t.array();
        var scale =
                //1/Math.sqrt(fanIn);
                //2/Math.sqrt(cols); //He init
                (1-sparsity) * 2/Math.sqrt(cols); //sparse He

        var noise = scale * noiseFloor;
        for (var i = 0; i < w.length; i++) {
            var r = Fuzzy.polarize(rng.nextFloat());
            w[i] = r * (rng.nextFloat() < sparsity ? noise : scale);
        }
    }

    /**
     * ObservationNormalizer using Exponential Moving Average (EMA) for running mean and variance.
     */
    static class ObservationNormalizer {
        private final Tensor mean, variance;
        private final float alpha; // Smoothing factor
        private static final double epsilon = 1e-6;

        /**
         * Constructor for ObservationNormalizer.
         *
         * @param size  The dimensionality of the observation space.
         * @param alpha The smoothing factor for EMA (0 < alpha <= 1).
         */
        ObservationNormalizer(int size, float alpha) {
            if (alpha <= 0 || alpha > 1)
                throw new IllegalArgumentException("Alpha should be in the range (0, 1].");

            this.mean = Tensor.zeros(1, size).fill(Double.NaN);
            this.variance = Tensor.ones(1, size);
            this.alpha = alpha;
        }

        /**
         * Updates the running mean and variance using EMA and normalizes the input.
         *
         * @param x The new observation tensor.
         * @return The normalized observation.
         */
        Tensor update(Tensor x) {
            if (this.mean.scalar()!=mean.scalar()) {
                //initialize
                this.mean.setData(x);
            } else {

                // Update running mean: mean_new = alpha * x + (1 - alpha) * mean_old
                //TODO var updatedMean = x.lerp(...)
                var updatedMean = x.mul(this.alpha).addMul(this.mean, 1 - this.alpha);
                this.mean.setData(updatedMean.data);

                // Update running variance: var_new = alpha * (x - mean_new)^2 + (1 - alpha) * var_old
                var diff = x.sub(updatedMean);
                var updatedVar = diff.mul(diff).mul(this.alpha).addMul(this.variance, 1 - this.alpha);
                this.variance.setData(updatedVar.data);
            }

            // Normalize the input
            return x.sub(this.mean).div(this.variance.sqrt().add(epsilon));
        }
    }
    /**
     * Online observation normalization using Welford's algorithm
     */
    static class ObservationNormalizerWelford {
        private final Tensor mean, meanDiff, variance;

        /** TODO periodically reset, or remove somehow */
        @Deprecated private int count;
        
        ObservationNormalizerWelford(int size) {
            this.mean = Tensor.zeros(1, size);
            this.meanDiff = Tensor.zeros(1, size);
            this.variance = Tensor.ones(1, size);
            this.count = 0;
        }

        Tensor update(Tensor x) {
            count++;
            var delta = x.sub(mean);
            mean.add(delta.div(count));
            meanDiff.add(delta.mul(delta).mul(count-1).div(count));
            variance.setData(meanDiff.add(x.variance().mul(count)).div(count));

            return x.sub(mean).div(variance.sqrt().add(1e-8));
        }
    }
    /**
     * Online observation normalization using Welford's algorithm
     */
    static class ObservationNormalizerRollingWelford {
        final FloatMeanWindow[] mean;

        private transient final Tensor MEAN, VAR;

        ObservationNormalizerRollingWelford(int dim, int history) {
            MEAN = Tensor.zeros(1, dim);
            VAR = Tensor.zeros(1, dim);

            this.mean = new FloatMeanWindow[dim];
            for (int i = 0; i < dim; i++)
                mean[i] = new FloatMeanWindow(history);
        }

        Tensor update(Tensor x) {
            int v = mean.length;
            double[] xx = x.array();
            for (int i = 0; i < v; i++) {
                double[] meanVar = mean[i].acceptAndGetMeanAndVariance((float) xx[i]);
                MEAN.data.set(i, meanVar[0]);
                VAR.data.set(i, meanVar[1]);
            }
            return x.sub(MEAN).div(VAR.sqrt().add(EPSILON));
        }

        public static final double EPSILON =
            1e-6;
            //1e-8;
    }

    static class ObGD implements OptimizerStep {
        /** kappa parameter - controls update rate */
        final double kappa;
        final EligibilityTraces traces;
        final FloatSupplier lr;
        final DoubleSupplier tdErr;
        @Nullable final OptimizerStep filter;
        public final float weightDecay =
            0; //DISABLED
            //1e-8f;

        public ObGD(FloatSupplier lr, double kappa, EligibilityTraces traces, DoubleSupplier tdErr, @Nullable OptimizerStep filter) {
            this.lr = lr;
            this.kappa = kappa;
            this.traces = traces;
            this.tdErr = tdErr;
            this.filter = filter;
        }

        @Override
        public void accept(List<Tensor> p) {
            traces.accumulate(p);

            if (filter!=null)
                filter.accept(p);

            double weightMult = 1 - weightDecay;
            var step = -step();
            var n = p.size();
            for (var i = 0; i < n; i++)
                addTo(p.get(i).array(), weightMult, traces.traces.get(i).array(), step);
        }

        static void addTo(double[] x, double w, double[] y, double m) {
            if (w==1)
                addTo(x, y, m);
            else {
                var n = x.length;
                for (var j = 0; j < n; j++)
                    x[j] = (w * x[j]) + (m * y[j]); //TODO fma
            }
        }
        static void addTo(double[] x, double[] y, double m) {
            if (m==1)
                Util.addTo(x, y);
            else {
                var n = x.length;
                for (var j = 0; j < n; j++)
                    x[j] = x[j] + (m * y[j]); //TODO fma
            }
        }


        private double step() {
            var tdError = tdErr.getAsDouble();

            var lr = this.lr.asFloat();

            var deltaBar =
                //1 + Math.abs(tdError);
                Math.max(Math.abs(tdError), 1); //PAPER
                //1;

            var tracesNorm =
                traces.normL1();  //PAPER
                //traces.normL1()/traces.volume;
                //Math.sqrt(traces.normL1());
                //Math.pow(traces.normL1(), 0.9);
                //Util.max(traces.normL1(), 1);
                //Util.clamp(traces.normL1(), 1, traces.volume); //learning rate lower bound: clip to parameter max range of 1
                //Util.max(traces.normL1() / Math.sqrt(traces.volume), 1);
                //1 + traces.normL1();

            var lrDivisor =
                kappa * deltaBar * tracesNorm; //PAPER

            var effectiveLr =
                lr / Math.max(lrDivisor, 1); //PAPER

            return effectiveLr * tdError; //PAPER
            //return effectiveLr * Util.clamp(tdError, -1, +1);
            //return effectiveLr * (tdError >= 0 ? +1 : -1);
            //return effectiveLr * Math.signum(tdError);
            //return effectiveLr;
        }

        @Deprecated public Tensor.Optimizer optimizer() {
            return new Tensor.Optimizer() {
                @Override
                protected void _run(List<Tensor> p) {
                    accept(p);
                }
            };
        }
    }

    /**
     * Eligibility traces with proper decay and accumulation
     */
    public static class EligibilityTraces {
//        private static final double TRACE_NORM_THRESHOLD = 1E3;

        private final FloatSupplier gamma, lambda;
        public final List<Tensor> traces = new CopyOnWriteArrayList<>();

//        private static final boolean clip = false;

        /** caches normL1 value */
        private transient double normL1;

        /** total # of elements */
        private transient int volume;

        EligibilityTraces(FloatSupplier gamma, FloatSupplier lambda) {
            this.gamma = gamma;
            this.lambda = lambda;
            reset();
        }
        
        void reset() {
            traces.clear();
        }
        void accumulate(List<Tensor> p) {
            if (traces.size()!=p.size()) {
                traces.clear();
                for (var pp : p)
                    traces.add(Tensor.zeros(pp.rows(), pp.cols()));
            }

            var gamma = this.gamma.asFloat();
            var lambda = this.lambda.asFloat();
            var decay = gamma * lambda;
            var s = traces.size();
            double normSum = 0;
            int volume = 0;
            for (var i = 0; i < s; i++) {
                var t = traces.get(i);
                var T = t.array();
                var G = p.get(i).grad.array();

                for (int j = 0; j < T.length; j++) {
                    double y =
                        Util.fma(T[j], decay, G[j]); //T[j] = T[j] * decay + G[j]; //PAPER
                        //Util.fma(T[j], decay, (1-decay)*G[j]); //T[j] = T[j] * decay + G[j] * (1-decay);
                        //Math.tanh(Util.fma(T[j], decay, G[j]));
                        //Util.clampSafePolar(Util.fma(T[j], decay, G[j]));
                        //Util.fma(T[j], decay, Util.clampSafePolar(G[j])); //clipped accumulation
                    T[j] = y;
                }

//                if (clip) Util.clampSafe(t.array(), -1, +1);

                var tNorm = Util.sumAbs(T);
                normSum += tNorm;

                var v = t.volume();
                volume += v;
            }

            this.volume = volume;
            this.normL1 = normSum;
        }

        public double normL1() {
            return normL1;
        }

    }

    public static class ObGDWithMomentum extends Optimizers.OptimizerStepWithMemory {
        /** kappa parameter - controls update rate */
        private final double kappa;
        private final EligibilityTraces traces;
        private final FloatSupplier lr;
        private final DoubleSupplier tdErr;
        @Nullable private final OptimizerStep filter;
        private final float weightDecay = 0; // DISABLED
        private final float momentum;

        /**
         * Constructs the ObGD optimizer with momentum.
         *
         * @param lr      Learning rate supplier
         * @param kappa   Kappa parameter controlling the update rate
         * @param traces  Eligibility traces
         * @param tdErr   Temporal difference error supplier
         * @param filter  Optional filter step
         * @param momentum Momentum factor (typically between 0 and 1)
         */
        public ObGDWithMomentum(FloatSupplier lr, double kappa, EligibilityTraces traces,
                                DoubleSupplier tdErr, @Nullable OptimizerStep filter, float momentum) {
            super(1); // Initialize with one memory slot per parameter tensor
            this.lr = lr;
            this.kappa = kappa;
            this.traces = traces;
            this.tdErr = tdErr;
            this.filter = filter;
            this.momentum = momentum;
        }

        @Override
        public void accept(List<Tensor> parameters) {
            // Accumulate eligibility traces
            traces.accumulate(parameters);

            // Apply optional filter if present
            if (filter != null)
                filter.accept(parameters);

            // Compute the step based on current TD error and learning rate
            double step = computeStep();

            // Weight multiplier for weight decay (if enabled)
            double weightMult = 1 - weightDecay;

            // Iterate over each parameter tensor
            for (int p = 0; p < parameters.size(); p++) {
                Tensor param = parameters.get(p);
                double[] paramArray = param.array();
                double[] traceArray = traces.traces.get(p).array();

                // Retrieve or initialize velocity for this parameter
                double[] velocity = data(p, paramArray.length);

                // Update velocity: v = momentum * v + step * trace
                for (int j = 0; j < paramArray.length; j++) {
                    velocity[j] = momentum * velocity[j] + step * traceArray[j];
                }

                // Apply weight decay if enabled
                if (weightDecay != 0) {
                    for (int j = 0; j < paramArray.length; j++) {
                        paramArray[j] *= weightMult;
                    }
                }

                // Update parameters: param -= learningRate * velocity
                float currentLr = lr.asFloat();
                for (int j = 0; j < paramArray.length; j++)
                    paramArray[j] -= currentLr * velocity[j];
            }
        }

        /**
         * Computes the effective step size based on the current TD error and learning rate.
         *
         * @return The computed step size.
         */
        private double computeStep() {
            double tdError = tdErr.getAsDouble();
            float currentLr = lr.asFloat();

            // Compute deltaBar as per the original logic
            double deltaBar = Math.max(Math.abs(tdError), 1);

            // Compute the norm of the eligibility traces
            double tracesNorm =
                    1 + traces.normL1();
                    //traces.normL1(); //PAPER

            // Compute the learning rate divisor
            double lrDivisor = kappa * deltaBar * tracesNorm;

            // Compute the effective learning rate
            double effectiveLr = currentLr / Math.max(lrDivisor, 1);

            // Return the step scaled by TD error
            return effectiveLr * tdError;
        }

        @Deprecated public Tensor.Optimizer optimizer() { return new Tensor.Optimizer() { @Override protected void _run(List<Tensor> p) {
                    accept(p);
                } }; }
    }
}
