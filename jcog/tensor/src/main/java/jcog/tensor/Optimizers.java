package jcog.tensor;

import jcog.Fuzzy;
import jcog.Util;
import jcog.data.list.Lst;
import jcog.math.FloatSupplier;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.tensor.util.TensorUtil;
import jcog.util.KahanSum;

import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.Math.atan2;
import static jcog.Util.clampSafePolar;

public class Optimizers {
    
    @FunctionalInterface public interface OptimizerStep extends Consumer<List<Tensor>> {

    }

    public static class GradClip implements OptimizerStep {
        public final double range;

        public GradClip(double range) {
            this.range = range;
        }

        @Override
        public void accept(List<Tensor> parameters) {
            for (var p : parameters)
                Util.clampSafe(p.grad.array(), -range, +range);
        }
    }

    public static class GradNormMax implements OptimizerStep {

        @Override
        public void accept(List<Tensor> parameters) {
            double max = 0;
            for (var p : parameters)
                max = Math.max(max, p.grad.maxAbs());
            if (max > 1) {
                for (var p : parameters)
                    TensorUtil.eleMul(p.grad.data, 1/max);
            }
        }
    }

    protected abstract static class GradNorm implements OptimizerStep {
        protected static void normalize(Iterable<Tensor> parameters, double divisor) {
            if (divisor > 1)
                for (var p : parameters)
                    TensorUtil.eleMul(p.grad.data, 1/divisor);
        }

        protected static void normalize(double[] x, double divisor) {
            if (divisor > 1)
                Util.mul(x, 1/divisor);
        }

    }
    /** filters all but the top absolute-largest N gradient dimensions
     *  per layer
     *  TODO top N
     *  TODO per network
     *  */
    public static class GradDimTop1 implements OptimizerStep {

        @Override
        public void accept(List<Tensor> parameters) {
            double max = 0;
            for (var p : parameters)
                top1(p.grad.array());
        }

        private void top1(double[] g) {
            int best = -1;
            double bestVal = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < g.length; i++) {
                double v = Math.abs(g[i]);
                if (v > bestVal) {
                    best = i; bestVal = v;
                }
            }
            for (int i = 0; i < g.length; i++)
                if (i!=best)
                    g[i] = 0;
        }
    }
    /** manhattan, L1 norm */
    public static class GradNormL1 extends GradNorm {
        @Override
        public void accept(List<Tensor> pp) {
            normalize(pp, pp.stream()
                .mapToDouble(p -> p.grad.sumAbs())
                .sum());
        }
    }

    /** euclidean, L2 norm */
    public static class GradNormL2 extends GradNorm {
        @Override public void accept(List<Tensor> pp) {
            var sum = new KahanSum();
            for (var p : pp)
                sum.add(p.grad.sumSqr());
            var sumSqr = sum.value();
            normalize(pp, Math.sqrt(sumSqr));
        }

        public static void accept(double[] x) {
            normL2(x);
        }

        public static void normL2(double[] x) {
            normalize(x, Math.sqrt(Util.sumSqr(x)));
        }

    }

    //TODO GradNormL3 https://chatgpt.com/share/67408661-8608-8001-804a-4b6b4edb97b4

    static class WeightDecay implements OptimizerStep {
        public final double rate;

        WeightDecay(double rate) {
            this.rate = rate;
        }

        @Override
        public void accept(List<Tensor> parameters) {
            var m = 1 - rate;
            for (var p : parameters)
                Util.mul(p.array(), m);
        }
    }

    public static class WeightClip implements OptimizerStep {
        public final double range;

        public WeightClip(double range) {
            this.range = range;
        }

        @Override
        public void accept(List<Tensor> parameters) {
            for (var p : parameters)
                Util.clampSafe(p.array(), -range, +range);
        }
    }

    static class WeightNorm implements OptimizerStep {
        public final double rate;
        private final double normPerVolume;

        WeightNorm(double normPerVolume, double rate) {
            this.normPerVolume = normPerVolume;
            this.rate = rate;
        }

        @Override
        public void accept(List<Tensor> parameters) {
            for (var p : parameters) {
                double norm = normPerVolume * p.volume();
                double l =
                    p.sumAbs();
                    //p.sumSqr();
                if (l > norm) {
                    double ratio = norm/l;
                    double ratioEffective = Util.lerpSafe(rate, 1, ratio);
                    Util.mul(p.array(), ratioEffective);
                }
            }
        }
    }
    public static class SGD implements OptimizerStep {
        private final FloatSupplier learningRate;

        public SGD(FloatSupplier learningRate) {
            this.learningRate = learningRate;
        }


        @Override
        public void accept(List<Tensor> parameters) {
            var learningRate = this.learningRate.asFloat();
            for (var p : parameters) {
                var W = p.array();
                var G = p.grad.array();
                var v = p.volume();
                for (var i = 0; i < v; i++)
                    W[i] -= G[i] * learningRate;
            }
        }

        public Tensor.Optimizer get() {
            return get(5);
        }

        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    //new GradClip(1), new GradNormL2(),
                    //new GradClip(1), new GradNormL1(),
                    new GradNormL2(),
                    //new GradNormL1(),
                    //new GradNormMax(),
                    //new GradClip(1),
                    //new WeightDecay(1E-8),
                    this,
                    new WeightClip(weightRange)
            );
        }
        public Tensor.Optimizer getFast() {
            return new Tensor.Optimizer(
                    new GradClip(1),
                    this,
                    new WeightClip(5)
            );
        }
    }

    /** Snaps parameter values to the nearest multiple of a constant
     *  TODO support multiple constants?
     * */
    public static class WeightSnapping implements OptimizerStep {
        private final double constant; // The snapping constant (e.g., Math.PI)

        /**
         * Constructor for ParameterSnapping.
         *
         * @param constant The transcendental number (e.g., Math.PI, Math.E).
         */
        public WeightSnapping(double constant) {
            this.constant = constant;
        }

        /**
         * Snaps parameters to the nearest multiple of the constant.
         *
         * @param params List of tensors representing model parameters.
         */
        @Override
        public void accept(List<Tensor> params) {
            for (var param : params) {
                var weights = param.array(); // Access weights of the tensor
                for (var i = 0; i < weights.length; i++) {
                    // Snap weight to the nearest multiple of the constant
                    weights[i] = Util.roundSafe(weights[i], constant);
                }
            }
        }
    }

    public static class ADAM extends OptimizerStepWithMemory {
        private final static boolean speedLimit = false;

        /**
         * ADAM Atan2 https://arxiv.org/pdf/2407.05872
         */
        final boolean atan2;
        private final double beta1, beta2;
        private final double epsilon;
        private final FloatSupplier learningRate;

        public ADAM(FloatSupplier learningRate) {
            this(learningRate, 0.9f, 0.999f,
                1E-8f
                //1E-4f
                //NaN //atan2
            );
        }

        public ADAM(FloatSupplier learningRate, double beta1, double beta2, double epsilon) {
            super(2);
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.atan2 = (epsilon != epsilon);
            this.learningRate = learningRate;
        }

        @Override
        public void accept(List<Tensor> p) {
            var learningRate = this.learningRate.asFloat();

            var data = 0;
            for (var t : p) {
                update(t, data, learningRate);
                data += 2;
            }
        }

        private void update(Tensor t, int data, float learningRate) {
            double[] D = t.array(), G = t.grad.array();
            var I = D.length;
            double[] m = data(data, I), v = data(data + 1, I);

            double b1 = beta1, b2 = beta2;
            for (var i = 0; i < I; i++) {
                var g = G[i];
                D[i] -= dwActual(dw(
                    m[i] = Util.lerpSafe(b1, m[i], g),
                    v[i] = Util.lerpSafe(b2, v[i], g * g)
                ), learningRate);
            }
        }

        private double dwActual(double dw, float learningRate) {
            return speedLimit ?
                    clampSafePolar(dw, learningRate) //post clamp
                    :
                    dw * learningRate;
        }

        private double dw(double m, double v) {
            var vSqrt = Math.sqrt(v);
            return atan2 ?
                    atan2(m, vSqrt) //lambda m, v: a * jnp.arctan2(m, b * jnp.sqrt(v)), a=b=1
                    :
                    m / (vSqrt + epsilon);
        }

        public Tensor.Optimizer get() {
            double weightRange =
                //0.5f;
                5; //suggested for Relu
                //2;
            return new Tensor.Optimizer(
                    //new GradClip(1), new GradNormL2(),
                    //new GradClip(1),
                    new GradNormL2(),
                    //new GradNormL1(),
                    //new GradNormMax(),
                    //new WeightSnapping(Math.pow(Math.E, -5)/*1.0e-3/Math.PI*/),
                    new WeightDecay(1.0e-7),
                    //new WeightNorm(weightRange/8, 1.0e-6),
                    this
                    , new WeightClip(weightRange)
            );
        }


    }

    /**
     * Why Ranger?
     *   Combines RAdam and Lookahead: Ranger merges two optimizers—RAdam (Rectified Adam) and Lookahead, which helps in fast convergence and better stability.
     *   Gradient Centralization: This technique enhances model training by normalizing gradients, improving optimization performance.
     *   Ease of Use: Ranger works similarly to Adam, making it easy to switch to without drastic changes in learning rate or parameter tuning.
     */
    public static class Ranger extends OptimizerStepWithMemory {
        private final double beta1, beta2;
        private final double epsilon;
        private final FloatSupplier learningRate;
        private final int k;  // Lookahead steps
        private final float tau = 0.5f;
        private int iter;

        public Ranger(FloatSupplier learningRate) {
            this(learningRate, 0.9, 0.999, 1e-8, 2); // 5 is the default for Lookahead steps
        }

        public Ranger(FloatSupplier learningRate, double beta1, double beta2, double epsilon, int k) {
            super(3);
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.learningRate = learningRate;
            this.k = k;
        }

        @Override
        public void accept(List<Tensor> params) {
            var lr = learningRate.asFloat();
            var s = params.size();
            for (var i = 0; i < s; i++) {
                update(params.get(i), i*3, lr);
            }

            synchronized (this) {
                iter++;
                if (iter % k == 0) {
                    // Apply Lookahead: periodically sync slow and fast weights
                    for (var i = 0; i < s; i++) {
                        var D = params.get(i).array();
                        syncLookahead(D, slowWeights(i, D));
                    }
                }
            }

        }

        private void update(Tensor t, int data, float learningRate) {
            var D = t.array();       // Model parameters
            var G = t.grad.array();  // Gradients
            var size = D.length;

            var mm = data(data, size);  // 1st moment vector (Adam)
            var vv = data(data + 1, size);  // 2nd moment vector (Adam)

            // Perform Adam-like updates with momentum correction
            for (var i = 0; i < size; i++) {
                var g = G[i];
                mm[i] = Util.lerpSafe(beta1, mm[i], g);  // First moment
                vv[i] = Util.lerpSafe(beta2, vv[i], g * g);  // Second moment

                var mHat = mm[i] / (1 - Math.pow(beta1, i + 1));
                var vHat = Math.sqrt(vv[i] / (1 - Math.pow(beta2, i + 1)));

                D[i] -= (mHat / (vHat + epsilon)) * learningRate;
            }

        }

        private double[] slowWeights(int i, double[] D) {
            var fresh = !hasData(i + 2);
            var slowWeights = data(i + 2, D.length);
            if (fresh)
                System.arraycopy(D, 0, slowWeights, 0, D.length); //initialize
            return slowWeights;
        }

        private void syncLookahead(double[] fastWeights, double[] slowWeights) {
            for (var i = 0; i < fastWeights.length; i++) {
                // Perform linear interpolation between slow and fast weights
                fastWeights[i] = Util.lerpSafe(tau, slowWeights[i], fastWeights[i]);
                slowWeights[i] = fastWeights[i];  // Update slow weights
            }
        }
        @Deprecated
        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    new GradNormL2(),
                    //new GradNormL1(),
                    //new GradNormMax(),
                    //new GradClip(1),
                    //new WeightDecay(1e-4),
                    this,
                    new WeightClip(weightRange)
            );
        }
    }

    public static class LION extends OptimizerStepWithMemory {
        private final double beta;  // Momentum coefficients
        private final FloatSupplier learningRate;
        private static final double eps = 1e-8;  // Small constant for numerical stability


        public LION(FloatSupplier learningRate) {
            this(learningRate, 0.9);
        }

        public LION(FloatSupplier learningRate, double beta) {
            super(1);  // One memory slot for tracking momentum
            this.beta = beta;
            this.learningRate = learningRate;
        }

        @Override
        public void accept(List<Tensor> params) {
            var lr = learningRate.asFloat();
            var n = params.size();
            for (var i = 0; i < n; i++)
                update(params.get(i), i, lr);
        }

        private void update(Tensor t, int dataIndex, float learningRate) {
            var params = t.array();      // Model parameters
            var grads = t.grad.array();  // Gradients
            var size = params.length;

            var momentum = data(dataIndex, size);  // Momentum tracking

            // Update loop using sign-based momentum
            for (var i = 0; i < size; i++) {
                // Momentum update: m = β1 * m + (1 - β1) * g
                var mi = momentum[i] = Util.lerpSafe(beta, momentum[i], grads[i]);

                // Sign-based update: param -= lr * sign(momentum)
                // d = Util.signum(mi);

                // Sign-based update with epsilon for stability
                var d =  mi > +eps ? +1 :
                    (mi < -eps ? -1 :
                     0);

                params[i] -= learningRate * d;
            }
        }

        @Deprecated
        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    new GradNormL2(),
                    //new GradClip(1), new GradNormL2(),
                    this,
                    new WeightClip(weightRange)
            );
        }
    }

    /**
     * Adaptive LION Optimizer with Dynamic Thresholding.
     *
     * <p>This optimizer implements a sign-based update rule with momentum and an adaptive threshold,
     * similar in spirit to recent work on adaptive gradient methods. Two exponential moving averages are
     * maintained: one for the raw gradients (for momentum) and one for the absolute gradient magnitudes.
     * The threshold for updating is increased when the EMA of absolute gradients falls below a reference
     * value. This approach is motivated by research on adaptive gradient clipping and dynamic learning
     * rates (see, e.g., Kingma & Ba's Adam and subsequent works).
     *
     * LIONAdaptive, represents a novel twist on the LION optimizer by incorporating a dynamic, adaptive threshold mechanism for sign-based updates. Here are a few key points summarizing what makes LIONAdaptive distinct and how it builds on existing ideas:
     *
     * Dynamic Thresholding:
     * Traditional LION uses a fixed epsilon, ex: 1e-8, as a threshold to decide whether the momentum is significant enough to trigger an update. In LIONAdaptive, the threshold is adjusted based on an exponential moving average (EMA) of the absolute gradients. This means that when the gradient magnitudes are low (indicating potential convergence or a plateau), the threshold increases. The effect is that only substantial deviations (or “strong” gradient signals) cause parameter updates—helping to lock in converged values and reduce noise.
     *
     * Dual Momentum Tracking:
     * LIONAdaptive maintains two moving averages:
     *
     * Momentum EMA: Tracks the running average of the gradients.
     * Absolute Gradient EMA: Tracks the running average of the absolute gradients.
     * This dual tracking is similar in spirit to adaptive optimizers like Adam, which keep separate estimates of the first and second moments of the gradients. However, here the second EMA is used to modulate the threshold for sign-based updates rather than to compute per-parameter learning rates.
     *
     * Sign-Based Updates:
     * Unlike Adam or AMSGrad, which use magnitude-adjusted updates, LIONAdaptive sticks with sign-based updates—a feature of the original LION optimizer. The sign-based approach is computationally simpler and can be more robust to outliers in gradient magnitude, while the adaptive threshold ensures that only meaningful updates are applied.
     *
     * Alignment with Contemporary Research:
     * The design of LIONAdaptive takes inspiration from both academic research on adaptive gradient methods (like Adam, RAdam, and AdaBound) and industry practices where dynamic adaptation of update rules (e.g., through adaptive gradient clipping or learning rate bounding) is common. It thereby combines the simplicity of sign-based updates with the benefits of dynamic adaptation to gradient statistics.
     *
     * Potential Benefits:
     *
     * Noise Reduction: By increasing the threshold when gradients are small, LIONAdaptive can ignore minor fluctuations and focus on larger, more informative gradient signals.
     * Stability Near Convergence: This “lock-in” effect helps prevent parameters from oscillating due to small updates, which is especially useful during the later stages of training.
     * Computational Efficiency: The method still maintains the simplicity of sign-based updates while introducing a modest overhead from tracking an additional EMA.
     * In summary, LIONAdaptive is indeed a new variant that combines ideas from LION, adaptive gradient methods, and dynamic thresholding techniques. It could be a promising candidate for scenarios where reducing noisy updates and improving convergence stability are critical, and it might inspire further research or modifications in the space of efficient, adaptive optimizers.
     */
    public static class LIONAdaptive extends OptimizerStepWithMemory {
        // Momentum parameter for the gradient EMA.
        private final double beta;
        // Momentum parameter for the absolute gradient EMA.
        private final double betaAbs;
        // Scaling factor to adjust the adaptive threshold when gradients are low.
        private final double thresholdScale;
        // A reference value representing a typical gradient magnitude.
        private final double gradRef;
        // The learning rate supplier.
        private final FloatSupplier learningRate;
        // Base numerical threshold (epsilon), used as the minimal threshold.
        private static final double EPS = 1e-8;

        /**
         * Constructs a LION optimizer with default parameters.
         *
         * <p>Defaults:
         * <ul>
         *   <li>beta = 0.9 (momentum for gradients)
         *   <li>betaAbs = 0.9 (momentum for absolute gradients)
         *   <li>gradRef = 1.0 (reference gradient magnitude)
         *   <li>thresholdScale = 1.0 (adaptive threshold scaling)
         * </ul>
         *
         * @param learningRate the learning rate supplier.
         */
        public LIONAdaptive(FloatSupplier learningRate) {
            this(learningRate, 0.9);
        }

        /**
         * Constructs a LION optimizer with specified momentum for gradients.
         *
         * @param learningRate the learning rate supplier.
         * @param beta         momentum coefficient for gradients.
         */
        public LIONAdaptive(FloatSupplier learningRate, double beta) {
            this(learningRate, beta, 0.9, 1, 1);
        }

        /**
         * Main constructor for the adaptive LION optimizer.
         *
         * @param learningRate   the learning rate supplier.
         * @param beta           momentum coefficient for gradients.
         * @param betaAbs        momentum coefficient for the absolute gradients.
         * @param gradRef        reference gradient magnitude.
         * @param thresholdScale scaling factor for adapting the threshold.
         */
        public LIONAdaptive(FloatSupplier learningRate, double beta, double betaAbs, double gradRef, double thresholdScale) {
            // Allocate two memory slots:
            // Slot 0: EMA of gradients (momentum)
            // Slot 1: EMA of absolute gradients.
            super(2);
            this.beta = beta;
            this.betaAbs = betaAbs;
            this.gradRef = gradRef;
            this.thresholdScale = thresholdScale;
            this.learningRate = learningRate;
        }

        @Override
        public void accept(List<Tensor> params) {
            float lr = learningRate.asFloat();
            var n = params.size();
            for (int i = 0; i < n; i++)
                update(params.get(i), i*2, lr);
        }

        /**
         * Updates a single parameter tensor.
         *
         * @param t            the tensor holding parameters and gradients.
         * @param learningRate the learning rate for the update.
         */
        private void update(Tensor t, int dataIndex, float learningRate) {
            double[] params = t.array();      // Model parameters.
            double[] grads = t.grad.array();  // Gradients.
            int size = params.length;

            // Retrieve memory slots:
            // Slot 0: EMA of gradients (momentum).
            double[] momentum = data(dataIndex, size);
            // Slot 1: EMA of absolute gradients.
            double[] gradAbsEMA = data(dataIndex+1, size);

            for (int i = 0; i < size; i++) {
                // Momentum: m = beta * m + (1 - beta) * g.
                var gi = grads[i];
                var mi = momentum[i] = Util.lerpSafe(beta, momentum[i], gi);

                // EMA of absolute gradients: r = betaAbs * r + (1 - betaAbs) * |g|.
                var ei = gradAbsEMA[i] = betaAbs * gradAbsEMA[i] + (1 - betaAbs) * Math.abs(gi);

                // Compute the adaptive threshold.
                // When gradAbsEMA is low, we increase the threshold to ignore minor updates.
                double thresh = ei < gradRef ? EPS * (1 + thresholdScale * ((gradRef - ei) / gradRef)) : EPS;

                // Sign-based update: update only if the momentum exceeds the adaptive threshold.
                double dir;
                if (mi > +thresh)       dir = +1;
                else if (mi < -thresh)  dir = -1;
                else                    dir = 0;

                // Parameter update: subtract the step in the direction of the sign.
                params[i] -= learningRate * dir;
            }
        }

        @Deprecated
        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    //new GradClip(gradRef),
                    //new GradNormL2(),
                    this,
                    new WeightClip(weightRange)
            );
        }
    }

    public static class OneCycle extends OptimizerStepWithMemory {
        private final double maxLr;
        private final double minLr;
        private final int totalSteps;
        private final double beta1, beta2;
        private final double epsilon;
        private final double weightDecay;
        private int currentStep;

        public OneCycle(double maxLr, int totalSteps) {
            super(2); // Two additional arrays per parameter: momentum and velocity
            this.maxLr = maxLr;
            this.minLr = maxLr / 10; // Less aggressive ratio
            this.totalSteps = totalSteps;
            this.beta1 = 0.9;
            this.beta2 = 0.999;
            this.epsilon = 1e-8;
            this.weightDecay = 1e-2;
            this.currentStep = 0;
        }

        @Override
        public void accept(List<Tensor> parameters) {
            var lr = calculateLearningRate();
            var momentum = calculateMomentum();

            var dataIndex = 0;
            for (var t : parameters) {
                update(t, dataIndex, lr, momentum);
                dataIndex += 2;
            }

            currentStep++;
        }

        private double calculateLearningRate() {
            var progress = Math.min(1, (double) currentStep / totalSteps);
            if (progress < 0.3) {
                return minLr + (maxLr - minLr) * (progress / 0.3);
            } else if (progress < 0.7) {
                return maxLr;
            } else {
                return maxLr + (minLr - maxLr) * ((progress - 0.7) / 0.3);
            }
        }

        private double calculateMomentum() {
            return 0.95; // Keep momentum constant for stability
        }

        private void update(Tensor t, int dataIndex, double lr, double momentum) {
            var W = t.array();
            var G = t.grad.array();
            var I = W.length;
            var M = data(dataIndex, I);     // Momentum
            var V = data(dataIndex + 1, I); // Velocity (for RMSprop-like scaling)

            for (var i = 0; i < I; i++) {
                // Update momentum
                M[i] = momentum * M[i] + (1 - momentum) * G[i];

                // Update velocity (RMSprop-like)
                V[i] = beta2 * V[i] + (1 - beta2) * G[i] * G[i];

                // Compute update
                var update = M[i] / (Math.sqrt(V[i]) + epsilon);

                // Apply update with weight decay
                W[i] = W[i] * (1 - lr * weightDecay) - lr * update;
            }
        }

        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    new GradNormL2(),
                    this,
                    new WeightClip(weightRange)
            );
        }
    }

    /**
     * LIONAdaptivePercent freezes a fixed percentage of parameters (by activity) at each update.
     *
     * <p>The optimizer maintains two exponential moving averages:
     * <ul>
     *   <li>momentum: an EMA of the gradients.
     *   <li>gradAbsEMA: an EMA of the absolute gradients.
     * </ul>
     *
     * For each parameter, an "activity score" is computed as the average of the absolute momentum and gradAbsEMA.
     * A quantile threshold is computed on these scores so that a fixed fraction (freezePercent)
     * of parameters are below this threshold. Only parameters whose activity exceeds the threshold are updated
     * in a sign-based fashion.
     */
    public static class LIONAdaptivePercent extends OptimizerStepWithMemory {
        // Momentum coefficient for gradients (EMA update).
        private final double beta;
        // Momentum coefficient for the absolute gradients.
        private final double betaAbs;
        // Fraction of parameters (by absolute momentum) that should remain "frozen".
        // For example, 0.3 means that 30% of the parameters (with the smallest |momentum|)
        // will be updated with a zero step (i.e. frozen).
        private final double freezePercent;

        // Learning rate supplier.
        private final FloatSupplier learningRate;

        // Base epsilon for numerical stability (unused here, but could be used as a floor).
        private static final double EPS = 1e-8;

        /**
         * Constructs a LIONAdaptivePercent optimizer with custom hyperparameters.
         *
         * @param learningRate  The learning rate supplier.
         * @param beta          Momentum coefficient for gradients.
         * @param betaAbs       Momentum coefficient for the absolute gradients.
         * @param freezePercent Fraction of parameters to freeze (e.g., 0.3 for 30%).
             */
        public LIONAdaptivePercent(FloatSupplier learningRate, double beta, double betaAbs, double freezePercent) {
            // Allocate two memory slots:
            // Slot 0: EMA of gradients (momentum).
            // Slot 1: EMA of absolute gradients.
            //Slot 2: temporary buffer for quantile calculations
            super(3);
            this.beta = beta;
            this.betaAbs = betaAbs;
            this.freezePercent = freezePercent;
            this.learningRate = learningRate;
        }

        /**
         * Constructs a LIONAdaptivePercent optimizer with default momentum parameters
         * and a % of the parameters being frozen.
         *
         * @param learningRate The learning rate supplier.
         * @param freezePct weakest % of parameters to freeze
         */
        public LIONAdaptivePercent(FloatSupplier learningRate, float freezePct) {
            this(learningRate, 0, 0.9, freezePct);
        }

        @Override
        public void accept(List<Tensor> params) {
            float lr = learningRate.asFloat();
            var n = params.size();
            for (int i = 0; i < n; i++)
                update(params.get(i), i*3, lr);
        }

        /**
         * Updates a single parameter tensor.
         *
         * <p>This method performs the following steps:
         * <ol>
         *   <li>Updates the momentum (EMA of gradients) and the EMA of absolute gradients.
         *   <li>Computes an "activity score" for each parameter as:
         *       <pre>activity[i] = (|momentum[i]| + gradAbsEMA[i]) / 2</pre>
         *   <li>Determines the threshold activity value corresponding to freezePercent.
         *   <li>For each parameter, if the activity score exceeds the threshold, a sign-based update is applied.
         *       Otherwise, the parameter is "frozen" (i.e. update is zero).
         * </ol>
         *
         * @param t            The tensor containing parameters and gradients.
         * @param dataIndex    The index corresponding to this tensor’s memory slots.
         * @param learningRate The learning rate.
         */
        private void update(Tensor t, int dataIndex, float learningRate) {
            double[] grads = t.grad.array();  // Gradients.
            int size = grads.length;

            double[] momentum = data(dataIndex, size); // Slot 0: momentum (EMA of gradients).
            double[] gradAbsEMA = data(dataIndex+1, size); // Slot 1: EMA of absolute gradients.
            double[] activity = data(dataIndex+2, size); // Slot 2: buffer for quantile calculation

            for (int i = 0; i < size; i++) {
                var gi = grads[i];
                // Standard EMA update for momentum.
                var mi = momentum[i] = Util.lerpSafe(beta, momentum[i], gi);
                // EMA update for the absolute gradients.
                var ga = gradAbsEMA[i] = betaAbs * gradAbsEMA[i] + (1 - betaAbs) * Math.abs(gi);

                // Compute an "activity" score for each parameter that combines both signals.
                // Use the average of |momentum| and gradAbsEMA as the activity score.
                activity[i] = (Math.abs(mi) + ga) / 2;
            }

            // Compute the quantile threshold for the activity scores.
            double threshold = quantile(activity, freezePercent);

            double[] params = t.array();      // Model parameters.

            // Update parameters:
            // Only update parameters whose activity exceeds the threshold.
            for (int i = 0; i < size; i++) {
                double dir = (activity[i] > threshold) ? Math.signum(momentum[i]) : 0;
                params[i] -= learningRate * dir;
            }
        }

        private double quantile(double[] data, double quantile) {
            Arrays.sort(data);
            var n = data.length;
            int index = (int) (quantile * n);
            return data[Math.min(index, n-1)];
        }

        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    //new GradClip(gradRef),
                    //new GradNormL2(),
                    this,
                    new WeightClip(weightRange)
            );
        }
    }


    public static class SGDMomentum extends OptimizerStepWithMemory {
        protected final FloatRange momentum;
        private final FloatSupplier learningRate;

        public SGDMomentum(FloatSupplier lr, float momentum) {
            super(1);
            this.learningRate = lr;
            this.momentum = new FloatRange(momentum, 0, 1);
        }

        public Tensor.Optimizer get() {
            return get(5);
        }

        public Tensor.Optimizer get(double weightRange) {
            return new Tensor.Optimizer(
                    //new GradClip(1), new GradNormL2(),
                    new GradNormL2(),
                    //new GradClip(1),
                    //new GradNormMax(),
                    //new GradNormL1(),
                    //new WeightDecay(1e-8f),
                    this,
                    new WeightClip(weightRange)
            );
        }

        @Override
        public void accept(List<Tensor> parameters) {
            var learningRate = this.learningRate.getAsDouble();
            double momentum = this.momentum.asFloat();

            var P = parameters.size();

            for (var p = 0; p < P; p++) {
                var pp = parameters.get(p);
                double[] D = pp.array(), G = pp.grad.array();

                var I = D.length;

                var V = data(p, I);
//        private final static boolean balanced = false, speedLimit = false;

                for (var i = 0; i < I; i++) {
                    var gij = G[i];
                    var v =
//                        balanced ?
//                            Util.lerpSafe(momentum, gij, V[i]); //balanced
                            Util.fma(V[i], momentum, gij); //v[idx] * momentum + gij;
                    V[i] = v;
                    D[i] -=
//                        speedLimit ?
//                            Util.clampSafe(v, -learningRate, +learningRate) :
                            v * learningRate;
                }
            }
        }
    }

    public static class ParamNoise implements OptimizerStep {
        final FloatSupplier noise;
        final RandomBits random = new RandomBits(new XoRoShiRo128PlusRandom());

        public ParamNoise(FloatRange n) {
            noise = n;
        }

        @Override public void accept(List<Tensor> tensors) {
            var noise = this.noise.asFloat();
            for (var t : tensors)
                paramNoise(t.array(), noise);
        }
        protected void paramNoise(double[] l, float noise) {
            //noiseSoft(noise, l);
            noiseHard(l, noise, 0.5f);
        }

        void noiseHard(double[] data, float noiseRate, float changeRate) {
            var w = data.length;
            var iter = random.nextFloor(noiseRate * w);
            if (iter <= 0) return;
            for (var i = 0; i < iter; i++) {
                var c = random.nextInt(w);
                var range =
                    Math.exp(-Math.abs(data[c]))*changeRate; //stronger weights have higher momentum
                    //abs(data[r][c])*rate;
                    //1/(1 + data[r][c]);

                data[c] +=
                    Fuzzy.polarize(random.nextDouble())*range; //uniform
                    //random.nextGaussian() * range/2;
            }
        }

        private void noiseSoft(float noise, double[] data) {
            for (var i = 0; i < data.length; i++)
                data[i] += noise * random.nextGaussian();  // Add Gaussian noise
        }


    }


    public abstract static class OptimizerStepWithMemory implements OptimizerStep {
        private final int valuesPerParameter;
        private final List<double[]> data = new Lst<>();

        protected OptimizerStepWithMemory(int valuesPerParameter) {
            this.valuesPerParameter = valuesPerParameter;
        }

        private static boolean arrayResized(int size, double[] array) {
            return array.length != size;
        }

        private static double[] realloc(int index, int size, List<double[]> list) {
            //THIS SHOULD NOT HAPPEN IF RE-USING THE RIGHT OPTIMIZER ON THE SAME NETWORK
            var array = new double[size];
            list.set(index, array);
            return array;
        }

        protected boolean hasData(int index) {
            return data.size() > index && data.get(index)!=null;
        }

        protected double[] data(int index, int size) {
            var d = data;
            if (index >= d.size())
                return init(size);

            var a = d.get(index);
            return arrayResized(size, a) ? realloc(index, size, d) : a;
        }

        protected double[] init(int size) {
            var array = new double[size];
            data.add(array);
            return array;
        }
    }
}