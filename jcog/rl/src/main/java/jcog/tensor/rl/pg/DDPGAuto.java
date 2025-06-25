package jcog.tensor.rl.pg;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.math.FloatMeanEwma;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.signal.IntRange;
import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.util.TensorUtil;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * An older implementation of DDPG with adaptive noise features.
 * This class is part of an older API.
 *
 * @deprecated This class is part of an older API. For DDPG or similar off-policy actor-critic algorithms,
 *             consider using or developing solutions based on the `jcog.tensor.rl.pg3` components and patterns.
 *             The `pg3` package is the focus for new development.
 */
@Deprecated
public class DDPGAuto extends AbstractPG {

    /** 'actor' */
    public final Models.Layers policy;

    /** 'critic' */
    public final Models.Layers value;

    public final FloatRange gamma = new FloatRange(0.9f, 0, 1);

    public final IntRange replays = new IntRange(8, 1, 256);

    public final FloatRange batchSize = new FloatRange(0.1f, 0, 1);

    public final FloatRange policyLR = new FloatRange(
            0.00003f //ADAM
            , 0, 0.01f);

    public final FloatRange valueLR = new FloatRange(policyLR.floatValue()*(10/3f), 0, 0.01f);

    // Base range for noise parameters
    public final FloatRange actionNoise = new FloatRange(1, 0, 1);


    /**
     * The rate at which the action noise scale is adjusted to try and reach the target.
     * Smaller = more stable adjustments; larger = faster but possibly noisier.
     */
    private final double actionNoiseAdaptRate = 0.01f;
    // New: adaptation rate for parameter noise adjustment
    private final double paramNoiseAdaptRate = 0.0000001f;

    // Previously, paramNoise was fixed; now we allow it to be adjusted dynamically.
    public final FloatRange paramNoise = new FloatRange(0, 0, 0.00001f);

    // Replay buffer
    private final int replayBufferCapacity = 1 * 1024;
    public final PrioritizedReplayBuffer memory = new PrioritizedReplayBuffer(replayBufferCapacity, 0.6f);

    // Optimizers for policy & value
    private final Tensor.Optimizer policyOpt, valueOpt;
    private final OUNoise ouNoise;

    // Grad queues (keep them separate for policy and value)
    private final Tensor.GradQueue valueCtx = new Tensor.GradQueue(), policyCtx = new Tensor.GradQueue();

    public double policyLoss, valueLoss, policyLossNext, valueLossNext;

    private double meanTDError = 0.0;

    // Keep track of last experience
    private Experience lastExperience;
    private Tensor lastState;
    double[] lastAction;

    private static final Tensor.Loss valueLossFn = Tensor.Loss.MeanSquared;

    final UnaryOperator<Tensor> valueOutputActivation = null, policyOutputActivation = Tensor::clipUnitPolar;



    // Define scaling factors for each metric
    // These can be tuned based on empirical results
    float valueLossFactor = 0.5f;
    float tdErrorFactor = 0.5f;

    private final int valueLossMovingAverageWindow = 16;
    private final int tdErrorMovingAverageWindow = valueLossMovingAverageWindow;

    final FloatMeanEwma
            valueLossMA = new FloatMeanEwma().period(valueLossMovingAverageWindow),
            tdErrMA = new FloatMeanEwma().period(tdErrorMovingAverageWindow);

    public DDPGAuto(int inputs, int outputs, int hiddenSize, int valueSize) {
        super(inputs, outputs);

        // Initialize OUNoise with adaptation rates
        ouNoise = new OUNoise(outputs, 0.005f, 0.005f);

        policy = model(inputs, hiddenSize, outputs, policyOutputActivation, true, 3);
        value = model(inputs + outputs, valueSize, 1, valueOutputActivation, true, 3);

        policy.train(false);
        value.train(false);

        policyOpt = new Optimizers.ADAM(policyLR).get();
        valueOpt  = new Optimizers.ADAM(valueLR).get();

        // Apply param noise if desired; note that paramNoise is now subject to adaptation.
        policyOpt.step.add(new Optimizers.ParamNoise(paramNoise));
        valueOpt.step.add(new Optimizers.ParamNoise(paramNoise));
    }

    private Models.Layers model(int inputSize, int hiddenSize, int outputSize,
                                UnaryOperator<Tensor> outputActivation,
                                boolean biasInLastLayer, int layerCount) {
        return new Models.Layers(
                Tensor.RELU_LEAKY,
                outputActivation,
                biasInLastLayer,
                Models.Layers.layerLerp(inputSize, hiddenSize, outputSize, layerCount)
        ) {
            @Override
            protected void afterLayer(int O, int l, int maxLayers) {
                // Optionally add dropout or other layer customizations here.
            }
        };
    }

    @Override
    protected double[] _action(Tensor state) {
        // raw (deterministic) policy output
        return policy(state).detach().array();
    }

    @Override
    public double[] act(double[] input, double reward) {
        return act(input, reward, false, true);
    }

    public double[] act(double[] _state, double reward, boolean done, boolean training) {
        var state = Tensor.row(_state);
        var isFirstStep = (lastState == null);

        // Store transition in memory if not the first call
        if (!isFirstStep) {
            remember(lastState, lastAction, reward, state, done);
        }

        // Always get the raw (noise-free) action from the policy
        var rawAction = policy(state).detach().array();

        // Optionally add exploration noise for training
        if (training) {
            addActionNoiseAndAdapt(rawAction);
            update();  // trigger learning updates
        }

        if (done) {
            lastState = null;
        } else {
            lastState = state;
            lastAction = rawAction;
        }

        return rawAction;
    }

    /**
     * Adds OU noise in-place to {@code action}, then adapts the "actionNoise" scale
     * based on the moving averages of Value Loss and TD Error.
     */
    private void addActionNoiseAndAdapt(double[] action) {
        // Remember the no-noise version so we can measure how big a difference the noise makes
        double[] cleanAction = Arrays.copyOf(action, action.length);

        // Apply OU noise scaled by actionNoise
        var scale = actionNoise.floatValue();
        var noiseSample = ouNoise.sample();
        for (int i = 0; i < action.length; i++)
            action[i] = Util.clampSafePolar(action[i] + scale * noiseSample[i]);

        // (Optional) one could also measure the difference here
    }

    /**
     * Store the experience in the replay buffer.
     */
    protected void remember(Tensor state, double[] action, double reward, Tensor nextState, boolean done) {
        var priority = memory.priMax();
        lastExperience = new Experience(state, action, reward, nextState.detach(), done);
        memory.add(lastExperience, priority);
    }

    @Override
    protected void reviseAction(double[] actionPrev) {
        if (lastAction != null)
            System.arraycopy(actionPrev, 0, lastAction, 0, actionPrev.length);
    }

    /**
     * Run a series of replay updates if enough experiences are in memory.
     */
    protected void update() {
        var replayCount = memory.size();
        if (replayCount == 0) return;

        var replays = Math.min(replayCount, this.replays.intValue());
        if (replays <= 0) return;

        var batchSize = Util.clamp(Math.round(this.batchSize.floatValue() * replays), 1, replays);
        var replaysRemain = replays;

        value.train(true);
        policy.train(true);

        synchronized (valueCtx) {
            valueLossNext = 0;
        }
        synchronized (policyCtx) {
            policyLossNext = 0;
        }

        // Reset mean TD Error for this update
        this.meanTDError = 0.0;

        // Break the replays into batches
        do {
            var batch = Math.min(batchSize, replaysRemain);
            updateBatch(batch);
            replaysRemain -= batch;
        } while (replaysRemain > 0);

        synchronized (valueCtx) {
            this.valueLoss = valueLossNext / replays;
        }
        synchronized (policyCtx) {
            this.policyLoss = policyLossNext / (outputs * replays);
        }

        // Update moving averages
        float valueLossMA = (float)this.valueLossMA.acceptAndGetMean(this.valueLoss);
        float tdErrMA = (float)this.tdErrMA.acceptAndGetMean(this.meanTDError);

        // Adjust actionNoise based on both moving averages
        updateNoise(valueLossMA, tdErrMA);

        // New: Adjust paramNoise in a similar way
        updateParamNoise(valueLossMA, tdErrMA);

        // Adapt OU noise parameters based on both moving averages
        ouNoise.adaptParameters(
                (float)this.valueLoss, valueLossMA,
                (float)this.meanTDError, tdErrMA);

        value.train(false);
        policy.train(false);
    }

    /**
     * Adjusts the action noise scale based on moving averages of Value Loss and TD Error.
     */
    private void updateNoise(float valueLossMA, float tdErrMA) {
        float valueLossDifference = (float)(this.valueLoss - valueLossMA);
        float tdErrorDifference = (float)(this.meanTDError - tdErrMA);

        float combinedDifference = valueLossFactor * valueLossDifference + tdErrorFactor * tdErrorDifference;

        float updatedNoise = actionNoise.floatValue() + (float)(actionNoiseAdaptRate * Util.clampSafePolar(combinedDifference));

        actionNoise.set(Util.unitize(updatedNoise));
    }

    /**
     * New method: Adjust the parameter noise scale based on moving averages.
     * This uses a similar update rule as for action noise.
     */
    private void updateParamNoise(float valueLossMA, float tdErrMA) {
        float valueLossDifference = (float)(this.valueLoss - valueLossMA);
        float tdErrorDifference = (float)(this.meanTDError - tdErrMA);

        float combinedDifference = valueLossFactor * valueLossDifference + tdErrorFactor * tdErrorDifference;

        float updatedParamNoise = paramNoise.floatValue() + (float)(paramNoiseAdaptRate * Util.clampSafePolar(combinedDifference));

        paramNoise.set(Util.unitize(updatedParamNoise));
        //System.out.println(paramNoise.asFloat());
    }

    private void updateBatch(int batchSize) {
        var sample = memory.sample(batchSize);
        var batch = sample.experiences;
        batchSize = batch.size(); // in case sample returned fewer

        var indices = sample.indices;

        // Accumulate TD Errors
        final double[] batchTDErrorSum = {0.0};

        TensorUtil.runParallel(0, batchSize, (s, e) -> {
            for (var i = s; i < e; i++) {
                var experience = batch.get(i);
                var tdError = update(experience);
                memory.update(indices.get(i), (float) (tdError));

                synchronized (this) {
                    batchTDErrorSum[0] += tdError;
                }
            }
        });

        // Compute mean TD Error for the batch
        double meanTDError = batchTDErrorSum[0] / batchSize;
        synchronized (this) {
            this.meanTDError = meanTDError;
        }

        valueCtx.div(batchSize).optimize(valueOpt);
        policyCtx.div(batchSize).optimize(policyOpt);
    }

    /** returns an error value */
    private double update(Experience e) {
        var valErr = updateValue(e);
        var polErr = updatePolicy(e.state);

        synchronized (policyCtx) {
            policyLossNext += polErr;
        }
        synchronized (valueCtx) {
            valueLossNext += valErr;
        }

        // Example: return L2 combined error
        return Math.sqrt(valErr * valErr + polErr * polErr);
    }

    /** @return err value for the value function */
    private double updateValue(Experience e) {
        var targetQ = value(e.nextState)
                .mul((e.done ? 0 : 1) * gamma.floatValue())
                .add(e.reward);

        var currentQ = value(e.state, e.action);

        var valueLoss = currentQ.loss(targetQ, valueLossFn).minimize(valueCtx);
        return Math.abs(valueLoss.scalar());
    }

    /** @return err value for the policy (magnitude of Q-value) */
    private double updatePolicy(Tensor state) {
        // let gradient flow through the policy
        var action = policy(state);
        var qValue = value(state, action).scalar();

        // maximize Q => minimize negative Q
        action.minimize(null, policyCtx, -qValue);

        return Math.abs(qValue);
    }

    private Tensor value(Tensor state) {
        // Detach so that target does not flow gradient back
        var detach = true;
        if (detach) state = state.detach();

        var actionEval = policy(state);
        if (detach) actionEval = actionEval.detach();

        var val = value(state, actionEval);
        if (detach) val = val.detach();

        return val;
    }

    private Tensor value(Tensor state, double[] action) {
        return value(state, Tensor.row(action));
    }

    private Tensor value(Tensor state, Tensor action) {
        return value.apply(state.concat(action));
    }

    private Tensor policy(Tensor state) {
        return policy.apply(state);
    }

    /**
     * Basic prioritized replay buffer.
     */
    public static class PrioritizedReplayBuffer {
        private final int capacity;
        private final float alpha; // how strongly to use prioritization
        private final float beta = 0.4f;
        private final float[] priorities;
        private final List<Experience> experiences;
        private int nextIndex;
        private volatile float maxPriority = Float.MIN_NORMAL;

        public PrioritizedReplayBuffer(int capacity, float alpha) {
            this.capacity = capacity;
            this.alpha = alpha;
            this.priorities = new float[capacity];
            this.experiences = new Lst<>(capacity);
            this.nextIndex = 0;
        }

        public synchronized void add(Experience experience, float priority) {
            if (size() < capacity) {
                experiences.add(experience);
                priorities[nextIndex] = priority;
                nextIndex = (nextIndex + 1) % capacity;
            } else {
                experiences.set(nextIndex, experience);
                priorities[nextIndex] = priority;
                nextIndex = (nextIndex + 1) % capacity;
            }
            if (priority > maxPriority) {
                maxPriority = priority;
            }
        }

        public synchronized SampleBatch sample(int batchSize) {
            var actualSize = size();
            if (actualSize == 0) {
                throw new IllegalStateException("Cannot sample from empty buffer");
            }
            var probabilities = new float[actualSize];
            var sum = 0.0f;
            for (var i = 0; i < actualSize; i++) {
                probabilities[i] = (float) Math.pow(priorities[i], alpha);
                sum += probabilities[i];
            }

            // Build cumulative sums
            var cumulativeSum = new float[actualSize];
            var cumsum = 0.0f;
            for (var i = 0; i < actualSize; i++) {
                cumsum += probabilities[i] / sum;
                cumulativeSum[i] = (float)cumsum;
            }

            var effectiveBatchSize = Math.min(batchSize, actualSize);

            List<Experience> batch = new Lst<>(effectiveBatchSize);
            IntArrayList indices = new IntArrayList(effectiveBatchSize);
            var isWeights = new float[effectiveBatchSize];

            var rng = ThreadLocalRandom.current();
            for (var i = 0; i < effectiveBatchSize; i++) {
                var r = rng.nextFloat();
                var index = binarySearch(cumulativeSum, r);
                // Clamp index just in case
                index = Math.min(actualSize - 1, index);
                batch.add(experiences.get(index));
                indices.add(index);

                // Compute importance sampling weight
                var p = probabilities[index] / sum;
                isWeights[i] = (float) Math.pow(effectiveBatchSize * p, -beta);
            }

            // Normalize IS weights
            var maxWeight = Util.max(isWeights);
            if (maxWeight > 0) {
                for (var i = 0; i < isWeights.length; i++) {
                    isWeights[i] /= maxWeight;
                }
            }

            return new SampleBatch(batch, indices, isWeights);
        }

        private static int binarySearch(float[] cumulativeSum, float value) {
            var low = 0;
            var high = cumulativeSum.length - 1;
            while (low <= high) {
                var mid = (low + high) >>> 1;
                if (value < cumulativeSum[mid]) {
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            }
            return low;
        }

        public synchronized void update(int index, float priority) {
            // small epsilon
            final float PRI_EPSILON = 1e-6f;
            priority += PRI_EPSILON;

            var old = priorities[index];
            priorities[index] = priority;

            // track max priority
            if (priority < old && old == maxPriority) {
                recalcMaxPriority();
            } else if (priority > maxPriority) {
                maxPriority = priority;
            }
        }

        private synchronized void recalcMaxPriority() {
            var nextMax = Float.MIN_NORMAL;
            var s = size();
            for (int i = 0; i < s; i++) {
                if (priorities[i] > nextMax) {
                    nextMax = priorities[i];
                }
            }
            maxPriority = nextMax;
        }

        public float priMax() {
            return maxPriority;
        }
        public synchronized double priMean() {
            return Util.mean(priorities, 0, size());
        }

        public int size() {
            return experiences.size();
        }

        public class SampleBatch {
            public final List<Experience> experiences;
            public final IntArrayList indices;
            public final float[] importanceSamplingWeights;

            public SampleBatch(List<Experience> experiences,
                               IntArrayList indices,
                               float[] isWeights) {
                this.experiences = experiences;
                this.indices = indices;
                this.importanceSamplingWeights = isWeights;
            }
        }
    }

    private static class Experience {
        final Tensor state;
        final Tensor action;
        final double reward;
        final Tensor nextState;
        final boolean done;

        Experience(Tensor state, double[] action, double reward, Tensor nextState, boolean done) {
            this.state = state;
            this.action = Tensor.row(action);
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }

    /**
     * Ornstein–Uhlenbeck noise for continuous exploration with autotuned parameters based on both Value Loss and TD Error.
     */
    private static class OUNoise {
        final RandomGenerator rng = new XoRoShiRo128PlusRandom();
        private double[] X;
        private double mu = 0.0;
        private double theta = 0.15;
        private double sigma = 0.4;
        private final double dt = 1.0;

        /** Distance from -1..+1 => radius 2 */
        private final double radius = 2.0;

        // Adaptation rates
        private final float thetaAdaptRate;
        private final float sigmaAdaptRate;

        OUNoise(int size, float thetaAdaptRate, float sigmaAdaptRate) {
            X = new double[size];
            reset();
            this.thetaAdaptRate = thetaAdaptRate;
            this.sigmaAdaptRate = sigmaAdaptRate;
        }

        void reset() {
            Arrays.fill(X, mu);
        }

        /**
         * dx = theta * (mu - x)*dt + sigma*sqrt(dt)*N(0,1)
         */
        double[] sample() {
            var n = X.length;
            var sqrtDT = Math.sqrt(dt);
            for (var i = 0; i < n; i++) {
                var x = X[i];
                var dx = dt * theta * (mu - x) + sigma * sqrtDT * rng.nextGaussian();
                X[i] = Util.clampSafePolar(x + dx, radius);
            }
            return X;
        }

        /**
         * Adjust theta and sigma based on the current value loss and TD error relative to their moving averages.
         * @param currentValueLoss The current mean value loss of the agent.
         * @param movingAverageValueLoss The moving average of the value loss.
         * @param currentTDError The current mean TD error of the agent.
         * @param movingAverageTDError The moving average of the TD error.
         */
        void adaptParameters(float currentValueLoss, float movingAverageValueLoss,
                             float currentTDError, float movingAverageTDError) {
            // Calculate loss differences
            float valueLossDifference = currentValueLoss - movingAverageValueLoss;
            float tdErrorDifference = currentTDError - movingAverageTDError;

            // Adjust theta and sigma based on both differences
            // Positive differences indicate higher errors -> increase exploration
            // Negative differences indicate lower errors -> decrease exploration
            theta += thetaAdaptRate * (valueLossDifference + tdErrorDifference);
            sigma += sigmaAdaptRate * (valueLossDifference + tdErrorDifference);

            // Clamp theta and sigma to ensure stability
            theta = Util.clamp(theta, 0.01, 1.0); // theta should be positive
            sigma = Util.clamp(sigma, 0.0, 5.0);   // sigma should be non-negative
        }
    }
}
