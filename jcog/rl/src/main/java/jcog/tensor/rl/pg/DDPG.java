package jcog.tensor.rl.pg;

import jcog.Util;
import jcog.data.list.Lst;
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
 * An older implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.
 * This class is part of an older API.
 *
 * @deprecated This class is part of an older API. For DDPG or similar off-policy actor-critic algorithms,
 *             consider using or developing solutions based on the `jcog.tensor.rl.pg3` components and patterns,
 *             or using the `jcog.tensor.rl.pg2.DDPGStrategy` if it's still maintained and suitable.
 *             The `pg3` package is the focus for new development.
 */
@Deprecated
public class DDPG extends AbstractPG {

    /** 'actor' */
    public final Models.Layers policy;

    /** 'critic' */
    public final Models.Layers value;

    public final FloatRange gamma = new FloatRange(0.9f, 0, 1);

    public final IntRange replays = new IntRange(8, 1, 256);

    public final FloatRange batchSize = new FloatRange(0.1f, 0, 1);

    public final FloatRange policyLR = new FloatRange(
            0.00003f //ADAM
            //0.003f //SGD
            , 0, 0.01f);

    public final FloatRange valueLR = new FloatRange(policyLR.floatValue()*(10/3f), 0, 0.01f);

    public final FloatRange actionNoise = new FloatRange(1/2f, 0, 1);
    public final FloatRange paramNoise = new FloatRange(0, 0, 0.01f);

    private final int replayBufferCapacity = 1 * 1024;
    public final PrioritizedReplayBuffer memory = new PrioritizedReplayBuffer(replayBufferCapacity, 0.6f);

    private final Tensor.Optimizer policyOpt, valueOpt;
    private final OUNoise ouNoise;

    private final Tensor.GradQueue valueCtx = new Tensor.GradQueue(), policyCtx = new Tensor.GradQueue();

    public double policyLoss, valueLoss, policyLossNext, valueLossNext;


    private Experience lastExperience;

    private Tensor lastState;
    double[] lastAction;

    private static final Tensor.Loss valueLossFn =
        Tensor.Loss.MeanSquared
        //Tensor.Loss.Huber
    ;

    final UnaryOperator<Tensor> valueOutputActivation = null;
    final UnaryOperator<Tensor> policyOutputActivation =
        Tensor::clipUnitPolar;
        //Tensor::clipTanh;

    public DDPG(int inputs, int outputs, int hiddenSize, int valueSize) {
        super(inputs, outputs);

        ouNoise = new OUNoise(outputs);

        policy = model(inputs, hiddenSize, outputs, policyOutputActivation, true, 2);
        value = model(inputs + outputs, valueSize, 1, valueOutputActivation, true, 2);

        policy.train(false);
        value.train(false);

        policyOpt =
            new Optimizers.ADAM(policyLR).get();
            //new Optimizers.ADAM(policyLR).get(1024);
            //new Optimizers.SGD(policyLR).get();
            //new Optimizers.SGDMomentum(policyLR, 0.99f).get(1024);
        valueOpt =
            new Optimizers.ADAM(valueLR).get();
            //new Optimizers.SGD(valueLR).get(1024);
            //new Optimizers.SGDMomentum(valueLR, 0.99f).get(1024);

        policyOpt.step.add(new Optimizers.ParamNoise(paramNoise));
        valueOpt.step.add(new Optimizers.ParamNoise(paramNoise));
    }

    private Models.Layers model(int inputSize, int hiddenSize, int outputSize, UnaryOperator<Tensor> outputActivation, boolean biasInLastLayer, int layerCount) {
        return new Models.Layers(
            Tensor.RELU,
            //Tensor.RELU_LEAKY,
            outputActivation,
            biasInLastLayer,
            Models.Layers.layerLerp(inputSize, hiddenSize, outputSize, layerCount)
        ) {
            @Override
            protected void afterLayer(int O, int l, int maxLayers) {
//                var innerLayer = l < maxLayers - 1;
//                if (innerLayer && O > 8)
//                    layer.add(new Models.Dropout(0.25f));
            }
        };
    }

    @Override
    protected double[] _action(Tensor state) {
        return policy(state).detach().array();
    }

    @Override
    public double[] act(double[] input, double reward) {
        return act(input, reward, false, true);
    }

    public double[] act(double[] _state, double reward, boolean done, boolean training) {
        var state = Tensor.row(_state);

        var isFirstStep = lastState == null;
        if (!isFirstStep)
            remember(lastState, lastAction, reward, state, done);

        var action = action(state);

        if (training) {
            actionNoise(action);
            update();
        }

        if (done)
            lastState = null;
        else {
            lastState = state;
            lastAction = action;
        }

        return action;
    }

    private void actionNoise(double[] action) {
        //System.out.println(Str.n2(action));
        var n = actionNoise.floatValue();
        var noise = ouNoise.sample();
        for (var i = 0; i < action.length; i++)
            //action[i] = Util.clampSafePolar(action[i] + n * noise[i]);
            action[i] = Math.tanh(action[i] + n * noise[i]);
    }

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

    protected void update() {
        var replayCount = memory.size();
        if (replayCount == 0) return;
        var replays = Math.min(replayCount, this.replays.intValue());
        if (replays <= 0) return;

        var batchSize = Util.clamp(Math.round(this.batchSize.floatValue() * replays), 1, replays);
        var replaysRemain = replays;

        value.train(true);
        policy.train(true);

        synchronized(valueCtx) { valueLossNext = 0; }
        synchronized(policyCtx) { policyLossNext = 0; }

        do {
            var batch = Math.min(batchSize, replaysRemain);
            updateBatch(batch);
            replaysRemain -= batch;
        } while (replaysRemain > 0);

        synchronized(valueCtx) { this.valueLoss = valueLossNext / replays; }
        synchronized(policyCtx) { this.policyLoss = policyLossNext / (outputs * replays); }

        value.train(false);
        policy.train(false);
    }

    private void updateBatch(int batchSize) {
        var sample = memory.sample(batchSize);

        var batch = sample.experiences;
        batchSize = batch.size(); //update in case different

        var indices = sample.indices;

        TensorUtil.runParallel(0, batchSize, (s, e) -> {
            for (var i = s; i < e; i++) {
                var experience = batch.get(i);
                var tdError = update(experience);
                memory.update(indices.get(i), (float)(tdError));
            }
        });

        valueCtx.div(batchSize).optimize(valueOpt);
        policyCtx.div(batchSize).optimize(policyOpt);
    }

    /** returns an error value */
    private double update(Experience e) {
        var valErr = updateValue(e);
        var polErr = updatePolicy(e.state);

        synchronized(policyCtx) { policyLossNext += polErr; }
        synchronized(valueCtx) { this.valueLossNext += valErr; }

        //return valErr + polErr; //L1
        return Math.sqrt(valErr*valErr + polErr*polErr); //L2
    }

    /** @return err value */
    private double updateValue(Experience e) {
        var targetQ = value(e.nextState)
                .mul((e.done ? 0 : 1) * gamma.floatValue())
                .add(e.reward);

        var currentQ = value(e.state, e.action);

        var valueLoss = currentQ.loss(targetQ, valueLossFn).minimize(valueCtx);

        return Math.abs(valueLoss.scalar());
        //return currentQ.sub(targetQ).scalar();
    }

    /** @return err value */
    private double updatePolicy(Tensor state) {
        var action = policy(state);  //gradients flow through the policy

        var qValue = value(state, action).scalar();

        action.minimize(null, policyCtx, -qValue);

        return Math.abs(qValue);
    }

    private Tensor value(Tensor state) {
        /* Detaching the target Q-value is common in Q-learning algorithms to stabilize training. It prevents gradients from flowing through the target network. */
        var detach = true;

        if (detach) state = state.detach();

        var action = policy(state);
        if (detach) action = action.detach();

        var value = value(state, action);
        if (detach) value = value.detach();

        return value;
    }

    private Tensor value(Tensor state, Tensor action) {
        return value.apply(state.concat(action));
    }

    private Tensor policy(Tensor state) {
        return policy.apply(state);
    }

    public static class PrioritizedReplayBuffer {
        private final int capacity;
        private final float alpha; // Determines how much prioritization is used
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
            if (priority > maxPriority)
                maxPriority = priority;
        }

        public synchronized SampleBatch sample(int batchSize) {
            var actualSize = size();
            if (actualSize == 0)
                throw new IllegalStateException("Cannot sample from empty buffer");

            // Calculate probabilities for the actual number of experiences
            var probabilities = new float[actualSize];
            var sum = 0.0f;
            for (var i = 0; i < actualSize; i++) {
                probabilities[i] = (float) Math.pow(priorities[i], alpha);
                sum += probabilities[i];
            }

            // Compute cumulative sum for sampling
            var cumulativeSum = new float[actualSize];
            var cumsum = 0.0f;
            for (var i = 0; i < actualSize; i++) {
                cumsum += probabilities[i] / sum; // Normalize while building cumsum
                cumulativeSum[i] = cumsum;
            }

            // Adjust batch size if it's larger than available experiences
            var effectiveBatchSize = Math.min(batchSize, actualSize);

            List<Experience> batch = new Lst<>(effectiveBatchSize);
            IntArrayList indices = new IntArrayList(effectiveBatchSize);
            var isWeights = new float[effectiveBatchSize];

            // Calculate current beta for annealing (example implementation)
            //float beta = calculateCurrentBeta(); // Should be implemented to anneal from 0.4 to 1.0

            var rng = ThreadLocalRandom.current();
            for (var i = 0; i < effectiveBatchSize; i++) {
                var random = rng.nextFloat();
                var index = binarySearch(cumulativeSum, random);
                index = Math.min(actualSize-1,index); //HACK
                batch.add(experiences.get(index));
                indices.add(index);

                // Compute importance sampling weight with annealed beta
                var p = probabilities[index] / sum;
                isWeights[i] = (float) Math.pow(effectiveBatchSize * p, -beta);
            }

            // Normalize IS weights
            var maxWeight = Util.max(isWeights);
            if (maxWeight > 0) {
                for (var i = 0; i < isWeights.length; i++)
                    isWeights[i] /= maxWeight;
            }

            return new SampleBatch(batch, indices, isWeights);
        }

        // Additional method to handle maxPriority maintenance
        private synchronized void recalculateMaxPriority() {
            var nextMaxPriority = Float.MIN_NORMAL;
            var s = size();
            for (var i = 0; i < s; i++) {
                var p = priorities[i];
                if (p > maxPriority)
                    nextMaxPriority = p;
            }
            this.maxPriority = nextMaxPriority;
        }

        private static final float PRI_EPSILON = 1e-6f;

        public synchronized void update(int index, float priority) {
            priority += PRI_EPSILON;
            var priBefore = priorities[index];
            priorities[index] = priority;
            if (priority < priBefore && priBefore==maxPriority)
                recalculateMaxPriority();
            else if (priority > maxPriority)
                maxPriority = priority;
        }

        public float priMax() {
            return maxPriority;
        }
        public synchronized double priMean() {
            return Util.mean(priorities, 0, size());
        }

        private static int binarySearch(float[] cumulativeSum, float value) {
            var low = 0;
            var high = cumulativeSum.length - 1;
            while (low <= high) {
                var mid = (low + high) / 2;
                if (value < cumulativeSum[mid])
                    high = mid - 1;
                else
                    low = mid + 1;
            }
            return low;
        }

        public int size() {
            return experiences.size();
        }

        /**
         * Helper class to store sampled batch data
         */
        public class SampleBatch {
            public final List<Experience> experiences;
            public final IntArrayList indices;
            public final float[] importanceSamplingWeights;

            public SampleBatch(List<Experience> experiences, IntArrayList indices, float[] isWeights) {
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

    static class OUNoise {
        final RandomGenerator rng = new XoRoShiRo128PlusRandom();
        private final double[] X;
        private final double mu = 0;
        double theta = 0.15;
        double sigma = 0.4;
        private final double dt = 1;

        /** dist from -1..+1 */
        private final double radius = +2;

        OUNoise(int size) {
            X = new double[size];
            reset();
        }

        void reset() {
            Arrays.fill(X, mu);
        }

        /**
         *   dx = theta * (mu - x) + sigma * rng_gaussian
         *   x += dx
         */
        double[] sample() {
            var n = X.length;
            var sqrtDT = Math.sqrt(dt);
            for (var i = 0; i < n; i++) {
                var x = X[i];
                var dx =
                    dt * theta * (mu - x) +
                    sqrtDT * sigma * rng.nextGaussian();
                X[i] = x + dx;
                //X[i] = Util.clampSafePolar(x + dx, radius);
            }
            return X;
        }
    }
}