//package jcog.tensor.rl.pg;
//
//import jcog.Util;
//import jcog.data.list.Lst;
//import jcog.math.FloatMeanEwma;
//import jcog.random.XoRoShiRo128PlusRandom;
//import jcog.signal.FloatRange;
//import jcog.tensor.Models;
//import jcog.tensor.Optimizers;
//import jcog.tensor.Tensor;
//import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
//
//import java.util.Arrays;
//import java.util.List;
//import java.util.concurrent.ThreadLocalRandom;
//import java.util.random.RandomGenerator;
//
///**
// * SACHybridAgent implements a state-of-the-art, off-policy actor–critic algorithm that
// * combines elements from Soft Actor–Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradients (TD3).
// *
// * <p>Key Features:
// * <ul>
// *   <li>Twin Q-networks with target networks to mitigate overestimation bias.
// *   <li>A stochastic actor outputting mean and log–standard deviation, trained with an entropy bonus.
// *   <li>Adaptive temperature parameter (alpha) that controls the tradeoff between reward and exploration.
// *   <li>Delayed policy updates and soft target updates (controlled by tau) to improve stability.
// *   <li>Adaptive exploration via OU noise for target policy smoothing and parameter noise attached to optimizers.
// * </ul>
// * </p>
// *
// * <p>This implementation is compact, self–documenting, and designed for clarity while reflecting current best practices.</p>
// */
//public class SACHybridAgent extends AbstractPG {
//
//    // ===== Network Definitions =====
//    // Actor networks: policy (online) and targetPolicy.
//    public final Models.Layers policy;
//    public final Models.Layers targetPolicy;
//
//    // Twin Q-networks (critics) and their target networks.
//    public final Models.Layers q1, q2;
//    public final Models.Layers targetQ1, targetQ2;
//
//    // ===== Adaptive Hyperparameters =====
//    public final FloatRange gamma = new FloatRange(0.99f, 0, 1);
//    public final FloatRange tau = new FloatRange(0.005f, 0, 0.1f); // Soft-update rate.
//    public final FloatRange alpha = new FloatRange(0.2f, 1e-4f, 5f);  // Temperature for entropy bonus.
//    private final float targetEntropy; // Typically set to -|actionDim|.
//
//    // Exploration noise parameters.
//    public final FloatRange actionNoise = new FloatRange(0.5f, 0, 1);
//    public final FloatRange paramNoise = new FloatRange(0.005f, 0, 0.01f);
//
//    // Learning rates.
//    public final FloatRange policyLR = new FloatRange(0.0003f, 1e-6f, 0.01f);
//    public final FloatRange criticLR = new FloatRange(0.001f, 1e-6f, 0.01f);
//    public final FloatRange temperatureLR = new FloatRange(0.0003f, 1e-6f, 0.01f);
//
//    // ===== Optimizers =====
//    private final Tensor.Optimizer policyOpt, q1Opt, q2Opt, temperatureOpt;
//
//    // ===== Replay Buffer =====
//    private final int replayBufferCapacity = 4096;
//    public final PrioritizedReplayBuffer memory = new PrioritizedReplayBuffer(replayBufferCapacity, 0.6f);
//
//    // ===== Update Scheduling =====
//    private int updateCounter = 0;
//    private final int policyUpdateDelay = 2; // Delayed policy updates.
//
//    // ===== Exploration Noise =====
//    private final OUNoise ouNoise;
//
//    // ===== Utilities =====
//    // For error tracking (if desired for adaptive noise adjustments).
//    private final FloatMeanEwma errorMA = new FloatMeanEwma().period(16);
//
//    // Temporary storage for the last experience (if needed).
//    private Tensor lastState, lastAction;
//
//    private final Tensor.GradQueue ctx1 = new Tensor.GradQueue();
//    private final Tensor.GradQueue ctx2 = new Tensor.GradQueue();
//    private final Tensor.GradQueue ctxp = new Tensor.GradQueue();
//
//    // ===== Constructor =====
//    public SACHybridAgent(int stateDim, int actionDim, int hiddenPolicy, int hiddenCritic) {
//        super(stateDim, actionDim);
//        // Set target entropy to -actionDim (a common heuristic).
//        this.targetEntropy = -actionDim;
//
//        // Build actor and critic networks.
//        policy = buildActor(stateDim, hiddenPolicy, actionDim);
//        targetPolicy = buildActor(stateDim, hiddenPolicy, actionDim);
//        copyParameters(policy, targetPolicy);
//
//        q1 = buildCritic(stateDim + actionDim, hiddenCritic);
//        q2 = buildCritic(stateDim + actionDim, hiddenCritic);
//        targetQ1 = buildCritic(stateDim + actionDim, hiddenCritic);
//        targetQ2 = buildCritic(stateDim + actionDim, hiddenCritic);
//        copyParameters(q1, targetQ1);
//        copyParameters(q2, targetQ2);
//
//        // Set networks to evaluation mode initially.
//        policy.train(false); targetPolicy.train(false);
//        q1.train(false); q2.train(false);
//        targetQ1.train(false); targetQ2.train(false);
//
//        // Build optimizers.
//        policyOpt = new Optimizers.ADAM(policyLR).get();
//        q1Opt = new Optimizers.ADAM(criticLR).get();
//        q2Opt = new Optimizers.ADAM(criticLR).get();
//        temperatureOpt = new Optimizers.ADAM(temperatureLR).get();
//
//        // Attach parameter noise to optimizers.
//        policyOpt.step.add(new Optimizers.ParamNoise(paramNoise));
//        q1Opt.step.add(new Optimizers.ParamNoise(paramNoise));
//        q2Opt.step.add(new Optimizers.ParamNoise(paramNoise));
//
//        // Initialize OU noise for exploration smoothing.
//        ouNoise = new OUNoise(actionDim, 0.005f, 0.005f);
//    }
//
//    // ===== Network Builders =====
//
//    /**
//     * Builds the stochastic actor network. The network outputs 2*actionDim values:
//     * the first half represents the mean and the second half the log–std.
//     */
//    private Models.Layers buildActor(int inputSize, int hiddenSize, int actionDim) {
//        // Use ReLU for hidden layers and tanh to squash outputs.
//        return new Models.Layers(
//                Tensor::relu,
//                SACHybridAgent::tanhActivation,
//                true,
//                Models.Layers.layerLerp(inputSize, hiddenSize, actionDim * 2, 3, 0.5f)
//        );
//    }
//
//    /**
//     * Builds a critic network that outputs a scalar Q–value.
//     */
//    private Models.Layers buildCritic(int inputSize, int hiddenSize) {
//        return new Models.Layers(
//                Tensor::relu,
//                null,
//                true,
//                Models.Layers.layerLerp(inputSize, hiddenSize, 1, 3, 0.5f)
//        );
//    }
//
//    // Utility: Copy parameters from source to target.
//    private void copyParameters(Models.Layers source, Models.Layers target) {
//        target.copyFrom(source);
//    }
//
//    // Squashing activation (e.g., tanh) for actor outputs.
//    private static Tensor tanhActivation(Tensor t) {
//        return t.tanh();
//    }
//
//    // ===== Action Selection =====
//
//    @Override
//    protected double[] _action(Tensor state) {
//        // Compute actor output and split into mean and logStd.
//        Tensor output = policy.apply(state).detach();
//        int actDim = output.cols() / 2;
//        Tensor mu = output.slice(0, actDim);
//        Tensor logStd = output.slice(actDim, actDim * 2).clip(-20, 2);
//        Tensor std = logStd.exp();
//        double[] eps = sampleStandardNormal(actDim);
//        Tensor actionSample = mu.add(std.mul(eps));
//
//        double[] actionArr = actionSample.array(); // Add exploration noise.
//        addOUActionNoise(actionArr);
//        Util.clamp(actionArr, actionArr, -1, +1);
//        return actionArr;
//    }
//
//    private double[] sampleStandardNormal(int dim) {
//        double[] eps = new double[dim];
//        for (int i = 0; i < dim; i++)
//            eps[i] = ThreadLocalRandom.current().nextGaussian();
//        return eps;
//    }
//
//    /**
//     * Adds OU noise to the action for exploration.
//     */
//    private void addOUActionNoise(double[] action) {
//        double[] noise = ouNoise.sample();
//        float scale = actionNoise.floatValue();
//        for (int i = 0; i < action.length; i++) {
//            action[i] = Util.clampSafePolar(action[i] + scale * noise[i]);
//        }
//    }
//
//    /**
//     * The public act() method. It receives the current state and reward,
//     * stores the experience, triggers an update, and returns the action.
//     */
//    @Override
//    public double[] act(double[] input, double reward) {
//        Tensor state = Tensor.row(input);
//        double[] action = _action(state);
//        // If there is a previous state, assume we have a transition.
//        if (lastState != null) {
//            // In practice, the next state should be obtained from the environment.
//            // Here we use the current state as a placeholder.
//            memory.add(new Experience(lastState, lastAction, reward, state, false), memory.priMax());
//        }
//        lastState = state;
//        lastAction = Tensor.row(action);
//        update();
//        return action;
//    }
//
//    // ===== Update Routine =====
//
//    /**
//     * Performs one update step: samples a mini-batch, updates critics,
//     * and, on a delayed schedule, updates the actor and temperature.
//     */
//    protected void update() {
//        final int batchSize = 64;
//        if (memory.size() < batchSize) return;
//
//        List<Experience> batch = memory.sample(batchSize).experiences;
//
//        // Update critics.
//        double criticLoss = 0.0;
//        for (Experience exp : batch) {
//            criticLoss += updateCritics(exp);
//        }
//        criticLoss /= batch.size();
//
//        ctx1.optimize(q1Opt);
//        ctx2.optimize(q2Opt);
//
//        // Delayed actor and temperature update.
//        updateCounter++;
//        if (updateCounter % policyUpdateDelay == 0) {
//            updateActor(batch);
//            updateTemperature(batch);
//            softUpdate(targetPolicy, policy, tau.floatValue());
//            softUpdate(targetQ1, q1, tau.floatValue());
//            softUpdate(targetQ2, q2, tau.floatValue());
//        }
//
//        ctxp.optimize(policyOpt);
//    }
//
//    /**
//     * Updates twin critics using the Bellman target with an entropy bonus.
//     */
//    private double updateCritics(Experience exp) {
//        // Sample next action from target policy.
//        Tensor nextOutput = targetPolicy.apply(exp.nextState).detach();
//        int actDim = nextOutput.cols() / 2;
//        Tensor nextMu = nextOutput.slice(0, actDim);
//        Tensor nextLogStd = nextOutput.slice(actDim, actDim * 2).clip(-20, 2);
//        Tensor nextStd = nextLogStd.exp();
//        double[] eps = sampleStandardNormal(actDim);
//        Tensor nextAction = nextMu.add(nextStd.mul(eps));
//        // Apply smoothing noise.
//        double[] smoothingNoise = ouNoise.sample();
//        for (int i = 0; i < nextAction.array().length; i++) {
//            nextAction.array()[i] = Util.clampSafePolar(nextAction.array()[i] + 0.2 * smoothingNoise[i]);
//        }
//        Tensor nextStateAction = exp.nextState.concat(nextAction);
//        Tensor targetQ = Tensor.min(targetQ1.apply(nextStateAction), targetQ2.apply(nextStateAction));
//        // Entropy term.
//        var logProb = computeLogProbability(nextMu, nextLogStd, nextAction);
//        var entropyBonus = logProb.mul(-alpha.floatValue());
//        Tensor target = Tensor.scalar(exp.reward)
//                            .add(targetQ.mul(gamma.floatValue()))
//                            .add(entropyBonus);
//        // Current Q estimates.
//        Tensor currentQ = exp.state.concat(exp.action);
//        Tensor currentQ1 = q1.apply(currentQ);
//        Tensor currentQ2 = q2.apply(currentQ);
//        var loss1 = currentQ1.loss(target, Tensor.Loss.MeanSquared).minimize(null);
//        var loss2 = currentQ2.loss(target, Tensor.Loss.MeanSquared).minimize(null);
//        loss1.minimize(q1Opt, ctx1);
//        loss2.minimize(q2Opt, ctx2);
//        return Math.sqrt(Util.sqr(loss1.scalar()) + Util.sqr(loss2.scalar()));
//    }
//
//    /**
//     * Updates the actor network using the reparameterization trick and an entropy bonus.
//     */
//    private void updateActor(List<Experience> batch) {
//        double totalLoss = 0.0;
//        for (Experience exp : batch) {
//            Tensor output = policy.apply(exp.state);
//            int actDim = output.cols() / 2;
//            Tensor mu = output.slice(0, actDim);
//            Tensor logStd = output.slice(actDim, actDim * 2).clip(-20, 2);
//            Tensor std = logStd.exp();
//            double[] eps = sampleStandardNormal(actDim);
//            Tensor actionSample = mu.add(std.mul(eps));
//            var logProb = computeLogProbability(mu, logStd, actionSample);
//            Tensor stateAction = exp.state.concat(actionSample);
//            Tensor qVal = Tensor.min(q1.apply(stateAction), q2.apply(stateAction));
//            // Actor loss: maximize Q + alpha * entropy ==> minimize negative of that.
//            var loss = (qVal.add(logProb.mul(alpha.floatValue())).neg();
//            totalLoss += Math.abs(loss.scalar());
//            actionSample.minimize(null, null, loss);
//            loss.minimize(q1Opt, ctxp);
//        }
//    }
//
//    /**
//     * Adjusts the temperature parameter alpha so that the average entropy moves toward targetEntropy.
//     */
//    private void updateTemperature(List<Experience> batch) {
//        double entropySum = 0.0;
//        for (Experience exp : batch) {
//            Tensor output = policy.apply(exp.state).detach();
//            int actDim = output.cols() / 2;
//            Tensor mu = output.slice(0, actDim);
//            Tensor logStd = output.slice(actDim, actDim * 2).clamp(-20, 2);
//            Tensor std = logStd.exp();
//            double[] eps = sampleStandardNormal(actDim);
//            Tensor actionSample = mu.add(std.mul(eps));
//            var logProb = computeLogProbability(mu, logStd, actionSample);
//            entropySum += logProb.scalar();
//        }
//        double avgEntropy = entropySum / batch.size();
//        double temperatureLoss = -alpha.floatValue() * (avgEntropy - targetEntropy);
//        // A simple gradient step update:
//        alpha.set(Util.unitize((float) (alpha.floatValue() - temperatureLR.floatValue() * temperatureLoss)));
//        temperatureOpt.step();
//    }
//
//    /**
//     * Computes the log-probability of an action under a diagonal Gaussian.
//     */
//    private Tensor computeLogProbability(Tensor mu, Tensor logStd, Tensor action) {
//        Tensor diff = action.sub(mu);
//        // log-probability per dimension.
//        Tensor logProb = diff.pow(2).div(logStd.exp().pow(2).mul(2))
//                .add(logStd)
//                .add(Math.log(Math.sqrt(2 * Math.PI)));
//        return logProb.sum().neg();
//    }
//
//    /**
//     * Performs a soft update of target network parameters.
//     */
//    private void softUpdate(Models.Layers target, Models.Layers source, float tau) {
//        target.softUpdateFrom(source, tau);
//    }
//
//    // ===== OU Noise Implementation =====
//
//    private static class OUNoise {
//        final RandomGenerator rng = ThreadLocalRandom.current();
//        private final double[] X;
//        private final double mu = 0.0;
//        private double theta = 0.15;
//        private double sigma = 0.4;
//        private final double dt = 1.0;
//        private final double radius = 2.0;
//
//        OUNoise(int size, float thetaAdaptRate, float sigmaAdaptRate) {
//            X = new double[size];
//            reset();
//        }
//
//        void reset() {
//            for (int i = 0; i < X.length; i++) {
//                X[i] = mu;
//            }
//        }
//
//        double[] sample() {
//            double sqrtDT = Math.sqrt(dt);
//            for (int i = 0; i < X.length; i++) {
//                double dx = dt * theta * (mu - X[i]) + sigma * sqrtDT * rng.nextGaussian();
//                X[i] = Util.clampSafePolar(X[i] + dx, radius);
//            }
//            return X;
//        }
//    }
//
//    // ===== Experience Class =====
//
//    public static class Experience {
//        public final Tensor state;
//        public final Tensor action;
//        public final double reward;
//        public final Tensor nextState;
//        public final boolean done;
//
//        public Experience(Tensor state, Tensor action, double reward, Tensor nextState, boolean done) {
//            this.state = state;
//            this.action = action;
//            this.reward = reward;
//            this.nextState = nextState;
//            this.done = done;
//        }
//    }
//
//    public static class PrioritizedReplayBuffer {
//        private final int capacity;
//        private final float alpha; // how strongly to use prioritization
//        private final float beta = 0.4f;
//        private final float[] priorities;
//        private final List<Experience> experiences;
//        private int nextIndex;
//        private volatile float maxPriority = Float.MIN_NORMAL;
//
//        public PrioritizedReplayBuffer(int capacity, float alpha) {
//            this.capacity = capacity;
//            this.alpha = alpha;
//            this.priorities = new float[capacity];
//            this.experiences = new Lst<>(capacity);
//            this.nextIndex = 0;
//        }
//
//        public synchronized void add(Experience experience, float priority) {
//            if (size() < capacity) {
//                experiences.add(experience);
//                priorities[nextIndex] = priority;
//                nextIndex = (nextIndex + 1) % capacity;
//            } else {
//                experiences.set(nextIndex, experience);
//                priorities[nextIndex] = priority;
//                nextIndex = (nextIndex + 1) % capacity;
//            }
//            if (priority > maxPriority) {
//                maxPriority = priority;
//            }
//        }
//
//        public synchronized PrioritizedReplayBuffer.SampleBatch sample(int batchSize) {
//            var actualSize = size();
//            if (actualSize == 0) {
//                throw new IllegalStateException("Cannot sample from empty buffer");
//            }
//            var probabilities = new float[actualSize];
//            var sum = 0.0f;
//            for (var i = 0; i < actualSize; i++) {
//                probabilities[i] = (float) Math.pow(priorities[i], alpha);
//                sum += probabilities[i];
//            }
//
//            // Build cumulative sums
//            var cumulativeSum = new float[actualSize];
//            var cumsum = 0.0f;
//            for (var i = 0; i < actualSize; i++) {
//                cumsum += probabilities[i] / sum;
//                cumulativeSum[i] = (float)cumsum;
//            }
//
//            var effectiveBatchSize = Math.min(batchSize, actualSize);
//
//            List<Experience> batch = new Lst<>(effectiveBatchSize);
//            IntArrayList indices = new IntArrayList(effectiveBatchSize);
//            var isWeights = new float[effectiveBatchSize];
//
//            var rng = ThreadLocalRandom.current();
//            for (var i = 0; i < effectiveBatchSize; i++) {
//                var r = rng.nextFloat();
//                var index = binarySearch(cumulativeSum, r);
//                // Clamp index just in case
//                index = Math.min(actualSize - 1, index);
//                batch.add(experiences.get(index));
//                indices.add(index);
//
//                // Compute importance sampling weight
//                var p = probabilities[index] / sum;
//                isWeights[i] = (float) Math.pow(effectiveBatchSize * p, -beta);
//            }
//
//            // Normalize IS weights
//            var maxWeight = Util.max(isWeights);
//            if (maxWeight > 0) {
//                for (var i = 0; i < isWeights.length; i++) {
//                    isWeights[i] /= maxWeight;
//                }
//            }
//
//            return new PrioritizedReplayBuffer.SampleBatch(batch, indices, isWeights);
//        }
//
//        private static int binarySearch(float[] cumulativeSum, float value) {
//            var low = 0;
//            var high = cumulativeSum.length - 1;
//            while (low <= high) {
//                var mid = (low + high) >>> 1;
//                if (value < cumulativeSum[mid]) {
//                    high = mid - 1;
//                } else {
//                    low = mid + 1;
//                }
//            }
//            return low;
//        }
//
//        public synchronized void update(int index, float priority) {
//            // small epsilon
//            final float PRI_EPSILON = 1e-6f;
//            priority += PRI_EPSILON;
//
//            var old = priorities[index];
//            priorities[index] = priority;
//
//            // track max priority
//            if (priority < old && old == maxPriority) {
//                recalcMaxPriority();
//            } else if (priority > maxPriority) {
//                maxPriority = priority;
//            }
//        }
//
//        private synchronized void recalcMaxPriority() {
//            var nextMax = Float.MIN_NORMAL;
//            var s = size();
//            for (int i = 0; i < s; i++) {
//                if (priorities[i] > nextMax) {
//                    nextMax = priorities[i];
//                }
//            }
//            maxPriority = nextMax;
//        }
//
//        public float priMax() {
//            return maxPriority;
//        }
//        public synchronized double priMean() {
//            return Util.mean(priorities, 0, size());
//        }
//
//        public int size() {
//            return experiences.size();
//        }
//
//        public class SampleBatch {
//            public final List<Experience> experiences;
//            public final IntArrayList indices;
//            public final float[] importanceSamplingWeights;
//
//            public SampleBatch(List<Experience> experiences,
//                               IntArrayList indices,
//                               float[] isWeights) {
//                this.experiences = experiences;
//                this.indices = indices;
//                this.importanceSamplingWeights = isWeights;
//            }
//        }
//    }
//
//    /**
//     * Ornstein–Uhlenbeck noise for continuous exploration with autotuned parameters based on both Value Loss and TD Error.
//     */
//    private static class OUNoise {
//        final RandomGenerator rng = new XoRoShiRo128PlusRandom();
//        private double[] X;
//        private double mu = 0.0;
//        private double theta = 0.15;
//        private double sigma = 0.4;
//        private final double dt = 1.0;
//
//        /** Distance from -1..+1 => radius 2 */
//        private final double radius = 2.0;
//
//        // Adaptation rates
//        private final float thetaAdaptRate;
//        private final float sigmaAdaptRate;
//
//        OUNoise(int size, float thetaAdaptRate, float sigmaAdaptRate) {
//            X = new double[size];
//            reset();
//            this.thetaAdaptRate = thetaAdaptRate;
//            this.sigmaAdaptRate = sigmaAdaptRate;
//        }
//
//        void reset() {
//            Arrays.fill(X, mu);
//        }
//
//        /**
//         * dx = theta * (mu - x)*dt + sigma*sqrt(dt)*N(0,1)
//         */
//        double[] sample() {
//            var n = X.length;
//            var sqrtDT = Math.sqrt(dt);
//            for (var i = 0; i < n; i++) {
//                var x = X[i];
//                var dx = dt * theta * (mu - x) + sigma * sqrtDT * rng.nextGaussian();
//                X[i] = Util.clampSafePolar(x + dx, radius);
//            }
//            return X;
//        }
//
//        /**
//         * Adjust theta and sigma based on the current value loss and TD error relative to their moving averages.
//         * @param currentValueLoss The current mean value loss of the agent.
//         * @param movingAverageValueLoss The moving average of the value loss.
//         * @param currentTDError The current mean TD error of the agent.
//         * @param movingAverageTDError The moving average of the TD error.
//         */
//        void adaptParameters(float currentValueLoss, float movingAverageValueLoss,
//                             float currentTDError, float movingAverageTDError) {
//            // Calculate loss differences
//            float valueLossDifference = currentValueLoss - movingAverageValueLoss;
//            float tdErrorDifference = currentTDError - movingAverageTDError;
//
//            // Adjust theta and sigma based on both differences
//            // Positive differences indicate higher errors -> increase exploration
//            // Negative differences indicate lower errors -> decrease exploration
//            theta += thetaAdaptRate * (valueLossDifference + tdErrorDifference);
//            sigma += sigmaAdaptRate * (valueLossDifference + tdErrorDifference);
//
//            // Clamp theta and sigma to ensure stability
//            theta = Util.clamp(theta, 0.01, 1.0); // theta should be positive
//            sigma = Util.clamp(sigma, 0.0, 5.0);   // sigma should be non-negative
//        }
//
//        // Optionally, methods to set or get theta and sigma can be added
//    }
//}
