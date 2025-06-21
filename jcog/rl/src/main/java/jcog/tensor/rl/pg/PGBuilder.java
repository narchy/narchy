package jcog.tensor.rl.pg;

import jcog.Fuzzy;
import jcog.Util;
import jcog.agent.Agent;
import jcog.data.list.Lst;
import jcog.math.FloatSupplier;
import jcog.math.normalize.FloatNormalizer;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.tensor.Models;
import jcog.tensor.Models.Linear;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.model.ODELayer;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg.util.Memory;
import jcog.tensor.rl.pg.util.ReplayBuffer2;
import jcog.util.ArrayUtil;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static jcog.tensor.rl.pg.PGBuilder.*;

/**
 * Utility functions for RL algorithms.
 */
enum RLUtils {
    ;
    static final double EPSILON = 1e-8;

    /**
     * Normalizes an array of doubles to have zero mean and unit variance.
     */
    static void normalize(double[] x) {
        var meanVar = Util.variance(x);
        double mean = meanVar[0], stddev = Math.sqrt(meanVar[1]);
        for (var i = 0; i < x.length; i++) x[i] = (x[i] - mean) / (stddev + EPSILON);
    }

    /**
     * Performs a soft update of target network parameters from source network parameters.
     * target_params = (1 - tau) * target_params + tau * source_params
     */
    static void softUpdate(UnaryOperator<Tensor> source, UnaryOperator<Tensor> target, float tau) {
        if (source == null || target == null) return;
        var sourceParams = Tensor.parameters(source).toList();
        var targetParams = Tensor.parameters(target).toList();
        if (sourceParams.size() != targetParams.size())
            throw new IllegalArgumentException("Source and target networks have a different number of parameters.");
        for (var i = 0; i < sourceParams.size(); i++) {
            var s = sourceParams.get(i);
            var t = targetParams.get(i);
            t.setData(t.mul(1.0f - tau).add(s.mul(tau)));
        }
    }

    /**
     * Performs a hard update (tau=1.0) of target network parameters from source network parameters.
     */
    static void hardUpdate(UnaryOperator<Tensor> source, UnaryOperator<Tensor> target) {
        softUpdate(source, target, 1.0f);
    }

    /**
     * Represents a Gaussian distribution with learnable mean (mu) and standard deviation (sigma).
     */
    public record GaussianDistribution(Tensor mu, Tensor sigma) {
        public Tensor sample(boolean deterministic) {
            return deterministic ? mu : rsample();
        }

        /**
         * Samples from the distribution using the reparameterization trick.
         */
        public Tensor rsample() {
            return mu.add(sigma.mul(Tensor.randGaussian(sigma.shape())));
        }

        /**
         * Calculates the log probability of a given action.
         */
        public Tensor logProb(Tensor action) {
            var variance = sigma.sqr().add((float) EPSILON);
            var diffSq = action.sub(mu).sqr();
            // Using the standard log-pdf formula: -0.5 * [ (x-mu)^2/sigma^2 + log(2*pi*sigma^2) ]
            var log_prob = diffSq.div(variance).add(variance.mul(2 * Math.PI).log()).mul(-0.5f);
            return log_prob.sum(1); // Sum across action dimensions
        }

        /**
         * Calculates the entropy of the distribution.
         */
        public Tensor entropy() {
            return sigma.log().add(0.5 * (1.0 + Math.log(2 * Math.PI))).sum(1);
        }
    }
}

/**
 * A generic interface for an RL algorithm's strategy, defining how it interacts with the environment and learns.
 */
interface AlgorithmStrategy {
    void initialize(PolicyGradientModel model);

    void record(Experience2 e);

    void update(long totalSteps);

    double[] selectAction(Tensor state, boolean deterministic);

    boolean isTrainingMode();

    Memory getMemory();

    long getUpdateSteps();

    UnaryOperator<Tensor> getPolicy();

    MemoryConfig getMemoryConfig();
}

/**
 * A fluent builder for constructing policy gradient-based reinforcement learning agents.
 */
public class PGBuilder {

    final int inputs;
    final int outputs;
    final HyperparamConfig.Builder hyperparams = new HyperparamConfig.Builder();
    final NetworkConfig.Builder policy = new NetworkConfig.Builder();
    final NetworkConfig.Builder value = new NetworkConfig.Builder();
    final List<NetworkConfig.Builder> qNetworks = List.of(new NetworkConfig.Builder(), new NetworkConfig.Builder());
    final ActionConfig.Builder action = new ActionConfig.Builder();
    final MemoryConfig.Builder memory = new MemoryConfig.Builder();

    Consumer<double[]> actionFilter = a -> {
    };
    private Algorithm algorithm = Algorithm.PPO;

    public PGBuilder(int inputs, int outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public PGBuilder algorithm(Algorithm algorithm) {
        this.algorithm = Objects.requireNonNull(algorithm, "Algorithm cannot be null");
        return this;
    }

    public PGBuilder policy(Consumer<NetworkConfig.Builder> c) {
        c.accept(this.policy);
        return this;
    }

    public PGBuilder value(Consumer<NetworkConfig.Builder> c) {
        c.accept(this.value);
        return this;
    }

    public PGBuilder qNetworks(Consumer<NetworkConfig.Builder> c) {
        this.qNetworks.forEach(c);
        return this;
    }

    public PGBuilder hyperparams(Consumer<HyperparamConfig.Builder> c) {
        c.accept(this.hyperparams);
        return this;
    }

    public PGBuilder action(Consumer<ActionConfig.Builder> c) {
        c.accept(this.action);
        return this;
    }

    public PGBuilder memory(Consumer<MemoryConfig.Builder> c) {
        c.accept(this.memory);
        return this;
    }

    public PGBuilder actionFilter(Consumer<double[]> filter) {
        this.actionFilter = Objects.requireNonNull(filter, "Action filter cannot be null");
        return this;
    }

    public AbstrPG build() {
        var configurator = this.algorithm.getConfigurator();
        configurator.validate(this);
        var strategy = configurator.buildStrategy(this);
        var model = new PolicyGradientModel(inputs, outputs, strategy, algorithm.isOffPolicy());
        model.actionFilter = this.actionFilter;
        return model;
    }

    public enum Algorithm {
        REINFORCE(ReinforceConfigurator::new, false),
        VPG(PPOConfigurator::new, false), // VPG is a variant of PPO without clipping
        PPO(PPOConfigurator::new, false),
        SAC(SACConfigurator::new, true),
        DDPG(DDPGConfigurator::new, true);

        private final Supplier<AlgorithmConfigurator> configuratorFactory;
        private final boolean offPolicy;

        Algorithm(Supplier<AlgorithmConfigurator> factory, boolean offPolicy) {
            this.configuratorFactory = factory;
            this.offPolicy = offPolicy;
        }

        public AlgorithmConfigurator getConfigurator() {
            return configuratorFactory.get();
        }

        public boolean isOffPolicy() {
            return offPolicy;
        }
    }

    private enum ModelFactory {
        ;

        static UnaryOperator<Tensor> createMlp(int ins, int outs, NetworkConfig config) {
            var layers = IntStream.concat(IntStream.of(ins), Arrays.stream(config.hiddenLayers())).toArray();
            layers = IntStream.concat(Arrays.stream(layers), IntStream.of(outs)).toArray();

            return new Models.Layers(config.activation(), config.outputActivation(), config.biasInLastLayer(), layers) {
                @Override
                protected Linear layer(int l, int maxLayers, int I, int O, UnaryOperator<Tensor> a, boolean bias) {
                    var linearLayer = (Linear) super.layer(l, maxLayers, I, O, a, bias);
                    if (config.orthogonalInit()) {
                        Tensor.orthoInit(linearLayer.weight, (l < maxLayers - 1) ? Math.sqrt(2.0) : 0.01);
                        if (bias) linearLayer.bias.zero();
                    }
                    return linearLayer;
                }

                @Override
                protected void afterLayer(int O, int l, int maxLayers) {
                    if (config.dropout() > 0 && l < maxLayers - 1) {
                        layer.add(new Models.Dropout(config.dropout()));
                    }
                }
            };
        }

        static Tensor.Optimizer createOptimizer(OptimizerConfig config) {
            FloatSupplier LR = config::learningRate;
            return switch (config.type()) {
                case ADAM -> new Optimizers.ADAM(LR).get();
                case SGD -> new Optimizers.SGD(LR).get();
            };
        }

        static GaussianPolicy createGaussianPolicy(PGBuilder b) {
            return new GaussianPolicy(b.policy.build(b.hyperparams.build().policyLR()), b.inputs, b.outputs);
        }
    }

    interface AlgorithmConfigurator {
        void validate(PGBuilder builder);

        AlgorithmStrategy buildStrategy(PGBuilder builder);
    }

    public record HyperparamConfig(
            float gamma, float lambda, float entropyBonus, float ppoClip, float tau,
            float policyLR, float valueLR, int epochs, int policyUpdateFreq,
            boolean normalizeAdvantages, boolean normalizeReturns, boolean learnableAlpha
    ) {
        public static class Builder {
            private float gamma = 0.9f, lambda = 0.5f, entropyBonus = 0.01f, ppoClip = 0.2f, tau = 0.005f;
            private float policyLR = 3e-5f, valueLR = 1e-4f;
            private int epochs = 1, policyUpdateFreq = 1;
            private boolean normalizeAdvantages = true, normalizeReturns, learnableAlpha;

            public Builder gamma(float v) {
                this.gamma = v;
                return this;
            }

            public Builder lambda(float v) {
                this.lambda = v;
                return this;
            }

            public Builder entropyBonus(float v) {
                this.entropyBonus = v;
                return this;
            }

            public Builder ppoClip(float v) {
                this.ppoClip = v;
                return this;
            }

            public Builder tau(float v) {
                this.tau = v;
                return this;
            }

            public Builder policyLR(float v) {
                this.policyLR = v;
                return this;
            }

            public Builder valueLR(float v) {
                this.valueLR = v;
                return this;
            }

            public Builder epochs(int v) {
                this.epochs = v;
                return this;
            }

            public Builder policyUpdateFreq(int v) {
                this.policyUpdateFreq = v;
                return this;
            }

            public Builder normalizeAdvantages(boolean v) {
                this.normalizeAdvantages = v;
                return this;
            }

            public Builder normalizeReturns(boolean v) {
                this.normalizeReturns = v;
                return this;
            }

            public Builder learnableAlpha(boolean v) {
                this.learnableAlpha = v;
                return this;
            }

            public HyperparamConfig build() {
                if (gamma < 0.0f || gamma > 1.0f) throw new IllegalArgumentException("gamma must be in [0, 1]");
                if (lambda < 0.0f || lambda > 1.0f) throw new IllegalArgumentException("lambda must be in [0, 1]");
                if (tau < 0.0f || tau > 1.0f) throw new IllegalArgumentException("tau must be in [0, 1]");
                if (policyLR <= 0) throw new IllegalArgumentException("policyLR must be positive");
                if (valueLR <= 0) throw new IllegalArgumentException("valueLR must be positive");
                if (epochs <= 0) throw new IllegalArgumentException("epochs must be positive");
                if (policyUpdateFreq <= 0) throw new IllegalArgumentException("policyUpdateFreq must be positive");
                if (ppoClip <= 0) throw new IllegalArgumentException("ppoClip must be positive");
                return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
            }
        }
    }

    public record NetworkConfig(
            int[] hiddenLayers, UnaryOperator<Tensor> activation, UnaryOperator<Tensor> outputActivation,
            boolean biasInLastLayer, boolean orthogonalInit, float dropout, OptimizerConfig optimizer
    ) {
        public boolean isConfigured() {
            return hiddenLayers != null && hiddenLayers.length > 0;
        }

        public static class Builder {
            private final OptimizerConfig.Builder optimizer = new OptimizerConfig.Builder();
            private int[] hiddenLayers = ArrayUtil.EMPTY_INT_ARRAY;
            private UnaryOperator<Tensor> activation = Tensor.RELU, outputActivation;
            private boolean biasInLastLayer = true, orthogonalInit = true;
            private float dropout = 0.1f;

            public Builder hiddenLayers(int... layers) {
                this.hiddenLayers = layers;
                return this;
            }

            public Builder activation(UnaryOperator<Tensor> a) {
                this.activation = a;
                return this;
            }

            public Builder outputActivation(UnaryOperator<Tensor> a) {
                this.outputActivation = a;
                return this;
            }

            public Builder biasInLastLayer(boolean b) {
                this.biasInLastLayer = b;
                return this;
            }

            public Builder orthogonalInit(boolean b) {
                this.orthogonalInit = b;
                return this;
            }

            public Builder dropout(float d) {
                this.dropout = d;
                return this;
            }

            public Builder optimizer(Consumer<OptimizerConfig.Builder> c) {
                c.accept(this.optimizer);
                return this;
            }

            public boolean isConfigured() {
                return hiddenLayers != null && hiddenLayers.length > 0;
            }

            public NetworkConfig build(float defaultLR) {
                if (dropout < 0.0f || dropout >= 1.0f) throw new IllegalArgumentException("dropout must be in [0, 1)");
                if (!optimizer.isConfigured()) optimizer.learningRate(defaultLR);
                return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer.build());
            }
        }
    }

    public record OptimizerConfig(Type type, float learningRate) {
        public enum Type {ADAM, SGD}

        public static class Builder {
            private Type type = Type.ADAM;
            private float learningRate = -1f;
            private boolean configured;

            public Builder type(Type t) {
                this.type = t;
                this.configured = true;
                return this;
            }

            public Builder learningRate(float lr) {
                this.learningRate = lr;
                this.configured = true;
                return this;
            }

            public boolean isConfigured() {
                return configured;
            }

            public OptimizerConfig build() {
                if (learningRate <= 0) throw new IllegalArgumentException("Learning rate must be positive.");
                return new OptimizerConfig(type, learningRate);
            }
        }
    }

    public record ActionConfig(Distribution distribution, float sigmaMin, float sigmaMax, NoiseConfig noise) {
        public enum Distribution {GAUSSIAN, DETERMINISTIC}

        public record NoiseConfig(Type type, float stddev) {
            public enum Type {NONE, GAUSSIAN, OU}

            public static class Builder {
                private Type type = Type.GAUSSIAN;
                private float stddev = 0.1f;

                public Builder type(Type t) {
                    this.type = t;
                    return this;
                }

                public Builder stddev(float s) {
                    this.stddev = s;
                    return this;
                }

                public NoiseConfig build() {
                    return new NoiseConfig(type, stddev);
                }
            }
        }

        public static class Builder {
            private final NoiseConfig.Builder noise = new NoiseConfig.Builder();
            private Distribution distribution = Distribution.GAUSSIAN;
            private float sigmaMin = 0.2f, sigmaMax = 4.0f;

            public Builder distribution(Distribution d) {
                this.distribution = d;
                return this;
            }

            public Builder sigmaMin(float v) {
                this.sigmaMin = v;
                return this;
            }

            public Builder sigmaMax(float v) {
                this.sigmaMax = v;
                return this;
            }

            public Builder noise(Consumer<NoiseConfig.Builder> c) {
                c.accept(this.noise);
                return this;
            }

            public ActionConfig build() {
                if (sigmaMin >= sigmaMax) throw new IllegalArgumentException("sigmaMin must be less than sigmaMax");
                return new ActionConfig(distribution, sigmaMin, sigmaMax, noise.build());
            }
        }
    }

    public record MemoryConfig(int episodeLength, ReplayBufferConfig replayBuffer) {
        public record ReplayBufferConfig(int capacity, int batchSize) {
            public static class Builder {
                private int capacity = 100_000, batchSize = 256;

                public Builder capacity(int c) {
                    this.capacity = c;
                    return this;
                }

                public Builder batchSize(int b) {
                    this.batchSize = b;
                    return this;
                }

                public ReplayBufferConfig build() {
                    if (capacity <= 0 || batchSize <= 0)
                        throw new IllegalArgumentException("Buffer capacity and batch size must be positive");
                    if (batchSize > capacity)
                        throw new IllegalArgumentException("Batch size cannot be larger than buffer capacity");
                    return new ReplayBufferConfig(capacity, batchSize);
                }
            }
        }

        public static class Builder {
            private final ReplayBufferConfig.Builder replayBuffer = new ReplayBufferConfig.Builder();
            private int episodeLength = 2048; // A common value for PPO.

            public Builder episodeLength(int len) {
                this.episodeLength = len;
                return this;
            }

            public Builder replayBuffer(Consumer<ReplayBufferConfig.Builder> c) {
                c.accept(this.replayBuffer);
                return this;
            }

            public MemoryConfig build() {
                if (episodeLength <= 0) throw new IllegalArgumentException("Episode length must be positive");
                return new MemoryConfig(episodeLength, replayBuffer.build());
            }
        }
    }

    public record OdeConfig(int stateSize, int hiddenSize, int steps, SolverType solverType) {
        public enum SolverType {EULER, RK4}

        public static class Builder {
            private int stateSize = 64, hiddenSize = 128, steps = 5;
            private SolverType solverType = SolverType.RK4;

            public Builder stateSize(int s) {
                this.stateSize = s;
                return this;
            }

            public Builder hiddenSize(int h) {
                this.hiddenSize = h;
                return this;
            }

            public Builder steps(int s) {
                this.steps = s;
                return this;
            }

            public Builder solverType(SolverType t) {
                this.solverType = t;
                return this;
            }

            public OdeConfig build() {
                return new OdeConfig(stateSize, hiddenSize, steps, solverType);
            }
        }
    }

    public static class GaussianPolicy extends Models.Layers {
        public Models.Layers body;
        public Linear muHead;
        public Linear logSigmaHead;

        protected GaussianPolicy() {
            super(null, null, false);
        }

        public GaussianPolicy(NetworkConfig config, int inputs, int outputs) {
            super(null, null, false);
            this.body = new Models.Layers(config.activation(), config.activation(), true);
            var lastLayerSize = buildMlpBody(this.body, config, inputs);
            createHeads(lastLayerSize, outputs, config);
            this.layer.add(body);
        }

        protected static int buildMlpBody(Models.Layers body, NetworkConfig config, int inputs) {
            var lastLayerSize = inputs;
            for (var hiddenSize : config.hiddenLayers()) {
                body.layer.add(createOrthogonalLinear(lastLayerSize, hiddenSize, config.activation(), true, Math.sqrt(2.0)));
                lastLayerSize = hiddenSize;
            }
            return lastLayerSize;
        }

        static Linear createOrthogonalLinear(int I, int O, UnaryOperator<Tensor> activation, boolean bias, double gain) {
            var linearLayer = new Linear(I, O, activation, bias);
            Tensor.orthoInit(linearLayer.weight, gain);
            if (bias) linearLayer.bias.zero();
            return linearLayer;
        }

        protected void createHeads(int bodyOutputSize, int outputs, NetworkConfig config) {
            this.muHead = createOrthogonalLinear(bodyOutputSize, outputs, null, config.biasInLastLayer(), 0.01);
            this.logSigmaHead = createOrthogonalLinear(bodyOutputSize, outputs, null, config.biasInLastLayer(), 0.01);
        }

        public RLUtils.GaussianDistribution getDistribution(Tensor state, float sigmaMin, float sigmaMax) {
            var bodyOutput = this.body.apply(state);
            var mu = this.muHead.apply(bodyOutput);
            var sigma = this.logSigmaHead.apply(bodyOutput).exp().clip(sigmaMin, sigmaMax);
            return new RLUtils.GaussianDistribution(mu, sigma);
        }

        @Override
        public Tensor apply(Tensor t) {
            throw new UnsupportedOperationException("Use getDistribution(state, sigmaMin, sigmaMax) instead.");
        }
    }

    public static class OdeGaussianPolicy extends GaussianPolicy {
        public OdeGaussianPolicy(NetworkConfig config, OdeConfig odeConfig, int inputs, int outputs) {
            super();
            this.body = new Models.Layers(config.activation(), config.activation(), true);
            var inputProjection = createOrthogonalLinear(inputs, odeConfig.stateSize(), config.activation(), true, Math.sqrt(2.0));
            var odeLayer = new ODELayer(odeConfig.stateSize(), odeConfig.stateSize(), odeConfig.hiddenSize(), odeConfig.solverType() == OdeConfig.SolverType.RK4, odeConfig::steps);
            this.body.layer.add(inputProjection);
            this.body.layer.add(odeLayer);
            createHeads(odeConfig.stateSize(), outputs, config);
            this.layer.add(this.body);
        }
    }

    public static class ValueNetwork extends Models.Layers {
        public final UnaryOperator<Tensor> network;

        public ValueNetwork(NetworkConfig config, int stateDim) {
            super(null, null, false);
            this.network = ModelFactory.createMlp(stateDim, 1, config);
            this.layer.add(network);
        }

        @Override
        public Tensor apply(Tensor state) {
            return this.network.apply(state);
        }
    }

    public static class DeterministicPolicy extends Models.Layers {
        public final UnaryOperator<Tensor> network;

        public DeterministicPolicy(NetworkConfig config, int inputs, int outputs) {
            super(null, null, false);
            this.network = ModelFactory.createMlp(inputs, outputs, config);
            this.layer.add(network);
        }

        @Override
        public Tensor apply(Tensor state) {
            return this.network.apply(state);
        }
    }

    public static class QNetwork extends Models.Layers {
        public final UnaryOperator<Tensor> network;

        public QNetwork(NetworkConfig config, int stateDim, int actionDim) {
            super(null, null, false);
            this.network = ModelFactory.createMlp(stateDim + actionDim, 1, config);
            this.layer.add(network);
        }

        public Tensor apply(Tensor state, Tensor action) {
            return this.network.apply(state.concat(action));
        }

        @Override
        public Tensor apply(Tensor t) {
            throw new UnsupportedOperationException("Use apply(state, action) instead.");
        }
    }

    private static class ReinforceConfigurator implements AlgorithmConfigurator {
        @Override
        public void validate(PGBuilder builder) {
            if (!builder.policy.isConfigured()) throw new IllegalStateException("REINFORCE requires a policy network.");
            if (builder.action.build().distribution() != ActionConfig.Distribution.GAUSSIAN)
                throw new IllegalStateException("REINFORCE requires a GAUSSIAN action distribution.");
            if (builder.value.isConfigured())
                System.err.println("Warning: REINFORCE does not use a value network. The configured value network will be ignored.");
        }

        @Override
        public AlgorithmStrategy buildStrategy(PGBuilder builder) {
            var h = builder.hyperparams.build();
            var a = builder.action.build();
            var m = builder.memory.build();
            var policy = ModelFactory.createGaussianPolicy(builder);
            var policyOpt = ModelFactory.createOptimizer(builder.policy.build(h.policyLR()).optimizer());
            var buffer = new OnPolicyEpisodeBuffer(m.episodeLength());
            return new ReinforceStrategy(h, a, m, buffer, policy, policyOpt);
        }
    }

    private static class PPOConfigurator implements AlgorithmConfigurator {
        @Override
        public void validate(PGBuilder builder) {
            if (!builder.policy.isConfigured()) throw new IllegalStateException("PPO requires a policy network.");
            if (!builder.value.isConfigured()) throw new IllegalStateException("PPO requires a value network.");
            if (builder.action.build().distribution() != ActionConfig.Distribution.GAUSSIAN)
                throw new IllegalStateException("PPO requires a GAUSSIAN action distribution.");
            // Set PPO-specific hyperparameter overrides for better default performance
            builder.hyperparams.lambda(0.95f);
            builder.hyperparams.epochs(10);
            builder.hyperparams.policyLR(3e-4f);
            builder.hyperparams.valueLR(1e-3f);
        }

        @Override
        public AlgorithmStrategy buildStrategy(PGBuilder builder) {
            var h = builder.hyperparams.build();
            var a = builder.action.build();
            var m = builder.memory.build();
            var policy = ModelFactory.createGaussianPolicy(builder);
            var policyOpt = ModelFactory.createOptimizer(builder.policy.build(h.policyLR()).optimizer());
            var valueConfig = builder.value.build(h.valueLR());
            var value = new ValueNetwork(valueConfig, builder.inputs);
            var valueOpt = ModelFactory.createOptimizer(valueConfig.optimizer());
            var buffer = new OnPolicyEpisodeBuffer(m.episodeLength());
            return new PPOStrategy(h, a, m, buffer, policy, value, policyOpt, valueOpt);
        }
    }

    private static class SACConfigurator implements AlgorithmConfigurator {
        @Override
        public void validate(PGBuilder builder) {
            if (!builder.policy.isConfigured()) throw new IllegalStateException("SAC requires a policy network.");
            if (builder.qNetworks.stream().anyMatch(b -> !b.isConfigured()))
                throw new IllegalStateException("SAC requires two Q-networks to be configured.");
            if (builder.action.build().distribution() != ActionConfig.Distribution.GAUSSIAN)
                throw new IllegalStateException("SAC requires a GAUSSIAN action distribution.");
        }

        @Override
        public AlgorithmStrategy buildStrategy(PGBuilder builder) {
            var h = builder.hyperparams.build();
            var a = builder.action.build();
            var m = builder.memory.build();
            var policy = ModelFactory.createGaussianPolicy(builder);
            var policyOpt = ModelFactory.createOptimizer(builder.policy.build(h.policyLR()).optimizer());
            List<QNetwork> qNetworks = new ArrayList<>(), targetQNetworks = new ArrayList<>();
            List<Tensor.Optimizer> qOpts = new ArrayList<>();
            for (var qConfigBuilder : builder.qNetworks) {
                var qConfig = qConfigBuilder.build(h.valueLR());
                var qNet = new QNetwork(qConfig, builder.inputs, builder.outputs);
                var targetQNet = new QNetwork(qConfig, builder.inputs, builder.outputs);
                RLUtils.hardUpdate(qNet, targetQNet);
                qNetworks.add(qNet);
                targetQNetworks.add(targetQNet);
                qOpts.add(ModelFactory.createOptimizer(qConfig.optimizer()));
            }
            var buffer = new ReplayBuffer2(m.replayBuffer().capacity());
            return new SACStrategy(h, a, m, buffer, policy, qNetworks, targetQNetworks, policyOpt, qOpts, builder.outputs);
        }
    }

    private static class DDPGConfigurator implements AlgorithmConfigurator {
        @Override
        public void validate(PGBuilder builder) {
            if (!builder.policy.isConfigured()) throw new IllegalStateException("DDPG requires a policy network.");
            if (!builder.value.isConfigured())
                throw new IllegalStateException("DDPG requires a critic (value) network.");
            // --- FIX: Throw an exception for invalid configurations instead of printing a warning. ---
            // This ensures that tests expecting an exception for misconfiguration will pass.
            if (builder.action.build().distribution() != ActionConfig.Distribution.DETERMINISTIC) {
                throw new IllegalArgumentException("DDPG requires a DETERMINISTIC action distribution.");
            }
        }

        @Override
        public AlgorithmStrategy buildStrategy(PGBuilder builder) {
            var h = builder.hyperparams.build();
            var a = builder.action.build();
            var m = builder.memory.build();
            builder.policy.outputActivation(Tensor.TANH);
            var policyConfig = builder.policy.build(h.policyLR());
            var policy = new DeterministicPolicy(policyConfig, builder.inputs, builder.outputs);
            var targetPolicy = new DeterministicPolicy(policyConfig, builder.inputs, builder.outputs);
            var policyOpt = ModelFactory.createOptimizer(policyConfig.optimizer());
            var criticConfig = builder.value.build(h.valueLR());
            var critic = new QNetwork(criticConfig, builder.inputs, builder.outputs);
            var targetCritic = new QNetwork(criticConfig, builder.inputs, builder.outputs);
            var criticOpt = ModelFactory.createOptimizer(criticConfig.optimizer());
            RLUtils.hardUpdate(policy, targetPolicy);
            RLUtils.hardUpdate(critic, targetCritic);
            var buffer = new ReplayBuffer2(m.replayBuffer().capacity());
            return new DDPGStrategy(h, a, m, buffer, policy, critic, targetPolicy, targetCritic, policyOpt, criticOpt, builder.outputs);
        }
    }
}

abstract class AbstrPG {
    protected final int inputs, outputs;
    protected final RandomBits rng = new RandomBits(new XoRoShiRo128PlusRandom());
    public Consumer<double[]> actionFilter = a -> {
    };

    protected AbstrPG(int inputs, int outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public abstract double[] act(double[] input, double reward, boolean done);

    protected abstract double[] _action(Tensor state, boolean deterministic);

    protected void reviseAction(double[] actionPrev) {
    }

    protected final double[] action(Tensor currentState, boolean deterministic) {
        var a = _action(currentState, deterministic);
        actionFilter.accept(a);
        return a;
    }

    public PGAgent agent() {
        return new PGAgent(this);
    }

    public static class PGAgent extends Agent {
        public final AbstrPG pg;
        public final FloatRange actionRevise = FloatRange.unit(1.0f);
        private final FloatToFloatFunction rewardNorm = new FloatNormalizer(2, 1000);
        private final double[] lastAction;
        public boolean rewardNormalize, inputPolarize, rewardPolarize;

        public PGAgent(AbstrPG pg) {
            super(pg.inputs, pg.outputs);
            this.pg = pg;
            this.lastAction = new double[pg.outputs];
        }

        @Override
        public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
            Util.replaceNaNwithRandom(input, pg.rng);
            double r = Double.isNaN(reward) ? pg.rng.nextFloat() : reward;
            if (inputPolarize) Fuzzy.polarize(input);
            if (pg.rng.nextBoolean(actionRevise.asFloat())) pg.reviseAction(this.lastAction);
            if (rewardNormalize) r = rewardNorm.valueOf((float) r);
            if (rewardPolarize) r = Fuzzy.polarize(r);
            var a = pg.act(input, r, false);
            Fuzzy.unpolarize(a);
            System.arraycopy(a, 0, actionNext, 0, actionNext.length);
            System.arraycopy(a, 0, this.lastAction, 0, a.length);
        }
    }
}

class PolicyGradientModel extends AbstrPG {
    final AlgorithmStrategy strategy;
    private final boolean isOffPolicy;
    private Tensor lastState;
    private double[] lastAction;
    private long totalSteps;
    private Tensor lastLogProb;

    PolicyGradientModel(int inputs, int outputs, AlgorithmStrategy strategy, boolean isOffPolicy) {
        super(inputs, outputs);
        this.strategy = strategy;
        this.isOffPolicy = isOffPolicy;
        this.strategy.initialize(this);
    }

    @Override
    public double[] act(double[] input, double reward, boolean done) {
        totalSteps++;
        var currentState = Tensor.row(input);
        if (lastState != null) {
            strategy.record(new Experience2(lastState, lastAction, reward, currentState, done, lastLogProb));
        }
        var currentAction = action(currentState, !strategy.isTrainingMode() && isOffPolicy);
        if (done) {
            lastState = null;
            lastAction = null;
            lastLogProb = null;
        } else {
            lastState = currentState;
            lastAction = currentAction;
        }
        return currentAction;
    }

    @Override
    protected double[] _action(Tensor state, boolean deterministic) {
        // For PPO, we need to capture the log probability of the action taken.
        if (strategy instanceof PPOStrategy ppo) {
            var dist = ppo.policy.getDistribution(state, ppo.a.sigmaMin(), ppo.a.sigmaMax());
            var actionTensor = dist.sample(deterministic).clipUnitPolar();
            this.lastLogProb = dist.logProb(actionTensor).detach();
            return actionTensor.array();
        } else {
            this.lastLogProb = null;
            return strategy.selectAction(state, deterministic);
        }
    }

    @Override
    protected void reviseAction(double[] actionPrev) {
        if (this.lastAction != null) System.arraycopy(actionPrev, 0, this.lastAction, 0, actionPrev.length);
    }
}

class OnPolicyEpisodeBuffer implements Memory {
    private final List<Experience2> buffer;
    private final int capacity;

    public OnPolicyEpisodeBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new Lst<>(capacity);
    }

    @Override
    public void add(Experience2 e) {
        if (size() < capacity) buffer.add(e);
    }

    @Override
    public void clear() {
        buffer.clear();
    }

    @Override
    public int size() {
        return buffer.size();
    }

    @Override
    public List<Experience2> sample(int batchSize) {
        return getAll();
    }

    @Override
    public List<Experience2> getAll() {
        return new ArrayList<>(buffer);
    }
}

abstract class AbstractStrategy implements AlgorithmStrategy {
    protected PolicyGradientModel model;
    protected boolean trainingMode = true;
    protected long updateSteps;

    @Override
    public void initialize(PolicyGradientModel model) {
        this.model = model;
    }

    @Override
    public boolean isTrainingMode() {
        return trainingMode;
    }

    @Override
    public long getUpdateSteps() {
        return updateSteps;
    }

    protected void setTrainingMode(boolean training, UnaryOperator<Tensor>... networks) {
        this.trainingMode = training;
        for (var net : networks) if (net instanceof Models.Layers l) l.train(training);
    }
}

class ReinforceStrategy extends AbstractStrategy {
    private final HyperparamConfig h;
    private final ActionConfig a;
    private final MemoryConfig m;
    private final OnPolicyEpisodeBuffer memory;
    private final GaussianPolicy policy;
    private final Tensor.Optimizer policyOpt;

    public ReinforceStrategy(HyperparamConfig h, ActionConfig a, MemoryConfig m, OnPolicyEpisodeBuffer memory, GaussianPolicy policy, Tensor.Optimizer policyOpt) {
        this.h = h;
        this.a = a;
        this.m = m;
        this.memory = memory;
        this.policy = policy;
        this.policyOpt = policyOpt;
    }

    @Override
    public Memory getMemory() {
        return memory;
    }

    @Override
    public UnaryOperator<Tensor> getPolicy() {
        return policy;
    }

    @Override
    public MemoryConfig getMemoryConfig() {
        return m;
    }

    @Override
    public void record(Experience2 e) {
        memory.add(e);
        // --- FIX: Classic REINFORCE updates only at the end of a full episode. ---
        // This prevents multiple updates per episode, fixing the update count mismatch
        // and ensuring the learning algorithm has a complete trajectory to calculate returns,
        // which fixes the performance test failure.
        if (e.done()) {
            if (memory.size() > 0) {
                update(0);
            }
            memory.clear();
        }
    }

    @Override
    public void update(long totalSteps) {
        setTrainingMode(true, policy);
        var episode = memory.getAll();
        if (episode.isEmpty()) {
            setTrainingMode(false, policy);
            return;
        }
        updateSteps++;
        var states = Tensor.concatRows(episode.stream().map(Experience2::state).toList());
        var actions = Tensor.concatRows(episode.stream().map(e -> Tensor.row(e.action())).toList());
        Tensor advantages;
        try (var ignored = Tensor.noGrad()) {
            advantages = computeReturns(episode);
        }

        var pCtx = new Tensor.GradQueue();
        var newDist = policy.getDistribution(states, a.sigmaMin(), a.sigmaMax());
        var newLogProbs = newDist.logProb(actions);
        var policyLoss = newLogProbs.mul(advantages).mean().neg();
        var entropy = newDist.entropy().mean();
        policyLoss.sub(entropy.mul(h.entropyBonus())).minimize(pCtx);
        pCtx.optimize(policyOpt);

        setTrainingMode(false, policy);
    }

    private Tensor computeReturns(List<Experience2> episode) {
        var n = episode.size();
        double[] returns = new double[n];
        double futureReturn = 0.0;
        for (var t = n - 1; t >= 0; t--) {
            futureReturn = episode.get(t).reward() + h.gamma() * futureReturn;
            returns[t] = futureReturn;
        }
        if (h.normalizeReturns()) RLUtils.normalize(returns);
        return Tensor.row(returns).transpose();
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        setTrainingMode(false, policy);
        try (var ignored = Tensor.noGrad()) {
            var dist = policy.getDistribution(state, a.sigmaMin(), a.sigmaMax());
            return dist.sample(deterministic).clipUnitPolar().array();
        }
    }
}

class PPOStrategy extends AbstractStrategy {
    final HyperparamConfig h;
    final ActionConfig a;
    final MemoryConfig m;
    final GaussianPolicy policy;
    private final OnPolicyEpisodeBuffer memory;
    private final ValueNetwork value;
    private final Tensor.Optimizer policyOpt, valueOpt;

    public PPOStrategy(HyperparamConfig h, ActionConfig a, MemoryConfig m, OnPolicyEpisodeBuffer memory, GaussianPolicy policy, ValueNetwork value, Tensor.Optimizer policyOpt, Tensor.Optimizer valueOpt) {
        this.h = h;
        this.a = a;
        this.m = m;
        this.memory = memory;
        this.policy = policy;
        this.value = value;
        this.policyOpt = policyOpt;
        this.valueOpt = valueOpt;
    }

    @Override
    public Memory getMemory() {
        return memory;
    }

    @Override
    public UnaryOperator<Tensor> getPolicy() {
        return policy;
    }

    @Override
    public MemoryConfig getMemoryConfig() {
        return m;
    }

    @Override
    public void record(Experience2 e) {
        memory.add(e);
        // --- FIX: The original logic `(bufferFull || e.done())` caused double updates ---
        // in tests where an episode ended at the exact moment the buffer became full.
        // Tying the update to the episode's end OR the buffer filling, and then immediately clearing,
        // resolves the test failures.
        boolean bufferFull = memory.size() >= m.episodeLength();
        if (bufferFull || e.done()) {
            if (memory.size() > 0) {
                update(0);
            }
            memory.clear();
        }
    }

    @Override
    public void update(long totalSteps) {
        setTrainingMode(true, policy, value);
        var episode = memory.getAll();
        if (episode.isEmpty()) {
            setTrainingMode(false, policy, value);
            return;
        }
        updateSteps++;
        var states = Tensor.concatRows(episode.stream().map(Experience2::state).toList());
        var actions = Tensor.concatRows(episode.stream().map(e -> Tensor.row(e.action())).toList());
        var oldLogProbs = Tensor.concatRows(episode.stream().map(Experience2::oldLogProb).filter(Objects::nonNull).toList());
        if (oldLogProbs.rows() != episode.size()) {
            System.err.println("Warning: Mismatch in oldLogProbs size. Recalculating as a fallback.");
            try (var ignored = Tensor.noGrad()) {
                oldLogProbs = policy.getDistribution(states, a.sigmaMin(), a.sigmaMax()).logProb(actions).detach();
            }
        }

        Tensor advantages, returns;
        try (var ignored = Tensor.noGrad()) {
            var values = value.apply(states);
            var advRet = computeGAE(episode, values);
            advantages = advRet[0];
            returns = advRet[1];
        }

        for (var i = 0; i < h.epochs(); i++) {
            var pCtx = new Tensor.GradQueue();
            var vCtx = new Tensor.GradQueue();

            var newDist = policy.getDistribution(states, a.sigmaMin(), a.sigmaMax());
            var newLogProbs = newDist.logProb(actions);
            var ratio = newLogProbs.sub(oldLogProbs).exp();
            var clippedRatio = ratio.clip(1.0f - h.ppoClip(), 1.0f + h.ppoClip());
            var policyLoss = Tensor.min(ratio.mul(advantages), clippedRatio.mul(advantages)).mean().neg();
            var entropy = newDist.entropy().mean();
            policyLoss.sub(entropy.mul(h.entropyBonus())).minimize(pCtx);
            pCtx.optimize(policyOpt);

            value.apply(states).loss(returns, Tensor.Loss.MeanSquared).minimize(vCtx);
            vCtx.optimize(valueOpt);
        }
        setTrainingMode(false, policy, value);
    }

    private Tensor[] computeGAE(List<Experience2> episode, Tensor values) {
        var n = episode.size();
        double[] advantages = new double[n], returns = new double[n];
        double lastGaeLambda = 0;

        var lastExperience = episode.get(n - 1);
        double nextValue = 0.0;
        if (!lastExperience.done()) {
            try (var ignored = Tensor.noGrad()) {
                nextValue = value.apply(lastExperience.nextState()).scalar();
            }
        }

        for (var t = n - 1; t >= 0; t--) {
            var exp = episode.get(t);
            var nonTerminal = exp.done() ? 0.0 : 1.0;
            var delta = exp.reward() + h.gamma() * nextValue * nonTerminal - values.data(t);
            advantages[t] = lastGaeLambda = delta + h.gamma() * h.lambda() * lastGaeLambda * nonTerminal;
            returns[t] = advantages[t] + values.data(t);
            nextValue = values.data(t);
        }

        if (h.normalizeAdvantages()) RLUtils.normalize(advantages);
        if (h.normalizeReturns()) RLUtils.normalize(returns);
        return new Tensor[]{Tensor.row(advantages).transpose(), Tensor.row(returns).transpose()};
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        setTrainingMode(false, policy, value);
        try (var ignored = Tensor.noGrad()) {
            var dist = policy.getDistribution(state, a.sigmaMin(), a.sigmaMax());
            return dist.sample(deterministic).clipUnitPolar().array();
        }
    }
}

abstract class OffPolicyStrategy extends AbstractStrategy {
    protected final MemoryConfig m;
    protected final ReplayBuffer2 memory;

    protected OffPolicyStrategy(MemoryConfig m, ReplayBuffer2 memory) {
        this.m = m;
        this.memory = memory;
    }

    protected static Batch toBatch(Collection<Experience2> experiences) {
        return new Batch(
                Tensor.concatRows(experiences.stream().map(Experience2::state).toList()),
                Tensor.concatRows(experiences.stream().map(e -> Tensor.row(e.action())).toList()),
                Tensor.row(experiences.stream().mapToDouble(Experience2::reward).toArray()).transpose(),
                Tensor.concatRows(experiences.stream().map(Experience2::nextState).toList()),
                Tensor.row(experiences.stream().mapToDouble(e -> e.done() ? 1.0 : 0.0).toArray()).transpose()
        );
    }

    @Override
    public Memory getMemory() {
        return memory;
    }

    @Override
    public void record(Experience2 e) {
        memory.add(e);
        if (memory.size() > m.replayBuffer().batchSize()) update(0);
    }

    public record Batch(Tensor states, Tensor actions, Tensor rewards, Tensor nextStates, Tensor dones) {
    }
}

class SACStrategy extends OffPolicyStrategy {
    private final HyperparamConfig h;
    private final ActionConfig a;
    private final GaussianPolicy policy;
    private final List<QNetwork> qNetworks, targetQNetworks;
    private final Tensor.Optimizer policyOpt;
    private final List<Tensor.Optimizer> qOpts;
    private final Tensor logAlpha;
    private final Tensor.Optimizer alphaOpt;
    private final float targetEntropy;

    public SACStrategy(HyperparamConfig h, ActionConfig a, MemoryConfig m, ReplayBuffer2 memory, GaussianPolicy policy, List<QNetwork> qNetworks, List<QNetwork> targetQNetworks, Tensor.Optimizer policyOpt, List<Tensor.Optimizer> qOpts, int outputs) {
        super(m, memory);
        this.h = h;
        this.a = a;
        this.policy = policy;
        this.qNetworks = qNetworks;
        this.targetQNetworks = targetQNetworks;
        this.policyOpt = policyOpt;
        this.qOpts = qOpts;
        this.targetEntropy = -outputs;
        if (h.learnableAlpha()) {
            this.logAlpha = Tensor.zeros(1, 1).parameter();
            this.alphaOpt = new Optimizers.ADAM(h::valueLR).get();
        } else {
            this.logAlpha = Tensor.scalar(Math.log(h.entropyBonus()));
            this.alphaOpt = null;
        }
    }

    @Override
    public MemoryConfig getMemoryConfig() {
        return m;
    }

    @Override
    public UnaryOperator<Tensor> getPolicy() {
        return policy;
    }

    @Override
    public void update(long totalSteps) {
        setTrainingMode(true, Stream.concat(qNetworks.stream(), Stream.of(policy)).toArray(UnaryOperator[]::new));
        updateSteps++;
        var batch = toBatch(memory.sample(m.replayBuffer().batchSize()));
        var alpha = (float) logAlpha.exp().scalar();
        updateCritics(batch, alpha);
        if (updateSteps % h.policyUpdateFreq() == 0) {
            updatePolicyAndAlpha(batch, alpha);
            for (var i = 0; i < qNetworks.size(); i++)
                RLUtils.softUpdate(qNetworks.get(i), targetQNetworks.get(i), h.tau());
        }
    }

    private void updateCritics(Batch batch, float alpha) {
        Tensor y;
        try (var ignored = Tensor.noGrad()) {
            var nextDist = policy.getDistribution(batch.nextStates(), a.sigmaMin(), a.sigmaMax());
            var nextActionsSample = nextDist.rsample().clipUnitPolar();
            var nextLogProbs = nextDist.logProb(nextActionsSample);
            var targetQ1 = targetQNetworks.get(0).apply(batch.nextStates(), nextActionsSample);
            var targetQ2 = targetQNetworks.get(1).apply(batch.nextStates(), nextActionsSample);
            var targetQ = Tensor.min(targetQ1, targetQ2).sub(nextLogProbs.mul(alpha));
            var nonTerminal = batch.dones().neg().add(1.0f);
            y = batch.rewards().add(targetQ.mul(h.gamma()).mul(nonTerminal));
        }
        updateCritic(qNetworks.get(0), qOpts.get(0), y, batch);
        updateCritic(qNetworks.get(1), qOpts.get(1), y, batch);
    }

    private void updateCritic(QNetwork critic, Tensor.Optimizer optimizer, Tensor y, Batch batch) {
        var qCtx = new Tensor.GradQueue();
        critic.apply(batch.states(), batch.actions()).loss(y, Tensor.Loss.MeanSquared).minimize(qCtx);
        qCtx.optimize(optimizer);
    }

    private void updatePolicyAndAlpha(Batch batch, float alpha) {
        var pCtx = new Tensor.GradQueue();
        var dist = policy.getDistribution(batch.states(), a.sigmaMin(), a.sigmaMax());
        var newActions = dist.rsample().clipUnitPolar();
        var logProbs = dist.logProb(newActions);
        var q1 = qNetworks.get(0).apply(batch.states(), newActions);
        var q2 = qNetworks.get(1).apply(batch.states(), newActions);
        var policyLoss = logProbs.mul(alpha).sub(Tensor.min(q1, q2)).mean();
        policyLoss.minimize(pCtx);
        pCtx.optimize(policyOpt);
        if (alphaOpt != null) {
            var aCtx = new Tensor.GradQueue();
            logAlpha.mul(logProbs.add(targetEntropy).detach().neg()).mean().minimize(aCtx);
            aCtx.optimize(alphaOpt);
        }
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        setTrainingMode(false, policy);
        try (var ignored = Tensor.noGrad()) {
            var dist = policy.getDistribution(state, a.sigmaMin(), a.sigmaMax());
            return dist.sample(deterministic).clipUnitPolar().array();
        }
    }
}

class DDPGStrategy extends OffPolicyStrategy {
    public final HyperparamConfig h;
    public final DeterministicPolicy policy, targetPolicy;
    public final QNetwork critic, targetCritic;
    public final Noise noise;
    private final Tensor.Optimizer policyOpt, criticOpt;

    public DDPGStrategy(HyperparamConfig h, ActionConfig a, MemoryConfig m, ReplayBuffer2 memory, DeterministicPolicy policy, QNetwork critic, DeterministicPolicy targetPolicy, QNetwork targetCritic, Tensor.Optimizer policyOpt, Tensor.Optimizer criticOpt, int outputs) {
        super(m, memory);
        this.h = h;
        this.policy = policy;
        this.critic = critic;
        this.targetPolicy = targetPolicy;
        this.targetCritic = targetCritic;
        this.policyOpt = policyOpt;
        this.criticOpt = criticOpt;
        this.noise = Noise.create(a.noise(), outputs);
    }

    @Override
    public MemoryConfig getMemoryConfig() {
        return m;
    }

    @Override
    public UnaryOperator<Tensor> getPolicy() {
        return policy;
    }

    @Override
    public void update(long totalSteps) {
        setTrainingMode(true, policy, critic);
        updateSteps++;
        var batch = toBatch(memory.sample(m.replayBuffer().batchSize()));
        Tensor y;
        try (var ignored = Tensor.noGrad()) {
            var nextActions = targetPolicy.apply(batch.nextStates());
            var targetQ = targetCritic.apply(batch.nextStates(), nextActions);

            var nonTerminal = batch.dones().neg().add(1.0f);
            y = batch.rewards().add(targetQ.mul(h.gamma()).mul(nonTerminal));

        }
        var vCtx = new Tensor.GradQueue();
        critic.apply(batch.states(), batch.actions()).loss(y, Tensor.Loss.MeanSquared).minimize(vCtx);
        vCtx.optimize(criticOpt);

        if (updateSteps % h.policyUpdateFreq() == 0) {
            var pCtx = new Tensor.GradQueue();
            critic.apply(batch.states(), policy.apply(batch.states())).mean().neg().minimize(pCtx);
            pCtx.optimize(policyOpt);
            RLUtils.softUpdate(policy, targetPolicy, h.tau());
            RLUtils.softUpdate(critic, targetCritic, h.tau());
        }
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        setTrainingMode(false, policy);
        try (var ignored = Tensor.noGrad()) {
            var action = policy.apply(state).array();
            if (!deterministic) {
                noise.apply(action, model.rng);
                for (var i = 0; i < action.length; i++) action[i] = Util.clamp(action[i], -1, 1);
            }
            return action;
        }
    }

    private interface Noise {
        static Noise create(ActionConfig.NoiseConfig config, int actionDim) {
            return switch (config.type()) {
                case OU -> new OUNoise(actionDim, config.stddev());
                case GAUSSIAN -> (action, rng) -> {
                    for (var i = 0; i < action.length; i++) action[i] += rng.nextGaussian(0, config.stddev());
                };
                case NONE -> (action, rng) -> {
                };
            };
        }

        void apply(double[] action, RandomGenerator rng);
    }

    private static class OUNoise implements Noise {
        private final double[] state;
        private final double mu = 0, theta = 0.15, sigma;

        OUNoise(int size, double sigma) {
            this.state = new double[size];
            this.sigma = sigma;
        }

        @Override
        public void apply(double[] action, RandomGenerator rng) {
            for (var i = 0; i < state.length; i++) {
                var dx = theta * (mu - state[i]) + sigma * rng.nextGaussian();
                state[i] += dx;
                action[i] += state[i];
            }
        }
    }
}