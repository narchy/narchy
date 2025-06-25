package jcog.tensor.rl.pg2;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.math.FloatSupplier;
import jcog.tensor.Models;
import jcog.tensor.Models.Linear;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.model.ODELayer;
import jcog.tensor.rl.pg.PolicyGradientModel;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg.util.Memory;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import jcog.tensor.rl.util.RLNetworkUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;
import java.util.stream.Stream;

import static jcog.tensor.rl.pg2.PGBuilder.*;

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
 * This class formerly contained a fluent builder for policy gradient agents.
 * It has been refactored to promote direct constructor-based dependency injection
 * for all components (configurations, networks, strategies).
 *
 * Users should now:
 * 1. Instantiate configuration records (e.g., HyperparamConfig, NetworkConfig).
 * 2. Instantiate network models (e.g., GaussianPolicy, ValueNetwork) using these configs.
 * 3. Instantiate optimizers (e.g., via OptimizerConfig.buildOptimizer()).
 * 4. Instantiate memory buffers (e.g., OnPolicyEpisodeBuffer, ReplayBuffer2).
 * 5. Instantiate the desired AlgorithmStrategy (e.g., PPOStrategy, SACStrategy)
 *    by passing all the above components directly to its constructor.
 * 6. Instantiate the PolicyGradientModel, passing the strategy.
 *
 * The static inner classes for configurations (HyperparamConfig, NetworkConfig, etc.)
 * and network models (GaussianPolicy, ValueNetwork, etc.) remain part of this file
 * as they define the components to be injected.
 *
 * The ModelFactory, Algorithm enum, and AlgorithmConfigurator infrastructure
 * have been removed as they were part of the old builder pattern.
 * Their logic will be handled by direct instantiation or moved to static factory methods
 * on the respective components if complex creation logic is still needed.
 */
public class PGBuilder { // Class name retained to minimize disruption, though it's no longer a builder.
                         // Consider renaming to something like PolicyGradientComponents or RLAgentFactoryUtils in the future.

    // createMlp method has been moved to jcog.tensor.rl.util.RLNetworkUtils

    // createOptimizer is now part of OptimizerConfig.buildOptimizer()
    // createGaussianPolicy is now new GaussianPolicy(...)

    // All former PGBuilder fluent methods, Algorithm enum, and AlgorithmConfigurator implementations are removed.
    // Configuration records and network classes remain below.

    //TODO make default constructor with sane defaults, then allow overrides with `with` keyword if Java records get it, or provide copy-constructors
    public record HyperparamConfig(
            float gamma, float lambda, float entropyBonus, float ppoClip, float tau,
            float policyLR, float valueLR, int epochs, int policyUpdateFreq,
            boolean normalizeAdvantages, boolean normalizeReturns, boolean learnableAlpha
    ) {
        // Default values
        public static final float DEFAULT_GAMMA = 0.9f;
        public static final float DEFAULT_LAMBDA = 0.5f;
        public static final float DEFAULT_ENTROPY_BONUS = 0.01f;
        public static final float DEFAULT_PPO_CLIP = 0.2f;
        public static final float DEFAULT_TAU = 0.005f;
        public static final float DEFAULT_POLICY_LR = 3e-5f;
        public static final float DEFAULT_VALUE_LR = 1e-4f;
        public static final int DEFAULT_EPOCHS = 1;
        public static final int DEFAULT_POLICY_UPDATE_FREQ = 1;
        public static final boolean DEFAULT_NORMALIZE_ADVANTAGES = true;
        public static final boolean DEFAULT_NORMALIZE_RETURNS = false;
        public static final boolean DEFAULT_LEARNABLE_ALPHA = false;

        // Constructor with all arguments for full customization
        public HyperparamConfig {
            if (gamma < 0.0f || gamma > 1.0f) throw new IllegalArgumentException("gamma must be in [0, 1]");
            if (lambda < 0.0f || lambda > 1.0f) throw new IllegalArgumentException("lambda must be in [0, 1]");
            if (tau < 0.0f || tau > 1.0f) throw new IllegalArgumentException("tau must be in [0, 1]");
            if (policyLR <= 0) throw new IllegalArgumentException("policyLR must be positive");
            if (valueLR <= 0) throw new IllegalArgumentException("valueLR must be positive");
            if (epochs <= 0) throw new IllegalArgumentException("epochs must be positive");
            if (policyUpdateFreq <= 0) throw new IllegalArgumentException("policyUpdateFreq must be positive");
            if (ppoClip <= 0) throw new IllegalArgumentException("ppoClip must be positive");
        }

        // Constructor with default values
        public HyperparamConfig() {
            this(DEFAULT_GAMMA, DEFAULT_LAMBDA, DEFAULT_ENTROPY_BONUS, DEFAULT_PPO_CLIP, DEFAULT_TAU,
                 DEFAULT_POLICY_LR, DEFAULT_VALUE_LR, DEFAULT_EPOCHS, DEFAULT_POLICY_UPDATE_FREQ,
                 DEFAULT_NORMALIZE_ADVANTAGES, DEFAULT_NORMALIZE_RETURNS, DEFAULT_LEARNABLE_ALPHA);
        }

        // "Wither" methods for individual parameter modification, returning a new instance
        public HyperparamConfig withGamma(float gamma) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withLambda(float lambda) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withEntropyBonus(float entropyBonus) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withPpoClip(float ppoClip) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withTau(float tau) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withPolicyLR(float policyLR) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withValueLR(float valueLR) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withEpochs(int epochs) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withPolicyUpdateFreq(int policyUpdateFreq) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withNormalizeAdvantages(boolean normalizeAdvantages) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withNormalizeReturns(boolean normalizeReturns) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
        public HyperparamConfig withLearnableAlpha(boolean learnableAlpha) {
            return new HyperparamConfig(gamma, lambda, entropyBonus, ppoClip, tau, policyLR, valueLR, epochs, policyUpdateFreq, normalizeAdvantages, normalizeReturns, learnableAlpha);
        }
    }

//    public record NetworkConfig(
//            int[] hiddenLayers, UnaryOperator<Tensor> activation, UnaryOperator<Tensor> outputActivation,
//            boolean biasInLastLayer, boolean orthogonalInit, float dropout, OptimizerConfig optimizer
//    ) {
//        // Defaults
//        public static final int[] DEFAULT_HIDDEN_LAYERS = ArrayUtil.EMPTY_INT_ARRAY;
//        public static final UnaryOperator<Tensor> DEFAULT_ACTIVATION = Tensor.RELU;
//        public static final UnaryOperator<Tensor> DEFAULT_OUTPUT_ACTIVATION = null; // often identity or specific to use-case
//        public static final boolean DEFAULT_BIAS_IN_LAST_LAYER = true;
//        public static final boolean DEFAULT_ORTHOGONAL_INIT = true;
//        public static final float DEFAULT_DROPOUT = 0.0f; // Default to no dropout
//
//        public NetworkConfig {
//            Objects.requireNonNull(hiddenLayers, "hiddenLayers cannot be null");
//            Objects.requireNonNull(activation, "activation function cannot be null");
//            // outputActivation can be null
//            Objects.requireNonNull(optimizer, "optimizer cannot be null");
//            if (dropout < 0.0f || dropout >= 1.0f) throw new IllegalArgumentException("dropout must be in [0, 1)");
//        }
//
//        // Constructor for minimal setup, typically for when an optimizer with a specific LR is needed.
//        public NetworkConfig(OptimizerConfig optimizer, int... hiddenLayers) {
//            this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
//                 DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT, optimizer);
//        }
//
//        // Constructor with common defaults, requiring explicit optimizer and default learning rate for it
//        public NetworkConfig(float defaultLearningRate, int... hiddenLayers) {
//            this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
//                 DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
//                 OptimizerConfig.of(defaultLearningRate)); // Create default optimizer if only LR is given
//        }
//
//        // Default constructor using all default values, including a default optimizer.
//        // NOTE: This defaultLearningRate is a bit arbitrary here, might be better to require OptimizerConfig explicitly.
//        // For now, matches typical PGBuilder use.
//        public NetworkConfig(float defaultLearningRate) {
//            this(DEFAULT_HIDDEN_LAYERS, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
//                 DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
//                 OptimizerConfig.of(defaultLearningRate));
//        }
//
//
//        public boolean isConfigured() { // Retained as it's used in PGBuilder logic
//            return hiddenLayers != null && hiddenLayers.length > 0;
//        }
//
//        // Wither methods

//        public NetworkConfig withOutputActivation(UnaryOperator<Tensor> outputActivation) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
//        }
//    }

    public record OptimizerConfig(Type type, float learningRate) {
        public enum Type {ADAM, SGD}

        // Default values
        public static final Type DEFAULT_OPTIMIZER_TYPE = Type.ADAM;
        // No default for learningRate, it should be specified or defaulted by the network using it.

        // Constructor with all arguments for full customization
        public OptimizerConfig {
            if (learningRate <= 0) throw new IllegalArgumentException("Learning rate must be positive.");
            Objects.requireNonNull(type, "Optimizer type cannot be null");
        }

        // Static factory for default type if learning rate is provided
        public static OptimizerConfig of(float learningRate) {
            return new OptimizerConfig(DEFAULT_OPTIMIZER_TYPE, learningRate);
        }

        // "Wither" methods
        public OptimizerConfig withType(Type type) {
            return new OptimizerConfig(type, learningRate);
        }

        public OptimizerConfig withLearningRate(float learningRate) {
            return new OptimizerConfig(type, learningRate);
        }

        public Tensor.Optimizer buildOptimizer() {
            FloatSupplier lrSupplier = this::learningRate;
            return switch (type) {
                case ADAM -> new Optimizers.ADAM(lrSupplier).get();
                case SGD -> new Optimizers.SGD(lrSupplier).get();
            };
        }
    }

    public record ActionConfig(Distribution distribution, float sigmaMin, float sigmaMax, NoiseConfig noise) {
        public enum Distribution {GAUSSIAN, DETERMINISTIC}

        public record NoiseConfig(Type type, float stddev) {
            public enum Type {NONE, GAUSSIAN, OU}

            // Defaults
            public static final Type DEFAULT_NOISE_TYPE = Type.GAUSSIAN;
            public static final float DEFAULT_NOISE_STDDEV = 0.1f;

            public NoiseConfig {
                Objects.requireNonNull(type, "Noise type cannot be null");
                if (stddev < 0 && type != Type.NONE) throw new IllegalArgumentException("Noise stddev must be non-negative for GAUSSIAN or OU noise.");
            }

            public NoiseConfig() {
                this(DEFAULT_NOISE_TYPE, DEFAULT_NOISE_STDDEV);
            }

            public NoiseConfig withType(Type type) {
                return new NoiseConfig(type, this.stddev);
            }

            public NoiseConfig withStddev(float stddev) {
                return new NoiseConfig(this.type, stddev);
            }
        }

        // Defaults for ActionConfig
        public static final Distribution DEFAULT_ACTION_DISTRIBUTION = Distribution.GAUSSIAN;
        public static final float DEFAULT_SIGMA_MIN = 0.2f;
        public static final float DEFAULT_SIGMA_MAX = 4.0f;
        public static final NoiseConfig DEFAULT_NOISE_CONFIG = new NoiseConfig();


        public ActionConfig {
            Objects.requireNonNull(distribution, "Action distribution cannot be null");
            Objects.requireNonNull(noise, "Noise config cannot be null");
            if (sigmaMin >= sigmaMax && distribution == Distribution.GAUSSIAN) {
                throw new IllegalArgumentException("sigmaMin must be less than sigmaMax for GAUSSIAN distribution");
            }
        }

        public ActionConfig() {
            this(DEFAULT_ACTION_DISTRIBUTION, DEFAULT_SIGMA_MIN, DEFAULT_SIGMA_MAX, DEFAULT_NOISE_CONFIG);
        }

        public ActionConfig withDistribution(Distribution distribution) {
            return new ActionConfig(distribution, sigmaMin, sigmaMax, noise);
        }

        public ActionConfig withSigmaMin(float sigmaMin) {
            return new ActionConfig(distribution, sigmaMin, sigmaMax, noise);
        }

        public ActionConfig withSigmaMax(float sigmaMax) {
            return new ActionConfig(distribution, sigmaMin, sigmaMax, noise);
        }

        public ActionConfig withNoiseConfig(NoiseConfig noise) {
            return new ActionConfig(distribution, sigmaMin, sigmaMax, noise);
        }

        public ActionConfig withNoiseType(NoiseConfig.Type noiseType) {
            return new ActionConfig(distribution, sigmaMin, sigmaMax, this.noise.withType(noiseType));
        }

        public ActionConfig withNoiseStddev(float noiseStddev) {
            return new ActionConfig(distribution, sigmaMin, sigmaMax, this.noise.withStddev(noiseStddev));
        }
    }

    public record MemoryConfig(int episodeLength, ReplayBufferConfig replayBuffer) {
        public record ReplayBufferConfig(int capacity, int batchSize) {
            // Defaults
            public static final int DEFAULT_CAPACITY = 100_000;
            public static final int DEFAULT_BATCH_SIZE = 256;

            public ReplayBufferConfig {
                if (capacity <= 0 || batchSize <= 0)
                    throw new IllegalArgumentException("Buffer capacity and batch size must be positive");
                if (batchSize > capacity)
                    throw new IllegalArgumentException("Batch size cannot be larger than buffer capacity");
            }

            public ReplayBufferConfig() {
                this(DEFAULT_CAPACITY, DEFAULT_BATCH_SIZE);
            }

            public ReplayBufferConfig withCapacity(int capacity) {
                return new ReplayBufferConfig(capacity, this.batchSize);
            }

            public ReplayBufferConfig withBatchSize(int batchSize) {
                return new ReplayBufferConfig(this.capacity, batchSize);
            }
        }

        // Defaults for MemoryConfig
        public static final int DEFAULT_EPISODE_LENGTH = 2048; // A common value for PPO.
        public static final ReplayBufferConfig DEFAULT_REPLAY_BUFFER_CONFIG = new ReplayBufferConfig();

        public MemoryConfig {
            if (episodeLength <= 0 && replayBuffer == null) { // episodeLength is for on-policy, replayBuffer for off-policy
                 throw new IllegalArgumentException("Episode length must be positive for on-policy, or replay buffer must be configured for off-policy.");
            }
             if (episodeLength <= 0 && (replayBuffer != null && (replayBuffer.capacity <=0 || replayBuffer.batchSize <=0))) {
                throw new IllegalArgumentException("Episode length must be positive for on-policy, or replay buffer parameters must be positive for off-policy.");
            }
            // Allow replayBuffer to be null for on-policy algorithms that don't use it.
            // Validation for replayBuffer itself is handled in its own constructor.
        }

        public MemoryConfig(int episodeLength) {
            this(episodeLength, null); // For on-policy that doesn't use a replay buffer
        }

        public MemoryConfig(ReplayBufferConfig replayBufferConfig) {
            this(0, replayBufferConfig); // For off-policy, episodeLength might not be relevant or can be defaulted/ignored
        }

        public MemoryConfig() {
            this(DEFAULT_EPISODE_LENGTH, DEFAULT_REPLAY_BUFFER_CONFIG);
        }


        public MemoryConfig withEpisodeLength(int episodeLength) {
            return new MemoryConfig(episodeLength, this.replayBuffer);
        }

        public MemoryConfig withReplayBufferConfig(ReplayBufferConfig replayBuffer) {
            return new MemoryConfig(this.episodeLength, replayBuffer);
        }

        public MemoryConfig withReplayBufferCapacity(int capacity) {
            ReplayBufferConfig currentRB = this.replayBuffer == null ? new ReplayBufferConfig() : this.replayBuffer;
            return new MemoryConfig(this.episodeLength, currentRB.withCapacity(capacity));
        }

        public MemoryConfig withReplayBufferBatchSize(int batchSize) {
            ReplayBufferConfig currentRB = this.replayBuffer == null ? new ReplayBufferConfig() : this.replayBuffer;
            return new MemoryConfig(this.episodeLength, currentRB.withBatchSize(batchSize));
        }
    }

    public record OdeConfig(int stateSize, int hiddenSize, int steps, SolverType solverType) {
        public enum SolverType {EULER, RK4}

        // Defaults
        public static final int DEFAULT_STATE_SIZE = 64;
        public static final int DEFAULT_HIDDEN_SIZE = 128;
        public static final int DEFAULT_STEPS = 5;
        public static final SolverType DEFAULT_SOLVER_TYPE = SolverType.RK4;

        public OdeConfig {
            if (stateSize <= 0) throw new IllegalArgumentException("stateSize must be positive.");
            if (hiddenSize <= 0) throw new IllegalArgumentException("hiddenSize must be positive.");
            if (steps <= 0) throw new IllegalArgumentException("steps must be positive.");
            Objects.requireNonNull(solverType, "solverType cannot be null.");
        }

        public OdeConfig() {
            this(DEFAULT_STATE_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_STEPS, DEFAULT_SOLVER_TYPE);
        }

        // Wither methods
        public OdeConfig withStateSize(int stateSize) {
            return new OdeConfig(stateSize, hiddenSize, steps, solverType);
        }
        public OdeConfig withHiddenSize(int hiddenSize) {
            return new OdeConfig(stateSize, hiddenSize, steps, solverType);
        }
        public OdeConfig withSteps(int steps) {
            return new OdeConfig(stateSize, hiddenSize, steps, solverType);
        }
        public OdeConfig withSolverType(SolverType solverType) {
            return new OdeConfig(stateSize, hiddenSize, steps, solverType);
        }
    }

    public static class GaussianPolicy extends Models.Layers {
        public final Models.Layers body;
        public final Linear muHead;
        public final Linear logSigmaHead;
        public final NetworkConfig config;
        public final int inputs;
        public final int outputs;

        // Comprehensive constructor for subclasses or advanced use
        public GaussianPolicy(NetworkConfig config, int inputs, int outputs, Models.Layers body, Linear muHead, Linear logSigmaHead) {
            super(null, null, false); // Base Models.Layers constructor
            this.config = Objects.requireNonNull(config, "NetworkConfig cannot be null");
            this.inputs = inputs;
            this.outputs = outputs;
            this.body = Objects.requireNonNull(body, "Body cannot be null");
            this.muHead = Objects.requireNonNull(muHead, "muHead cannot be null");
            this.logSigmaHead = Objects.requireNonNull(logSigmaHead, "logSigmaHead cannot be null");

            this.layer.add(this.body);
            // muHead and logSigmaHead are used in getDistribution, not added to this.layer directly
        }

        // Standard constructor that builds an MLP body
        public GaussianPolicy(NetworkConfig config, int inputs, int outputs) {
            this(config, inputs, outputs,
                 buildDefaultMlpBody(config, inputs), // Call static method to build body
                 createOrthogonalLinear( // Directly create muHead
                    getLastLayerSize(config, inputs), // Calculate last layer size for head input
                    outputs, null, config.biasInLastLayer(), 0.01),
                 createOrthogonalLinear( // Directly create logSigmaHead
                    getLastLayerSize(config, inputs), // Calculate last layer size for head input
                    outputs, null, config.biasInLastLayer(), 0.01)
            );
        }

        // Helper to calculate the size of the last layer of the MLP body, needed for head creation
        private static int getLastLayerSize(NetworkConfig config, int inputs) {
            if (config.hiddenLayers() == null || config.hiddenLayers().length == 0) {
                return inputs;
            }
            return config.hiddenLayers()[config.hiddenLayers().length - 1];
        }

        // Static method to build the default MLP body, used by the standard constructor
        private static Models.Layers buildDefaultMlpBody(NetworkConfig config, int inputs) {
            Models.Layers newBody = new Models.Layers(config.activation(), config.activation(), true);
            int lastLayerSize = inputs;
            if (config.hiddenLayers() != null) {
                for (int hiddenSize : config.hiddenLayers()) {
                    newBody.layer.add(createOrthogonalLinear(lastLayerSize, hiddenSize, config.activation(), true, Math.sqrt(2.0)));
                    lastLayerSize = hiddenSize;
                }
            }
            return newBody;
        }

        // Renamed from buildMlpBody to avoid confusion, now specifically for the default MLP policy body construction.
        // This method is now part of the static buildDefaultMlpBody.
        // public static int configureMlpBodyLayers(Models.Layers bodyToConfigure, NetworkConfig config, int inputs) {
        //     var lastLayerSize = inputs;
        //     for (var hiddenSize : config.hiddenLayers()) {
        //         bodyToConfigure.layer.add(createOrthogonalLinear(lastLayerSize, hiddenSize, config.activation(), true, Math.sqrt(2.0)));
        //         lastLayerSize = hiddenSize;
        //     }
        //     return lastLayerSize;
        // }

        static Linear createOrthogonalLinear(int I, int O, UnaryOperator<Tensor> activation, boolean bias, double gain) {
            var linearLayer = new Linear(I, O, activation, bias);
            Tensor.orthoInit(linearLayer.weight, gain);
            if (bias) linearLayer.bias.zero();
            return linearLayer;
        }

        // createHeads logic is now incorporated into the main constructor(s)

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
        public final OdeConfig odeConfig;

        public OdeGaussianPolicy(NetworkConfig config, OdeConfig odeConfig, int inputs, int outputs) {
            // 1. Construct the specific body for OdeGaussianPolicy
            Models.Layers constructedBody = new Models.Layers(
                    Objects.requireNonNull(config, "NetworkConfig cannot be null for OdeGaussianPolicy").activation(),
                    config.activation(),
                    true); // Assuming bias in body layers based on typical MLP structure

            var inputProjection = createOrthogonalLinear(inputs,
                    Objects.requireNonNull(odeConfig, "OdeConfig cannot be null for OdeGaussianPolicy").stateSize(),
                    config.activation(), true, Math.sqrt(2.0));
            var odeL = new ODELayer(odeConfig.stateSize(), odeConfig.stateSize(), odeConfig.hiddenSize(),
                    odeConfig.solverType() == OdeConfig.SolverType.RK4, odeConfig::steps);
            constructedBody.layer.add(inputProjection);
            constructedBody.layer.add(odeL);

            // 2. Determine the output size of this body (which is the input size for the heads)
            int bodyOutputSize = odeConfig.stateSize();

            // 3. Construct the heads
            Linear constructedMuHead = createOrthogonalLinear(bodyOutputSize, outputs, null, config.biasInLastLayer(), 0.01);
            Linear constructedLogSigmaHead = createOrthogonalLinear(bodyOutputSize, outputs, null, config.biasInLastLayer(), 0.01);

            // 4. Call the comprehensive super constructor from GaussianPolicy
            super(config, inputs, outputs, constructedBody, constructedMuHead, constructedLogSigmaHead);

            // 5. Initialize own final fields
            this.odeConfig = odeConfig;
        }
    }

    public static class ValueNetwork extends Models.Layers {
        public final UnaryOperator<Tensor> network;
        public final NetworkConfig config;
        public final int stateDim;

        public ValueNetwork(NetworkConfig config, int stateDim) {
            super(null, null, false); // Base class constructor
            this.config = Objects.requireNonNull(config, "NetworkConfig cannot be null");
            this.stateDim = stateDim;
            this.network = RLNetworkUtils.createMlp(stateDim, 1, config); // Changed to RLNetworkUtils.createMlp
            this.layer.add(network); // Add the created MLP to this Models.Layers instance
        }

        @Override
        public Tensor apply(Tensor state) {
            return this.network.apply(state);
        }
    }

    public static class DeterministicPolicy extends Models.Layers {
        public final UnaryOperator<Tensor> network;
        public final NetworkConfig config;
        public final int inputs;
        public final int outputs;

        public DeterministicPolicy(NetworkConfig config, int inputs, int outputs) {
            super(null, null, false); // Base class constructor
            this.config = Objects.requireNonNull(config, "NetworkConfig cannot be null");
            this.inputs = inputs;
            this.outputs = outputs;
            this.network = RLNetworkUtils.createMlp(inputs, outputs, config); // Changed to RLNetworkUtils.createMlp
            this.layer.add(network); // Add the created MLP to this Models.Layers instance
        }

        @Override
        public Tensor apply(Tensor state) {
            return this.network.apply(state);
        }
    }

    public static class QNetwork extends Models.Layers {
        public final UnaryOperator<Tensor> network;
        public final NetworkConfig config;
        public final int stateDim;
        public final int actionDim;

        public QNetwork(NetworkConfig config, int stateDim, int actionDim) {
            super(null, null, false); // Base class constructor
            this.config = Objects.requireNonNull(config, "NetworkConfig cannot be null");
            this.stateDim = stateDim;
            this.actionDim = actionDim;
            this.network = RLNetworkUtils.createMlp(stateDim + actionDim, 1, config); // Changed to RLNetworkUtils.createMlp
            this.layer.add(network); // Add the created MLP to this Models.Layers instance
        }

        public Tensor apply(Tensor state, Tensor action) {
            return this.network.apply(state.concat(action));
        }

        @Override
        public Tensor apply(Tensor t) {
            throw new UnsupportedOperationException("Use apply(state, action) instead.");
        }
    }

    // Former AlgorithmConfigurator implementations (ReinforceConfigurator, PPOConfigurator, SACConfigurator, DDPGConfigurator)
    // are removed. Their logic for validation and strategy construction will now be handled by:
    // 1. Direct instantiation of strategies with all required components.
    // 2. Validation logic can be part of strategy constructors or dedicated validation methods if needed.
    // 3. Helper/factory methods for creating common strategy setups can be added if desired,
    //    but the primary path is direct instantiation.
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

abstract class AbstractStrategy implements PGStrategy {
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
    public final HyperparamConfig h;
    public final ActionConfig a;
    public final MemoryConfig m;
    public final Memory memory; // Changed to Memory interface
    public final GaussianPolicy policy;
    public final Tensor.Optimizer policyOpt;

    public ReinforceStrategy(HyperparamConfig h, ActionConfig a, MemoryConfig m,
                             Memory memory, // Changed to Memory interface
                             GaussianPolicy policy, Tensor.Optimizer policyOpt) {
        this.h = Objects.requireNonNull(h);
        this.a = Objects.requireNonNull(a);
        this.m = Objects.requireNonNull(m);
        this.memory = Objects.requireNonNull(memory);
        this.policy = Objects.requireNonNull(policy);
        this.policyOpt = Objects.requireNonNull(policyOpt);
    }

    @Override
    public Memory getMemory() {
        return memory; // Already returns Memory type
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

abstract class OffPolicyStrategy extends AbstractStrategy {
    public final MemoryConfig m;
    public final Memory memory; // Changed to Memory interface

    protected OffPolicyStrategy(MemoryConfig m, Memory memory) { // Changed to Memory interface
        this.m = Objects.requireNonNull(m);
        this.memory = Objects.requireNonNull(memory);
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
    public final HyperparamConfig h;
    public final ActionConfig a;
    public final GaussianPolicy policy;
    public final List<QNetwork> qNetworks;
    public final List<QNetwork> targetQNetworks;
    public final Tensor.Optimizer policyOpt;
    public final List<Tensor.Optimizer> qOpts;
    public final Tensor logAlpha; // Initialized based on h.learnableAlpha()
    public final Tensor.Optimizer alphaOpt; // Can be null
    public final float targetEntropy;
    public final int outputs; // Storing for clarity, as it's used for targetEntropy

    public SACStrategy(HyperparamConfig h, ActionConfig a, MemoryConfig m,
                       Memory memory, // Changed to Memory interface
                       GaussianPolicy policy,
                       List<QNetwork> qNetworks, List<QNetwork> targetQNetworks,
                       Tensor.Optimizer policyOpt, List<Tensor.Optimizer> qOpts,
                       int outputs) {
        super(m, memory);
        this.h = Objects.requireNonNull(h);
        this.a = Objects.requireNonNull(a);
        this.policy = Objects.requireNonNull(policy);
        this.qNetworks = Objects.requireNonNull(qNetworks);
        this.targetQNetworks = Objects.requireNonNull(targetQNetworks);
        this.policyOpt = Objects.requireNonNull(policyOpt);
        this.qOpts = Objects.requireNonNull(qOpts);
        this.outputs = outputs;

        if (qNetworks.size() != targetQNetworks.size() || qNetworks.size() != qOpts.size() || qNetworks.isEmpty()) {
            throw new IllegalArgumentException("qNetworks, targetQNetworks, and qOpts must be non-empty and of the same size.");
        }

        this.targetEntropy = -outputs; // Cast to float
        if (h.learnableAlpha()) {
            this.logAlpha = Tensor.zeros(1, 1).parameter();
            // Use valueLR from HyperparamConfig for alpha optimizer's learning rate
            this.alphaOpt = new Optimizers.ADAM(h::valueLR).get();
        } else {
            this.logAlpha = Tensor.scalar(Math.log(h.entropyBonus()));
            this.alphaOpt = null;
        }
    }

    @Override
    public boolean isOffPolicy() {
        return true;
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

