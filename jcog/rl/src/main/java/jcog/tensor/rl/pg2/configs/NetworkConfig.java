package jcog.tensor.rl.pg2.configs;

import jcog.signal.FloatRange;
import jcog.tensor.Tensor;
import jcog.util.ArrayUtil;

import java.util.Objects;
import java.util.function.UnaryOperator;

public record NetworkConfig(
    int[] hiddenLayers, // Keep as int[]
    UnaryOperator<Tensor> activation, // Keep as is
    UnaryOperator<Tensor> outputActivation, // Keep as is
    boolean biasInLastLayer, // Keep as boolean
    boolean orthogonalInit, // Keep as boolean
    FloatRange dropout, // Changed from float to FloatRange
    OptimizerConfig optimizer // Already a record, assumed to be the new version
) {
    public static final int[] DEFAULT_HIDDEN_LAYERS = ArrayUtil.EMPTY_INT_ARRAY; // Ensure ArrayUtil is accessible
    public static final UnaryOperator<Tensor> DEFAULT_ACTIVATION = Tensor.RELU;
    public static final UnaryOperator<Tensor> DEFAULT_OUTPUT_ACTIVATION = null;
    public static final boolean DEFAULT_BIAS_IN_LAST_LAYER = true;
    public static final boolean DEFAULT_ORTHOGONAL_INIT = true;
    public static final FloatRange DEFAULT_DROPOUT = new FloatRange(0.0f, 0.0f, 0.9f); // Range from 0 to <1

    public NetworkConfig {
        Objects.requireNonNull(hiddenLayers, "hiddenLayers cannot be null");
        Objects.requireNonNull(activation, "activation function cannot be null");
        // outputActivation can be null
        Objects.requireNonNull(optimizer, "optimizer cannot be null");
        Objects.requireNonNull(dropout, "dropout cannot be null");
        if (dropout.floatValue() < 0.0f || dropout.floatValue() >= 1.0f) { // Dropout should be [0, 1)
            throw new IllegalArgumentException("dropout must be in [0, 1)");
        }
    }

    public NetworkConfig(OptimizerConfig optimizer, int... hiddenLayers) {
        this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT, optimizer);
    }

    public NetworkConfig(float defaultLearningRateValue, int... hiddenLayers) {
        this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
             OptimizerConfig.of(defaultLearningRateValue));
    }

    public NetworkConfig(FloatRange defaultLearningRateRange, int... hiddenLayers) {
        this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
             OptimizerConfig.of(defaultLearningRateRange));
    }

    public NetworkConfig() {
        this(DEFAULT_HIDDEN_LAYERS, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
             new OptimizerConfig() // Uses OptimizerConfig's default constructor
            );
    }

    // Wither methods (selected examples)
    public NetworkConfig withHiddenLayers(int... hiddenLayers) {
        return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
    }

    public NetworkConfig withOptimizer(OptimizerConfig optimizer) {
        return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
    }

    public NetworkConfig withDropout(FloatRange dropout) {
        return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
    }

    public NetworkConfig withDropout(float dropoutValue) {
        return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit,
                 new FloatRange(dropoutValue, this.dropout.min, this.dropout.max), optimizer);
    }
    public NetworkConfig withOutputActivation(UnaryOperator<Tensor> outputActivation) {
        return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
    }

//        public NetworkConfig withHiddenLayers(int... hiddenLayers) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
//        }
        public NetworkConfig withActivation(UnaryOperator<Tensor> activation) {
            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
        }
//        public NetworkConfig withBiasInLastLayer(boolean biasInLastLayer) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
//        }
//        public NetworkConfig withOrthogonalInit(boolean orthogonalInit) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
//        }
//        public NetworkConfig withDropout(float dropout) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
//        }
//        public NetworkConfig withOptimizer(OptimizerConfig optimizer) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, optimizer);
//        }
//        public NetworkConfig withOptimizerLearningRate(float learningRate) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, this.optimizer.withLearningRate(learningRate));
//        }
//        public NetworkConfig withOptimizerType(OptimizerConfig.Type type) {
//            return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit, dropout, this.optimizer.withType(type));
//        }

}
