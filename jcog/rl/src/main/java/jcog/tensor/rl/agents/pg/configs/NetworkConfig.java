package jcog.tensor.rl.agents.pg.configs;

import jcog.signal.FloatRange; // Assuming this exists
import jcog.tensor.Tensor;
import jcog.util.ArrayUtil; // Assuming this is available

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
    // Defaults
    public static final int[] DEFAULT_HIDDEN_LAYERS = ArrayUtil.EMPTY_INT_ARRAY; // Ensure ArrayUtil is accessible
    public static final UnaryOperator<Tensor> DEFAULT_ACTIVATION = Tensor.RELU;
    public static final UnaryOperator<Tensor> DEFAULT_OUTPUT_ACTIVATION = null;
    public static final boolean DEFAULT_BIAS_IN_LAST_LAYER = true;
    public static final boolean DEFAULT_ORTHOGONAL_INIT = true;
    public static final FloatRange DEFAULT_DROPOUT = new FloatRange(0.0f, 0.0f, 0.9f, 0.0f); // Range from 0 to <1

    // Compact constructor
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

    // Constructor for minimal setup with a specific OptimizerConfig
    public NetworkConfig(OptimizerConfig optimizer, int... hiddenLayers) {
        this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT, optimizer);
    }

    // Constructor with common defaults, creating an OptimizerConfig from a learning rate value
    public NetworkConfig(float defaultLearningRateValue, int... hiddenLayers) {
        this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
             OptimizerConfig.of(defaultLearningRateValue));
    }

    // Constructor with common defaults, creating an OptimizerConfig from a FloatRange learning rate
    public NetworkConfig(FloatRange defaultLearningRateRange, int... hiddenLayers) {
        this(hiddenLayers, DEFAULT_ACTIVATION, DEFAULT_OUTPUT_ACTIVATION,
             DEFAULT_BIAS_IN_LAST_LAYER, DEFAULT_ORTHOGONAL_INIT, DEFAULT_DROPOUT,
             OptimizerConfig.of(defaultLearningRateRange));
    }

    // Default constructor using all default values, including a default optimizer.
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
        // Assuming FloatRange constructor (value, min, max, default)
        return new NetworkConfig(hiddenLayers, activation, outputActivation, biasInLastLayer, orthogonalInit,
                                 new FloatRange(dropoutValue, this.dropout.min(), this.dropout.max(), dropoutValue), optimizer);
    }
}
