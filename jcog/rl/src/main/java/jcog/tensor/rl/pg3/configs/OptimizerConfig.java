package jcog.tensor.rl.pg3.configs;

import jcog.math.FloatSupplier;
import jcog.signal.FloatRange;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;

import java.util.Objects;

public record OptimizerConfig(
    Type type,
    FloatRange learningRate // Changed from float to FloatRange
) {
    public enum Type {ADAM, SGD, LION, RANGER } // Added ADAMW, LION, RANGER as they are common

    // Default values
    public static final Type DEFAULT_OPTIMIZER_TYPE = Type.ADAM;
    public static final FloatRange DEFAULT_LEARNING_RATE = new FloatRange(1e-3f, 1e-6f, 1e-1f);

    public OptimizerConfig {
        Objects.requireNonNull(type, "Optimizer type cannot be null");
        Objects.requireNonNull(learningRate, "Learning rate cannot be null");
        if (learningRate.floatValue() <= 0) throw new IllegalArgumentException("Learning rate must be positive.");
    }

    // Static factory for default type if learning rate FloatRange is provided
    public static OptimizerConfig of(FloatRange learningRate) {
        return new OptimizerConfig(DEFAULT_OPTIMIZER_TYPE, learningRate);
    }

    // Static factory for default type if learning rate float is provided
    public static OptimizerConfig of(float learningRateValue) {
        // Assuming FloatRange constructor can take (value, min, max, default) or similar
        // For now, creating a new FloatRange with potentially different min/max if not provided explicitly
        return new OptimizerConfig(DEFAULT_OPTIMIZER_TYPE, new FloatRange(learningRateValue, DEFAULT_LEARNING_RATE.min, DEFAULT_LEARNING_RATE.max));
    }

    // Default constructor
    public OptimizerConfig() {
        this(DEFAULT_OPTIMIZER_TYPE, DEFAULT_LEARNING_RATE);
    }

    // Wither methods
    public OptimizerConfig withType(Type type) {
        return new OptimizerConfig(type, learningRate);
    }

    public OptimizerConfig withLearningRate(FloatRange learningRate) {
        return new OptimizerConfig(type, learningRate);
    }

    public OptimizerConfig withLearningRate(float learningRateValue) {
        return new OptimizerConfig(type, new FloatRange(learningRateValue, this.learningRate.min, this.learningRate.max));
    }

    public Tensor.Optimizer build() {
        FloatSupplier lrSupplier = learningRate::floatValue; // Method reference to get the float
        return switch (type) {
            case ADAM -> new Optimizers.ADAM(lrSupplier).get();
            case SGD -> new Optimizers.SGD(lrSupplier).get();
            case LION -> new Optimizers.LION(lrSupplier).get(5);
            case RANGER -> new Optimizers.Ranger(lrSupplier).get(5);
        };
    }
}
