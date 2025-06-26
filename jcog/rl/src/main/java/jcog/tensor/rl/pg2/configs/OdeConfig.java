package jcog.tensor.rl.pg2.configs;

import jcog.signal.IntRange;

import java.util.Objects;

public record OdeConfig(
    IntRange stateSize,    // Changed from int
    IntRange hiddenSize,   // Changed from int
    IntRange steps,        // Changed from int
    SolverType solverType
) {
    public enum SolverType {EULER, RK4}

    // Defaults
    public static final IntRange DEFAULT_STATE_SIZE = new IntRange(64, 8, 256);
    public static final IntRange DEFAULT_HIDDEN_SIZE = new IntRange(128, 16, 512);
    public static final IntRange DEFAULT_STEPS = new IntRange(5, 1, 20);
    public static final SolverType DEFAULT_SOLVER_TYPE = SolverType.RK4;

    public OdeConfig {
        Objects.requireNonNull(stateSize, "stateSize cannot be null");
        Objects.requireNonNull(hiddenSize, "hiddenSize cannot be null");
        Objects.requireNonNull(steps, "steps cannot be null");
        Objects.requireNonNull(solverType, "solverType cannot be null.");

        if (stateSize.intValue() <= 0) throw new IllegalArgumentException("stateSize must be positive.");
        if (hiddenSize.intValue() <= 0) throw new IllegalArgumentException("hiddenSize must be positive.");
        if (steps.intValue() <= 0) throw new IllegalArgumentException("steps must be positive.");
    }

    public OdeConfig() {
        this(DEFAULT_STATE_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_STEPS, DEFAULT_SOLVER_TYPE);
    }

    // Wither methods
    public OdeConfig withStateSize(IntRange stateSize) {
        return new OdeConfig(stateSize, hiddenSize, steps, solverType);
    }
    public OdeConfig withStateSize(int value) {
        // Assuming IntRange constructor (value, min, max, default)
        return new OdeConfig(new IntRange(value, this.stateSize.min, this.stateSize.max), hiddenSize, steps, solverType);
    }

    public OdeConfig withHiddenSize(IntRange hiddenSize) {
        return new OdeConfig(stateSize, hiddenSize, steps, solverType);
    }
    public OdeConfig withHiddenSize(int value) {
        return new OdeConfig(stateSize, new IntRange(value, this.hiddenSize.min, this.hiddenSize.max), steps, solverType);
    }

    public OdeConfig withSteps(IntRange steps) {
        return new OdeConfig(stateSize, hiddenSize, steps, solverType);
    }
    public OdeConfig withSteps(int value) {
        return new OdeConfig(stateSize, hiddenSize, new IntRange(value, this.steps.min, this.steps.max), solverType);
    }

    public OdeConfig withSolverType(SolverType solverType) {
        return new OdeConfig(stateSize, hiddenSize, steps, solverType);
    }
}
