package jcog.tensor.rl.env;

import java.util.Arrays;
import java.util.random.RandomGenerator;

public class ContinuousSeekEnv implements SyntheticEnv {

    private final int dimensions;
    private final double[] agentPos;
    private final double[] targetPos;
    private final double[] startPos;
    private final double successThreshold;
    private final double stepPenaltyFactor;
    private final double maxAction;

    private int currentSteps;

    public ContinuousSeekEnv(int dimensions, double[] startPos, double[] targetPos,
                             double successThreshold, double stepPenaltyFactor,
                             double maxAction) {
        this.dimensions = dimensions;
        this.startPos = Arrays.copyOf(startPos, startPos.length);
        this.agentPos = Arrays.copyOf(startPos, startPos.length);
        this.targetPos = Arrays.copyOf(targetPos, targetPos.length);
        this.successThreshold = successThreshold;
        this.stepPenaltyFactor = stepPenaltyFactor; // Added to reward, so should be negative
        this.maxAction = maxAction;

        if (startPos.length != dimensions || targetPos.length != dimensions) {
            throw new IllegalArgumentException("Start and target positions must match dimensions.");
        }
    }

    // Default 1D constructor
    public ContinuousSeekEnv() {
        this(1, new double[]{0.0}, new double[]{1.0}, 0.05, -0.01, 0.1);
    }

    // Default 2D constructor
    public static ContinuousSeekEnv W2D() {
        return new ContinuousSeekEnv(2, new double[]{0.0, 0.0}, new double[]{1.0, 1.0}, 0.05, -0.01, 0.1);
    }


    @Override
    public int stateDimension() {
        return dimensions;
    }

    @Override
    public int actionDimension() {
        return dimensions;
    }

    @Override
    public boolean isDiscreteActionSpace() {
        return false;
    }

    @Override
    public double[] reset(RandomGenerator rng) {
        System.arraycopy(startPos, 0, agentPos, 0, dimensions);
        currentSteps = 0;
        return Arrays.copyOf(agentPos, agentPos.length);
    }

    @Override
    public StepResult step(double[] action) {
        if (action.length != dimensions) {
            throw new IllegalArgumentException("Action dimension mismatch.");
        }

        for (int i = 0; i < dimensions; i++) {
            // Clip action to maxAction bounds
            double clippedAction = Math.max(-maxAction, Math.min(maxAction, action[i]));
            agentPos[i] += clippedAction;
            // Optionally, clip agentPos to be within certain bounds, e.g., [-2, 2] or [0, 2] if target is at 1
             agentPos[i] = Math.max(-2.0, Math.min(2.0, agentPos[i]));
        }

        currentSteps++;

        double distance = 0;
        for (int i = 0; i < dimensions; i++) {
            distance += Math.pow(agentPos[i] - targetPos[i], 2);
        }
        distance = Math.sqrt(distance);

        float reward = (float) (-distance * 1.0 + stepPenaltyFactor); // Penalize distance, small penalty for steps

        if (distance < successThreshold) {
            // Optional: give a bonus for reaching the target
            reward += 1f;
        }



        return new StepResult(Arrays.copyOf(agentPos, agentPos.length), reward);
    }

    @Override
    public StepResult step(int action) {
        return null;
    }

}
