package jcog.tensor.rl.env;

import java.util.Arrays;
import java.util.Random;
import java.util.random.RandomGenerator;

public class SimpleGridEnv implements SyntheticEnv {

    private final int width;
    private final int height;
    private final int[] startState;
    private final int[] goalState;
    private final double goalReward;
    private final double wallPenalty;
    public final double stepPenalty;


    private int[] currentState;
    private int currentStep;
    private final Random random;

    public static final int ACTION_UP = 0;
    public static final int ACTION_DOWN = 1;
    public static final int ACTION_LEFT = 2;
    public static final int ACTION_RIGHT = 3;

    public SimpleGridEnv(int width, int height, int[] startState, int[] goalState,
                         double goalReward, double wallPenalty, double stepPenalty) {
        this.width = width;
        this.height = height;
        this.startState = Arrays.copyOf(startState, startState.length);
        this.goalState = Arrays.copyOf(goalState, goalState.length);
        this.goalReward = goalReward;
        this.wallPenalty = wallPenalty;
        this.stepPenalty = stepPenalty;
        this.random = new Random(); // Or allow seeding
        reset(random);
    }

    public SimpleGridEnv() {
        this(5, 5, new int[]{0, 0}, new int[]{4, 4}, 1.0, -0.5, -0.01);
    }

    @Override
    public int stateDimension() {
        return 2; // x, y coordinates
    }

    @Override
    public int actionDimension() {
        return 4; // Up, Down, Left, Right
    }

    @Override
    public boolean isDiscreteActionSpace() {
        return true;
    }

    @Override
    public double[] reset(RandomGenerator rng) {
        this.currentState = Arrays.copyOf(startState, startState.length);
        this.currentStep = 0;
        return Arrays.stream(this.currentState).asDoubleStream().toArray();
    }

    @Override
    public StepResult step(double[] action) {
        throw new UnsupportedOperationException("SimpleGridWorld uses discrete actions. Call step(int action).");
    }

    @Override
    public StepResult step(int action) {
        if (action < 0 || action >= actionDimension()) {
            throw new IllegalArgumentException("Invalid action: " + action);
        }

        currentStep++;
        double reward = stepPenalty;
        boolean done = false;

        int[] nextStateRaw = Arrays.copyOf(currentState, currentState.length);

        switch (action) {
            case ACTION_UP:    nextStateRaw[1]--; break;
            case ACTION_DOWN:  nextStateRaw[1]++; break;
            case ACTION_LEFT:  nextStateRaw[0]--; break;
            case ACTION_RIGHT: nextStateRaw[0]++; break;
        }

        // Check bounds
        if (nextStateRaw[0] < 0 || nextStateRaw[0] >= width ||
            nextStateRaw[1] < 0 || nextStateRaw[1] >= height) {
            reward += wallPenalty;
            // Agent stays in the same place if it hits a wall
        } else {
            currentState = nextStateRaw;
        }

        // Check if goal reached
        if (Arrays.equals(currentState, goalState)) {
            reward += goalReward;
        }

        return new StepResult(Arrays.stream(currentState).asDoubleStream().toArray(), reward);
    }

    // Optional: A method to render or print the grid (useful for debugging)
    public String render() {
        StringBuilder sb = new StringBuilder();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (x == currentState[0] && y == currentState[1]) {
                    sb.append("A "); // Agent
                } else if (x == goalState[0] && y == goalState[1]) {
                    sb.append("G "); // Goal
                } else {
                    sb.append(". "); // Empty
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
