package net.nwrn.jcog.rl.env;

import java.util.Arrays;
import java.util.random.RandomGenerator;

public class SimpleGridWorld implements GameAdapter {

    private final int gridSize;
    private final int[] agentPos;
    private final int[] goalPos;
    private final int[] pitPos; // Optional pit
    private final double goalReward;
    private final double pitReward;
    private final double stepPenalty;
    private final int maxSteps;
    private int currentSteps;

    public SimpleGridWorld(int gridSize, int[] startPos, int[] goalPos, int[] pitPos,
                           double goalReward, double pitReward, double stepPenalty, int maxSteps) {
        this.gridSize = gridSize;
        this.agentPos = Arrays.copyOf(startPos, startPos.length);
        this.goalPos = Arrays.copyOf(goalPos, goalPos.length);
        this.pitPos = (pitPos != null) ? Arrays.copyOf(pitPos, pitPos.length) : null;
        this.goalReward = goalReward;
        this.pitReward = pitReward;
        this.stepPenalty = stepPenalty;
        this.maxSteps = maxSteps;
    }

    public SimpleGridWorld() {
        this(3, new int[]{0, 0}, new int[]{2, 2}, new int[]{1, 1}, 10.0, -10.0, -0.1, 20);
    }

    @Override
    public int observationDim() {
        return 2; // x, y coordinates
    }

    @Override
    public int actionDim() {
        return 4; // 0:Up, 1:Down, 2:Left, 3:Right
    }

    @Override
    public double[] reset(RandomGenerator rng) {
        agentPos[0] = 0; // Reset to start position, e.g., [0,0]
        agentPos[1] = 0;
        currentSteps = 0;
        return getObservation();
    }

    @Override
    public StepResult step(double[] action) {
        if (action.length != 1 && !isContinuousActions()) {
            throw new IllegalArgumentException("Discrete SimpleGridWorld expects a single action index.");
        }
        int actionIdx = (int) action[0];

        int prevX = agentPos[0];
        int prevY = agentPos[1];

        switch (actionIdx) {
            case 0: // Up
                agentPos[1] = Math.min(gridSize - 1, agentPos[1] + 1);
                break;
            case 1: // Down
                agentPos[1] = Math.max(0, agentPos[1] - 1);
                break;
            case 2: // Left
                agentPos[0] = Math.max(0, agentPos[0] - 1);
                break;
            case 3: // Right
                agentPos[0] = Math.min(gridSize - 1, agentPos[0] + 1);
                break;
            default:
                // Invalid action, stay in place or throw error
                break;
        }

        currentSteps++;
        boolean done = false;
        float reward = (float) stepPenalty;

        if (Arrays.equals(agentPos, goalPos)) {
            reward = (float) goalReward;
            done = true;
        } else if (pitPos != null && Arrays.equals(agentPos, pitPos)) {
            reward = (float) pitReward;
            done = true;
        }

        if (currentSteps >= maxSteps) {
            done = true;
        }

        return new StepResult(getObservation(), reward, done);
    }

    private double[] getObservation() {
        // Normalize coordinates to [0, 1] if desired, or use raw coords
        return new double[]{
            (double) agentPos[0] / (gridSize -1),
            (double) agentPos[1] / (gridSize -1)
        };
    }

    @Override
    public boolean isContinuousActions() {
        return false; // Discrete actions
    }
}
