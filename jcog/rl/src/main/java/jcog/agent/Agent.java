package jcog.agent;

import jcog.decide.Decide;
import org.jetbrains.annotations.Nullable;

/**
 * reinforcement learning agent interface
 */
public abstract class Agent {

    public final int inputs, actions;
    public final double[] actionPrev, actionNext, inputPrev;

    public transient float reward = Float.NaN;


    protected Agent(int inputs, int actions) {
        this.inputs = inputs;
        this.actions = actions;
        this.inputPrev = new double[inputs];
        this.actionPrev = new double[actions];
        this.actionNext = new double[actions];
        reset();
    }

    /**
     * @param inputPrev null if first iteration
     * @param actionPrev actions actually acted in previous cycle
     * @param reward     reward associated with the previous cycle's actions
     * @param input      next sensory observation
     */
    public abstract void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext);

    private void reset() {
        inputPrev[0] = Double.NaN; //HACK
    }

    /**
     * @param actionPrev (input)
     * @param reward
     * @param input
     * @param actionNext      (output)
     */
    public final void act(double[] actionPrev, float reward, double[] input, double[] actionNext) {
        if (reward == reward) {
            apply(start() ? inputPrev : null, actionPrev, reward, input, actionNext);
            actionFilter(actionNext);
        }

        this.reward = reward;
        System.arraycopy(input, 0, inputPrev, 0, input.length);
    }

    private boolean start() {  double i = inputPrev[0];  return i == i;  }

    /**
     * optional post-processing of action vector
     */
    protected void actionFilter(double[] actionNext) { }

    @Override
    public String toString() {
        return summary();
    }

    public String summary() {
        return getClass().getSimpleName() + "<in=" + inputs + ", act=" + actions + '>';
    }

    @Deprecated
    public final int actDiscrete(float reward, double[] input, Decide d) {
        throw new UnsupportedOperationException();
    }
}