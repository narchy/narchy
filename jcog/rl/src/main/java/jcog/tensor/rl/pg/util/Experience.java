package jcog.tensor.rl.pg.util;

public class Experience {
    public final double[] state;
    public final double[] action;
    public final double reward;
    public final double[] nextState;
    public final boolean done;

    public Experience(double[] state, double[] action, double reward, double[] nextState, boolean done) {
        this.state = state;
        this.action = action;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}
