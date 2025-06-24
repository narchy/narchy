package jcog.tensor.rl.env;

import java.util.Map;

public interface SyntheticEnvironment {
    int stateDimension();
    int actionDimension(); // For continuous actions, this is the dimension; for discrete, it's the count.
    boolean isDiscreteActionSpace();

    double[] reset(); // Returns initial state

    StepResult step(double[] action); // For continuous actions
    StepResult step(int action);    // For discrete actions

    class StepResult {
        public final double[] nextState;
        public final double reward;
        public final boolean done;
        public final Map<String, Object> info;

        public StepResult(double[] nextState, double reward, boolean done, Map<String, Object> info) {
            this.nextState = nextState;
            this.reward = reward;
            this.done = done;
            this.info = info;
        }

        public StepResult(double[] nextState, double reward, boolean done) {
            this(nextState, reward, done, null);
        }
    }
}
