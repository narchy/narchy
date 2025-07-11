package jcog.tensor.rl.env;

import java.util.Map;
import java.util.random.RandomGenerator;

public interface SyntheticEnv {
    int stateDimension();
    int actionDimension(); // For continuous actions, this is the dimension; for discrete, it's the count.
    boolean isDiscreteActionSpace();

    double[] reset(RandomGenerator rng); // Returns initial state

    StepResult step(double[] action); // For continuous actions
    StepResult step(int action);    // For discrete actions

    record StepResult(double[] nextState, double reward, Map<String, Object> info) {

        public StepResult(double[] nextState, double reward) {
                this(nextState, reward, null);
            }
        }
}
