package jcog.tensor.rl.pg2;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.PolicyGradientModel;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg.util.Memory;

import java.util.function.UnaryOperator;

/**
 * A generic interface for an RL algorithm's strategy, defining how it interacts with the environment and learns.
 */
public interface PGStrategy {
    void initialize(PolicyGradientModel model);

    void record(Experience2 e);

    void update(long totalSteps);

    double[] selectAction(Tensor state, boolean deterministic);

    boolean isTrainingMode();

    Memory getMemory();

    long getUpdateSteps();

    UnaryOperator<Tensor> getPolicy();

    PGBuilder.MemoryConfig getMemoryConfig();

    default boolean isOffPolicy() {
        return false;
    }
}
