package jcog.tensor.rl.pg2;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.PolicyGradientModel;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg.util.Memory;

import java.util.function.UnaryOperator;

/**
 * A generic interface for an RL algorithm's strategy, defining how it interacts with the environment and learns.
 * This interface is part of the `pg2` system.
 *
 * @deprecated This interface and the `pg2` strategy system are superseded by the `pg3` agent architecture.
 *             See {@link jcog.tensor.rl.pg3.PolicyGradientAgent} and its implementations (e.g., {@link jcog.tensor.rl.pg3.PPOAgent}).
 */
@Deprecated
public interface PGStrategy {
    @Deprecated // Initialization was tied to the old PolicyGradientModel
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
