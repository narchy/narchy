package jcog.tensor.rl.pg.util;

import java.util.List;

/**
 * A generic interface for experience replay buffers.
 * @deprecated Use {@link jcog.tensor.rl.pg3.memory.AgentMemory} instead.
 */
@Deprecated
public interface Memory {
    void add(Experience2 e);

    void clear();

    int size();

    List<Experience2> sample(int batchSize);

    List<Experience2> getAll();
}
