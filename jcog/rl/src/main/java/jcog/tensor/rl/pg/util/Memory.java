package jcog.tensor.rl.pg.util;

import java.util.List;

/**
 * A generic interface for experience replay buffers.
 */
public interface Memory {
    void add(Experience2 e);

    void clear();

    int size();

    List<Experience2> sample(int batchSize);

    List<Experience2> getAll();
}
