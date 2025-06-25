package jcog.tensor.rl.pg3.memory;

import jcog.tensor.rl.pg.util.Experience2;

import java.util.List;

/**
 * Interface for memory buffers used by RL agents.
 */
public interface AgentMemory {

    /**
     * Adds a new experience to the memory.
     * @param experience The experience to add.
     */
    void add(Experience2 experience);

    /**
     * Samples a batch of experiences from the memory.
     * The definition of a "batch" can vary:
     * - For on-policy, it might return all experiences since the last clear.
     * - For off-policy, it might return a random minibatch.
     * @param batchSize The desired size of the batch (may be ignored by some implementations, e.g., on-policy).
     * @return A list of experiences.
     */
    List<Experience2> sample(int batchSize);

    /**
     * Returns all experiences currently in the buffer.
     * Useful for on-policy algorithms that process the entire episode/batch.
     * @return A list of all experiences.
     */
    List<Experience2> getAll();

    /**
     * Clears all experiences from the memory.
     */
    void clear();

    /**
     * Returns the current number of experiences stored in the memory.
     * @return The size of the memory.
     */
    int size();

    /**
     * Returns the capacity of the memory buffer.
     * @return The maximum number of experiences the buffer can hold.
     */
    int capacity();
}
