package jcog.tensor.rl.pg3.memory;

//import jcog.tensor.rl.pg.util.Experience2; // Old import
import jcog.tensor.Tensor; // Added for ExperienceBatch
import jcog.tensor.rl.pg3.memory.Experience2; // New import

import java.util.Collection; // Added for toBatch
import java.util.List;
import java.util.stream.Collectors; // Added for toBatch

/**
 * Interface for memory buffers used by RL agents.
 */
public interface AgentMemory {

    /**
     * Represents a batch of experiences, converted to Tensors for efficient processing by networks.
     */
    record ExperienceBatch(
        Tensor states,
        Tensor actions,
        Tensor rewards,
        Tensor nextStates,
        Tensor dones, // 1.0 for done, 0.0 for not done
        @Nullable Tensor oldLogProbs // Optional, may be null if not available/needed
    ) {
        // Constructor for batches without oldLogProbs
        public ExperienceBatch(Tensor states, Tensor actions, Tensor rewards, Tensor nextStates, Tensor dones) {
            this(states, actions, rewards, nextStates, dones, null);
        }
    }

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

    /**
     * Converts a collection of {@link Experience2} objects into an {@link ExperienceBatch} of Tensors.
     * This default implementation handles common conversions.
     *
     * @param experiences The collection of experiences to convert.
     * @return An {@link ExperienceBatch} containing Tensors for states, actions, rewards, next states, and done flags.
     *         Returns null if the input collection is null or empty.
     */
    default ExperienceBatch toBatch(Collection<Experience2> experiences) {
        if (experiences == null || experiences.isEmpty()) {
            return null;
        }

        List<Tensor> states = experiences.stream().map(Experience2::state).collect(Collectors.toList());
        List<Tensor> actions = experiences.stream().map(e -> Tensor.row(e.action())).collect(Collectors.toList());
        double[] rewardsArray = experiences.stream().mapToDouble(Experience2::reward).toArray();
        List<Tensor> nextStates = experiences.stream().map(Experience2::nextState).collect(Collectors.toList());
        // Ensure nextStates are not null; if an episode terminated, nextState might be a dummy or special tensor.
        // For now, assuming valid tensors. If nulls are possible, they need careful handling (e.g. replacing with zeros).
        double[] donesArray = experiences.stream().mapToDouble(e -> e.done() ? 1.0 : 0.0).toArray();
        List<Tensor> oldLogProbsList = experiences.stream()
                                                  .map(Experience2::oldLogProb)
                                                  .filter(java.util.Objects::nonNull) // Filter out null log probs
                                                  .collect(Collectors.toList());

        Tensor oldLogProbsTensor = null;
        if (!oldLogProbsList.isEmpty() && oldLogProbsList.size() == experiences.size()) {
            // Only create tensor if all experiences had a logProb
            oldLogProbsTensor = Tensor.concatRows(oldLogProbsList);
        } else if (!oldLogProbsList.isEmpty() && oldLogProbsList.size() != experiences.size()) {
            // Log a warning if some experiences have logProbs and others don't, as this is usually an issue.
            System.err.println("Warning: Mixing experiences with and without oldLogProb in a single batch. oldLogProbs will be null.");
        }


        return new ExperienceBatch(
            Tensor.concatRows(states),
            Tensor.concatRows(actions),
            Tensor.row(rewardsArray).transpose(),
            Tensor.concatRows(nextStates),
            Tensor.row(donesArray).transpose(),
            oldLogProbsTensor
        );
    }
}
