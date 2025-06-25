package jcog.tensor.rl.pg3.memory;

import jcog.list.FasterShuffle;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * A simple replay buffer for off-policy reinforcement learning agents.
 * It stores experiences and allows random sampling of batches.
 * This version is adapted for use with {@link Experience2} objects.
 */
public class ReplayBuffer implements AgentMemory {

    private final List<Experience2> buffer;
    private final int capacity;
    private int position;
    private final Random random;

    /**
     * Constructs a ReplayBuffer.
     *
     * @param capacity The maximum number of experiences the buffer can hold.
     */
    public ReplayBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive.");
        }
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
        this.position = 0;
        this.random = new Random(); // Consider making Random injectable for reproducibility
    }

    /**
     * Adds an experience to the buffer. If the buffer is full, the oldest experience is overwritten.
     *
     * @param experience The experience to add.
     */
    @Override
    public void add(Experience2 experience) {
        if (this.buffer.size() < this.capacity) {
            this.buffer.add(experience);
        } else {
            this.buffer.set(this.position, experience);
        }
        this.position = (this.position + 1) % this.capacity;
    }

    /**
     * Samples a batch of experiences randomly from the buffer.
     *
     * @param batchSize The number of experiences to sample.
     * @return A list of sampled experiences. Returns an empty list if the buffer contains fewer
     *         experiences than the batch size.
     */
    @Override
    public List<Experience2> sample(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive.");
        }
        if (this.buffer.size() < batchSize) {
            // Not enough samples in buffer yet, return empty or partial batch based on desired behavior.
            // Returning empty list if not enough, common approach.
            return Collections.emptyList();
        }

        List<Experience2> batch = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            batch.add(this.buffer.get(this.random.nextInt(this.buffer.size())));
        }
        return batch;
    }

    /**
     * Samples a batch of experiences randomly from the buffer using a faster shuffle.
     * This is an alternative to repeated random index selection.
     *
     * @param batchSize The number of experiences to sample.
     * @return A list of sampled experiences.
     */
    public List<Experience2> sampleFaster(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive.");
        }
        int currentSize = this.buffer.size();
        if (currentSize < batchSize) {
            return Collections.emptyList(); // Or handle as desired, e.g., return all available
        }

        // Efficiently sample 'batchSize' distinct elements
        // Create a temporary list of indices or shuffle a sublist if buffer is very large
        // For moderately sized buffers, shuffling a copy of references is okay.
        List<Experience2> shuffledBuffer = new ArrayList<>(this.buffer);
        FasterShuffle.shuffle(shuffledBuffer, this.random); // Or Collections.shuffle

        return new ArrayList<>(shuffledBuffer.subList(0, batchSize));
    }


    /**
     * Returns all experiences currently in the buffer.
     * This is typically used by on-policy algorithms or for specific debugging/analysis,
     * less so for standard off-policy replay buffer sampling.
     *
     * @return A list containing all experiences in the buffer.
     */
    @Override
    public List<Experience2> getAll() {
        return new ArrayList<>(this.buffer); // Return a copy
    }

    /**
     * Returns the current number of experiences in the buffer.
     *
     * @return The size of the buffer.
     */
    @Override
    public int size() {
        return this.buffer.size();
    }

    /**
     * Returns the capacity of the buffer.
     *
     * @return The maximum capacity.
     */
    public int capacity() {
        return this.capacity;
    }

    /**
     * Clears all experiences from the buffer.
     */
    @Override
    public void clear() {
        this.buffer.clear();
        this.position = 0;
    }

    /**
     * Converts a list of Experience2 objects into separate lists of Tensors for states, actions, etc.
     * This is a utility method that can be helpful for preparing batches for network processing.
     *
     * @param batch The list of Experience2 objects.
     * @return A BatchTuple containing Tensors for states, actions, rewards, nextStates, dones, and oldLogProbs.
     */
    public static BatchTuple experiencesToBatchTuple(List<Experience2> batch) {
        if (batch == null || batch.isEmpty()) {
            return new BatchTuple(null, null, null, null, null, null);
        }

        List<Tensor> states = new ArrayList<>(batch.size());
        List<Tensor> actions = new ArrayList<>(batch.size());
        List<Float> rewards = new ArrayList<>(batch.size());
        List<Tensor> nextStates = new ArrayList<>(batch.size());
        List<Boolean> dones = new ArrayList<>(batch.size());
        List<Tensor> oldLogProbs = new ArrayList<>(batch.size()); // May be null for some experiences

        for (Experience2 exp : batch) {
            states.add(exp.state());
            actions.add(Tensor.row(exp.action())); // Assuming action is double[]
            rewards.add(exp.reward());
            nextStates.add(exp.nextState());
            dones.add(exp.done());
            if (exp.oldLogProb() != null) {
                oldLogProbs.add(exp.oldLogProb());
            }
        }

        Tensor statesTensor = states.isEmpty() ? null : Tensor.concatRows(states);
        Tensor actionsTensor = actions.isEmpty() ? null : Tensor.concatRows(actions);
        Tensor rewardsTensor = rewards.isEmpty() ? null : Tensor.vector(rewards.stream().mapToDouble(f -> f).toArray()).transpose();
        Tensor nextStatesTensor = nextStates.stream().anyMatch(java.util.Objects::isNull) ? null : Tensor.concatRows(nextStates.stream().filter(java.util.Objects::nonNull).toList());

        // Handle cases where not all experiences have nextState (e.g. terminal states might have null nextState in Experience2)
        // This requires careful handling by the caller if nextStatesTensor can have fewer rows.
        // A more robust way is to ensure nextState is always present, perhaps a zero tensor for terminal.
        // For now, we filter nulls, which might lead to mismatched batch sizes if not handled.
        // A common practice: if exp.done(), nextState might be irrelevant or a conventional zero tensor.
        // Let's assume for now that if nextState is null, it's a terminal state.
        // The `donesTensor` will indicate this.

        Tensor donesTensor = dones.isEmpty() ? null : Tensor.vector(dones.stream().mapToDouble(b -> b ? 1.0 : 0.0).toArray()).transpose();
        Tensor oldLogProbsTensor = oldLogProbs.isEmpty() ? null : Tensor.concatRows(oldLogProbs);

        // Adjust nextStatesTensor to ensure it has the same number of rows, padding with zeros for terminal states if necessary.
        // This simplified version just concatenates non-null nextStates. The user of BatchTuple must be aware.
        // A more robust solution would involve ensuring all lists in BatchTuple have the same row count.

        return new BatchTuple(statesTensor, actionsTensor, rewardsTensor, nextStatesTensor, donesTensor, oldLogProbsTensor);
    }

    /**
     * A tuple to hold batched experiences as Tensors.
     */
    public record BatchTuple(
        Tensor states,
        Tensor actions,
        Tensor rewards,
        Tensor nextStates,
        Tensor dones,
        Tensor oldLogProbs // Relevant for PPO, might be null for DDPG experiences
    ) {}
}
