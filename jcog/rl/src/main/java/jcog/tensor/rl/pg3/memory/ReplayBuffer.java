package jcog.tensor.rl.pg3.memory;

import jcog.tensor.Tensor;
//import jcog.tensor.rl.pg.util.Experience2; // Old import
import jcog.tensor.rl.pg3.memory.Experience2; // New import
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

/**
 * A standard Experience Replay Buffer for off-policy reinforcement learning agents.
 * It stores experiences ({@link Experience2} records: state, action, reward, nextState, done, oldLogProb)
 * and allows for random sampling of mini-batches.
 * This implementation acts as a circular buffer: once full, adding a new experience causes the oldest one to be removed.
 *
 * <p>This implementation is based on concepts from `jcog.tensor.rl.pg.util.ReplayBuffer2.java`.
 */
public class ReplayBuffer implements AgentMemory {

    private final int capacity;
    private final Deque<Experience2> buffer;
    private final Random random;

    public ReplayBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Replay buffer capacity must be positive.");
        }
        this.capacity = capacity;
        this.buffer = new ArrayDeque<>(capacity);
        this.random = ThreadLocalRandom.current();
    }

    /**
     * Adds an experience to the buffer. If the buffer is full, the oldest experience is removed.
     *
     * @param experience The experience to add.
     */
    @Override
    public synchronized void add(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null");
        if (buffer.size() == capacity) {
            buffer.removeFirst(); // Remove the oldest experience
        }
        buffer.addLast(experience); // Add the new experience
    }

    /**
     * Samples a mini-batch of experiences randomly from the buffer.
     *
     * @param batchSize The number of experiences to sample.
     * @return A list of sampled experiences. Returns an empty list if the buffer is empty or
     *         batchSize is non-positive. Returns all experiences if batchSize is larger than
     *         current buffer size.
     */
    @Override
    public synchronized List<Experience2> sample(int batchSize) {
        if (batchSize <= 0 || buffer.isEmpty()) {
            return Collections.emptyList();
        }

        int sampleSize = Math.min(batchSize, buffer.size());
        List<Experience2> sampledBatch = new ArrayList<>(sampleSize);

        // Convert Deque to List for efficient random access if needed,
        // or iterate and randomly select. For moderate sizes, converting might be simpler.
        // For very large buffers and frequent sampling, more optimized sampling might be needed.
        List<Experience2> bufferSnapshot = new ArrayList<>(buffer);
        Collections.shuffle(bufferSnapshot, random); // Shuffle the snapshot

        for (int i = 0; i < sampleSize; i++) {
            sampledBatch.add(bufferSnapshot.get(i));
        }

        return sampledBatch;
    }

    /**
     * Returns all experiences currently in the buffer.
     * This is typically used by on-policy algorithms or for specific inspection needs.
     * For off-policy replay, `sample(batchSize)` is preferred.
     *
     * @return A list containing all experiences in the buffer.
     */
    @Override
    public synchronized List<Experience2> getAll() {
        return new ArrayList<>(buffer); // Return a copy
    }

    /**
     * Clears all experiences from the buffer.
     */
    @Override
    public synchronized void clear() {
        buffer.clear();
    }

    /**
     * Gets the current number of experiences in the buffer.
     *
     * @return The size of the buffer.
     */
    @Override
    public synchronized int size() {
        return buffer.size();
    }

    /**
     * Gets the maximum capacity of the buffer.
     *
     * @return The capacity of the buffer.
     */
    @Override
    public int capacity() {
        return capacity;
    }

    /**
     * {@inheritDoc}
     *
     * This method converts the stored {@link Experience2} objects into a more structured
     * {@link ExperienceBatch} which separates states, actions, rewards, etc., into their own Tensors.
     * This is often more convenient for direct input into neural networks.
     *
     * @param batch The list of {@link Experience2} objects to convert.
     * @return An {@link ExperienceBatch} containing the data in batched Tensor format.
     */
    @Override
    public ExperienceBatch toBatch(@NotNull List<Experience2> batch) {
        if (batch.isEmpty()) {
            return new ExperienceBatch(Tensor.EMPTY, Tensor.EMPTY, Tensor.EMPTY, Tensor.EMPTY, Tensor.EMPTY, Tensor.EMPTY);
        }

        List<Tensor> states = batch.stream().map(Experience2::state).collect(Collectors.toList());
        List<Tensor> actions = batch.stream().map(e -> Tensor.row(e.action())).collect(Collectors.toList());
        List<Double> rewards = batch.stream().map(Experience2::reward).collect(Collectors.toList());
        List<Tensor> nextStates = batch.stream().map(Experience2::nextState).collect(Collectors.toList());
        List<Boolean> dones = batch.stream().map(Experience2::done).collect(Collectors.toList());
        // oldLogProbs might be null for algorithms like DDPG, handle this gracefully.
        List<Tensor> oldLogProbs = batch.stream()
                                        .map(Experience2::oldLogProb)
                                        .map(logProb -> logProb != null ? logProb : Tensor.EMPTY) // Use a placeholder if null
                                        .collect(Collectors.toList());

        // Filter out empty tensors for oldLogProbs before concatenation if some are truly missing
        List<Tensor> validOldLogProbs = oldLogProbs.stream().filter(t -> t != Tensor.EMPTY && t.size > 0).collect(Collectors.toList());
        Tensor oldLogProbsTensor = validOldLogProbs.isEmpty() ? Tensor.EMPTY : Tensor.concatRows(validOldLogProbs);


        return new ExperienceBatch(
                Tensor.concatRows(states),
                Tensor.concatRows(actions),
                Tensor.doubles(rewards.stream().mapToDouble(Double::doubleValue).toArray()).transpose(),
                Tensor.concatRows(nextStates),
                Tensor.booleans(dones.stream().map(b -> (byte)(b ? 1 : 0)).collect(Collectors.toList())).transpose().asFloats(), // Convert boolean to float tensor (0.0 or 1.0)
                oldLogProbsTensor
        );
    }
}
