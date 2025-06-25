package jcog.tensor.rl.pg3.memory;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.list.FasterList;
import jcog.tensor.Tensor;
//import jcog.tensor.rl.pg.util.Experience2; // Old import
import jcog.tensor.rl.pg3.memory.Experience2; // New import
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;


/**
 * A Prioritized Experience Replay (PER) buffer.
 * Experiences are sampled with probabilities proportional to their TD error (priority).
 * This helps the agent focus on "surprising" or more informative experiences.
 *
 * <p>Reference: "Prioritized Experience Replay" by Schaul et al. (2015).</p>
 *
 * <p>This implementation currently uses a simplified proportional sampling method
 * by iterating through priorities. For optimal performance with very large buffers (e.g., >100k samples),
 * a more sophisticated data structure like a SumTree or Segment Tree would be required for O(log N) sampling
 * and priority updates. The current approach is O(N) for sampling and O(1) for single priority update after finding index.
 * </p>
 */
public class PrioritizedReplayBuffer implements AgentMemory {

    private final int capacity;
    private final List<Experience2> buffer; // Using List for easier index management here
    private final double[] priorities;    // Stores the priority of each experience

    private int position = 0; // Current position to insert next experience
    private volatile double maxPriority = 1.0; // Max priority seen so far, for new experiences

    private final double alpha; // Controls how much prioritization is used (0: uniform, 1: full priority)
    private final double beta;  // Importance-sampling exponent, anneals from beta0 to 1.0
    private final double betaIncrementPerSampling;
    private double currentBeta;

    private static final double PRIORITY_EPSILON = 1e-5; // Small constant to ensure non-zero priority

    private final Random random;

    public PrioritizedReplayBuffer(int capacity, double alpha, double beta0, double betaAnnealingSteps) {
        if (capacity <= 0) throw new IllegalArgumentException("Capacity must be positive.");
        if (alpha < 0) throw new IllegalArgumentException("Alpha must be non-negative.");
        if (beta0 < 0) throw new IllegalArgumentException("Initial beta (beta0) must be non-negative.");

        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity); // Initialize with nulls or manage size dynamically
        for (int i = 0; i < capacity; i++) {
            this.buffer.add(null); // Pre-fill to allow direct index setting
        }
        this.priorities = new double[capacity]; // Initialized to 0

        this.alpha = alpha;
        this.beta = beta0; // Initial beta
        this.currentBeta = beta0;
        if (betaAnnealingSteps > 0) {
            this.betaIncrementPerSampling = (1.0 - beta0) / betaAnnealingSteps;
        } else {
            this.betaIncrementPerSampling = 0; // Beta does not anneal if steps is 0
        }
        this.random = ThreadLocalRandom.current();
    }

    @Override
    public synchronized void add(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null");

        double priorityToAssign = maxPriority; // New experiences get max priority to ensure they are seen

        buffer.set(position, experience);
        priorities[position] = Math.pow(priorityToAssign, alpha); // Store p^alpha

        position = (position + 1) % capacity;
    }

    @Override
    public synchronized List<Experience2> sample(int batchSize) {
        return sampleWithDetails(batchSize).experiences();
    }

    /**
     * Samples a mini-batch of experiences and returns details including indices and IS weights.
     *
     * @param batchSize The number of experiences to sample.
     * @return A {@link PrioritizedSampleBatch} containing experiences, indices, and IS weights.
     */
    public synchronized PrioritizedSampleBatch sampleWithDetails(int batchSize) {
        if (batchSize <= 0) {
            return new PrioritizedSampleBatch(Collections.emptyList(), new IntArrayList(0), Tensor.EMPTY);
        }

        int currentSize = size();
        if (currentSize == 0) {
            return new PrioritizedSampleBatch(Collections.emptyList(), new IntArrayList(0), Tensor.EMPTY);
        }

        batchSize = Math.min(batchSize, currentSize);
        List<Experience2> sampledExperiences = new FasterList<>(batchSize);
        IntArrayList sampledIndices = new IntArrayList(batchSize);
        double[] importanceSamplingWeights = new double[batchSize];

        // Calculate sum of priorities (p_i^alpha) for sampling distribution
        // Only consider filled slots up to `currentSize` if buffer is not full, or all `capacity` if full.
        double totalPriority = 0;
        for (int i = 0; i < currentSize; i++) {
            totalPriority += priorities[i]; // priorities already store p^alpha
        }

        if (totalPriority <= 0) { // Should not happen if PRIORITY_EPSILON and maxPriority > 0
             // Fallback to uniform sampling if all priorities are zero for some reason
            System.err.println("Warning: Total priority is zero in PER. Falling back to uniform sampling for this batch.");
            List<Experience2> allExperiences = getAll(); // gets filled experiences
            Collections.shuffle(allExperiences, random);
            for (int i = 0; i < batchSize; i++) {
                sampledExperiences.add(allExperiences.get(i));
                // Cannot determine original index easily here for uniform fallback,
                // so IS weights and index updates are problematic.
                // This path should ideally be avoided.
                // For simplicity, we'll just return uniform samples without proper IS weights or indices.
            }
            // This fallback needs proper index tracking if it's to be robust.
            // For now, let's assume totalPriority > 0.
             if (currentSize > 0) { // If fallback is hit, and currentSize is 0, this is an issue.
                // A simple fix for indices in fallback, though not perfectly robust for IS weights:
                for(int i=0; i < batchSize; i++) sampledIndices.add(i % currentSize);
                Arrays.fill(importanceSamplingWeights, 1.0); // Uniform weights
                return new PrioritizedSampleBatch(sampledExperiences, sampledIndices, Tensor.doubles(importanceSamplingWeights));
            } else {
                return new PrioritizedSampleBatch(Collections.emptyList(), new IntArrayList(0), Tensor.EMPTY);
            }
        }


        // Proportional sampling
        for (int k = 0; k < batchSize; k++) {
            double sampleValue = random.nextDouble() * totalPriority;
            double currentSum = 0;
            int sampledIdx = -1;
            for (int i = 0; i < currentSize; i++) {
                currentSum += priorities[i];
                if (currentSum > sampleValue) {
                    sampledIdx = i;
                    break;
                }
            }
            // Handle potential floating point issues where sampleValue might be exactly totalPriority
            if (sampledIdx == -1) {
                sampledIdx = currentSize - 1;
            }

            sampledExperiences.add(buffer.get(sampledIdx));
            sampledIndices.add(sampledIdx);

            // Calculate importance sampling weight: (N * P(i))^-beta / max_w
            // P(i) = priority_i / totalPriority
            double probability = priorities[sampledIdx] / totalPriority;
            importanceSamplingWeights[k] = Math.pow(currentSize * probability, -currentBeta);
        }

        // Normalize IS weights by dividing by max_w (max IS weight in the batch)
        double maxISWeight = 0;
        for (double weight : importanceSamplingWeights) {
            if (weight > maxISWeight) {
                maxISWeight = weight;
            }
        }
        if (maxISWeight > 0) { // Avoid division by zero if all weights are zero
            for (int i = 0; i < batchSize; i++) {
                importanceSamplingWeights[i] /= maxISWeight;
            }
        }

        // Anneal beta
        currentBeta = Math.min(1.0, currentBeta + betaIncrementPerSampling);

        return new PrioritizedSampleBatch(sampledExperiences, sampledIndices, Tensor.doubles(importanceSamplingWeights));
    }

    /**
     * Updates the priorities of the experiences at the given indices.
     *
     * @param indices        The indices of the experiences to update.
     * @param newTDErrorAbs  A tensor or array of new absolute TD errors for these experiences.
     *                       These errors are used to calculate new priorities.
     */
    public synchronized void updatePriorities(MutableIntList indices, Tensor newTDErrorAbs) {
        if (indices.size() != newTDErrorAbs.size()) {
            throw new IllegalArgumentException("Number of indices must match number of TD errors. Got " +
                                               indices.size() + " indices and " + newTDErrorAbs.size() + " errors.");
        }

        for (int i = 0; i < indices.size(); i++) {
            int index = indices.get(i);
            if (index < 0 || index >= capacity) {
                System.err.println("Warning: Invalid index " + index + " in updatePriorities. Skipping.");
                continue;
            }
            // Ensure experience at index is not null before updating, though sampling should only return valid indices
            if (buffer.get(index) == null) {
                 System.err.println("Warning: Attempting to update priority for a null experience at index " + index + ". Skipping.");
                continue;
            }

            double error = newTDErrorAbs.data(i); // Assuming newTDErrorAbs is a flat list/tensor of errors
            double newPriority = Math.abs(error) + PRIORITY_EPSILON; // Add epsilon to avoid zero priority

            priorities[index] = Math.pow(newPriority, alpha);

            if (newPriority > maxPriority) {
                maxPriority = newPriority; // Update overall max priority
            }
        }
    }


    @Override
    public synchronized List<Experience2> getAll() {
        // Return only filled experiences
        int currentSize = size();
        if (position < currentSize && currentSize < capacity) { // Buffer not full, position is the count
             return Collections.unmodifiableList(new ArrayList<>(buffer.subList(0, position)));
        } else { // Buffer is full or position wrapped around
            return Collections.unmodifiableList(new ArrayList<>(buffer));
        }
    }

    @Override
    public synchronized void clear() {
        for (int i = 0; i < capacity; i++) {
            buffer.set(i, null);
            priorities[i] = 0;
        }
        position = 0;
        maxPriority = 1.0; // Reset max priority
        currentBeta = beta; // Reset beta to initial
    }

    @Override
    public synchronized int size() {
        // If buffer has not been filled to capacity yet
        if (buffer.get(capacity -1) == null && position < capacity) {
            boolean trulyNotFull = false;
            for(int i = position; i < capacity; i++) {
                if (buffer.get(i) == null) {
                    trulyNotFull = true;
                    break;
                }
            }
            if (trulyNotFull) return position;
        }
        return capacity; // Otherwise, it's full
    }

    @Override
    public int capacity() {
        return capacity;
    }

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
        List<Tensor> oldLogProbs = batch.stream()
                                        .map(Experience2::oldLogProb)
                                        .map(logProb -> logProb != null ? logProb : Tensor.EMPTY)
                                        .collect(Collectors.toList());

        List<Tensor> validOldLogProbs = oldLogProbs.stream().filter(t -> t != Tensor.EMPTY && t.size > 0).collect(Collectors.toList());
        Tensor oldLogProbsTensor = validOldLogProbs.isEmpty() ? Tensor.EMPTY : Tensor.concatRows(validOldLogProbs);

        return new ExperienceBatch(
                Tensor.concatRows(states),
                Tensor.concatRows(actions),
                Tensor.doubles(rewards.stream().mapToDouble(Double::doubleValue).toArray()).transpose(),
                Tensor.concatRows(nextStates),
                Tensor.booleans(dones.stream().map(b -> (byte)(b ? 1 : 0)).collect(Collectors.toList())).transpose().asFloats(),
                oldLogProbsTensor
        );
    }

    /**
     * Holds a batch of experiences sampled from PER, along with their original indices
     * in the buffer and their importance sampling (IS) weights.
     */
    public record PrioritizedSampleBatch(
        List<Experience2> experiences,
        MutableIntList indices, // Indices in the main buffer
        Tensor importanceSamplingWeights // IS weights for these experiences
    ) {}

}
