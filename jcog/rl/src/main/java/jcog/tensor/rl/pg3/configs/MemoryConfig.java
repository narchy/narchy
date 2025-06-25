package jcog.tensor.rl.pg3.configs;

import jcog.signal.IntRange;

import java.util.Objects;

/**
 * Configuration for the memory system of an RL agent.
 * This can define settings for on-policy data collection (like episode or batch length)
 * and/or for off-policy replay buffers (like capacity and sampling batch size).
 *
 * @param episodeLength For on-policy algorithms, this defines the number of steps collected
 *                      before an update is performed (e.g., PPO's rollout length).
 *                      Can be {@code null} if the agent is purely off-policy and uses only a replay buffer.
 * @param replayBuffer Configuration for an off-policy replay buffer.
 *                     Can be {@code null} if the agent is purely on-policy.
 */
public record MemoryConfig(
    IntRange episodeLength,
    ReplayBufferConfig replayBuffer
) {
    /**
     * Configuration for an off-policy experience replay buffer.
     *
     * @param capacity The maximum number of experiences the buffer can store.
     * @param batchSize The number of experiences to sample from the buffer during each learning update.
     * @param updateEveryNSteps Frequency of learning updates (e.g., update networks every N environment steps).
     * @param gradientStepsPerUpdate Number of gradient descent steps to perform per learning update.
     */
    public record ReplayBufferConfig(
        IntRange capacity,
        IntRange batchSize,
        IntRange updateEveryNSteps,
        IntRange gradientStepsPerUpdate
    ) {
        /** Default replay buffer capacity: 100,000. Range [1000, 1,000,000]. */
        public static final IntRange DEFAULT_CAPACITY = new IntRange(100_000, 1_000, 1_000_000);
        /** Default replay buffer sample batch size: 256. Range [32, 2048]. */
        public static final IntRange DEFAULT_BATCH_SIZE = new IntRange(256, 32, 2048);
        /** Default update frequency: update every 1 environment step. Range [1, 1000]. */
        public static final IntRange DEFAULT_UPDATE_EVERY_N_STEPS = new IntRange(1, 1, 1000);
        /** Default gradient steps per update: 1. Range [1, 100]. */
        public static final IntRange DEFAULT_GRADIENT_STEPS_PER_UPDATE = new IntRange(1, 1, 100);


        /**
         * Validates replay buffer configuration.
         */
        public ReplayBufferConfig {
            Objects.requireNonNull(capacity, "Capacity cannot be null");
            Objects.requireNonNull(batchSize, "Batch size cannot be null");
            Objects.requireNonNull(updateEveryNSteps, "updateEveryNSteps cannot be null");
            Objects.requireNonNull(gradientStepsPerUpdate, "gradientStepsPerUpdate cannot be null");

            if (capacity.intValue() <= 0) throw new IllegalArgumentException("Buffer capacity must be positive");
            if (batchSize.intValue() <= 0) throw new IllegalArgumentException("Batch size must be positive");
            if (batchSize.intValue() > capacity.intValue()) {
                throw new IllegalArgumentException("Batch size cannot be larger than buffer capacity");
            }
            if (updateEveryNSteps.intValue() <= 0) throw new IllegalArgumentException("updateEveryNSteps must be positive.");
            if (gradientStepsPerUpdate.intValue() <= 0) throw new IllegalArgumentException("gradientStepsPerUpdate must be positive.");
        }

        /**
         * Creates a default {@code ReplayBufferConfig}.
         */
        public ReplayBufferConfig() {
            this(DEFAULT_CAPACITY, DEFAULT_BATCH_SIZE, DEFAULT_UPDATE_EVERY_N_STEPS, DEFAULT_GRADIENT_STEPS_PER_UPDATE);
        }

        /** Returns a new ReplayBufferConfig with the specified capacity range. */
        public ReplayBufferConfig withCapacity(IntRange capacity) {
            return new ReplayBufferConfig(capacity, this.batchSize, this.updateEveryNSteps, this.gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified capacity value, retaining current min/max from its range. */
        public ReplayBufferConfig withCapacity(int capacityValue) {
            return new ReplayBufferConfig(new IntRange(capacityValue, this.capacity.min, this.capacity.max), this.batchSize, this.updateEveryNSteps, this.gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified batch size range. */
        public ReplayBufferConfig withBatchSize(IntRange batchSize) {
            return new ReplayBufferConfig(this.capacity, batchSize, this.updateEveryNSteps, this.gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified batch size value, retaining current min/max from its range. */
        public ReplayBufferConfig withBatchSize(int batchSizeValue) {
            return new ReplayBufferConfig(this.capacity, new IntRange(batchSizeValue, this.batchSize.min, this.batchSize.max), this.updateEveryNSteps, this.gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified updateEveryNSteps range. */
        public ReplayBufferConfig withUpdateEveryNSteps(IntRange updateEveryNSteps) {
            return new ReplayBufferConfig(this.capacity, this.batchSize, updateEveryNSteps, this.gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified updateEveryNSteps value. */
        public ReplayBufferConfig withUpdateEveryNSteps(int value) {
            return new ReplayBufferConfig(this.capacity, this.batchSize, new IntRange(value, this.updateEveryNSteps.min, this.updateEveryNSteps.max), this.gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified gradientStepsPerUpdate range. */
        public ReplayBufferConfig withGradientStepsPerUpdate(IntRange gradientStepsPerUpdate) {
            return new ReplayBufferConfig(this.capacity, this.batchSize, this.updateEveryNSteps, gradientStepsPerUpdate);
        }
        /** Returns a new ReplayBufferConfig with the specified gradientStepsPerUpdate value. */
        public ReplayBufferConfig withGradientStepsPerUpdate(int value) {
            return new ReplayBufferConfig(this.capacity, this.batchSize, this.updateEveryNSteps, new IntRange(value, this.gradientStepsPerUpdate.min, this.gradientStepsPerUpdate.max));
        }
    }

    /** Default episode/batch length for on-policy data collection: 2048. Range [64, 8192]. */
    public static final IntRange DEFAULT_EPISODE_LENGTH = new IntRange(2048, 64, 8192);
    /** Default replay buffer configuration (used if an agent supports off-policy learning). */
    public static final ReplayBufferConfig DEFAULT_REPLAY_BUFFER_CONFIG = new ReplayBufferConfig();

    /**
     * Validates memory configuration.
     * Ensures that at least one memory mechanism (on-policy episode length or off-policy replay buffer) is configured.
     */
    public MemoryConfig {
        if (episodeLength == null && replayBuffer == null) {
            throw new IllegalArgumentException("Either episodeLength or replayBuffer must be configured.");
        }
        if (episodeLength != null && episodeLength.intValue() <= 0 && replayBuffer == null) {
            throw new IllegalArgumentException("Episode length must be positive if replay buffer is not used.");
        }
    }

    /**
     * Constructor for on-policy agents, specifying only the episode length.
     * @param episodeLength The number of steps to collect before an update.
     */
    public MemoryConfig(IntRange episodeLength) {
        this(episodeLength, null);
    }

    /**
     * Constructor for on-policy agents, specifying only the episode length as an integer.
     * Uses default min/max for the IntRange.
     * @param episodeLengthValue The number of steps to collect before an update.
     */
    public MemoryConfig(int episodeLengthValue) {
        this(new IntRange(episodeLengthValue, DEFAULT_EPISODE_LENGTH.min, DEFAULT_EPISODE_LENGTH.max), null);
    }

    /**
     * Constructor for off-policy agents, specifying only the replay buffer configuration.
     * @param replayBufferConfig Configuration for the replay buffer.
     */
    public MemoryConfig(ReplayBufferConfig replayBufferConfig) {
        this(null, replayBufferConfig);
    }

    /**
     * Default constructor. Initializes with a default episode length (for on-policy)
     * and a default replay buffer configuration (for potential off-policy use).
     * Specific agents should typically use a more tailored constructor.
     */
    public MemoryConfig() {
        this(DEFAULT_EPISODE_LENGTH, DEFAULT_REPLAY_BUFFER_CONFIG);
    }

    // Withers for MemoryConfig
    /** Returns a new MemoryConfig with the specified episode length range. */
    public MemoryConfig withEpisodeLength(IntRange episodeLength) { return new MemoryConfig(episodeLength, this.replayBuffer); }
    /** Returns a new MemoryConfig with the specified episode length value. */
    public MemoryConfig withEpisodeLength(int episodeLengthValue) {
        IntRange currentMinMax = this.episodeLength != null ? this.episodeLength : DEFAULT_EPISODE_LENGTH;
        return new MemoryConfig(new IntRange(episodeLengthValue, currentMinMax.min, currentMinMax.max), this.replayBuffer);
    }
    /** Returns a new MemoryConfig with the specified replay buffer configuration. */
    public MemoryConfig withReplayBufferConfig(ReplayBufferConfig replayBuffer) { return new MemoryConfig(this.episodeLength, replayBuffer); }
    /** Returns a new MemoryConfig with updated replay buffer capacity. */
    public MemoryConfig withReplayBufferCapacity(int capacityValue) {
        ReplayBufferConfig currentRB = this.replayBuffer == null ? new ReplayBufferConfig() : this.replayBuffer;
        return new MemoryConfig(this.episodeLength, currentRB.withCapacity(capacityValue));
    }
    /** Returns a new MemoryConfig with updated replay buffer batch size. */
    public MemoryConfig withReplayBufferBatchSize(int batchSizeValue) {
        ReplayBufferConfig currentRB = this.replayBuffer == null ? new ReplayBufferConfig() : this.replayBuffer;
        return new MemoryConfig(this.episodeLength, currentRB.withBatchSize(batchSizeValue));
    }
}
