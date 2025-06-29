package jcog.tensor.rl.pg2.configs;

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
     */
    public record ReplayBufferConfig(
        IntRange capacity,
        IntRange batchSize
    ) {
        /** Default replay buffer capacity: 100,000. Range [1000, 1,000,000]. */
        public static final IntRange DEFAULT_CAPACITY = new IntRange(100_000, 1_000, 1_000_000);
        /** Default replay buffer sample batch size: 256. Range [32, 2048]. */
        public static final IntRange DEFAULT_BATCH_SIZE = new IntRange(256, 32, 2048);

        /**
         * Validates replay buffer configuration.
         */
        public ReplayBufferConfig {
            Objects.requireNonNull(capacity, "Capacity cannot be null");
            Objects.requireNonNull(batchSize, "Batch size cannot be null");
            if (capacity.intValue() <= 0) throw new IllegalArgumentException("Buffer capacity must be positive");
            if (batchSize.intValue() <= 0) throw new IllegalArgumentException("Batch size must be positive");
            if (batchSize.intValue() > capacity.intValue()) {
                throw new IllegalArgumentException("Batch size cannot be larger than buffer capacity");
            }
        }

        /**
         * Creates a default {@code ReplayBufferConfig}.
         */
        public ReplayBufferConfig() {
            this(DEFAULT_CAPACITY, DEFAULT_BATCH_SIZE);
        }

        /** Returns a new ReplayBufferConfig with the specified capacity range. */
        public ReplayBufferConfig withCapacity(IntRange capacity) { return new ReplayBufferConfig(capacity, this.batchSize); }
        /** Returns a new ReplayBufferConfig with the specified capacity value, retaining current min/max from its range. */
        public ReplayBufferConfig withCapacity(int capacityValue) {
            return new ReplayBufferConfig(new IntRange(capacityValue, this.capacity.min, this.capacity.max), this.batchSize);
        }
        /** Returns a new ReplayBufferConfig with the specified batch size range. */
        public ReplayBufferConfig withBatchSize(IntRange batchSize) { return new ReplayBufferConfig(this.capacity, batchSize); }
        /** Returns a new ReplayBufferConfig with the specified batch size value, retaining current min/max from its range. */
        public ReplayBufferConfig withBatchSize(int batchSizeValue) {
            return new ReplayBufferConfig(this.capacity, new IntRange(batchSizeValue, this.batchSize.min, this.batchSize.max));
        }
    }

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
        this(new IntRange(episodeLengthValue, 1, 256), null);
    }

    /**
     * Constructor for off-policy agents, specifying only the replay buffer configuration.
     * @param replayBufferConfig Configuration for the replay buffer.
     */
    public MemoryConfig(ReplayBufferConfig replayBufferConfig) {
        this(null, replayBufferConfig);
    }

    // Withers for MemoryConfig
    /** Returns a new MemoryConfig with the specified episode length range. */
    public MemoryConfig withEpisodeLength(IntRange episodeLength) { return new MemoryConfig(episodeLength, this.replayBuffer); }
    /** Returns a new MemoryConfig with the specified episode length value. */
    public MemoryConfig withEpisodeLength(int episodeLengthValue) {
        IntRange currentMinMax = this.episodeLength;
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
