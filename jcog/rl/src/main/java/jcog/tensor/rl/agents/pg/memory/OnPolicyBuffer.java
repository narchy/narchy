package jcog.tensor.rl.agents.pg.memory;

import jcog.data.list.Lst; // Assuming Lst is accessible, as used in PGBuilder's version
import jcog.tensor.rl.pg.util.Experience2; // Still using existing Experience2

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class OnPolicyBuffer implements AgentMemory {

    public final int capacity; // Made public final for GUI access/inspection
    private final List<Experience2> buffer;

    public OnPolicyBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive.");
        }
        this.capacity = capacity;
        // PGBuilder's OnPolicyEpisodeBuffer used Lst. Assuming it's a custom list type.
        // If Lst is not available or appropriate, java.util.ArrayList could be used.
        this.buffer = new Lst<>(capacity);
    }

    @Override
    public void add(Experience2 experience) {
        Objects.requireNonNull(experience, "Experience cannot be null.");
        if (buffer.size() < this.capacity) {
            buffer.add(experience);
        }
        // If buffer is full, new experiences are not added.
        // The agent is expected to process and clear the buffer.
    }

    @Override
    public List<Experience2> sample(int batchSize) {
        // For this on-policy buffer, sample(batchSize) returns all current experiences,
        // ignoring batchSize, as is typical for on-policy learning phases.
        return getAll();
    }

    @Override
    public List<Experience2> getAll() {
        // Return a defensive copy to prevent external modification of the internal buffer.
        return new ArrayList<>(buffer);
    }

    @Override
    public void clear() {
        buffer.clear();
    }

    @Override
    public int size() {
        return buffer.size();
    }

    @Override
    public int capacity() {
        return this.capacity;
    }
}
