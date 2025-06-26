package jcog.tensor.rl.pg2.memory;

import jcog.data.list.Lst;
import jcog.tensor.rl.pg.util.Experience2;

import java.util.ArrayList;
import java.util.List;

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
        while (buffer.size() + 1 >= this.capacity)
            buffer.removeFirst();

        buffer.add(experience);
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
