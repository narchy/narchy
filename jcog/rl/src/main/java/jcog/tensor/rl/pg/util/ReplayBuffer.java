package jcog.tensor.rl.pg.util;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Random;

public class ReplayBuffer {
    private final Deque<Experience> buffer;
    private final int capacity;

    public ReplayBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new ArrayDeque<>(capacity);
    }

    public void add(Experience e) {
        if (buffer.size() + 1 > capacity)
            buffer.removeFirst();
        buffer.addLast(e);
    }

    public Experience sample(Random rng) {
        return buffer.stream().skip(rng.nextInt(buffer.size())).findFirst().orElseThrow();
    }

    public int size() {
        return buffer.size();
    }

}

