package jcog.tensor.rl.pg.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class ReplayBuffer2 implements Memory {
    private final Experience2[] buffer;
    private final int capacity;
    private int index, size;

    public ReplayBuffer2(int capacity) {
        this.capacity = capacity;
        this.buffer = new Experience2[capacity];
    }

    @Override
    public void add(Experience2 e) {
        buffer[index] = e;
        index = (index + 1) % capacity;
        if (size < capacity) size++;
    }

    @Override
    public void clear() {
        index = 0;
        size = 0;
        Arrays.fill(buffer, null);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public List<Experience2> sample(int batchSize) {
        if (size == 0) return Collections.emptyList();
        List<Experience2> batch = new ArrayList<>(batchSize);
        var rng = ThreadLocalRandom.current();
        for (var i = 0; i < batchSize; i++) batch.add(buffer[rng.nextInt(size)]);
        return batch;
    }

    @Override
    public List<Experience2> getAll() {
        return Arrays.asList(buffer).subList(0, size);
    }
}
