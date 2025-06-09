package jcog.tensor.rl.pg.util;

import jcog.data.list.Lst;

import java.util.List;
import java.util.Random;

/** untested */
public class ReplayBufferPrioritized {
    private final List<Experience> buffer;
    private final List<Double> priorities;
    private final int capacity;
    private double maxPriority;

    public ReplayBufferPrioritized(int capacity) {
        this.capacity = capacity;
        this.buffer = new Lst<>(capacity);
        this.priorities = new Lst<>(capacity);
        this.maxPriority = 1.0;
    }

    public void add(Experience e, double priority) {
        if (buffer.size() + 1 > capacity) {
            buffer.removeFirst();
            priorities.removeFirst();
        }
        buffer.addLast(e);
        priorities.addLast(priority);
        maxPriority = Math.max(maxPriority, priority);
    }

    public Experience sample(Random rng) {
        double sum = priorities.stream().mapToDouble(p -> Math.pow(p / maxPriority, 0.6)).sum();
        double prefixSum = 0.0;
        double randomValue = rng.nextDouble() * sum;
        for (int i = 0; i < priorities.size(); i++) {
            prefixSum += Math.pow(priorities.get(i) / maxPriority, 0.6);
            if (prefixSum >= randomValue) {
                return buffer.get(i);
            }
        }
        throw new RuntimeException("Unreachable code");
    }

    public int size() {
        return buffer.size();
    }

}