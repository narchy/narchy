package nars.pri;

import jcog.data.list.Lst;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Thread-safe token-based budgeting system for data structures
 * @param <T> Target data structure type that will receive tokens
 *
 * Untested, may need changes before ready to use
 */
public abstract class Resources<T> {
    protected final ConcurrentHashMap<T, Resource<T>> resources = new ConcurrentHashMap<>();
    protected final AtomicInteger totalTokens = new AtomicInteger();
    protected final int maxTokens;

    /** Configuration parameters */
    protected float tenureFactor = 0.7f, accessFactor = 0.3f;
    
    /** Statistics */
    protected final AtomicLong totalAccesses = new AtomicLong(), totalRecalls = new AtomicLong();

    abstract protected long now();

    protected static class Resource<T> {
        final T target;
        final AtomicInteger tokens = new AtomicInteger();
        final AtomicLong accessCount = new AtomicLong();
        volatile long lastAccess;  // Updated atomically via CAS would be overkill
        volatile double priority;  // Cached priority, updated periodically

        Resource(T target, long now) {
            this.target = target;
            this.lastAccess = now;
        }

        void updatePriority(long now, float tenureFactor, float accessFactor) {
            var timeSinceAccess = now - lastAccess;
            priority = (tenureFactor * tokens.get()) + 
                      (accessFactor * accessCount.get() / (1.0 + timeSinceAccess/1000.0));
        }
    }

    public Resources(int initialTokens) {
        this.totalTokens.set(initialTokens);
        this.maxTokens = initialTokens * 2;
    }

    // Lifecycle visitors (not synchronized - should be thread-safe if needed)
    protected Consumer<T> onGive = t -> {};
    protected Consumer<T> onTake = t -> {};

    public void setOnGive(Consumer<T> onGive) { this.onGive = onGive; }
    public void setOnTake(Consumer<T> onTake) { this.onTake = onTake; }

    public void access(T target) {
        var entry = resources.computeIfAbsent(target, t -> new Resource<>(t, now()));
        entry.accessCount.incrementAndGet();
        entry.lastAccess = now();
        totalAccesses.incrementAndGet();
        adjustTokens();
    }

    public void adjustSupply(int newTotal) {
        if (newTotal > maxTokens) newTotal = maxTokens;

        var current = totalTokens.get();
        var delta = newTotal - current;
        
        if (delta > 0) {
            if (totalTokens.compareAndSet(current, newTotal)) {
                distributeTokens(delta);
            }
        } else if (delta < 0) {
            if (totalTokens.compareAndSet(current, newTotal)) {
                recallTokens(-delta);
            }
        }
    }

    protected void distributeTokens(int tokens) {
        List<Resource<T>> candidates = new Lst<>(resources.values());
        candidates.sort((a, b) -> Double.compare(b.priority, a.priority));

        var perTarget = tokens / Math.max(1, candidates.size());
        for (var entry : candidates) {
            if (tokens <= 0) break;
            var allocation = Math.min(perTarget, tokens);
            entry.tokens.addAndGet(allocation);
            tokens -= allocation;
            onGive.accept(entry.target);
        }
    }

    protected synchronized void recallTokens(int tokens) {
        var candidates = new Lst<>(resources.values());
        shuffle(candidates);

        candidates.sort(Comparator.comparingDouble(a -> a.priority));
        
        for (var entry : candidates) {
            if (tokens <= 0) break;
            var current = entry.tokens.get();
            var toTake = Math.min(current, tokens);
            if (toTake > 0 && entry.tokens.compareAndSet(current, current - toTake)) {
                tokens -= toTake;
                onTake.accept(entry.target);
                totalRecalls.incrementAndGet();
            }
        }
    }

    @Deprecated private static void shuffle(List candidates) {
        Collections.shuffle(candidates, ThreadLocalRandom.current());
    }

    public Iterator<T> targetIterator() {
        var targets = new Lst<T>(resources.keySet());
        shuffle(targets);
        return targets.iterator();
    }

    public Map<String, Number> getStats() {
        Map<String, Number> stats = new HashMap<>();
        stats.put("totalTokens", totalTokens.get());
        stats.put("activeTargets", resources.size());
        stats.put("totalAccesses", totalAccesses.get());
        stats.put("totalRecalls", totalRecalls.get());
        return stats;
    }

    protected void adjustTokens() {
        var now = now();
        resources.values().forEach(r -> r.updatePriority(now, tenureFactor, accessFactor));
    }

    /**
     * Memory pressure-aware subclass that adjusts tokens based on available memory
     */
    abstract public static class MemoryAwareResources<T> extends Resources<T> {
        private final long minFreeMemory;
        private final double shrinkFactor = 0.8;  // Reduce by 20% when pressured
        
        public MemoryAwareResources(int initialTokens, long minFreeMemory) {
            super(initialTokens);
            this.minFreeMemory = minFreeMemory;
            startMemoryMonitoring();
        }

        private void startMemoryMonitoring() {
            new Thread(() -> {
                while (true) {
                    checkMemoryPressure();
                    try {
                        Thread.sleep(1000);  // Check every second
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }).start();
        }

        private void checkMemoryPressure() {
            var rt = Runtime.getRuntime();
            var freeMemory = rt.freeMemory();
            if (freeMemory < minFreeMemory) {
                var currentTokens = totalTokens.get();
                var newTokens = (int)(currentTokens * shrinkFactor);
                adjustSupply(newTokens);
            }
        }
    }

    // Concrete simple implementation
    abstract public static class SimpleResources<T> extends Resources<T> {
        public SimpleResources(int initialTokens) {
            super(initialTokens);
        }
    }
}

// Example usage
class Example {
    static class DataStructure {
        String name;
        DataStructure(String name) { this.name = name; }
        @Override public String toString() { return name; }
    }

    public static void main(String[] args) {
        // Use memory-aware version
        var mgr =
            new Resources.MemoryAwareResources<DataStructure>(100, 100_000_000) {
                @Override protected long now() {
                    return System.currentTimeMillis();
                }
            }; // 100MB min
        
        mgr.setOnGive(ds -> System.out.println("Gave tokens to " + ds));
        mgr.setOnTake(ds -> System.out.println("Took tokens from " + ds));

        var ds1 = new DataStructure("DS1");
        var ds2 = new DataStructure("DS2");
        
        // Simulate concurrent access
        new Thread(() -> {
            for (var i = 0; i < 10; i++) {
                mgr.access(ds1);
                try { Thread.sleep(100); } catch (InterruptedException e) {}
            }
        }).start();

        new Thread(() -> {
            for (var i = 0; i < 10; i++) {
                mgr.access(ds2);
                try { Thread.sleep(100); } catch (InterruptedException e) {}
            }
        }).start();

        // Wait and adjust supply
        try { Thread.sleep(2000); } catch (InterruptedException e) {}
        mgr.adjustSupply(80);
        mgr.adjustSupply(120);
        
        System.out.println(mgr.getStats());

        var it = mgr.targetIterator();
        while (it.hasNext()) {
            System.out.println("Visiting: " + it.next());
        }
    }
}