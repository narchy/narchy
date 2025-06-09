package jcog.evolve;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ObjenTests {

    @Test
    void testBasicDependencyInjection() {
        var algo = new Objen()
            .any(Algorithm.class, FastAlgorithm.class)
            .get(Algorithm.class);
        assertNotNull(algo);
        assertInstanceOf(FastAlgorithm.class, algo);
    }

    @Test
    void testParameterEvolution() {
        var service = new Objen().evolve(
            OptimizableService.class,
            OptimizableService::score,
            new Objen.GeneticOptimizer.GeneticConfig(100, 20, 0.1, 3, 2)
        ).get();

        assertNotNull(service);
        // Test evolved parameters are within bounds
        assertTrue(service.learningRate >= 0.0001 && service.learningRate <= 0.1);
        assertTrue(service.batchSize >= 1 && service.batchSize <= 32);
    }

    @Test
    void testCustomConfiguration() {
        var service = new Objen().evolve(
                OptimizableService.class,
                OptimizableService::score,
                new Objen.GeneticOptimizer.GeneticConfig(
                        200,    // population
                        50,     // generations
                        0.15,   // mutation rate
                        4,      // tournament size
                        3       // elite count
                )
        ).get();

        assertEquals(OptimizableService.class, service.getClass());
        assertNotNull(service);
    }

    @Test
    void testCyclicDependencyDetection() {
        var o = new Objen().any(A.class).any(B.class);

        assertThrows(RuntimeException.class, () -> o.get(A.class));
    }

    @Test
    void testRangeStepValues() {
        var params = new Objen().evolve(
                SteppedParameters.class,
                p -> -Math.abs(0.5 - p.value)  // Optimize towards 0.5
        ).get();

        assertNotNull(params);
        // Value should be multiple of 0.1
        assertEquals(0, (params.value * 10) % 1, 0.000001);
    }



    @Test
    void testEnumEvolution() {
        var processor = new Objen().evolve(
            ConfigurableProcessor.class,
            ConfigurableProcessor::score,
            new Objen.GeneticOptimizer.GeneticConfig(200, 100, 0.1, 4, 2)
        ).get();

        assertNotNull(processor);
        assertNotNull(processor.strategy);
        assertNotNull(processor.cachePolicy);

        // Verify integer parameters are within bounds
        assertTrue(processor.batchSize >= 1 && processor.batchSize <= 32);
        assertTrue(processor.cacheSize >= 100 && processor.cacheSize <= 1000);
        // Verify cache size is multiple of step
        assertEquals(0, processor.cacheSize % 100);
    }

    @Test
    void testIntegerParameterEvolution() {
        var config = new Objen().evolve(
                IntOptimization.class,
                IntOptimization::score,
                new Objen.GeneticOptimizer.GeneticConfig(150, 75, 0.15, 3, 2)
        ).get();

        assertNotNull(config);

        // Verify integer parameters
        assertTrue(config.workers >= 1 && config.workers <= 16);

        // Verify queue size steps
        assertTrue(config.queueSize >= 100 && config.queueSize <= 1000);
        assertEquals(0, config.queueSize % 50);

        // Verify timeout steps
        assertTrue(config.timeoutMs >= 1000 && config.timeoutMs <= 5000);
        assertEquals(0, config.timeoutMs % 500);
    }

    @Test
    void testMultiObjectiveEvolution() {
        // Evolution with weighted objectives
        var system = new Objen().evolve(MultiObjectiveSystem.class,
                sys -> {
                    var costScore = 1000 - sys.calculateCost() / 100; // Lower cost is better
                    var reliabilityScore = sys.calculateReliability() * 1000; // Higher reliability is better
                    return costScore * 0.4 + reliabilityScore * 0.6; // 40% cost, 60% reliability
                },
                new Objen.GeneticOptimizer.GeneticConfig(100, 50, 0.1, 4, 3)
        ).get();

        assertNotNull(system);

        // Verify constraints
        assertTrue(system.componentCount >= 1 && system.componentCount <= 10);
        assertTrue(system.redundancy >= 0 && system.redundancy <= 3);
        assertTrue(system.checkInterval >= 100 && system.checkInterval <= 1000);
        assertEquals(0, system.checkInterval % 100);

        // Verify reasonable trade-offs
        var cost = system.calculateCost();
        var reliability = system.calculateReliability();

        assertTrue(cost > 0);
        assertTrue(reliability > 0);

        // If high reliability, expect higher cost
        if (system.reliabilityLevel == Reliability.HIGH) {
            assertTrue(cost > 500);
        }

        // If high redundancy, expect higher reliability
        if (system.redundancy > 1) {
            assertTrue(reliability > 1.0);
        }
    }




    // Test enum evolution
    enum ProcessingStrategy {
        FAST, BALANCED, ACCURATE
    }

    enum CachePolicy {
        NONE, LRU, LFU, FIFO
    }


    // Test combined integer and enum evolution with multiple objectives
    enum Reliability {
        LOW, MEDIUM, HIGH
    }

    // Test interfaces and classes
    interface Algorithm {
        double compute(double input);
    }

    public static class FastAlgorithm implements Algorithm {
        public FastAlgorithm() {
        } // Added public constructor

        @Override
        public double compute(double input) {
            return input * 2; // Simple, fast implementation
        }
    }

    public static class AccurateAlgorithm implements Algorithm {
        public AccurateAlgorithm() {
        } // Added public constructor

        @Override
        public double compute(double input) {
            return Math.pow(input, 2); // More complex, accurate implementation
        }
    }

    public static class OptimizableService {
        private final double learningRate;
        private final double batchSize;

        public OptimizableService(
                @Objen.Range(min = 0.0001, max = 0.1) double learningRate,
                @Objen.Range(min = 1, max = 32) int batchSize
        ) {
            this.learningRate = learningRate;
            this.batchSize = batchSize;
        }

        public double score() {
            // Simulate performance metric
            return (1.0 / learningRate) + (batchSize / 32.0);
        }
    }

    static class SteppedParameters {
        private final double value;

        public SteppedParameters(@Objen.Range(min = 0, max = 1, step = 0.1) double value) { // Made constructor public
            this.value = value;
        }
    }

    static class ConfigurableProcessor {
        private final ProcessingStrategy strategy;
        private final int batchSize;
        private final CachePolicy cachePolicy;
        private final int cacheSize;

        public ConfigurableProcessor(
                @Objen.Range(min = 0, max = 2, enumClass = ProcessingStrategy.class) ProcessingStrategy strategy,
                @Objen.Range(min = 1, max = 32) int batchSize,
                @Objen.Range(min = 0, max = 3, enumClass = CachePolicy.class) CachePolicy cachePolicy,
                @Objen.Range(min = 100, max = 1000, step = 100) int cacheSize
        ) {
            this.strategy = strategy;
            this.batchSize = batchSize;
            this.cachePolicy = cachePolicy;
            this.cacheSize = cacheSize;
        }

        public double score() {
            // Simulate performance based on configuration
            double score = 0;

            // Strategy score
            switch (strategy) {
                case FAST:
                    score += 100;
                    break;
                case BALANCED:
                    score += 50;
                    break;
                case ACCURATE:
                    score += 25;
                    break;
            }

            // Batch size penalty
            score -= Math.abs(16 - batchSize); // Optimal at 16

            // Cache configuration score
            if (cachePolicy == CachePolicy.NONE) {
                score += 10;
            } else {
                score += 30;
                // Optimal cache size around 500
                score -= Math.abs(500 - cacheSize) / 100.0;
            }

            return score;
        }
    }

    // Test integer parameters with steps
    public static class IntOptimization {
        final int workers;
        final int queueSize;
        final int timeoutMs;

        public IntOptimization(
                @Objen.Range(min = 1, max = 16) int workers,
                @Objen.Range(min = 100, max = 1000, step = 50) int queueSize,
                @Objen.Range(min = 1000, max = 5000, step = 500) int timeoutMs
        ) {
            this.workers = workers;
            this.queueSize = queueSize;
            this.timeoutMs = timeoutMs;
        }

        public double score() {
            // Simulate system performance
            double score = 1000;

            // Worker count penalty (optimal at 8)
            score -= Math.abs(8 - workers) * 50;

            // Queue size penalty (optimal at 500)
            score -= Math.abs(500 - queueSize) * 0.5;

            // Timeout penalty (optimal at 2500)
            score -= Math.abs(2500 - timeoutMs) * 0.1;

            return score;
        }
    }

    static class MultiObjectiveSystem {
        private final int componentCount;
        private final Reliability reliabilityLevel;
        private final int redundancy;
        private final int checkInterval;

        public MultiObjectiveSystem(
                @Objen.Range(min = 1, max = 10) int componentCount,
                @Objen.Range(min = 0, max = 2, enumClass = Reliability.class) Reliability reliabilityLevel,
                @Objen.Range(min = 0, max = 3) int redundancy,
                @Objen.Range(min = 100, max = 1000, step = 100) int checkInterval
        ) {
            this.componentCount = componentCount;
            this.reliabilityLevel = reliabilityLevel;
            this.redundancy = redundancy;
            this.checkInterval = checkInterval;
        }

        public double calculateCost() {
            double baseCost = componentCount * 100;

            switch (reliabilityLevel) {
                case LOW:
                    baseCost *= 1.0;
                    break;
                case MEDIUM:
                    baseCost *= 1.5;
                    break;
                case HIGH:
                    baseCost *= 2.0;
                    break;
            }

            baseCost += redundancy * componentCount * 50;
            baseCost += (1000 - checkInterval) * 0.1; // More frequent checks cost more

            return baseCost;
        }

        public double calculateReliability() {
            var baseReliability = 0.9;

            switch (reliabilityLevel) {
                case LOW:
                    baseReliability *= 1.0;
                    break;
                case MEDIUM:
                    baseReliability *= 1.2;
                    break;
                case HIGH:
                    baseReliability *= 1.5;
                    break;
            }

            baseReliability += redundancy * 0.05;
            baseReliability *= (1000.0 / checkInterval); // More frequent checks increase reliability

            return baseReliability;
        }
    }

    // Create cyclic dependency
    class A {
        public A(@Objen.Range(min = 0, max = 1) double param, B b) {
        } // Made constructor public
    }

    class B {
        public B(A a) {
        } // Made constructor public
    }
}