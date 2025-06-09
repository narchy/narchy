package jcog.evolve;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static jcog.evolve.ObjenTests.*;
import static org.junit.jupiter.api.Assertions.*;

public class ObjenEvolveTest {

    /** Test parameter distribution */
    @Test void testParameterDistribution() {
        var o = new Objen();
        var runs = 100;

        // Collect evolved parameters over multiple runs
        var distributions = IntStream.range(0, runs)
                .mapToObj(i -> o.evolve(IntOptimization.class, IntOptimization::score).get())
                .collect(Collectors.groupingBy(c -> c.workers, Collectors.counting()));

        // Verify reasonable distribution of values
        assertTrue(distributions.size() > 1, "Evolution should explore multiple values");

        // Most evolved configurations should be near optimal value (8)
        var optimal = distributions.entrySet().stream()
                .filter(e -> Math.abs(e.getKey() - 8) <= 2)
                .mapToLong(Map.Entry::getValue)
                .sum();

        assertTrue(optimal >= runs * 0.6,
                "insufficient optimal configurations: " + optimal + " of " + runs);
    }

    @Test
    void testSimpleParameterEvolution() {
        var y = new Objen().evolve(SimpleOptimizable.class, SimpleOptimizable::score).get();
        assertNotNull(y);
        assertTrue(y.value >= 0 && y.value <= 10);
        // Should be close to 5 (optimal value)
        assertTrue(Math.abs(y.value - 5) < 2, "not close");
    }

    @Test
    void testImplEvolution() {

        // Evolve implementation with fitness function favoring accuracy
        var algo = new Objen()
            //.any(Algorithm.class, FastAlgorithm.class, AccurateAlgorithm.class)
            .all(Algorithm.class)
            .evolve(Algorithm.class, a -> {
                double error = 0;
                for (double x = 0; x < 10; x++) {
                    error += Math.abs(a.compute(x) - Math.pow(x, 2));
                }
                return -error; // Negative error as fitness (higher is better)
            }).get();

        // Should prefer AccurateAlgorithm due to fitness function
        assertInstanceOf(AccurateAlgorithm.class, algo);
    }



    static class SimpleOptimizable {
        private final double value;

        public SimpleOptimizable(@Objen.Range(min = 0, max = 10) double value) {
            this.value = value;
        }

        public double score() {
            return -(value - 5) * (value - 5); // Optimal at value = 5
        }
    }

}
