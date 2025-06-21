package jcog.optimize;

import jcog.Util;
import jcog.random.XoRoShiRo128PlusRandom;
import org.junit.jupiter.api.Test;

import java.util.function.ToDoubleFunction;

import static jcog.Str.n2;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BayesOptimizerTest {
    @Test
    void handlesNoisyObservations() {
        var mins = new double[]{0.0};
        var maxs = new double[]{1.0};

        var rand = new XoRoShiRo128PlusRandom(42);
        ToDoubleFunction<double[]> f = x ->
                Util.sqr(x[0]) + 0.1 * rand.nextFloat();

        var o = new BayesOptimizer(32, mins, maxs);
        for (int i = 0; i < 100; i++) {
            o.put(f);
            System.out.println(i + "\t" + n2(o.best()));
        }

        var bestPoint = o.best();
        assertTrue(Math.abs(bestPoint[0]) < 0.2);
    }


    public static void main(String[] args) {
        // Example 1: Optimizing a simple 2D function
        System.out.println("Example 1: Optimizing 2D Ackley function");
        optimize2DAckley();

//        // Example 2: Optimizing hyperparameters for ML model
//        System.out.println("\nExample 2: ML Hyperparameter Optimization");
//        optimizeHyperparameters();
    }

    private static void optimize2DAckley() {
        // Define bounds for 2D Ackley function
        double[] lowerBounds = {-2.0, -2.0};
        double[] upperBounds = {2.0, 2.0};

        // Create optimizer
        BayesOptimizer o = new BayesOptimizer(32, lowerBounds, upperBounds);

        int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            double[] x = o.next();

            o.put(x,
                //ackley(x)
                new MyCMAESOptimizerTest.Rosen().value(x)
            );

            if ((i + 1) % 1 == 0) {
                System.out.printf("%5d:\t%.6f @ %s%n", i + 1, o.bestValue(), n2(o.best()));
            }
        }
    }

    private static void optimizeHyperparameters() {
        // Example of optimizing learning rate and regularization parameter
        double[] lowerBounds = {1e-4, 1e-4};  // learning rate, regularization
        double[] upperBounds = {1.0, 1.0};

        BayesOptimizer optimizer = new BayesOptimizer(8, lowerBounds, upperBounds);

        // Simulate ML model training with different hyperparameters
        for (int i = 0; i < 30; i++) {
            double[] hyperparams = optimizer.next();
            double validationError = simulateMLTraining(hyperparams[0], hyperparams[1]);
            optimizer.put(hyperparams, validationError);

            if ((i + 1) % 5 == 0) {
                System.out.printf("Iteration %d: Best validation error = %.6f%n",
                        i + 1, optimizer.bestValue());
                double[] bestHyperparams = optimizer.best();
                System.out.printf("Best hyperparameters: learning_rate=%.6f, reg=%.6f%n",
                        bestHyperparams[0], bestHyperparams[1]);
            }
        }
    }

    /**
     * 2D Ackley function - a common optimization benchmark function
     * Global minimum at f(0,0) = 0
     */
    private static double ackley(double[] x) {
        double a = 20;
        double b = 0.2;
        double c = 2 * Math.PI;

        double sum1 = 0;
        double sum2 = 0;
        for (double xi : x) {
            sum1 += xi * xi;
            sum2 += Math.cos(c * xi);
        }

        sum1 = -b * Math.sqrt(sum1 / x.length);
        sum2 = sum2 / x.length;

        return -a * Math.exp(sum1) - Math.exp(sum2) + a + Math.E;
    }

    /**
     * Simulates training an ML model and returning validation error
     * This is a synthetic example using a simple function
     */
    private static double simulateMLTraining(double learningRate, double regularization) {
        // Synthetic function to simulate validation error
        // Optimal values around: learning_rate = 0.1, regularization = 0.01
        double lr_component = Math.pow(Math.log10(learningRate) + 1, 2);
        double reg_component = Math.pow(Math.log10(regularization) + 2, 2);

        // Add some noise to simulate real-world variation
        double noise = (Math.random() - 0.5) * 0.1;

        return lr_component + reg_component + noise;
    }

//    @Test
//    void testRosenbrock() {
//        var b = new BayesianOptimization(2, new double[] { -1, -1}, new double[] { +1, +1 });
//        b.
//    }
}

//TODO translate some tests from here
//class BayesianOptimizerTest {
//
//    @Test
//    void optimizeSphereFunction() {
//        var names = new String[]{"x", "y"};
//        var mins = new double[]{-5.12, -5.12};
//        var maxs = new double[]{5.12, 5.12};
//        var isLog = new boolean[]{false, false};
//
//        Function<double[], Double> sphere = x ->
//                Arrays.stream(x).map(xi -> xi * xi).sum();
//
//        var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog, sphere);
//        var result = optimizer.optimize(100);
//
//        assertAll(
//                () -> assertTrue(result.y() < 0.1, "Sphere function optimization failed to find near-zero minimum."),
//                () -> assertTrue(Arrays.stream(result.x()).allMatch(xi -> Math.abs(xi) < 0.1), "Sphere function optimization did not converge to origin.")
//        );
//    }
//
//    @Test
//    void optimizeRosenbrockFunction() {
//        var names = new String[]{"x", "y"};
//        var mins = new double[]{-2.048, -2.048};
//        var maxs = new double[]{2.048, 2.048};
//        var isLog = new boolean[]{false, false};
//
//        Function<double[], Double> rosenbrock = x -> {
//            double a = 1.0, b = 100.0;
//            return Math.pow(a - x[0], 2) + b * Math.pow(x[1] - x[0] * x[0], 2);
//        };
//
//        var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog, rosenbrock);
//        var result = optimizer.optimize(100);
//
//        assertAll(
//                () -> assertTrue(result.y() < 0.1, "Rosenbrock function optimization failed to find near-zero minimum."),
//                () -> assertTrue(Math.abs(result.x()[0] - 1.0) < 0.1, "Rosenbrock function optimization did not converge x to 1."),
//                () -> assertTrue(Math.abs(result.x()[1] - 1.0) < 0.1, "Rosenbrock function optimization did not converge y to 1.")
//        );
//    }
//
//
//    private static final double DELTA = 1e-10;
//    private static final double OPTIMIZATION_DELTA = 1e-4;
//
//    @Nested
//    class CoreFunctionality {
//        @Test
//        void optimizesSphereFunction() {
//            var names = new String[]{"x", "y"};
//            var mins = new double[]{-5.12, -5.12};
//            var maxs = new double[]{5.12, 5.12};
//            var isLog = new boolean[]{false, false};
//
//            Function<double[], Double> sphere = x ->
//                    Arrays.stream(x).map(xi -> xi * xi).sum();
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog, sphere);
//            optimizer.initialize();
//            var result = optimizer.optimize(50);
//
//            assertAll(
//                    () -> assertTrue(Math.abs(result.y()) < 0.1),
//                    () -> assertTrue(Arrays.stream(result.x())
//                            .allMatch(x -> Math.abs(x) < 0.1))
//            );
//        }
//
//        @Test
//        void optimizesRosenbrockFunction() {
//            var names = new String[]{"x", "y"};
//            var mins = new double[]{-2.048, -2.048};
//            var maxs = new double[]{+2.048, +2.048};
//            var isLog = new boolean[]{false, false};
//
//            Function<double[], Double> rosenbrock = x -> {
//                double sum = 0;
//                for (var i = 0; i < x.length - 1; i++) {
//                    sum += 100 * Math.pow(x[i+1] - x[i]*x[i], 2) +
//                            Math.pow(1 - x[i], 2);
//                }
//                return sum;
//            };
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog, rosenbrock);
//            optimizer.initialize();
//            var result = optimizer.optimize(100);
//
//            assertAll(
//                    () -> assertTrue(result.y() < 0.1),
//                    () -> assertTrue(Math.abs(result.x()[0] - 1.0) < 0.1),
//                    () -> assertTrue(Math.abs(result.x()[1] - 1.0) < 0.1)
//            );
//        }
//    }
//
//    @Nested
//    class GaussianProcess {
//        private BayesianOptimizer0.GP newTestGP() {
//            var samples = new BayesianOptimizer0.Sample[] {
//                    new BayesianOptimizer0.Sample(new double[]{0.0}, 0.0),
//                    new BayesianOptimizer0.Sample(new double[]{0.5}, 0.25),
//                    new BayesianOptimizer0.Sample(new double[]{1.0}, 1.0)
//            };
//            return new BayesianOptimizer0(
//                    new String[]{"x"},
//                    new double[]{0},
//                    new double[]{1},
//                    new boolean[]{false},
//                    x -> x[0]
//            ).new GP(samples);
//        }
//
////        @Disabled @Test
////        void predictionsMatchTrainingPoints() {
////            var gp = newTestGP();
////            double[][] points = {{0.0}, {0.5}, {1.0}};
////            double[] expected = {0.0, 0.25, 1.0};
////
////            for (int i = 0; i < points.length; i++) {
////                var pred = gp.predict(points[i]);
////                assertEquals(expected[i], pred.mean(), DELTA);
////                assertTrue(pred.std() < DELTA);
////            }
////        }
////
////        @Test
////        void uncertaintyIncreasesAwayFromSamples() {
////            var gp = newTestGP();
////            double std1 = gp.predict(new double[]{0.25}).std();
////            double std2 = gp.predict(new double[]{2.0}).std();
////            assertTrue(std1 < std2);
////        }
//    }
//
//    @Nested
//    class NumericalStability {
//        @Test
//        void handlesClosePoints() {
//            var names = new String[]{"x"};
//            var mins = new double[]{0.0};
//            var maxs = new double[]{1.0};
//            var isLog = new boolean[]{false};
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog,
//                    x -> x[0] * x[0]);
//
//            // Add very close points
//            optimizer.addSample(new double[]{0.1}, 0.01);
//            optimizer.addSample(new double[]{0.1 + 1e-10}, 0.01);
//
//            assertDoesNotThrow(() -> optimizer.newGP().predict(new double[]{0.5}));
//        }
//
//    }
//
//    @Disabled
//    @Nested
//    class Parallelization {
//        @Test
//        void handlesConcurrentEvaluations() {
//            var names = new String[]{"x"};
//            var mins = new double[]{0.0};
//            var maxs = new double[]{1.0};
//            var isLog = new boolean[]{false};
//
//            var latch = new CountDownLatch(10);
//            Function<double[], Double> slowFunction = x -> {
//                try {
//                    Thread.sleep(100);
//                    latch.countDown();
//                    return x[0] * x[0];
//                } catch (InterruptedException e) {
//                    throw new RuntimeException(e);
//                }
//            };
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog, slowFunction);
//
//            var start = System.currentTimeMillis();
//            optimizer.initialize();
//            optimizer.optimize(10);
//            var duration = System.currentTimeMillis() - start;
//
//            assertTrue(latch.getCount() == 0);
//            assertTrue(duration < 1000); // Should be parallel
//        }
//    }
//
//    @Nested
//    class EdgeCases {
//        @ParameterizedTest
//        @ValueSource(ints = {1, 2, 3/*, 10*/})
//        void handlesVariousDimensions(int dims) {
//            var names = IntStream.range(0, dims)
//                    .mapToObj(i -> "x" + i)
//                    .toArray(String[]::new);
//            var mins = new double[dims];
//            var maxs = Arrays.stream(mins).map(x -> 1.0).toArray();
//            var isLog = new boolean[dims];
//
//            Function<double[], Double> sumSquares = x ->
//                    Arrays.stream(x).map(xi -> xi * xi).sum();
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog,
//                    sumSquares);
//            optimizer.initialize();
//            var result = optimizer.optimize(100 * dims);
//
//            assertTrue(result.y() < 0.1 * dims);
//        }
//
//        @Test
//        void handlesLogScaleParameters() {
//            var names = new String[]{"learning_rate"};
//            var mins = new double[]{1e-5};
//            var maxs = new double[]{1.0};
//            var isLog = new boolean[]{true};
//
//            Function<double[], Double> f = x ->
//                    Math.pow(Math.log10(x[0]) + 3, 2);
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog, f);
//            optimizer.initialize();
//            var result = optimizer.optimize(50);
//
//            var optimalLR = 1e-3;
//            assertTrue(Math.abs(Math.log10(result.x()[0]) -
//                    Math.log10(optimalLR)) < 1.0);
//        }
//
//        @Test
//        void handlesDiscontinuousFunction() {
//            var names = new String[]{"x"};
//            var mins = new double[]{-1.0};
//            var maxs = new double[]{1.0};
//            var isLog = new boolean[]{false};
//
//            Function<double[], Double> discontinuous = x ->
//                    x[0] < 0 ? x[0] * x[0] : x[0] + 1;
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog,
//                    discontinuous);
//            optimizer.initialize();
//            var result = optimizer.optimize(50);
//
//            assertTrue(result.x()[0] < 0);
//            assertTrue(result.y() < 0.1);
//        }
//    }
//
//    @Nested
//    class Constraints {
//        @Test
//        void respectsBoundaryConstraints() {
//            var names = new String[]{"x"};
//            var mins = new double[]{-1.0};
//            var maxs = new double[]{1.0};
//            var isLog = new boolean[]{false};
//
//            var optimizer = new BayesianOptimizer0(names, mins, maxs, isLog,
//                    x -> x[0] * x[0]);
//            optimizer.initialize();
//            var result = optimizer.optimize(50);
//
//            assertAll(
//                    () -> assertTrue(result.x()[0] >= mins[0]),
//                    () -> assertTrue(result.x()[0] <= maxs[0])
//            );
//        }
//    }
//}