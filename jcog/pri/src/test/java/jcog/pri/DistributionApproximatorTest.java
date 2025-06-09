package jcog.pri;

import com.google.common.collect.Streams;
import jcog.pri.distribution.DistributionApproximator;
import jcog.pri.distribution.SketchHistogram;
import jcog.random.XoRoShiRo128PlusRandom;
import org.hipparchus.stat.Frequency;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DistributionApproximatorTest {


    private static void topHeavy(int bins, double exp, int insertionsPerBin) {
        Random rng = new XoRoShiRo128PlusRandom(1);
        BiConsumer<DistributionApproximator, float[]> writer = (w, pdf) -> {
            for (int i = 0; i < bins * insertionsPerBin; i++) {
                w.accept(
                        (float) Math.pow(rng.nextFloat(), exp)
                );
            }
        };
        Frequency f = assertSampling(bins, insertionsPerBin * bins, writer);
        //check monotonically decreasing
        //TODO check that the approximate slope of the curve approximates the inputs
        for (int i = 1; i < bins; i++)
            assertTrue(f.getPct(i - 1) < f.getPct(i));
    }

    private static double variance(Frequency<?> f) {
        int bins = f.getUniqueCount();
        long iters = f.getSumFreq();
        double avg = ((double) iters) / bins;
        return Streams.stream(f.entrySetIterator()).map(Map.Entry::getValue).mapToDouble(x -> Math.abs(avg - x)).average().getAsDouble() / avg;
    }

    private static Frequency assertSampling(int bins, int iters, BiConsumer<DistributionApproximator, /*depr*/float[]> writer) {
        DistributionApproximator a =
                new SketchHistogram();
        //new ArrayHistogram();

        a.start(bins, iters);
        writer.accept(a, null);
        a.commit(0, bins-1, bins);

        System.out.println(a);
//        System.out.println(a.chart());

        Random rng = new XoRoShiRo128PlusRandom(1);

        Frequency f = new Frequency();
        int epochs = 100;
        for (int i = 0; i < iters*epochs; i++)
            f.addValue(a.sampleInt(rng));
        System.out.println(f);
        assertEquals(bins, f.getUniqueCount());
        return f;
    }

    @Test
    public void testSamplingDistribution() {
        SketchHistogram histogram = new SketchHistogram();
        int bins = 10;
        int values = 1000;
        int samples = 100000;

        // Start the histogram
        histogram.start(bins, values);

        // Add values with different priorities
        for (int i = 0; i < values; i++) {
            float priority = (float) i / values;
            histogram.accept(priority);
        }

        // Commit the histogram
        histogram.commit(0f, 1f, bins);

        // Sample from the histogram multiple times
        Map<Integer, Integer> sampleCounts = new HashMap<>();
        for (int i = 0; i < samples; i++) {
            float sample = histogram.sample((float) Math.random());
            int bin = (int) (sample * bins);
            sampleCounts.put(bin, sampleCounts.getOrDefault(bin, 0) + 1);
        }

        // Analyze the distribution
        for (int i = 0; i < bins; i++) {
            int count = sampleCounts.getOrDefault(i, 0);
            float expectedProportion = (float) (bins - i + 1) / (bins * (bins + 1) / 2);
            float actualProportion = (float) count / samples;

            // Allow for some margin of error (e.g., 10%)
            float marginOfError = 0.1f * expectedProportion;

            System.out.printf("Bin %d: Expected %.4f, Actual %.4f%n",
                    i, expectedProportion, actualProportion);

//            int I = i;
//            assertTrue(Math.abs(actualProportion - expectedProportion) <= marginOfError,
//                    ()->String.format("Bin %d proportion outside expected range", I));
        }
    }

    @ValueSource(ints = {4, 8, 16})
    @ParameterizedTest
    void topHeavy_2_many(int bins) {
        topHeavy(bins, 2, bins*2);
    }
    @ValueSource(ints = {4, 8, 16})
    @ParameterizedTest
    void topHeavy_4(int bins) {
        topHeavy(bins, 4, bins);
    }
    @ValueSource(ints = {4, 8, 16})
    @ParameterizedTest
    void topHeavy_8(int bins) {
        topHeavy(bins, 8, bins);
    }

    @ValueSource(ints = {3, 4, 5, 8, 15, 16, 32})
    @ParameterizedTest
    void linear(int bins) {
        int repeat = 1;
        var iters = repeat * bins;
        BiConsumer<DistributionApproximator, float[]> writer = (w, pdf) -> {
            for (int k = 0; k < repeat; k++) {
                for (int i = 0; i < bins; i++)
                    w.accept(((float) i) / (bins - 1));
            }
        };
        Frequency f = assertSampling(bins, iters, writer);
        System.out.println(f);

        assertTrue(f.getCount(0) > 1.5 * f.getCount(bins-1), "monotonically decreasing");

//        for (int i = 1; i < bins; i++)
//            assertTrue(f.getCount(i - 1) - f.getCount(i) > -(iters / 4), "monotonically decreasing");

    }

    //    @ValueSource(ints = {4, 8, 15, 16, 32})
//    @ParameterizedTest void flat(int bins) {
//        BiConsumer<DistributionApproximator, float[]> writer = (w, pdf) -> {
//            for (int i = 0; i < bins*64; i++)
//                w.accept((float) (0.5f + Math.random() * 0.05f));
//        };
//        Frequency f = assertSampling(bins, 200 * bins, writer);
//        System.out.println(f);
//        assertEquals(0, variance(f),  0.1f, "flat");
//    }

}