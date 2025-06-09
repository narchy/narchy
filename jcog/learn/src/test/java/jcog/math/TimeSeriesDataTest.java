package jcog.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TimeSeriesDataTest {
    @Test
    void testCompressionEfficiencyAndAccuracy() {
        TimeSeriesData data = new TimeSeriesData();
        // Simulate input with large data set
        for (int i = 0; i < 1000; i++)
            data.append(Math.sin(i % 360 * Math.PI / 180), i, i + 1);

        assertTrue(data.tree.nodes.size() > 1);

        // Check the estimated size after compression
        int compressedSize = data.estimateTotalDataSize();
        System.out.println("Compressed Size: " + compressedSize);

        // Ensure the size is reduced
        assertTrue(compressedSize < 1000);

        // Verify the accuracy of data retrieval from compressed data
        for (int i = 0; i < 1000; i += 10) { // Check every 10th point
            double expected = Math.sin(i % 360 * Math.PI / 180);
            double actual = data.estimateValue(i);
            assertEquals(expected, actual, 0.1, "Estimated value does not match the expected value");
        }
    }
    @Test
    void testFFTBlockCreation() {
        TimeSeriesData data = new TimeSeriesData();
        // Append periodic data that should trigger FFT compression
        for (int i = 0; i < 100; i++) {
            double value = Math.sin(2 * Math.PI * i / 30); // Periodic data with a period of 30
            data.append(value, i, i + 1);
        }

        // Check if any FFT block was created
        assertTrue(data.tree.nodes.stream().anyMatch(n -> n.block instanceof TimeSeriesData.FFTBlock));
    }
}