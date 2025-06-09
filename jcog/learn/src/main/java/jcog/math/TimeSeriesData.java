package jcog.math;

import jcog.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import static jcog.Str.n2;

public class TimeSeriesData {

    final IntervalTree tree = new IntervalTree();

    @Override
    public String toString() {
        return tree.toString();
    }

    static final int BUFFER_SIZE = 4;
    private List<Double> buffer = new ArrayList<>(BUFFER_SIZE);

    public void append(double value, double start, double end) {
        buffer.add(value);
        if (buffer.size() >= BUFFER_SIZE) { // BUFFER_SIZE can be the desired block size, say 20 or 50
            CompressionBlock block = selectCompressionType(buffer);
            buffer.forEach(block::append);
            tree.insert(new Range(start, start + buffer.size()), block);
            buffer.clear(); // Reset buffer after processing
        }
    }

    public CompressionBlock selectCompressionType(List<Double> values) {
        if (values.size() < 2) {
            return new DirectBlock(); // Not enough data to determine a pattern
        }

        // Check for repeating patterns that may indicate periodicity
        double mean = values.stream().mapToDouble(v -> v).average().orElse(0.0);
        double variance = values.stream().mapToDouble(v -> Math.pow(v - mean, 2)).average().orElse(0.0);

        if (variance < 0.01) {
            return new RepeatBlock(); // Low variance suggests constant or near-constant values
        } else if (isPeriodic(values)) {
            return new FFTBlock(); // Use FFT if data appears to be periodic
        } else {
            return new DirectBlock(); // Default to direct if no pattern is clear
        }
    }

    private boolean isPeriodic(List<Double> values) {
        // Simple periodicity check based on autocorrelation or Fourier transform
        // This is a placeholder for a real implementation
        double threshold = 0.8; // Threshold for determining strong periodicity
        double maxCorrelation = autocorrelation(values);
        return maxCorrelation > threshold;
    }

    private double autocorrelation(List<Double> values) {
        double mean = values.stream().mapToDouble(v -> v).average().orElse(0.0);
        double variance = values.stream().mapToDouble(v -> Math.pow(v - mean, 2)).sum();
        if (variance == 0) return 1.0;

        double autocorrelation = 0.0;
        for (int lag = 1; lag < values.size(); lag++) {
            double cov = 0.0;
            for (int i = lag; i < values.size(); i++) {
                cov += (values.get(i) - mean) * (values.get(i - lag) - mean);
            }
            autocorrelation = Math.max(autocorrelation, cov / variance);
        }
        return autocorrelation;
    }

    public int estimateTotalDataSize() {
        int totalSize = 0;
        for (IntervalNode node : tree.nodes) {
            totalSize += node.block.estimatedDataSize();
        }
        return totalSize;
    }

    public double estimateValue(double index) {
        List<IntervalNode> relevantNodes = tree.search(index);
        if (relevantNodes.isEmpty()) {
            return Double.NaN; // Return NaN if no relevant data blocks are found
        }

        // Find the closest points on either side of the index
        Double leftValue = null, rightValue = null;
        double leftIndex = Double.NEGATIVE_INFINITY, rightIndex = Double.POSITIVE_INFINITY;

        for (IntervalNode node : relevantNodes) {
            if (node.range.includes(index))
                return node.block.estimateValue(index);

            // Find the closest left point
            if (node.range.end <= index && node.range.end > leftIndex) {
                leftIndex = node.range.end;
                leftValue = node.block.estimateValue(leftIndex);
            }
            // Find the closest right point
            if (node.range.start >= index && node.range.start < rightIndex) {
                rightIndex = node.range.start;
                rightValue = node.block.estimateValue(rightIndex);
            }
        }

        // If exact match or only one side found, return the found value
        if (leftIndex == index || rightIndex == index || leftValue == null) {
            if (rightValue == null && leftValue == null)
                return Double.NaN;
            return rightValue != null ? rightValue : leftValue;
        } else if (rightValue == null) {
            return leftValue;
        }

        // Perform linear interpolation
        double fraction = (index - leftIndex) / (rightIndex - leftIndex);
        return leftValue + fraction * (rightValue - leftValue);
    }

    public double[] resample(int newSize) {
        if (tree.nodes.isEmpty()) {
            return new double[0]; // Return empty array if no data
        }

        double minIndex = tree.nodes.get(0).range.start;
        double maxIndex = tree.nodes.get(tree.nodes.size() - 1).range.end;
        double[] resampled = new double[newSize];
        double step = (maxIndex - minIndex) / (newSize - 1);

        for (int i = 0; i < newSize; i++) {
            double index = minIndex + i * step;
            resampled[i] = estimateValue(index);
        }
        return resampled;
    }

    public static void main(String[] args) {
        TimeSeriesData data = new TimeSeriesData();
        for (int i = 0; i < 20; i++)
            data.append(i % 5, i, i + 1); // Simulating repeating patterns with modulo

        System.out.println(data);
        double[] resampledData = data.resample(10);
        for (double d : resampledData) {
            System.out.println("Resampled value: " + n2(d));
        }
    }

    /** TODO use DoubleArrayList */
    static class DirectBlock implements CompressionBlock {
        private List<Double> samples = new LinkedList<>();
        private int capacity = 10;

        @Override
        public void append(double value) {
            if (samples.size() >= capacity) {
                compressData();
            }
            samples.add(value);
        }

        @Override
        public int estimatedDataSize() {
            return samples.size();
        }

        private void compressData() {
            List<Double> compressed = new ArrayList<>();
            for (int i = 0; i < samples.size(); i += 2) {
                if (i + 1 < samples.size()) {
                    compressed.add((samples.get(i) + samples.get(i + 1)) / 2);
                } else {
                    compressed.add(samples.get(i));
                }
            }
            samples = compressed;
        }

        @Override
        public double estimateValue(double index) {
            int i = Util.clamp((int) index, 0, samples.size()-1);
            return samples.get(i);
        }
    }
    static class RepeatBlock implements CompressionBlock {
        private double value;
        private int count;

        @Override
        public void append(double value) {
            if (this.count == 0 || this.value == value) {
                this.value = value;
                this.count++;
            } else {
                this.count = 1; // Reset for new value
            }
        }
        @Override
        public int estimatedDataSize() {
            return 1; // Only one value and a count are stored
        }
        @Override
        public double estimateValue(double index) {
            return value;
        }
    }
    public static class FFTBlock implements CompressionBlock {
        private double[] frequencyComponents = new double[0];

        @Override
        public void append(double value) {
            // Simplified: storing values directly for demo
            frequencyComponents = Arrays.copyOf(frequencyComponents, frequencyComponents.length + 1);
            frequencyComponents[frequencyComponents.length - 1] = value;
        }
        @Override
        public int estimatedDataSize() {
            return frequencyComponents.length;
        }
        @Override
        public double estimateValue(double index) {
            return frequencyComponents[(int) index];
        }

    }
}

interface CompressionBlock {
    double estimateValue(double index);
    void append(double value);
    int estimatedDataSize();
}

class IntervalNode {
    Range range;
    CompressionBlock block;

    IntervalNode(Range range, CompressionBlock block) {
        this.range = range;
        this.block = block;
    }

    @Override
    public String toString() {
        return "IntervalNode{" +
                "range=" + range +
                ", block=" + block +
                '}';
    }
}

class IntervalTree {
    List<IntervalNode> nodes = new ArrayList<>();

    @Override
    public String toString() {
        return "IntervalTree{" +
                "nodes=" + nodes +
                '}';
    }

    void insert(Range range, CompressionBlock block) {
        nodes.add(new IntervalNode(range, block));
    }

    List<IntervalNode> search(double index) {
        List<IntervalNode> result = new ArrayList<>();
        for (IntervalNode node : nodes) {
            if (node.range.includes(index)) {
                result.add(node);
            }
        }
        return result;
    }
}

class Range {
    double start, end;

    Range(double start, double end) {
        this.start = start;
        this.end = end;
    }

    boolean includes(double point) {
        return point >= start && point <= end;
    }
}

