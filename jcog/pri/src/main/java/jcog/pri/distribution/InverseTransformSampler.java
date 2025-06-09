//package jcog.pri.distribution;
//
//import java.util.Arrays;
//import java.util.Random;
//
///** TODO untested */
//public class InverseTransformSampler extends DistributionApproximator {
//    private float[] values;
//    private float[] cumulative;
//    private int size;
//    private int binCount;
//
//    @Override
//    public void clear() {
//        values = null;
//        cumulative = null;
//        size = 0;
//        binCount = 0;
//    }
//
//    @Override
//    public void start(int bins, int values) {
//        this.binCount = bins;
//        this.values = new float[values];
//        cumulative = new float[bins];
//        size = 0;
//    }
//
//    @Override
//    public void accept(float v) {
//        if (size >= binCount) {
//            throw new IllegalStateException("Too many values accepted");
//        }
//        values[size++] = v;
//    }
//
//    @Override
//    public void commit(float lo, float hi, int outBins) {
//        if (size == 0) {
//            throw new IllegalStateException("No values to commit");
//        }
//
//        Arrays.sort(values, 0, size);
//
//        // Compute cumulative distribution
//        float sum = 0;
//        for (int i = 0; i < size; i++) {
//            sum += values[i];
//            cumulative[i] = sum;
//        }
//
//        // Normalize cumulative distribution to [0, 1]
//        for (int i = 0; i < size; i++) {
//            cumulative[i] /= sum;
//        }
//
//    }
//
//    @Override
//    public void commit2(float lo, float hi) {
//        commit(lo, hi, size);
//    }
//
//    @Override
//    public float sample(float q) {
//        // Binary search to find the correct bin
//        int idx = Arrays.binarySearch(cumulative, 0, size, q);
//        if (idx < 0) idx = -idx - 1;
//        if (idx >= size) idx = size - 1;
//
//        return idx;
//    }
//
//    public static void main(String[] args) {
//        Random rand = new Random();
//        InverseTransformSampler sampler = new InverseTransformSampler();
//
////        sampler.start(10);
////        for (int i = 0; i < 10; i++) {
////            sampler.accept(rand.nextFloat());
////        }
//        sampler.start(3, 3);
//        sampler.accept(1);
//        sampler.accept(0.5f);
//        sampler.accept(0.01f);
//
//        sampler.commit(0, 1, 3);
//
//        for (int i = 0; i < 1000; i++) {
//            System.out.println(sampler.sample(rand.nextFloat()));
//        }
//    }
//}