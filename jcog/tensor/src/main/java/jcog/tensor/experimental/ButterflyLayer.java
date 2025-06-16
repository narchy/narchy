package jcog.tensor.experimental;

import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;

import java.util.function.UnaryOperator;

/**
 * ButterflyLayer implements a structured linear layer with O(N·logN) operations.
 * TODO test
 * 
 * For an input of dimension N (which must be a power-of-2), the layer factors the
 * full transformation into log₂(N) stages. At each stage s, groups of 2^(s+1) elements
 * are split into pairs and transformed by independent 2×2 matrices.
 *
 * Each stage has exactly N/2 pairs regardless of s.
 *
 * The forward pass applies, for each pair:
 *
 *    [y₀; y₁] = [a  b; c  d] [x₀; x₁]
 *
 * The backward pass reverses the stages and computes gradients for both the input and
 * each weight parameter.
 */
public class ButterflyLayer implements UnaryOperator<Tensor> {
    private final int N;         // Input dimension (must be a power of 2)
    private final int stages;    // Number of stages = log2(N)
    public final Tensor[] weights;  // One weight tensor per stage (each shape: (N/2, 4))

    /**
     * Constructs a ButterflyLayer for inputs of size N.
     * @param N must be a power of 2.
     */
    public ButterflyLayer(int N) {
        if (Integer.bitCount(N) != 1)
            throw new IllegalArgumentException("N must be a power of 2");
        this.N = N;
        this.stages = Integer.numberOfTrailingZeros(N);
        this.weights = new Tensor[stages];
        for (var s = 0; s < stages; s++)
            weights[s] = initButterflyWeight(N);
    }

    /**
     * Initializes a butterfly weight tensor for one stage.
     * Each row holds a 2×2 block stored as [a, b, c, d], where we initialize
     * a = d = 1 and b = c = 0 so the layer starts as the identity.
     */
    private static Tensor initButterflyWeight(int N) {
        var rows = N / 2;
        var cols = 4;
        var data = new double[rows * cols];
        for (var i = 0; i < rows; i++) {
            var o = i * cols;
            data[o]     = 1; // a
            data[o + 1] = 0; // b
            data[o + 2] = 0; // c
            data[o + 3] = 1; // d
        }
        return new Tensor(new SimpleMatrix(rows, cols, true, data), true).parameter();
    }

    /**
     * Forward pass.
     * Input tensor x must have shape (batch, N). The butterfly is applied stage-by-stage.
     */
    @Override
    public Tensor apply(Tensor x) {
        if (x.cols() != N)
            throw new IllegalArgumentException("Input tensor must have " + N + " columns");

        var batch = x.rows();
        var total = batch * N;
        var L = stages;

        // Save intermediate outputs: stageOutputs[0] is a copy of the input.
        var stageOutputs = new double[L + 1][];
        var inputData = x.array();
        stageOutputs[0] = new double[total];
        System.arraycopy(inputData, 0, stageOutputs[0], 0, total);

        // Forward: process each stage sequentially.
        for (var s = 0; s < L; s++) {
            var current = stageOutputs[s];
            var next = new double[total];
            var groupSize = 1 << (s + 1);      // 2^(s+1)
            var halfGroup = groupSize >> 1;      // groupSize / 2
            var numGroups = N / groupSize;       // Number of groups per row

            // Process each row in the batch.
            for (var b = 0; b < batch; b++) {
                var rowOffset = b * N;
                for (var g = 0; g < numGroups; g++) {
                    var groupStart = rowOffset + g * groupSize;
                    for (var i = 0; i < halfGroup; i++) {
                        var idx0 = groupStart + i;
                        var idx1 = groupStart + i + halfGroup;
                        var pairIndex = g * halfGroup + i;
                        // Cache the 2x2 block from the weight tensor for this stage.
                        var w = weights[s].array();
                        var wOffset = pairIndex * 4;
                        var a = w[wOffset];
                        var bVal = w[wOffset + 1];
                        var c = w[wOffset + 2];
                        var d = w[wOffset + 3];
                        var x0 = current[idx0];
                        var x1 = current[idx1];
                        next[idx0] = a * x0 + bVal * x1;
                        next[idx1] = c * x0 + d * x1;
                    }
                }
            }
            stageOutputs[s + 1] = next;
        }

        // Create output tensor from final stage result.
        var y = new Tensor(new SimpleMatrix(batch, N, true, stageOutputs[L]), x.hasGrad());
        if (y.hasGrad())
            y.op = new ButterflyOp(x, weights, stageOutputs, batch, N, L);

        return y;
    }

    /**
     * Custom TensorOp to perform backpropagation for the butterfly network.
     * parents[0] is the input, parents[1..stages] are the butterfly weight tensors.
     */
    private static class ButterflyOp extends Tensor.TensorOp {
        private final double[][] stageOutputs;
        private final int batch;
        private final int N;
        private final int stages;

        public ButterflyOp(Tensor input, Tensor[] weights, double[][] stageOutputs, int batch, int N, int stages) {
            super(buildParents(input, weights));
            this.stageOutputs = stageOutputs;
            this.batch = batch;
            this.N = N;
            this.stages = stages;
        }

        private static Tensor[] buildParents(Tensor input, Tensor[] weights) {
            var parents = new Tensor[weights.length + 1];
            parents[0] = input;
            System.arraycopy(weights, 0, parents, 1, weights.length);
            return parents;
        }

        /**
         * Backward pass:
         * - Reverses the butterfly stages,
         * - For each pair computes local gradients for the weight parameters using the saved stage outputs,
         * - Propagates input gradients backward.
         */
        @Override
        public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
            var batches = batch;
            var dCurrent = new double[batches * N];
            System.arraycopy(grad.getDDRM().data, 0, dCurrent, 0, dCurrent.length);

            // Allocate accumulators for each stage's weight gradients.
            var weightGrads = new double[stages][];
            for (var s = 0; s < stages; s++)
                weightGrads[s] = new double[(N / 2) * 4];


            // Process stages in reverse order.
            for (var s = stages - 1; s >= 0; s--) {
                var currentStageOutput = stageOutputs[s];
                var dPrev = new double[batches * N];
                var groupSize = 1 << (s + 1);
                var halfGroup = groupSize >> 1;
                var numGroups = N / groupSize;
                var w = parents[s + 1].array();

                for (var batch = 0; batch < batches; batch++) {
                    var rowOffset = batch * N;
                    for (var g = 0; g < numGroups; g++) {
                        var groupStart = rowOffset + g * groupSize;
                        for (var i = 0; i < halfGroup; i++) {
                            var idx0 = groupStart + i;
                            var idx1 = groupStart + i + halfGroup;
                            var pairIndex = g * halfGroup + i;
                            var wOffset = pairIndex * 4;
                            var a = w[wOffset];
                            var b = w[wOffset + 1];
                            var c = w[wOffset + 2];
                            var d = w[wOffset + 3];
                            var x0 = currentStageOutput[idx0];
                            var x1 = currentStageOutput[idx1];
                            var grad0 = dCurrent[idx0];
                            var grad1 = dCurrent[idx1];
                            // Accumulate gradients for the 2x2 weight block.
                            var wgs = weightGrads[s];
                            wgs[wOffset]     += grad0 * x0;
                            wgs[wOffset + 1] += grad0 * x1;
                            wgs[wOffset + 2] += grad1 * x0;
                            wgs[wOffset + 3] += grad1 * x1;
                            // Propagate gradients to the previous stage.
                            dPrev[idx0] += a * grad0 + c * grad1;
                            dPrev[idx1] += b * grad0 + d * grad1;
                        }
                    }
                }
                dCurrent = dPrev;
            }

            // Write the accumulated input gradient into the parent's gradient.
            if (gradOut[0] != null)
                System.arraycopy(dCurrent, 0, Tensor.array(gradOut[0]), 0, dCurrent.length);

            // Write each stage's weight gradients.
            for (var s = 0; s < stages; s++) {
                if (gradOut[s + 1] != null)
                    System.arraycopy(weightGrads[s], 0, Tensor.array(gradOut[s + 1]), 0, weightGrads[s].length);
            }
        }
    }
}
