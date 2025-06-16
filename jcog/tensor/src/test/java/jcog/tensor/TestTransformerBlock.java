package jcog.tensor;

import jcog.model.LayerNorm; // For checking norm layer parameters
import org.junit.jupiter.api.Test;
// import java.util.function.UnaryOperator; // Not directly needed for tests, Tensor.RELU is static
import static org.junit.jupiter.api.Assertions.*;

public class TestTransformerBlock {

    private static final double EPS = 1e-5;

    private void assertTensorShape(Tensor t, int rows, int cols) {
        if (t == null) {
            fail("Tensor is null. Expected shape: [" + rows + ", " + cols + "]");
        }
        assertEquals(rows, t.rows(), "Mismatch in rows for tensor. Expected " + rows + ", got " + t.rows());
        assertEquals(cols, t.cols(), "Mismatch in cols for tensor. Expected " + cols + ", got " + t.cols());
    }

    private boolean areTensorDataDifferent(Tensor t1, Tensor t2) {
        if (t1 == null || t2 == null) return t1 != t2; // If one is null and other isn't, they are different
        if (t1.volume() != t2.volume()) return true;
        double[] d1 = t1.array();
        double[] d2 = t2.array();
        for (int i = 0; i < d1.length; i++) {
            if (Math.abs(d1[i] - d2[i]) > EPS) return true;
        }
        return false;
    }

    private void assertNonNullGradient(Tensor tensor, String name) {
        assertNotNull(tensor, name + " tensor itself is null, cannot check gradient.");
        assertNotNull(tensor.grad(), name + " gradient is null");
        // Optionally, check if sum of absolute values of grad is > EPS
        assertTrue(tensor.grad().abs().sum().scalar() > EPS, name + " gradient sum of abs values is not > EPS, might be all zeros.");
    }

    private void assertHasGrad(Tensor tensor, String name, boolean expectedGradStatus) {
        assertNotNull(tensor, name + " tensor itself is null, cannot check grad status.");
        assertEquals(expectedGradStatus, tensor.hasGrad(), name + " hasGrad() status mismatch.");
        if (expectedGradStatus) {
            assertTrue(tensor.isParameter(), name + " should be a parameter if it requires grad.");
        }
    }


    @Test
    void testTransformerBlockInitialization() {
        int dModel = 4, numHeads = 2, dff = 8;
        // Constructor: int dModel, int numHeads, int dff, UnaryOperator<Tensor> ffnActivation,
        // float attentionDropoutRate, float ffnDropoutRate, boolean biasForProjections, boolean requiresGrad
        TransformerBlock block = new TransformerBlock(dModel, numHeads, dff, Tensor::relu, 0.0f, 0.0f, true, true);

        assertNotNull(block.mha, "MHA should be initialized");
        assertNotNull(block.ffn, "FFN should be initialized");
        assertNotNull(block.norm1, "Norm1 should be initialized");
        assertNotNull(block.norm2, "Norm2 should be initialized");

        // Check MHA components' grad status (assuming biasForProjections=true, requiresGrad=true)
        assertHasGrad(block.mha.wq.weight, "MHA wq.weight", true);
        assertHasGrad(block.mha.wq.bias, "MHA wq.bias", true);
        assertHasGrad(block.mha.wk.weight, "MHA wk.weight", true);
        assertHasGrad(block.mha.wk.bias, "MHA wk.bias", true);
        assertHasGrad(block.mha.wv.weight, "MHA wv.weight", true);
        assertHasGrad(block.mha.wv.bias, "MHA wv.bias", true);
        assertHasGrad(block.mha.wo.weight, "MHA wo.weight", true);
        assertHasGrad(block.mha.wo.bias, "MHA wo.bias", true);

        // Check FFN components' grad status
        assertHasGrad(block.ffn.linear1.weight, "FFN linear1.weight", true);
        assertHasGrad(block.ffn.linear1.bias, "FFN linear1.bias", true);
        assertHasGrad(block.ffn.linear2.weight, "FFN linear2.weight", true);
        assertHasGrad(block.ffn.linear2.bias, "FFN linear2.bias", true);

        // Check LayerNorm components' grad status (LayerNorm internally makes gamma/beta parameters if owning model requires grad)
        // LayerNorm itself doesn't take a requiresGrad flag, but its parameters should be learnable.
        // The TransformerBlock's requiresGrad should influence this.
        // Assuming LayerNorm's gamma/beta are parameters by default.
        assertHasGrad(block.norm1.gamma, "Norm1 gamma", true);
        assertHasGrad(block.norm1.beta, "Norm1 beta", true);
        assertHasGrad(block.norm2.gamma, "Norm2 gamma", true);
        assertHasGrad(block.norm2.beta, "Norm2 beta", true);
    }

    @Test
    void testTransformerBlockForwardPass() {
        int dModel = 4, numHeads = 2, dff = 8, seqLen = 3;
        TransformerBlock block = new TransformerBlock(dModel, numHeads, dff, Tensor::relu, 0.0f, 0.0f, true, false); // requiresGrad=false
        Tensor input = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor output = block.forward(input, null); // No mask
        assertTensorShape(output, seqLen, dModel);
    }

    @Test
    void testTransformerBlockForwardWithCausalMask() {
        int dModel = 4, numHeads = 2, dff = 8, seqLen = 3;
        TransformerBlock block = new TransformerBlock(dModel, numHeads, dff, Tensor::relu, 0.0f, 0.0f, true, false);
        Tensor input = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor causalMask = Tensor.matrix(new double[][]{{0, -1e9, -1e9}, {0, 0, -1e9}, {0, 0, 0}});
        causalMask.grad(false); // Mask does not require grad

        Tensor output = block.forward(input, causalMask);
        assertTensorShape(output, seqLen, dModel);
    }

    @Test
    void testTransformerBlockFFNDropoutTrainingMode() {
        int dModel = 4, numHeads = 2, dff = 8, seqLen = 3;
        float ffnDropoutRate = 0.5f;
        TransformerBlock block = new TransformerBlock(dModel, numHeads, dff, Tensor::relu, 0.0f, ffnDropoutRate, true, false);

        assertNotNull(block.ffn.dropout, "FFN dropout layer should exist if rate > 0");

        Tensor input = Tensor.ones(seqLen, dModel);

        block.train(true); // Set training mode (dropout enabled)
        assertTrue(block.ffn.dropout.training, "FFN Dropout training flag should be true after block.train(true)");
        Tensor outputTrainTrue = block.forward(input.copy(), null);

        block.train(false); // Set evaluation mode (dropout disabled)
        assertFalse(block.ffn.dropout.training, "FFN Dropout training flag should be false after block.train(false)");
        Tensor outputTrainFalse = block.forward(input.copy(), null);

        // If weights are non-zero and dropout rate is >0 and <1, outputs should differ.
        // This check assumes that the default random weights in FFN are not all zero.
        if (block.ffn.linear1.weight.sum().scalar() != 0 || block.ffn.linear2.weight.sum().scalar() != 0) {
             assertTrue(areTensorDataDifferent(outputTrainTrue, outputTrainFalse),
                   "Outputs should differ between train(true) and train(false) if FFN dropout is active and weights are non-trivial.");
        }
    }

    @Test
    void testTransformerBlockGradient() {
        int dModel = 2, numHeads = 1, dff = 4, seqLen = 1; // Minimal viable dimensions
        TransformerBlock block = new TransformerBlock(dModel, numHeads, dff, Tensor::relu, 0.0f, 0.0f, true, true); // requiresGrad=true
        Tensor input = Tensor.matrix(new double[][]{{0.1, 0.2}}).grad(true);

        Tensor output = block.forward(input, null);
        Tensor loss = output.sum();
        loss.minimize(); // Default optimizer and learning rate

        assertNonNullGradient(input, "input");

        // MHA Gradients
        assertNonNullGradient(block.mha.wq.weight, "MHA wq.weight");
        if (block.mha.wq.bias != null) assertNonNullGradient(block.mha.wq.bias, "MHA wq.bias");
        assertNonNullGradient(block.mha.wk.weight, "MHA wk.weight");
        if (block.mha.wk.bias != null) assertNonNullGradient(block.mha.wk.bias, "MHA wk.bias");
        assertNonNullGradient(block.mha.wv.weight, "MHA wv.weight");
        if (block.mha.wv.bias != null) assertNonNullGradient(block.mha.wv.bias, "MHA wv.bias");
        assertNonNullGradient(block.mha.wo.weight, "MHA wo.weight");
        if (block.mha.wo.bias != null) assertNonNullGradient(block.mha.wo.bias, "MHA wo.bias");

        // FFN Gradients
        assertNonNullGradient(block.ffn.linear1.weight, "FFN linear1.weight");
        if (block.ffn.linear1.bias != null) assertNonNullGradient(block.ffn.linear1.bias, "FFN linear1.bias");
        assertNonNullGradient(block.ffn.linear2.weight, "FFN linear2.weight");
        if (block.ffn.linear2.bias != null) assertNonNullGradient(block.ffn.linear2.bias, "FFN linear2.bias");

        // LayerNorm Gradients
        assertNonNullGradient(block.norm1.gamma, "norm1.gamma");
        assertNonNullGradient(block.norm1.beta, "norm1.beta");
        assertNonNullGradient(block.norm2.gamma, "norm2.gamma");
        assertNonNullGradient(block.norm2.beta, "norm2.beta");
    }
}
