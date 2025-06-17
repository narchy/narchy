package jcog.tensor;

import jcog.model.Layers; // Needed to access transformerBlocks.layer
import org.junit.jupiter.api.Test;
// import java.util.function.UnaryOperator; // Not directly needed for tests
import static org.junit.jupiter.api.Assertions.*;

public class TestModels {

    private static final double EPS = 1e-5;

    private void assertTensorShape(Tensor t, int rows, int cols, String msg) {
        assertNotNull(t, msg + ": Tensor is null");
        assertEquals(rows, t.rows(), msg + ": Mismatch in rows. Expected " + rows + ", got " + t.rows());
        assertEquals(cols, t.cols(), msg + ": Mismatch in cols. Expected " + cols + ", got " + t.cols());
    }

    private void assertNonNullGradient(Tensor tensor, String name) {
        assertNotNull(tensor, name + " tensor itself is null");
        assertTrue(tensor.hasGrad(), name + " does not require grad, so grad() will be null.");
        assertNotNull(tensor.grad(), name + " gradient is null");
        assertTrue(tensor.grad().abs().sum().scalar() > EPS, name + " gradient sum of abs values is not > EPS, indicating it might be all zeros.");
    }

    private void assertHasGrad(Tensor tensor, String name, boolean expectedGradStatus) {
        assertNotNull(tensor, name + " tensor itself is null");
        assertEquals(expectedGradStatus, tensor.hasGrad(), name + " hasGrad() status mismatch.");
        if (expectedGradStatus) {
            // Parameters should also be marked as such.
            assertTrue(tensor.isParameter(), name + " should be a parameter if it requires grad.");
        }
    }

    private boolean areTensorDataApproximatelyEqual(Tensor t1, Tensor t2, double epsilon) {
        if (t1.rows() != t2.rows() || t1.cols() != t2.cols()) return false;
        double[] d1 = t1.array();
        double[] d2 = t2.array();
        for (int i = 0; i < d1.length; i++) {
            if (Math.abs(d1[i] - d2[i]) > epsilon) return false;
        }
        return true;
    }


    // --- GPTLanguageModel Tests ---
    @Test
    void testGPTLanguageModelInitialization() {
        int vocabSize = 10, dModel = 4, maxSeqLen = 5, numLayers = 1, numHeads = 1, dff = 8;
        GPTLanguageModel model = new GPTLanguageModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                    Tensor::relu, 0.0f, false, true, true); // learnedPos=false, bias=true, reqGrad=true

        assertNotNull(model.tokenEmbedding, "GPT tokenEmbedding");
        assertNotNull(model.positionalEncoding, "GPT positionalEncoding");
        assertNotNull(model.transformerBlocks, "GPT transformerBlocks list");
        assertFalse(model.transformerBlocks.layer.isEmpty(), "GPT transformerBlocks list should not be empty");
        assertTrue(model.transformerBlocks.layer.get(0) instanceof TransformerBlock, "GPT transformerBlocks item type");
        assertNotNull(model.lmHead, "GPT lmHead");

        assertHasGrad(model.tokenEmbedding.weight, "GPT tokenEmbedding.weight", true);
        // Positional encoding (learned=false) weights should not require grad
        assertHasGrad(model.positionalEncoding.encodings, "GPT positionalEncoding.encodings (learned=false)", false);

        TransformerBlock block = (TransformerBlock) model.transformerBlocks.layer.get(0);
        assertHasGrad(block.mha.wq.weight, "GPT MHA wq.weight", true);
        assertHasGrad(block.ffn.linear1.weight, "GPT FFN linear1.weight", true);
        assertHasGrad(block.norm1.gamma, "GPT Block norm1.gamma", true);

        assertHasGrad(model.lmHead.projection.weight, "GPT lmHead.projection.weight", true);
    }

    @Test
    void testGPTLanguageModelForwardPass() {
        int vocabSize = 10, dModel = 4, maxSeqLen = 5, numLayers = 1, numHeads = 1, dff = 8, seqLen = 3;
        GPTLanguageModel model = new GPTLanguageModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                    Tensor::relu, 0.0f, false, true, false); // requiresGrad=false
        Tensor inputIds = Tensor.matrix(new double[][]{{1}, {2}, {3}}); // Shape [3,1]

        Tensor logits = model.forward(inputIds);
        assertTensorShape(logits, seqLen, vocabSize, "GPT ForwardPass output shape");
    }

    @Test
    void testGPTLanguageModelForwardCausalMaskEffect() {
        int vocabSize = 10, dModel = 4, maxSeqLen = 5, numLayers = 1, numHeads = 1, dff = 8;
        GPTLanguageModel model = new GPTLanguageModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                    Tensor::relu, 0.0f, false, true, false); // requiresGrad=false

        Tensor inputIds1 = Tensor.matrix(new double[][]{{1}, {2}, {3}}); // seqLen=3
        Tensor logits1 = model.forward(inputIds1);

        Tensor inputIds2 = Tensor.matrix(new double[][]{{1}, {2}, {9}}); // Change only the last token
        Tensor logits2 = model.forward(inputIds2);

        // Logits for the first two tokens (output rows 0 and 1) should be identical or very close
        // because of causal masking (they don't depend on the third token).
        Tensor logits1_first_two = logits1.slice(0, 2, 0, vocabSize);
        Tensor logits2_first_two = logits2.slice(0, 2, 0, vocabSize);
        assertTrue(areTensorDataApproximatelyEqual(logits1_first_two, logits2_first_two, EPS),
                   "First two token logits should be same due to causal mask");

        // Logits for the third token (output row 2) should differ.
        Tensor logits1_third = logits1.slice(2, 3, 0, vocabSize);
        Tensor logits2_third = logits2.slice(2, 3, 0, vocabSize);
        assertFalse(areTensorDataApproximatelyEqual(logits1_third, logits2_third, EPS),
                    "Third token logits should differ as input changed");
    }

    @Test
    void testGPTLanguageModelGradient() {
        int vocabSize = 3, dModel = 2, maxSeqLen = 3, numLayers = 1, numHeads = 1, dff = 4, seqLen = 2;
        GPTLanguageModel model = new GPTLanguageModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                    Tensor::relu, 0.0f, false, true, true); // requiresGrad=true
        Tensor inputIds = Tensor.matrix(new double[][]{{0},{1}});
        Tensor logits = model.forward(inputIds);
        Tensor loss = logits.sum();
        loss.minimize();

        assertNonNullGradient(model.tokenEmbedding.weight, "GPT tokenEmbedding.weight grad");

        TransformerBlock block = (TransformerBlock) model.transformerBlocks.layer.get(0);
        assertNonNullGradient(block.mha.wq.weight, "GPT MHA wq.weight grad");
        assertNonNullGradient(block.ffn.linear1.weight, "GPT FFN linear1.weight grad");
        assertNonNullGradient(block.norm1.gamma, "GPT Block norm1.gamma grad");

        assertNonNullGradient(model.lmHead.projection.weight, "GPT lmHead.projection.weight grad");
    }

    @Test
    void testGPTLanguageModelTrainingMode() {
        int vocabSize = 3, dModel = 2, maxSeqLen = 3, numLayers = 1, numHeads = 1, dff = 4;
        GPTLanguageModel model = new GPTLanguageModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                    Tensor::relu, 0.5f, false, true, true); // ffnDropoutRate=0.5f

        TransformerBlock block = (TransformerBlock) model.transformerBlocks.layer.get(0);
        assertNotNull(block.ffn.dropout, "FFN dropout in GPT block should exist");

        model.train(true);
        assertTrue(block.ffn.dropout.training, "GPT Block FFN dropout training mode true");

        model.train(false);
        assertFalse(block.ffn.dropout.training, "GPT Block FFN dropout training mode false");
    }

    // --- BERTEncoderModel Tests ---
    @Test
    void testBERTEncoderModelInitialization() {
        int vocabSize = 10, dModel = 4, maxSeqLen = 5, numLayers = 1, numHeads = 1, dff = 8, numSegments = 2;
        BERTEncoderModel model = new BERTEncoderModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                      Tensor::relu, 0.0f, false, numSegments, true, true, true);
                                                      // learnedPos=false, biasProj=true, finalNorm=true, reqGrad=true

        assertNotNull(model.tokenEmbedding, "BERT tokenEmbedding");
        assertNotNull(model.segmentEmbedding, "BERT segmentEmbedding");
        assertNotNull(model.positionalEncoding, "BERT positionalEncoding");
        assertNotNull(model.transformerBlocks, "BERT transformerBlocks list");
        assertFalse(model.transformerBlocks.layer.isEmpty(), "BERT transformerBlocks non-empty");
        assertNotNull(model.normLayer, "BERT finalNormLayer");

        assertHasGrad(model.tokenEmbedding.weight, "BERT tokenEmbedding.weight", true);
        assertHasGrad(model.segmentEmbedding.weight, "BERT segmentEmbedding.weight", true);
        assertHasGrad(model.positionalEncoding.encodings, "BERT positionalEncoding.encodings (learned=false)", false);

        TransformerBlock block = (TransformerBlock) model.transformerBlocks.layer.get(0);
        assertHasGrad(block.mha.wq.weight, "BERT MHA wq.weight", true);
        assertHasGrad(model.normLayer.gamma, "BERT finalNormLayer.gamma", true);
    }

    @Test
    void testBERTEncoderModelForwardPass() {
        int vocabSize = 10, dModel = 4, maxSeqLen = 5, numLayers = 1, numHeads = 1, dff = 8, numSegments = 2, seqLen = 3;
        BERTEncoderModel model = new BERTEncoderModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                      Tensor::relu, 0.0f, false, numSegments, true, true, false); // reqGrad=false
        Tensor inputIds = Tensor.matrix(new double[][]{{1}, {2}, {3}});
        Tensor segmentIds = Tensor.matrix(new double[][]{{0}, {0}, {1}});

        Tensor outputEmbeds = model.forward(inputIds, null, segmentIds);
        assertTensorShape(outputEmbeds, seqLen, dModel, "BERT ForwardPass output shape");
    }

    @Test
    void testBERTEncoderModelForwardWithPaddingMask() {
        int vocabSize=10, dModel=4, maxSeqLen=5, numLayers=1, numHeads=1, dff=8, numSegments=2, seqLen=3;
        BERTEncoderModel model = new BERTEncoderModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                      Tensor::relu, 0.0f, false, numSegments, true, true, false);
        Tensor inputIds = Tensor.matrix(new double[][]{{1}, {2}, {0}}); // Assume 0 is pad_token_id
        Tensor segmentIds = Tensor.matrix(new double[][]{{0}, {0}, {0}}); // Not relevant for mask test itself

        // Mask for padding: if key j is a pad token, column j of mask should be -1e9.
        // Here, token at index 2 (value 0) is padding. So, key at index 2 (column 2) masked.
        Tensor paddingMask = Tensor.matrix(new double[][]{
            {0, 0, -1e9},
            {0, 0, -1e9},
            {0, 0, -1e9}
        }); // Shape [seqLen, seqLen]

        Tensor outputEmbeds = model.forward(inputIds, paddingMask, segmentIds);
        assertTensorShape(outputEmbeds, seqLen, dModel, "BERT Masked Forward output shape");
        // Verifying the actual effect of the mask is complex without reference values or specific weight setup.
        // The primary check here is that it runs and produces the correct shape.
    }

    @Test
    void testBERTEncoderModelGradient() {
        int vocabSize=3, dModel=2, maxSeqLen=3, numLayers=1, numHeads=1, dff=4, numSegments=2, seqLen=2;
        BERTEncoderModel model = new BERTEncoderModel(vocabSize, dModel, maxSeqLen, numLayers, numHeads, dff,
                                                      Tensor::relu, 0.0f, false, numSegments, true, true, true); // reqGrad=true
        Tensor inputIds = Tensor.matrix(new double[][]{{0},{1}});
        Tensor segmentIds = Tensor.matrix(new double[][]{{0},{0}});

        Tensor outputEmbeds = model.forward(inputIds, null, segmentIds);
        Tensor loss = outputEmbeds.sum();
        loss.minimize();

        assertNonNullGradient(model.tokenEmbedding.weight, "BERT tokenEmbedding.weight grad");
        assertNonNullGradient(model.segmentEmbedding.weight, "BERT segmentEmbedding.weight grad");

        TransformerBlock block = (TransformerBlock) model.transformerBlocks.layer.get(0);
        assertNonNullGradient(block.mha.wq.weight, "BERT MHA wq.weight grad");
        assertNonNullGradient(block.ffn.linear1.weight, "BERT FFN linear1.weight grad");
        assertNonNullGradient(block.norm1.gamma, "BERT Block norm1.gamma grad");

        if (model.normLayer != null) {
            assertNonNullGradient(model.normLayer.gamma, "BERT finalNormLayer.gamma grad");
        }
    }
}
