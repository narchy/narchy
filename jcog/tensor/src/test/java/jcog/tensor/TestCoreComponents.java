package jcog.tensor;

import jcog.model.LayerNorm;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestCoreComponents {

    private static final double EPS = 1e-5;

    private void assertTensorShape(Tensor t, int rows, int cols, String messagePrefix) {
        assertEquals(rows, t.rows(), messagePrefix + ": Mismatch in rows");
        assertEquals(cols, t.cols(), messagePrefix + ": Mismatch in cols");
    }

    private void assertTensorShape(Tensor t, int rows, int cols) {
        assertTensorShape(t, rows, cols, "Shape assertion failed");
    }


    private void assertTensorDataEquals(double[][] expected, Tensor actual, String messagePrefix) {
        assertTensorShape(actual, expected.length, expected.length > 0 ? expected[0].length : 0, messagePrefix);
        double[] actualData = actual.array();
        int actualCols = actual.cols();
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(expected[i][j], actualData[i * actualCols + j], EPS,
                             messagePrefix + ": Data mismatch at [" + i + "," + j + "]");
            }
        }
    }
    private void assertTensorDataEquals(double[][] expected, Tensor actual) {
        assertTensorDataEquals(expected, actual, "Data assertion failed");
    }


    private void assertArrayApproximatelyEquals(double[] expected, double[] actual, String messagePrefix) {
        assertEquals(expected.length, actual.length, messagePrefix + ": Array length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], EPS, messagePrefix + ": Data mismatch at index " + i);
        }
    }


    // --- TokenEmbedding Tests ---
    @Test
    void testTokenEmbeddingInitialization() {
        TokenEmbedding emb1 = new TokenEmbedding(10, 4, true);
        assertNotNull(emb1.weight, "emb1.weight should not be null");
        assertTensorShape(emb1.weight, 10, 4, "emb1.weight shape");
        assertTrue(emb1.weight.hasGrad(), "emb1.weight should require grad");
        assertTrue(emb1.weight.isParameter(), "emb1.weight should be a parameter");

        TokenEmbedding emb2 = new TokenEmbedding(5, 3, false);
        assertNotNull(emb2.weight, "emb2.weight should not be null");
        assertTensorShape(emb2.weight, 5, 3, "emb2.weight shape");
        assertFalse(emb2.weight.hasGrad(), "emb2.weight should not require grad");
        assertFalse(emb2.weight.isParameter(), "emb2.weight should not be a parameter by default if not grad");
    }

    @Test
    void testTokenEmbeddingForward() {
        TokenEmbedding emb = new TokenEmbedding(3, 2, false);
        emb.weight.data = Tensor.getDoubles(new double[][]{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}, emb.weight.data);


        Tensor tokenIds1 = Tensor.matrix(new double[][]{{0}, {2}}); // Column vector
        Tensor output1 = emb.forward(tokenIds1);
        assertTensorShape(output1, 2, 2, "output1 shape");
        assertTensorDataEquals(new double[][]{{0.1, 0.2}, {0.5, 0.6}}, output1, "output1 data");

        Tensor tokenIds2 = Tensor.vector(1, 0); // Row vector (interpreted as list of IDs)
        Tensor output2 = emb.forward(tokenIds2);
        assertTensorShape(output2, 2, 2, "output2 shape"); // numTokens = 2
        assertTensorDataEquals(new double[][]{{0.3, 0.4}, {0.1, 0.2}}, output2, "output2 data");
    }

    @Test
    void testTokenEmbeddingForwardEmpty() {
        TokenEmbedding emb = new TokenEmbedding(3, 2, false);
        Tensor tokenIdsEmpty = Tensor.zeros(0, 1);
        Tensor outputEmpty = emb.forward(tokenIdsEmpty);
        assertTensorShape(outputEmpty, 0, emb.embeddingDim, "outputEmpty shape");
    }

    @Test
    void testTokenEmbeddingOutOfBounds() {
        TokenEmbedding emb = new TokenEmbedding(3, 2, false);
        Tensor tokenIdsInvalidUpper = Tensor.scalar(3);
        assertThrows(IllegalArgumentException.class, () -> emb.forward(tokenIdsInvalidUpper), "Upper bound check");

        Tensor tokenIdsInvalidLower = Tensor.scalar(-1);
        assertThrows(IllegalArgumentException.class, () -> emb.forward(tokenIdsInvalidLower), "Lower bound check");
    }

    @Test
    void testTokenEmbeddingGradient() {
        TokenEmbedding emb = new TokenEmbedding(3, 2, true);
        // emb.weight initialized randomly, that's fine.

        Tensor tokenIds = Tensor.matrix(new double[][]{{0}, {1}, {0}}); // Token 0 is repeated
        Tensor output = emb.forward(tokenIds); // Shape [3, 2]
        assertTensorShape(output, 3, 2, "Output shape");
        assertTrue(output.hasGrad(), "Output should require grad because weight does");

        Tensor loss = output.sum(); // Dummy loss
        loss.minimize(); // Uses Tensor.GLOBAL_LEARNING_RATE and new SGD() by default

        assertNotNull(emb.weight.grad(), "emb.weight.grad should not be null");
        assertTensorShape(emb.weight.grad(), 3, 2, "emb.weight.grad shape");

        // Gradient for output.sum() is 1.0 for each element of output.
        // Token 0 (emb.weight row 0) contributes to output row 0 and output row 2.
        // Token 1 (emb.weight row 1) contributes to output row 1.
        // Token 2 (emb.weight row 2) is unused.
        double[] expectedGradW0 = {1.0 + 1.0, 1.0 + 1.0}; // Grad for token 0, for each embedding dim
        double[] expectedGradW1 = {1.0, 1.0};         // Grad for token 1
        double[] expectedGradW2 = {0.0, 0.0};         // Grad for token 2

        assertArrayApproximatelyEquals(expectedGradW0, emb.weight.grad().slice(0,1,0,2).array(), "Grad for token 0 (row 0)");
        assertArrayApproximatelyEquals(expectedGradW1, emb.weight.grad().slice(1,2,0,2).array(), "Grad for token 1 (row 1)");
        assertArrayApproximatelyEquals(expectedGradW2, emb.weight.grad().slice(2,3,0,2).array(), "Grad for token 2 (row 2)");
    }

    // --- FactorizedTokenEmbedding Tests ---
    @Test
    void testFactorizedTokenEmbeddingInitialization() {
        int vocabSize = 10, embeddingDimE = 4, hiddenDimH = 6;

        FactorizedTokenEmbedding fte1 = new FactorizedTokenEmbedding(vocabSize, embeddingDimE, hiddenDimH, true);
        assertNotNull(fte1.embeddingLayerE.weights, "fte1.embeddingLayerE.weights should not be null");
        assertTensorShape(fte1.embeddingLayerE.weights, vocabSize, embeddingDimE, "fte1.embeddingLayerE.weights shape");
        assertTrue(fte1.embeddingLayerE.weights.hasGrad(), "fte1.embeddingLayerE.weights should require grad");
        assertTrue(fte1.embeddingLayerE.weights.isParameter(), "fte1.embeddingLayerE.weights should be a parameter");

        assertNotNull(fte1.projectionLayerH.weight, "fte1.projectionLayerH.weight should not be null");
        assertTensorShape(fte1.projectionLayerH.weight, embeddingDimE, hiddenDimH, "fte1.projectionLayerH.weight shape");
        assertTrue(fte1.projectionLayerH.weight.hasGrad(), "fte1.projectionLayerH.weight should require grad");
        assertTrue(fte1.projectionLayerH.weight.isParameter(), "fte1.projectionLayerH.weight should be a parameter");
        assertNotNull(fte1.projectionLayerH.bias, "fte1.projectionLayerH.bias should not be null");
        assertTensorShape(fte1.projectionLayerH.bias, 1, hiddenDimH, "fte1.projectionLayerH.bias shape");
        assertTrue(fte1.projectionLayerH.bias.hasGrad(), "fte1.projectionLayerH.bias should require grad");
        assertTrue(fte1.projectionLayerH.bias.isParameter(), "fte1.projectionLayerH.bias should be a parameter");

        FactorizedTokenEmbedding fte2 = new FactorizedTokenEmbedding(vocabSize, embeddingDimE, hiddenDimH, false);
        assertNotNull(fte2.embeddingLayerE.weights, "fte2.embeddingLayerE.weights should not be null");
        assertFalse(fte2.embeddingLayerE.weights.hasGrad(), "fte2.embeddingLayerE.weights should not require grad");
        assertFalse(fte2.embeddingLayerE.weights.isParameter(), "fte2.embeddingLayerE.weights should not be a parameter");

        assertNotNull(fte2.projectionLayerH.weight, "fte2.projectionLayerH.weight should not be null");
        assertFalse(fte2.projectionLayerH.weight.hasGrad(), "fte2.projectionLayerH.weight should not require grad (inherited from Linear)");
        // projectionLayerH is a Models.Linear, which initializes its parameters as grad(true) by default if not specified otherwise.
        // The 'requiresGrad' in FactorizedTokenEmbedding is only passed to TokenEmbedding_E.
        // This is a slight inconsistency. For this test, we'll assume Models.Linear defaults to grad=true for its params.
        // If FactorizedTokenEmbedding is meant to control projectionLayerH's grad status, its constructor needs modification.
        // For now, asserting based on current Models.Linear behavior.
        assertTrue(fte2.projectionLayerH.weight.isParameter(), "fte2.projectionLayerH.weight should be parameter by default in Linear");
        assertTrue(fte2.projectionLayerH.bias.isParameter(), "fte2.projectionLayerH.bias should be parameter by default in Linear");
    }

    @Test
    void testFactorizedTokenEmbeddingForward() {
        int vocabSize = 3, embeddingDimE = 2, hiddenDimH = 2;
        FactorizedTokenEmbedding fte = new FactorizedTokenEmbedding(vocabSize, embeddingDimE, hiddenDimH, false);

        // Manually set weights for TokenEmbedding_E
        // E_weights: [vocabSize, embeddingDimE] = [3, 2]
        fte.embeddingLayerE.weights.data = Tensor.getDoubles(new double[][]{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}, fte.embeddingLayerE.weights.data);

        // Manually set weights for projectionLayerH (Models.Linear)
        // H_weight: [embeddingDimE, hiddenDimH] = [2, 2]
        fte.projectionLayerH.weight.data = Tensor.getDoubles(new double[][]{{1.0, 2.0}, {3.0, 4.0}}, fte.projectionLayerH.weight.data);
        // H_bias: [1, hiddenDimH] = [1, 2]
        fte.projectionLayerH.bias.data = Tensor.getDoubles(new double[]{0.5, -0.5}, fte.projectionLayerH.bias.data);

        Tensor tokenIds = Tensor.matrix(new double[][]{{0}, {2}}); // Use token 0 and token 2
        // Expected intermediate embeddings (output of embeddingLayerE):
        // Token 0: [0.1, 0.2]
        // Token 2: [0.5, 0.6]
        // So, embeddedE = [[0.1, 0.2], [0.5, 0.6]]

        // Expected final output (output of projectionLayerH.apply(embeddedE)):
        // For [0.1, 0.2]:
        // matmul: [0.1, 0.2] @ [[1,2],[3,4]].T = [0.1, 0.2] @ [[1,3],[2,4]]
        //         = [0.1*1 + 0.2*3, 0.1*2 + 0.2*4] = [0.1+0.6, 0.2+0.8] = [0.7, 1.0]
        // add bias: [0.7, 1.0] + [0.5, -0.5] = [1.2, 0.5]
        // For [0.5, 0.6]:
        // matmul: [0.5, 0.6] @ [[1,3],[2,4]]
        //         = [0.5*1 + 0.6*3, 0.5*2 + 0.6*4] = [0.5+1.8, 1.0+2.4] = [2.3, 3.4]
        // add bias: [2.3, 3.4] + [0.5, -0.5] = [2.8, 2.9]
        double[][] expectedOutputData = {{1.2, 0.5}, {2.8, 2.9}};

        Tensor output = fte.forward(tokenIds);
        assertTensorShape(output, 2, hiddenDimH, "FTE Forward: Output shape");
        assertTensorDataEquals(expectedOutputData, output, "FTE Forward: Output data");
    }

    @Test
    void testFactorizedTokenEmbeddingForwardEmpty() {
        FactorizedTokenEmbedding fte = new FactorizedTokenEmbedding(3, 2, 2, false);
        Tensor tokenIdsEmpty = Tensor.zeros(0, 1); // Or Tensor.vector() if that makes an empty tensor
        Tensor outputEmpty = fte.forward(tokenIdsEmpty);
        assertTensorShape(outputEmpty, 0, fte.hiddenDimH, "FTE ForwardEmpty: outputEmpty shape");
    }

    @Test
    void testFactorizedTokenEmbeddingOutOfBounds() {
        FactorizedTokenEmbedding fte = new FactorizedTokenEmbedding(3, 2, 2, false);
        Tensor tokenIdsInvalidUpper = Tensor.scalar(3); // Vocab is 0, 1, 2
        assertThrows(IllegalArgumentException.class, () -> fte.forward(tokenIdsInvalidUpper), "FTE OutOfBounds: Upper bound check");

        Tensor tokenIdsInvalidLower = Tensor.scalar(-1);
        assertThrows(IllegalArgumentException.class, () -> fte.forward(tokenIdsInvalidLower), "FTE OutOfBounds: Lower bound check");
    }

    @Test
    void testFactorizedTokenEmbeddingGradient() {
        int vocabSize = 3, embeddingDimE = 2, hiddenDimH = 2;
        FactorizedTokenEmbedding fte = new FactorizedTokenEmbedding(vocabSize, embeddingDimE, hiddenDimH, true);
        // Weights are randomly initialized and require grad.

        Tensor tokenIds = Tensor.matrix(new double[][]{{0}, {1}});
        Tensor output = fte.forward(tokenIds);
        assertTensorShape(output, 2, hiddenDimH, "FTE Gradient: Output shape");
        assertTrue(output.hasGrad(), "FTE Gradient: Output should require grad");

        Tensor loss = output.sum();
        loss.minimize();

        assertNotNull(fte.embeddingLayerE.weights.grad(), "FTE embeddingLayerE.weights.grad should not be null");
        assertTensorShape(fte.embeddingLayerE.weights.grad(), vocabSize, embeddingDimE, "FTE embeddingLayerE.weights.grad shape");

        assertNotNull(fte.projectionLayerH.weight.grad(), "FTE projectionLayerH.weight.grad should not be null");
        assertTensorShape(fte.projectionLayerH.weight.grad(), embeddingDimE, hiddenDimH, "FTE projectionLayerH.weight.grad shape");
        assertNotNull(fte.projectionLayerH.bias.grad(), "FTE projectionLayerH.bias.grad should not be null");
        assertTensorShape(fte.projectionLayerH.bias.grad(), 1, hiddenDimH, "FTE projectionLayerH.bias.grad shape");

        // Check if gradients are non-zero (sum of abs values)
        assertTrue(fte.embeddingLayerE.weights.grad().abs().sum().scalar() > EPS, "FTE embeddingLayerE.weights.grad has non-zero values");
        assertTrue(fte.projectionLayerH.weight.grad().abs().sum().scalar() > EPS, "FTE projectionLayerH.weight.grad has non-zero values");
        assertTrue(fte.projectionLayerH.bias.grad().abs().sum().scalar() > EPS, "FTE projectionLayerH.bias.grad has non-zero values");
    }


    // --- PositionalEncoding Tests ---
    @Test
    void testPositionalEncodingInitializationLearned() {
        PositionalEncoding pe = new PositionalEncoding(10, 4, true, true);
        assertNotNull(pe.encodings, "pe.encodings should not be null (learned)");
        assertTensorShape(pe.encodings, 10, 4, "pe.encodings shape (learned)");
        assertTrue(pe.encodings.hasGrad(), "pe.encodings should require grad (learned)");
        assertTrue(pe.encodings.isParameter(), "pe.encodings should be a parameter (learned)");
    }

    @Test
    void testPositionalEncodingInitializationSinusoidal() {
        PositionalEncoding pe = new PositionalEncoding(3, 4, false, false); // learned=false
        assertNotNull(pe.encodings, "pe.encodings should not be null (sinusoidal)");
        assertTensorShape(pe.encodings, 3, 4, "pe.encodings shape (sinusoidal)");
        assertFalse(pe.encodings.hasGrad(), "pe.encodings should not require grad (sinusoidal)");

        // Values from problem description, PE(pos, i)
        // PE(pos, 2i) = sin(pos / 10000^(2i / embDim))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i / embDim))
        // Corrected formula used in PositionalEncoding.java:
        // divTerm = Math.pow(10000.0, (double)(2 * (j / 2)) / embDim); angle = (double)pos / divTerm;
        // if j % 2 == 0 -> sin(angle), else cos(angle)

        // Pos 0: angle for (0,1) = 0 / 10000^(0/4) = 0. angle for (2,3) = 0 / 10000^(2/4) = 0.
        assertEquals(Math.sin(0), pe.encodings.data(0,0), EPS, "PE(0,0)"); // sin(0)
        assertEquals(Math.cos(0), pe.encodings.data(0,1), EPS, "PE(0,1)"); // cos(0)
        assertEquals(Math.sin(0), pe.encodings.data(0,2), EPS, "PE(0,2)"); // sin(0)
        assertEquals(Math.cos(0), pe.encodings.data(0,3), EPS, "PE(0,3)"); // cos(0)

        // Pos 1, embDim=4
        double divTerm_01 = Math.pow(10000.0, (double)(2*(0/2))/4.0); // = 10000^0 = 1
        double angle_01 = 1.0 / divTerm_01; // = 1
        double divTerm_23 = Math.pow(10000.0, (double)(2*(2/2))/4.0); // = 10000^(2/4) = 10000^0.5 = 100
        double angle_23 = 1.0 / divTerm_23; // = 0.01

        assertEquals(Math.sin(angle_01), pe.encodings.data(1,0), EPS, "PE(1,0)");
        assertEquals(Math.cos(angle_01), pe.encodings.data(1,1), EPS, "PE(1,1)");
        assertEquals(Math.sin(angle_23), pe.encodings.data(1,2), EPS, "PE(1,2)");
        assertEquals(Math.cos(angle_23), pe.encodings.data(1,3), EPS, "PE(1,3)");
    }

    @Test
    void testPositionalEncodingForwardSinusoidal() {
        PositionalEncoding pe = new PositionalEncoding(5, 4, false, false);
        Tensor inputEmbeds = Tensor.zeros(3, 4); // seqLen=3
        Tensor output = pe.forward(inputEmbeds);
        assertTensorShape(output, 3, 4, "Output shape sinusoidal forward");

        Tensor expectedOutput = pe.encodings.slice(0, 3, 0, 4);
        assertTensorDataEquals(expectedOutput.data2D(), output, "Output data sinusoidal forward");
    }

    @Test
    void testPositionalEncodingForwardLearned() {
        PositionalEncoding pe = new PositionalEncoding(5, 2, true, false); // learned=true
        double[][] peData = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}, {0.7, 0.8}, {0.9, 1.0}};
        pe.encodings.data = Tensor.getDoubles(peData, pe.encodings.data);


        Tensor inputEmbeds = Tensor.ones(2, 2); // seqLen=2
        Tensor output = pe.forward(inputEmbeds);
        assertTensorShape(output, 2, 2, "Output shape learned forward");

        double[][] expectedOutputData = {
            {1.0 + 0.1, 1.0 + 0.2}, // 1.1, 1.2
            {1.0 + 0.3, 1.0 + 0.4}  // 1.3, 1.4
        };
        assertTensorDataEquals(expectedOutputData, output, "Output data learned forward");
    }

    @Test
    void testPositionalEncodingForwardSeqLenTooLong() {
        PositionalEncoding pe = new PositionalEncoding(5, 4, false, false);
        Tensor inputEmbeds = Tensor.zeros(6, 4); // seqLen=6, maxSeqLen=5
        assertThrows(IllegalArgumentException.class, () -> pe.forward(inputEmbeds), "Sequence length too long");
    }

    @Test
    void testPositionalEncodingDimMismatch() {
        PositionalEncoding pe = new PositionalEncoding(5, 4, false, false);
        Tensor inputEmbeds = Tensor.zeros(3, 3); // embeddingDim=3, PE expects 4
        assertThrows(IllegalArgumentException.class, () -> pe.forward(inputEmbeds), "Embedding dimension mismatch");
    }


    @Test
    void testPositionalEncodingGradientLearned() {
        PositionalEncoding pe = new PositionalEncoding(5, 2, true, true); // learned=true, requiresGrad=true
        Tensor inputEmbeds = Tensor.randGaussian(3, 2, 0.1).grad(true); // seqLen=3

        Tensor output = pe.forward(inputEmbeds);
        assertTensorShape(output, 3, 2, "Output shape for gradient test");
        assertTrue(output.hasGrad(), "Output should require grad for gradient test");

        Tensor loss = output.sum();
        loss.minimize();

        assertNotNull(pe.encodings.grad(), "pe.encodings.grad should not be null");
        assertTensorShape(pe.encodings.grad(), 5, 2, "pe.encodings.grad shape");

        // Grads for output.sum() are 1.0.
        // output = inputEmbeds.add(pe_slice)
        // So, grad for pe_slice elements should be 1.0.
        // pe_slice is encodings.slice(0, 3, 0, 2)
        double[] gradValues = pe.encodings.grad().array();
        int numCols = pe.encodings.cols();

        // Check first 3 rows (involved in forward pass)
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < numCols; c++) {
                assertEquals(1.0, gradValues[r * numCols + c], EPS,
                             "Gradient for pe.encodings[" + r + "," + c + "]");
            }
        }
        // Check last 2 rows (not involved)
        for (int r = 3; r < 5; r++) {
            for (int c = 0; c < numCols; c++) {
                assertEquals(0.0, gradValues[r * numCols + c], EPS,
                             "Gradient for pe.encodings[" + r + "," + c + "] (should be zero)");
            }
        }

        assertNotNull(inputEmbeds.grad(), "inputEmbeds.grad should not be null");
        assertTensorShape(inputEmbeds.grad(), 3, 2, "inputEmbeds.grad shape");
        double[] inputGradValues = inputEmbeds.grad().array();
        for(int i=0; i < inputGradValues.length; i++) {
            assertEquals(1.0, inputGradValues[i], EPS, "Gradient for inputEmbeds at index " + i);
        }
    }

    // --- RotaryPositionalEncoding Tests ---
    @Test
    void testRotaryPositionalEncodingPos0Identity() {
        int dim = 4, max_seq_len = 10;
        double theta_base = 10000.0;
        RotaryPositionalEncoding rope = new RotaryPositionalEncoding(dim, max_seq_len, theta_base);

        int seq_len = 5;
        Tensor q_or_k = Tensor.randGaussian(seq_len, dim, 0.1);
        Tensor q_or_k_copy = q_or_k.copy();

        // Apply RoPE with position_offset = 0
        // For position 0, the rotation should be an identity transformation.
        // However, apply_rotary_pos_emb applies it for positions [offset, offset + seq_len -1]
        // So, if seq_len > 1 and offset = 0, only the first row (pos 0) will be identity.
        // The test asks to assert that `rotated` is element-wise equal to `q_or_k_copy`.
        // This will only be true if all positions processed by RoPE are effectively 0.
        // Let's test for a single token at position 0.
        Tensor single_token_at_pos0 = q_or_k.slice(0, 1, 0, dim); // seq_len = 1
        Tensor single_token_copy = single_token_at_pos0.copy();
        Tensor rotated_single_token = rope.apply_rotary_pos_emb(single_token_at_pos0, 1, 0);

        assertTensorDataEquals(single_token_copy.data2D(), rotated_single_token, "RoPE at pos 0 should be identity for that token");

        // If we apply to the whole sequence with offset 0, only the first row should match original.
        Tensor rotated_full = rope.apply_rotary_pos_emb(q_or_k.copy(), seq_len, 0);
        assertTensorDataEquals(q_or_k.slice(0,1,0,dim).data2D(), rotated_full.slice(0,1,0,dim), "First row (pos 0) of RoPE'd tensor should be identity");

        // If seq_len > 1, the other rows should differ from original if RoPE is working.
        if (seq_len > 1 && dim > 0) {
            boolean otherRowsDiffer = false;
            for (int i = 1; i < seq_len; i++) {
                Tensor original_row = q_or_k.slice(i, i + 1, 0, dim);
                Tensor rotated_row = rotated_full.slice(i, i + 1, 0, dim);
                for(int j=0; j < original_row.volume(); j++) {
                    if (Math.abs(original_row.array()[j] - rotated_row.array()[j]) > EPS) {
                        otherRowsDiffer = true;
                        break;
                    }
                }
                if (otherRowsDiffer) break;
            }
            assertTrue(otherRowsDiffer, "RoPE at subsequent positions ( > 0) should change the input values.");
        }
    }

    @Test
    void testRotaryPositionalEncodingKnownValues() {
        // Based on example in RotaryPositionalEncoding.main()
        // For pos=1, inv_freq[0] = 1.0 / (10000^(0/2)) = 1.0 -> this seems wrong calculation in comment.
        // inv_freq[i] = 1.0 / Math.pow(theta_base, (double) (2 * i) / dim);
        // For dim=2, i=0: inv_freq[0] = 1.0 / Math.pow(10000, 0/2) = 1.0 / 1.0 = 1.0
        // t (positions) = [[1]] (for position_offset=1, seq_len=1)
        // freqs = t.matmul(inv_freq) = [[1]] @ [[1.0]] = [[1.0]] (shape [1,1])
        // emb_val_for_pair = freqs.slice(0,1,0,1) = [[1.0]]
        // cos(1.0) = 0.54030230586
        // sin(1.0) = 0.8414709848
        // x_j = [1], x_j_plus_1 = [2]
        // out_col_j = x_j * cos_val - x_j_plus_1 * sin_val = 1*0.5403 - 2*0.8414 = 0.5403 - 1.6829 = -1.1426
        // out_col_j_plus_1 = x_j * sin_val + x_j_plus_1 * cos_val = 1*0.8414 + 2*0.5403 = 0.8414 + 1.0806 = 1.9220
        RotaryPositionalEncoding rope = new RotaryPositionalEncoding(2, 10, 10000.0);
        Tensor input = Tensor.matrix(new double[][]{{1, 2}}); // seq_len=1, dim=2
        Tensor output = rope.apply_rotary_pos_emb(input, 1, 1); // position_offset=1 to test position 1

        double[][] expectedOutputData = {{-1.1426046117, 1.922041939}}; // More precise values
        assertTensorDataEquals(expectedOutputData, output, "RoPE known values (dim=2, pos=1)");
    }


    // --- AttentionMechanisms.scaledDotProductAttention Tests ---
    @Test
    void testSDPAttentionBasic() {
        Tensor q = Tensor.matrix(new double[][]{{1, 0}}); // [1, 2]
        Tensor k = Tensor.matrix(new double[][]{{1, 0}}); // [1, 2]
        Tensor v = Tensor.matrix(new double[][]{{0, 1}}); // [1, 2]
        // d_k = 2. scores = q @ k.T = [[1]] * [[1],[0]] = [[1]]. scaledScores = [[1 / sqrt(2)]].
        // attentionWeights = softmax([[1/sqrt(2)]]) = [[1]]. context = [[1]] @ [[0,1]] = [[0,1]].
        Tensor context = AttentionMechanisms.scaledDotProductAttention(q, k, v, null, false);
        assertTensorShape(context, 1, 2, "SDPA Basic: Output shape");
        assertTensorDataEquals(new double[][]{{0, 1}}, context, "SDPA Basic: Output data");
    }

    @Test
    void testSDPAttentionWithMask() {
        Tensor q = Tensor.matrix(new double[][]{{10, 0}, {0, 10}}); // Using large values to make softmax sharper
        Tensor k = Tensor.matrix(new double[][]{{10, 0}, {0, 10}});
        Tensor v = Tensor.matrix(new double[][]{{1, 2}, {3, 4}});
        Tensor mask = Tensor.matrix(new double[][]{{0, -1e9}, {0, 0}}); // Q0 cannot attend to K1

        // Q0: Query is [10,0]. Keys are K0=[10,0], K1=[0,10].
        // Scores for Q0: (Q0.K0^T)/sqrt(d_k) = (100)/sqrt(2) approx 70.7. (Q0.K1^T)/sqrt(d_k) = 0.
        // Masked scores for Q0: [70.7, -1e9]. Softmax -> [~1, ~0].
        // Context for Q0: ~1 * v[0] + ~0 * v[1] = v[0] = [1,2].

        // Q1: Query is [0,10]. Keys are K0=[10,0], K1=[0,10].
        // Scores for Q1: (Q1.K0^T)/sqrt(d_k) = 0. (Q1.K1^T)/sqrt(d_k) = (100)/sqrt(2) approx 70.7.
        // Masked scores for Q1: [0, 70.7] (mask is [0,0] for this row). Softmax -> [~0, ~1].
        // Context for Q1: ~0 * v[0] + ~1 * v[1] = v[1] = [3,4].
        // Wait, if Q1 attends to K0 and K1 equally (e.g. Q1=[10,10], K0=[10,0], K1=[0,10])
        // Then scores might be [~35, ~35]. Softmax [0.5, 0.5]. Output 0.5*v0 + 0.5*v1.
        // With Q1=[0,10], K0=[10,0], K1=[0,10]. Scores before mask: [0, 70.7]. Mask [0,0]. Softmax [~0, ~1]. Output = v[1].
        // The example calculation was: Query 1: attends to Key 0 and Key 1. Weights `[~0.5, ~0.5]`. This is if Q1 was like [10,10] and K0,K1 were orthogonal.
        // For Q1=[0,10], K0=[10,0], K1=[0,10]: Scores are [0*10+10*0, 0*0+10*10] = [0, 100]. Scaled: [0, 100/sqrt(2)]. Mask: [0,0].
        // Softmax([0, 70.7]) -> [small_val, large_val] -> approx [0,1]. So Q1 context is approx v[1].

        Tensor context = AttentionMechanisms.scaledDotProductAttention(q, k, v, mask, true); // debug printing true
        assertTensorShape(context, 2, 2, "SDPA Mask: Output shape");
        // Expected: [[1,2], [3,4]]
        assertTensorDataEquals(new double[][]{{1, 2}, {3, 4}}, context, "SDPA Mask: Output data");
    }

    @Test
    void testSDPAttentionGradient() {
        Tensor q = Tensor.matrix(new double[][]{{1, 0}}).grad(true);
        Tensor k = Tensor.matrix(new double[][]{{1, 0}}).grad(true);
        Tensor v = Tensor.matrix(new double[][]{{0, 1}}).grad(true);

        Tensor output = AttentionMechanisms.scaledDotProductAttention(q, k, v, null, false);
        Tensor loss = output.sum();
        loss.minimize();

        assertNotNull(q.grad(), "q.grad should not be null");
        assertNotNull(k.grad(), "k.grad should not be null");
        assertNotNull(v.grad(), "v.grad should not be null");
        // Exact gradient values depend on the chain rule through softmax and matmuls, complex to verify manually here.
        // Non-null check is a good first step.
        assertTrue(q.grad().sum().scalar() != 0, "q.grad sum should be non-zero");
        // k.grad can be zero if softmax output for it is zero, or if its contribution to loss is zero.
        // v.grad should be non-zero as it directly contributes to output.
        assertTrue(v.grad().sum().scalar() != 0, "v.grad sum should be non-zero");
    }


    // --- MultiHeadAttention Tests ---
    @Test
    void testMHAInitialization() {
        MultiHeadAttention mha = new MultiHeadAttention(4, 2, true, true, false);
        assertEquals(2, mha.d_k, "MHA d_k initialization");
        assertTrue(mha.wq.weight.hasGrad(), "MHA wq.weight grad status");
        assertTrue(mha.wk.weight.hasGrad(), "MHA wk.weight grad status");
        assertTrue(mha.wv.weight.hasGrad(), "MHA wv.weight grad status");
        assertTrue(mha.wo.weight.hasGrad(), "MHA wo.weight grad status");
        assertTrue(mha.wq.bias.hasGrad(), "MHA wq.bias grad status");
    }

    @Test
    void testMHAForwardPass() {
        int dModel = 4, numHeads = 2, seqLen = 3;
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads, true, false, false);
        Tensor query = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor key = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor value = Tensor.randGaussian(seqLen, dModel, 0.1);

        Tensor output = mha.forward(query, key, value, null);
        assertTensorShape(output, seqLen, dModel, "MHA Forward: Output shape");
    }

    @Test
    void testMHAForwardWithMask() {
        int dModel = 4, numHeads = 2, seqLen = 3;
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads, true, false, true); // debug true
        Tensor query = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor key = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor value = Tensor.randGaussian(seqLen, dModel, 0.1);
        // Causal mask for seqLen=3. Mask is [seq_len_q, seq_len_k]
        Tensor mask = Tensor.matrix(new double[][]{
            {0, -1e9, -1e9},
            {0, 0, -1e9},
            {0, 0, 0}
        });

        Tensor output = mha.forward(query, key, value, mask);
        assertTensorShape(output, seqLen, dModel, "MHA Masked Forward: Output shape");
    }

    @Test
    void testMHAGradient() {
        int dModel = 2, numHeads = 1, seqLen = 1; // Simplest case: 1 head, 1 item in sequence
        MultiHeadAttention mha = new MultiHeadAttention(dModel, numHeads, true, true, false, null); // No RoPE

        Tensor query = Tensor.matrix(new double[][]{{0.1, 0.2}}).grad(true);
        Tensor key = Tensor.matrix(new double[][]{{0.3, 0.4}}).grad(true);
        Tensor value = Tensor.matrix(new double[][]{{0.5, 0.6}}).grad(true);

        Tensor output = mha.forward(query, key, value, null);
        Tensor loss = output.sum(); // Sum all elements in the output tensor
        loss.minimize();

        assertNotNull(mha.wq.weight.grad(), "MHA wq.weight.grad");
        assertNotNull(mha.wk.weight.grad(), "MHA wk.weight.grad");
        assertNotNull(mha.wv.weight.grad(), "MHA wv.weight.grad");
        assertNotNull(mha.wo.weight.grad(), "MHA wo.weight.grad");
        assertNotNull(mha.wq.bias.grad(), "MHA wq.bias.grad");


        assertNotNull(query.grad(), "MHA query.grad");
        assertNotNull(key.grad(), "MHA key.grad");
        assertNotNull(value.grad(), "MHA value.grad");

        // Check if any gradient is actually non-zero (sum of absolute values)
        assertTrue(mha.wq.weight.grad().abs().sum().scalar() > EPS, "MHA wq.weight.grad has non-zero values");
        assertTrue(mha.wk.weight.grad().abs().sum().scalar() > EPS, "MHA wk.weight.grad has non-zero values");
        assertTrue(mha.wv.weight.grad().abs().sum().scalar() > EPS, "MHA wv.weight.grad has non-zero values");
        assertTrue(mha.wo.weight.grad().abs().sum().scalar() > EPS, "MHA wo.weight.grad has non-zero values");

        assertTrue(query.grad().abs().sum().scalar() > EPS, "MHA query.grad has non-zero values");
        // key.grad can sometimes be zero or very small if its influence on output is nullified by softmax
        // assertTrue(key.grad().abs().sum().scalar() > EPS, "MHA key.grad has non-zero values");
        assertTrue(value.grad().abs().sum().scalar() > EPS, "MHA value.grad has non-zero values");
    }

    @Test
    void testMHAForwardWithRoPE() {
        int dModel = 4, numHeads = 2, seqLen = 3;
        boolean bias = true, grad = false, debug = false;
        int d_k = dModel / numHeads; // Should be 2

        RotaryPositionalEncoding rope = new RotaryPositionalEncoding(d_k, seqLen, 10000.0);
        MultiHeadAttention mhaWithRope = new MultiHeadAttention(dModel, numHeads, bias, grad, debug, rope);
        MultiHeadAttention mhaWithoutRope = new MultiHeadAttention(dModel, numHeads, bias, grad, debug, null);

        // Ensure weights are identical for a fair comparison
        mhaWithRope.wq.weight.data = mhaWithoutRope.wq.weight.data.clone();
        mhaWithRope.wk.weight.data = mhaWithoutRope.wk.weight.data.clone();
        mhaWithRope.wv.weight.data = mhaWithoutRope.wv.weight.data.clone();
        mhaWithRope.wo.weight.data = mhaWithoutRope.wo.weight.data.clone();
        if (bias) {
            mhaWithRope.wq.bias.data = mhaWithoutRope.wq.bias.data.clone();
            mhaWithRope.wk.bias.data = mhaWithoutRope.wk.bias.data.clone();
            mhaWithRope.wv.bias.data = mhaWithoutRope.wv.bias.data.clone();
            mhaWithRope.wo.bias.data = mhaWithoutRope.wo.bias.data.clone();
        }


        Tensor query = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor key = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor value = Tensor.randGaussian(seqLen, dModel, 0.1);

        Tensor outputWithRope = mhaWithRope.forward(query.copy(), key.copy(), value.copy(), null);
        Tensor outputWithoutRope = mhaWithoutRope.forward(query.copy(), key.copy(), value.copy(), null);

        assertTensorShape(outputWithRope, seqLen, dModel, "MHA with RoPE: Output shape");
        assertTensorShape(outputWithoutRope, seqLen, dModel, "MHA without RoPE: Output shape");

        // If RoPE had an effect, the outputs should differ (unless all weights are zero, or by sheer chance)
        // RoPE at pos 0 is identity, so if seqLen=1, outputs should be same.
        // If seqLen > 1, RoPE is applied with offset 0, so pos 0 is identity, others are rotated.
        // Thus, if seqLen > 1, outputs should differ.
        boolean outputsDiffer = false;
        for (int i = 0; i < outputWithRope.volume(); i++) {
            if (Math.abs(outputWithRope.array()[i] - outputWithoutRope.array()[i]) > EPS) {
                outputsDiffer = true;
                break;
            }
        }
        if (seqLen > 1 && d_k > 0) {
             assertTrue(outputsDiffer, "MHA outputs with and without RoPE should differ for seqLen > 1");
        } else if (seqLen == 1 && d_k > 0) {
            assertFalse(outputsDiffer, "MHA outputs with and without RoPE should be identical for seqLen == 1 (RoPE is identity at pos 0)");
        }
    }

     @Test
    void testMHAForwardWithRoPE_Pos0IdentityEffect() {
        int seqLen = 1;
        int dModel = 4;
        int numHeads = 2;
        int d_k = dModel / numHeads; // 2
        boolean bias = true, grad = false, debug = false;

        RotaryPositionalEncoding rope = new RotaryPositionalEncoding(d_k, seqLen, 10000.0);
        MultiHeadAttention mhaWithRope = new MultiHeadAttention(dModel, numHeads, bias, grad, debug, rope);
        MultiHeadAttention mhaWithoutRope = new MultiHeadAttention(dModel, numHeads, bias, grad, debug, null);

        // Critical: Ensure weights are IDENTICAL for this test.
        // Simplest: copy from one to the other after random initialization.
        mhaWithRope.wq.weight.data = mhaWithoutRope.wq.weight.data.clone();
        mhaWithRope.wk.weight.data = mhaWithoutRope.wk.weight.data.clone();
        mhaWithRope.wv.weight.data = mhaWithoutRope.wv.weight.data.clone();
        mhaWithRope.wo.weight.data = mhaWithoutRope.wo.weight.data.clone();
        if (bias) {
            mhaWithRope.wq.bias.data = mhaWithoutRope.wq.bias.data.clone();
            mhaWithRope.wk.bias.data = mhaWithoutRope.wk.bias.data.clone();
            mhaWithRope.wv.bias.data = mhaWithoutRope.wv.bias.data.clone();
            mhaWithRope.wo.bias.data = mhaWithoutRope.wo.bias.data.clone();
        }


        Tensor query = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor key = Tensor.randGaussian(seqLen, dModel, 0.1);
        Tensor value = Tensor.randGaussian(seqLen, dModel, 0.1);

        Tensor outputWithRope = mhaWithRope.forward(query.copy(), key.copy(), value.copy(), null);
        Tensor outputWithoutRope = mhaWithoutRope.forward(query.copy(), key.copy(), value.copy(), null);

        assertTensorDataEquals(outputWithoutRope.data2D(), outputWithRope,
                "MHA output with RoPE should be identical to without RoPE for seqLen=1 (pos 0 RoPE is identity)");
    }


    // --- FeedForwardNetwork Tests ---
    @Test
    void testFFNInitialization() {
        FeedForwardNetwork ffn1 = new FeedForwardNetwork(4, 8, Tensor::relu, 0.1f, true, true);
        assertTrue(ffn1.linear1.weight.hasGrad(), "FFN1 linear1.weight grad status");
        assertTrue(ffn1.linear1.bias.hasGrad(), "FFN1 linear1.bias grad status");
        assertTrue(ffn1.linear2.weight.hasGrad(), "FFN1 linear2.weight grad status");
        assertTrue(ffn1.linear2.bias.hasGrad(), "FFN1 linear2.bias grad status");
        assertNotNull(ffn1.dropout, "FFN1 dropout layer should exist");

        FeedForwardNetwork ffn2 = new FeedForwardNetwork(4, 8, Tensor::relu, 0.0f, false, false);
        assertFalse(ffn2.linear1.weight.hasGrad(), "FFN2 linear1.weight grad status (false)");
        assertNull(ffn2.linear1.bias, "FFN2 linear1.bias should be null"); // biasLinear = false
        assertFalse(ffn2.linear2.weight.hasGrad(), "FFN2 linear2.weight grad status (false)");
        assertNull(ffn2.linear2.bias, "FFN2 linear2.bias should be null"); // biasLinear = false
        assertNull(ffn2.dropout, "FFN2 dropout layer should be null");
    }

    @Test
    void testFFNForwardPass() {
        int dModel = 2, dff = 4, seqLen = 1;
        FeedForwardNetwork ffn = new FeedForwardNetwork(dModel, dff, Tensor::relu, 0.0f, true, false);

        // Set weights for predictable output
        // linear1: [dff, dModel] = [4,2]
        // linear2: [dModel, dff] = [2,4]
        ffn.linear1.weight.data = Tensor.getDoubles(new double[][]{{1,0},{0,1},{1,1},{0,0}}, ffn.linear1.weight.data); // dff x dModel
        ffn.linear1.bias.data = Tensor.getDoubles(new double[]{0,0,0,0}, ffn.linear1.bias.data); // dff

        ffn.linear2.weight.data = Tensor.getDoubles(new double[][]{{1,0,1,0},{0,1,0,1}}, ffn.linear2.weight.data); // dModel x dff
        ffn.linear2.bias.data = Tensor.getDoubles(new double[]{0,0}, ffn.linear2.bias.data); // dModel

        Tensor input = Tensor.matrix(new double[][]{{2, 3}}); // [seqLen, dModel]
        // linear1.apply(input): input @ linear1.weight.T + linear1.bias
        // [[2,3]] @ [[1,0,1,0],[0,1,1,0]].T = [[2,3]] @ [[1,0],[0,1],[1,1],[0,0]]
        // = [[2*1+3*0, 2*0+3*1, 2*1+3*1, 2*0+3*0]] = [[2, 3, 5, 0]]
        // After ReLU: [[2, 3, 5, 0]]
        // linear2.apply(hidden_activated): hidden_activated @ linear2.weight.T + linear2.bias
        // [[2,3,5,0]] @ [[1,0],[0,1],[1,0],[0,1]].T = [[2,3,5,0]] @ [[1,0,1,0],[0,1,0,1]]
        // = [[2*1+3*0+5*1+0*0, 2*0+3*1+5*0+0*1]] = [[2+5, 3]] = [[7, 3]]

        Tensor output = ffn.forward(input);
        assertTensorShape(output, seqLen, dModel, "FFN Forward: Output shape");
        assertTensorDataEquals(new double[][]{{7, 3}}, output, "FFN Forward: Output data");
    }

    @Test
    void testFFNForwardWithDropout() {
        int dModel = 4, dff = 8, seqLen = 10;
        FeedForwardNetwork ffn = new FeedForwardNetwork(dModel, dff, Tensor::relu, 0.5f, true, false);
        Tensor input = Tensor.ones(seqLen, dModel);

        ffn.train(true); // Enable dropout
        Tensor outputTrainTrue = ffn.forward(input.copy()); // Use copy to avoid modifying input if forward pass does in-place ops internally on rare occasion

        ffn.train(false); // Disable dropout
        Tensor outputTrainFalse = ffn.forward(input.copy());

        assertTensorShape(outputTrainTrue, seqLen, dModel, "FFN Dropout True: Output shape");
        assertTensorShape(outputTrainFalse, seqLen, dModel, "FFN Dropout False: Output shape");

        // It's hard to check exact values due to randomness.
        // Check if the sum of outputs is different, indicating dropout likely zeroed some elements.
        // Or, if input is all ones, and weights are positive, outputTrainFalse should be all positive (after ReLU).
        // outputTrainTrue might have zeros due to dropout if intermediate results become zero.
        // A simpler check: if dropout is effective, the outputs should differ.
        boolean areDifferent = false;
        double[] trueData = outputTrainTrue.array();
        double[] falseData = outputTrainFalse.array();
        for(int i=0; i<trueData.length; i++) {
            if (Math.abs(trueData[i] - falseData[i]) > EPS) {
                areDifferent = true;
                break;
            }
        }
        // This test might be flaky if, by chance, dropout doesn't change the output significantly or if all weights are zero.
        // For robust testing, one might need to check statistics or set specific weights.
        // Given dropout is random, we expect different results. If not, it might indicate dropout isn't working.
        // This test is more of a sanity check.
         if (ffn.linear1.weight.sum().scalar() != 0 && ffn.linear2.weight.sum().scalar() != 0) { // Avoid issues if weights are all zero
            assertTrue(areDifferent, "Outputs with dropout enabled and disabled should differ if dropout rate > 0 and weights are non-trivial.");
        }
    }

    @Test
    void testFFNGradient() {
        int dModel = 2, dff = 4, seqLen = 1;
        FeedForwardNetwork ffn = new FeedForwardNetwork(dModel, dff, Tensor::relu, 0.0f, true, true); // No dropout, reqGrad=true

        Tensor input = Tensor.matrix(new double[][]{{0.1, 0.2}}).grad(true);
        Tensor output = ffn.forward(input);
        Tensor loss = output.sum();
        loss.minimize();

        assertNotNull(ffn.linear1.weight.grad(), "FFN linear1.weight.grad");
        assertNotNull(ffn.linear1.bias.grad(), "FFN linear1.bias.grad");
        assertNotNull(ffn.linear2.weight.grad(), "FFN linear2.weight.grad");
        assertNotNull(ffn.linear2.bias.grad(), "FFN linear2.bias.grad");
        assertNotNull(input.grad(), "FFN input.grad");

        assertTrue(ffn.linear1.weight.grad().abs().sum().scalar() > EPS, "FFN linear1.weight.grad has non-zero values");
        assertTrue(ffn.linear2.weight.grad().abs().sum().scalar() > EPS, "FFN linear2.weight.grad has non-zero values");
        assertTrue(input.grad().abs().sum().scalar() > EPS, "FFN input.grad has non-zero values");

    }

    // --- jcog.model.LayerNorm Tests ---
    @Test
    void testLayerNormInitialization() {
        int dim = 4;
        double epsilon = 1e-5;
        LayerNorm ln = new LayerNorm(dim, epsilon);

        assertNotNull(ln.gamma, "LayerNorm gamma should not be null");
        assertNotNull(ln.beta, "LayerNorm beta should not be null");

        assertTensorShape(ln.gamma, 1, dim, "LayerNorm gamma shape");
        assertTensorShape(ln.beta, 1, dim, "LayerNorm beta shape");

        // Check initialization values (gamma=1, beta=0 by default in jcog.model.LayerNorm)
        double[] expectedGammaData = new double[dim];
        double[] expectedBetaData = new double[dim];
        for (int i = 0; i < dim; i++) {
            expectedGammaData[i] = 1.0;
            expectedBetaData[i] = 0.0;
        }
        assertArrayApproximatelyEquals(expectedGammaData, ln.gamma.array(), "LayerNorm gamma initial data");
        assertArrayApproximatelyEquals(expectedBetaData, ln.beta.array(), "LayerNorm beta initial data");

        assertTrue(ln.gamma.hasGrad(), "LayerNorm gamma should require grad by default");
        assertTrue(ln.gamma.isParameter(), "LayerNorm gamma should be a parameter by default");
        assertTrue(ln.beta.hasGrad(), "LayerNorm beta should require grad by default");
        assertTrue(ln.beta.isParameter(), "LayerNorm beta should be a parameter by default");
    }

    @Test
    void testLayerNormForwardPassKnownValues() {
        int dim = 2;
        double epsilon = 1e-5; // LayerNorm uses 1e-5f, which is float.
        LayerNorm ln = new LayerNorm(dim, epsilon);

        // Default gamma = [1,1], beta = [0,0]
        // ln.gamma.data = Tensor.getDoubles(new double[][]{{1.0, 1.0}}, ln.gamma.data); // Ensure defaults if needed
        // ln.beta.data = Tensor.getDoubles(new double[][]{{0.0, 0.0}}, ln.beta.data);

        Tensor input = Tensor.matrix(new double[][]{{1, 3}, {5, 7}});
        // Row 0: x1 = [1, 3]. Mean = 2. Var = ((1-2)^2 + (3-2)^2)/2 = 1. Std = sqrt(1+eps) approx 1.
        //        Norm = [(1-2)/1, (3-2)/1] = [-1, 1]. Output = 1*[-1,1] + 0 = [-1, 1]
        // Row 1: x2 = [5, 7]. Mean = 6. Var = ((5-6)^2 + (7-6)^2)/2 = 1. Std = sqrt(1+eps) approx 1.
        //        Norm = [(5-6)/1, (7-6)/1] = [-1, 1]. Output = 1*[-1,1] + 0 = [-1, 1]
        double[][] expectedOutputData = {{-1.0, 1.0}, {-1.0, 1.0}};

        Tensor output = ln.apply(input);
        assertTensorShape(output, 2, dim, "LayerNorm KnownValues: Output shape");
        assertTensorDataEquals(expectedOutputData, output, "LayerNorm KnownValues: Output data");
    }

    @Test
    void testLayerNormForwardPassDifferentGammaBeta() {
        int dim = 2;
        double epsilon = 1e-5;
        LayerNorm ln = new LayerNorm(dim, epsilon);

        ln.gamma.data = Tensor.getDoubles(new double[][]{{0.5, 2.0}}, ln.gamma.data);
        ln.beta.data = Tensor.getDoubles(new double[][]{{0.1, 0.2}}, ln.beta.data);

        Tensor input = Tensor.matrix(new double[][]{{1, 3}}); // Mean=2, Var=1, Norm=[-1,1]
        // Expected: gamma * norm + beta = [0.5, 2.0] * [-1, 1] + [0.1, 0.2]
        //         = [-0.5, 2.0] + [0.1, 0.2] = [-0.4, 2.2]
        double[][] expectedOutputData = {{-0.4, 2.2}};

        Tensor output = ln.apply(input);
        assertTensorShape(output, 1, dim, "LayerNorm DiffGammaBeta: Output shape");
        assertTensorDataEquals(expectedOutputData, output, "LayerNorm DiffGammaBeta: Output data");
    }

    @Test
    void testLayerNormGradient() {
        int dim = 2;
        double epsilon = 1e-5;
        LayerNorm ln = new LayerNorm(dim, epsilon); // gamma and beta are params by default

        Tensor input = Tensor.matrix(new double[][]{{1, 3}}).grad(true);
        Tensor output = ln.apply(input);
        Tensor loss = output.sum();
        loss.minimize();

        assertNotNull(ln.gamma.grad(), "LayerNorm gamma.grad should not be null");
        assertTensorShape(ln.gamma.grad(), 1, dim, "LayerNorm gamma.grad shape");
        assertTrue(ln.gamma.grad().abs().sum().scalar() > EPS, "LayerNorm gamma.grad has non-zero values");

        assertNotNull(ln.beta.grad(), "LayerNorm beta.grad should not be null");
        assertTensorShape(ln.beta.grad(), 1, dim, "LayerNorm beta.grad shape");
        assertTrue(ln.beta.grad().abs().sum().scalar() > EPS, "LayerNorm beta.grad has non-zero values");

        assertNotNull(input.grad(), "LayerNorm input.grad should not be null");
        assertTensorShape(input.grad(), 1, dim, "LayerNorm input.grad shape");
        // Input grad can be zero if its change doesn't affect the loss, e.g. if all inputs are same.
        // Here, for {{1,3}}, mean is 2, normalized is [-1,1]. If input changes slightly, output changes.
        // So input grad should be non-zero.
        assertTrue(input.grad().abs().sum().scalar() > EPS, "LayerNorm input.grad has non-zero values");
    }
}
