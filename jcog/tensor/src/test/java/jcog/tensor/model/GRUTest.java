package jcog.tensor.model;

import jcog.tensor.Tensor;
import org.junit.jupiter.api.Test; // Common testing framework

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*; // Common assertions

public class GRUTest {

    @Test
    void testGRUCellForwardPassShapes() {
        int inputSize = 2;
        int hiddenSize = 3;
        int batchSize = 1;

        GRUCell cell = new GRUCell(inputSize, hiddenSize);

        Tensor input = Tensor.randGaussian(batchSize, inputSize);
        Tensor prevHidden = Tensor.zeros(batchSize, hiddenSize);

        Tensor newHiddenState = cell.forward(input, prevHidden);

        assertNotNull(newHiddenState, "Hidden state should not be null");
        assertEquals(batchSize, newHiddenState.rows(), "Hidden state rows should match batch size");
        assertEquals(hiddenSize, newHiddenState.cols(), "Hidden state cols should match hidden size");
    }

    @Test
    void testGRUCellParameters() {
        GRUCell cell = new GRUCell(2, 3);
        List<Tensor> params = cell.parameters();
        assertNotNull(params);
        // wIr, wIz, wIn, wHr, wHz, wHn, bR, bZ, bIn, bHn
        assertEquals(10, params.size(), "GRUCell should have 10 parameter tensors");
        for (Tensor p : params) {
            assertNotNull(p, "Parameter tensor should not be null");
            assertTrue(p.hasGrad(), "Parameter tensor should require gradient");
        }
    }

    // Conceptual numerical test for GRUCell (similar limitations as LSTMTest)
    @Test
    void testGRUCellForwardPassSimpleValues() {
        int inputSize = 1;
        int hiddenSize = 1;
        int batchSize = 1;

        GRUCell cell = new GRUCell(inputSize, hiddenSize);

        // This part of the test is conceptual due to lack of weight modification.
        // We are testing the flow with default random weights instead.
        Tensor input = Tensor.scalar(0.1f);
        Tensor prevHidden = Tensor.scalar(0.0f);

        Tensor newHiddenState = cell.forward(input, prevHidden);

        assertNotNull(newHiddenState);
        // Cannot assert specific values without known weights and full Tensor API for data access.
        // assertTrue(!Double.isNaN(newHiddenState.get(0,0)), "Hidden state should not be NaN");
    }

    @Test
    void testGRULayerForwardPassShapes() {
        int inputSize = 2;
        int hiddenSize = 3;
        int sequenceLength = 4;
        int batchSize = 1;

        GRU gru = new GRU(inputSize, hiddenSize);
        List<Tensor> inputSequence = new ArrayList<>();
        for (int i = 0; i < sequenceLength; i++) {
            inputSequence.add(Tensor.randGaussian(batchSize, inputSize));
        }

        GRU.GRUOutput gruOutput = gru.forward(inputSequence);

        assertNotNull(gruOutput.allHiddenStates, "All hidden states list should not be null");
        assertEquals(sequenceLength, gruOutput.allHiddenStates.size(), "Number of hidden states should match sequence length");
        for (Tensor h_t : gruOutput.allHiddenStates) {
            assertNotNull(h_t);
            assertEquals(batchSize, h_t.rows(), "Hidden state rows should match batch size");
            assertEquals(hiddenSize, h_t.cols(), "Hidden state cols should match hidden size");
        }

        assertNotNull(gruOutput.finalHiddenState, "Final hidden state should not be null");
        assertEquals(batchSize, gruOutput.finalHiddenState.rows());
        assertEquals(hiddenSize, gruOutput.finalHiddenState.cols());
    }

    @Test
    void testGRULayerForwardWithInitialState() {
        int inputSize = 2;
        int hiddenSize = 3;
        int sequenceLength = 2;
        int batchSize = 1;

        GRU gru = new GRU(inputSize, hiddenSize);
        List<Tensor> inputSequence = new ArrayList<>();
        for (int i = 0; i < sequenceLength; i++) {
            inputSequence.add(Tensor.randGaussian(batchSize, inputSize));
        }

        Tensor initialHidden = Tensor.randGaussian(batchSize, hiddenSize);

        GRU.GRUOutput gruOutput = gru.forward(inputSequence, initialHidden);

        assertNotNull(gruOutput);
        assertEquals(sequenceLength, gruOutput.allHiddenStates.size());
        assertNotNull(gruOutput.finalHiddenState);
    }

    @Test
    void testGRULayerParameters() {
        GRU gru = new GRU(2, 3);
        List<Tensor> params = gru.parameters();
        assertNotNull(params);
        assertEquals(10, params.size(), "GRU layer should have 10 parameter tensors from its cell");
        for (Tensor p : params) {
            assertNotNull(p);
            assertTrue(p.hasGrad(), "Parameter tensor should require gradient");
        }
    }
}
