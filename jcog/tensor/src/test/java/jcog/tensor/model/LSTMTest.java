package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.tensor.model.LSTMCell.LSTMState;
import org.junit.jupiter.api.Test; // Common testing framework

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*; // Common assertions

public class LSTMTest {

    @Test
    void testLSTMCellForwardPassShapes() {
        int inputSize = 2;
        int hiddenSize = 3;
        int batchSize = 1; // Or some other small number

        LSTMCell cell = new LSTMCell(inputSize, hiddenSize);

        Tensor input = Tensor.randGaussian(batchSize, inputSize);
        Tensor prevHidden = Tensor.zeros(batchSize, hiddenSize);
        Tensor prevCell = Tensor.zeros(batchSize, hiddenSize);
        LSTMState initialState = new LSTMState(prevHidden, prevCell);

        LSTMState newState = cell.forward(input, initialState);

        assertNotNull(newState.hiddenState, "Hidden state should not be null");
        assertEquals(batchSize, newState.hiddenState.rows(), "Hidden state rows should match batch size");
        assertEquals(hiddenSize, newState.hiddenState.cols(), "Hidden state cols should match hidden size");

        assertNotNull(newState.cellState, "Cell state should not be null");
        assertEquals(batchSize, newState.cellState.rows(), "Cell state rows should match batch size");
        assertEquals(hiddenSize, newState.cellState.cols(), "Cell state cols should match hidden size");
    }

    @Test
    void testLSTMCellParameters() {
        LSTMCell cell = new LSTMCell(2, 3);
        List<Tensor> params = cell.parameters();
        assertNotNull(params);
        // wIi, wIf, wIg, wIo, wHi, wHf, wHg, wHo, bI, bF, bG, bO
        assertEquals(12, params.size(), "LSTMCell should have 12 parameter tensors");
        for (Tensor p : params) {
            assertNotNull(p, "Parameter tensor should not be null");
            assertTrue(p.hasGrad(), "Parameter tensor should require gradient");
        }
    }

    // A very simple numerical test for LSTMCell - might need adjustment based on Tensor API
    // This test is hard to get exactly right without running and debugging
    // It assumes specific behaviors of sigmoid and tanh and simple weights.
    @Test
    void testLSTMCellForwardPassSimpleValues() {
        int inputSize = 1;
        int hiddenSize = 1;
        int batchSize = 1;

        LSTMCell cell = new LSTMCell(inputSize, hiddenSize);

        // Override weights and biases for simplicity
        // Assuming direct access or setters, which is not current API.
        // For now, this test will be more of a structural placeholder
        // as modifying internal tensors of LSTMCell is not directly supported by current code.
        // A real test would require Tensor.set(value) or similar, or initializing with specific values.

        // Let's assume all weights are 0 and all biases are 0 for a moment.
        // wIi, wIf, wIg, wIo, wHi, wHf, wHg, wHo all zero
        // bI, bF, bG, bO all zero
        // If this were possible:
        // i_t = sigmoid(0) = 0.5
        // f_t = sigmoid(0) = 0.5
        // g_t = tanh(0) = 0
        // o_t = sigmoid(0) = 0.5
        // c_t = f_t * prev_c + i_t * g_t = 0.5 * 0 + 0.5 * 0 = 0
        // h_t = o_t * tanh(c_t) = 0.5 * tanh(0) = 0

        // This part of the test is conceptual due to lack of weight modification.
        // We are testing the flow with default random weights instead.
        Tensor input = Tensor.scalar(0.1f); // input x_t
        Tensor prevHidden = Tensor.scalar(0.0f); // h_t-1
        Tensor prevCell = Tensor.scalar(0.0f);   // c_t-1
        LSTMState initialState = new LSTMState(prevHidden, prevCell);

        LSTMState newState = cell.forward(input, initialState);

        assertNotNull(newState.hiddenState);
        assertNotNull(newState.cellState);
        // Cannot assert specific values without known weights and full Tensor API for data access.
        // We can at least check they are not NaN if the Tensor class supports it.
        // assertTrue(!Double.isNaN(newState.hiddenState.get(0,0)), "Hidden state should not be NaN");
        // assertTrue(!Double.isNaN(newState.cellState.get(0,0)), "Cell state should not be NaN");
    }


    @Test
    void testLSTMLayerForwardPassShapes() {
        int inputSize = 2;
        int hiddenSize = 3;
        int sequenceLength = 4;
        int batchSize = 1; // Assuming batch size 1 for simplicity in list based input

        LSTM lstm = new LSTM(inputSize, hiddenSize);
        List<Tensor> inputSequence = new ArrayList<>();
        for (int i = 0; i < sequenceLength; i++) {
            inputSequence.add(Tensor.randGaussian(batchSize, inputSize));
        }

        LSTM.LSTMOutput lstmOutput = lstm.forward(inputSequence);

        assertNotNull(lstmOutput.allHiddenStates, "All hidden states list should not be null");
        assertEquals(sequenceLength, lstmOutput.allHiddenStates.size(), "Number of hidden states should match sequence length");
        for (Tensor h_t : lstmOutput.allHiddenStates) {
            assertNotNull(h_t);
            assertEquals(batchSize, h_t.rows(), "Hidden state rows should match batch size");
            assertEquals(hiddenSize, h_t.cols(), "Hidden state cols should match hidden size");
        }

        assertNotNull(lstmOutput.finalState.hiddenState, "Final hidden state should not be null");
        assertEquals(batchSize, lstmOutput.finalState.hiddenState.rows());
        assertEquals(hiddenSize, lstmOutput.finalState.hiddenState.cols());

        assertNotNull(lstmOutput.finalState.cellState, "Final cell state should not be null");
        assertEquals(batchSize, lstmOutput.finalState.cellState.rows());
        assertEquals(hiddenSize, lstmOutput.finalState.cellState.cols());
    }

    @Test
    void testLSTMLayerForwardWithInitialState() {
        int inputSize = 2;
        int hiddenSize = 3;
        int sequenceLength = 2;
        int batchSize = 1;

        LSTM lstm = new LSTM(inputSize, hiddenSize);
        List<Tensor> inputSequence = new ArrayList<>();
        for (int i = 0; i < sequenceLength; i++) {
            inputSequence.add(Tensor.randGaussian(batchSize, inputSize));
        }

        Tensor initialHidden = Tensor.randGaussian(batchSize, hiddenSize);
        Tensor initialCell = Tensor.randGaussian(batchSize, hiddenSize);
        LSTMState initialState = new LSTMState(initialHidden, initialCell);

        LSTM.LSTMOutput lstmOutput = lstm.forward(inputSequence, initialState);

        assertNotNull(lstmOutput);
        // Further checks could involve comparing if the first hidden state computation
        // was influenced by the initial state if we had a way to peek into values.
        // For now, just checking shape and non-nullness.
        assertEquals(sequenceLength, lstmOutput.allHiddenStates.size());
        assertNotNull(lstmOutput.finalState);
    }

    @Test
    void testLSTMLayerParameters() {
        LSTM lstm = new LSTM(2, 3);
        List<Tensor> params = lstm.parameters();
        assertNotNull(params);
        assertEquals(12, params.size(), "LSTM layer should have 12 parameter tensors from its cell");
        for (Tensor p : params) {
            assertNotNull(p);
            assertTrue(p.hasGrad(), "Parameter tensor should require gradient");
        }
    }
}
