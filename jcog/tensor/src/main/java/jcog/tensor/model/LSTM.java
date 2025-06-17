package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.tensor.model.LSTMCell.LSTMState; // Import the inner class

import java.util.ArrayList;
import java.util.List;

/**
 * LSTM Layer
 *
 * Processes a sequence of inputs using an LSTMCell.
 */
public class LSTM {

    private final LSTMCell cell;
    private final int hiddenSize;

    public LSTM(int inputSize, int hiddenSize) {
        this.cell = new LSTMCell(inputSize, hiddenSize);
        this.hiddenSize = hiddenSize;
    }

    /**
     * Represents the output of an LSTM layer after processing a sequence.
     */
    public static class LSTMOutput {
        public final List<Tensor> allHiddenStates; // List of Ht for each time step
        public final LSTMState finalState;         // Final (Ht, Ct)

        public LSTMOutput(List<Tensor> allHiddenStates, LSTMState finalState) {
            this.allHiddenStates = allHiddenStates;
            this.finalState = finalState;
        }
    }

    /**
     * Processes an input sequence.
     *
     * @param inputSequence A list of Tensors, where each Tensor represents input x_t
     *                      for a time step. Each x_t is expected to be [batchSize, inputSize].
     * @param initialStates Optional initial hidden and cell states. If null, defaults to zeros.
     *                      Batch size is inferred from the first input tensor.
     * @return LSTMOutput containing all hidden states and the final cell state.
     */
    public LSTMOutput forward(List<Tensor> inputSequence, LSTMState initialStates) {
        if (inputSequence == null || inputSequence.isEmpty()) {
            throw new IllegalArgumentException("Input sequence cannot be null or empty.");
        }

        // Assuming all tensors in the sequence have the same batch size.
        // Infer batch size from the first input tensor (e.g., inputSequence.get(0).rows())
        // For simplicity, this example assumes batch processing is handled within Tensor ops
        // or that input is single instance (batch size 1). Let's assume batch size 1 for now.
        // If batch > 1, initial states need to match batch size.
        int batchSize = inputSequence.get(0).rows(); // Or a more robust way to get batch size

        LSTMState currentState;
        if (initialStates != null) {
            currentState = initialStates;
        } else {
            // Default initial states (zeros)
            // Assuming Tensor.zeros(rows, cols) exists
            Tensor h0 = Tensor.zeros(batchSize, hiddenSize);
            Tensor c0 = Tensor.zeros(batchSize, hiddenSize);
            currentState = new LSTMState(h0, c0);
        }

        List<Tensor> allHiddenStates = new ArrayList<>();

        for (Tensor inputAtT : inputSequence) {
            currentState = cell.forward(inputAtT, currentState);
            allHiddenStates.add(currentState.hiddenState);
        }

        return new LSTMOutput(allHiddenStates, currentState);
    }

    /**
     * Convenience overload for forward pass with default initial zero states.
     */
    public LSTMOutput forward(List<Tensor> inputSequence) {
        return forward(inputSequence, null);
    }

    /**
     * Returns a list of all learnable parameters in this LSTM layer (from its cell).
     */
    public List<Tensor> parameters() {
        return cell.parameters();
    }
}
