package jcog.tensor.model;

import jcog.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * GRU Layer
 *
 * Processes a sequence of inputs using a GRUCell.
 */
public class GRU {

    private final GRUCell cell;
    private final int hiddenSize;

    public GRU(int inputSize, int hiddenSize) {
        this.cell = new GRUCell(inputSize, hiddenSize);
        this.hiddenSize = hiddenSize;
    }

    /**
     * Represents the output of a GRU layer after processing a sequence.
     */
    public static class GRUOutput {
        public final List<Tensor> allHiddenStates; // List of Ht for each time step
        public final Tensor finalHiddenState;      // Final Ht

        public GRUOutput(List<Tensor> allHiddenStates, Tensor finalHiddenState) {
            this.allHiddenStates = allHiddenStates;
            this.finalHiddenState = finalHiddenState;
        }
    }

    /**
     * Processes an input sequence.
     *
     * @param inputSequence A list of Tensors, where each Tensor represents input x_t
     *                      for a time step. Each x_t is expected to be [batchSize, inputSize].
     * @param initialHiddenState Optional initial hidden state. If null, defaults to zeros.
     *                           Batch size is inferred from the first input tensor.
     * @return GRUOutput containing all hidden states and the final hidden state.
     */
    public GRUOutput forward(List<Tensor> inputSequence, Tensor initialHiddenState) {
        if (inputSequence == null || inputSequence.isEmpty()) {
            throw new IllegalArgumentException("Input sequence cannot be null or empty.");
        }

        int batchSize = inputSequence.get(0).rows();

        Tensor currentHiddenState;
        if (initialHiddenState != null) {
            currentHiddenState = initialHiddenState;
        } else {
            // Default initial hidden state (zeros)
            currentHiddenState = Tensor.zeros(batchSize, hiddenSize);
        }

        List<Tensor> allHiddenStates = new ArrayList<>();

        for (Tensor inputAtT : inputSequence) {
            currentHiddenState = cell.forward(inputAtT, currentHiddenState);
            allHiddenStates.add(currentHiddenState);
        }

        return new GRUOutput(allHiddenStates, currentHiddenState);
    }

    /**
     * Convenience overload for forward pass with default initial zero state.
     */
    public GRUOutput forward(List<Tensor> inputSequence) {
        return forward(inputSequence, null);
    }

    /**
     * Returns a list of all learnable parameters in this GRU layer (from its cell).
     */
    public List<Tensor> parameters() {
        return cell.parameters();
    }
}
