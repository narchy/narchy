package jcog.tensor.model;

import jcog.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * LSTM Cell
 *
 * Takes input Xt, previous hidden state Ht-1, and previous cell state Ct-1
 * Outputs hidden state Ht and cell state Ct
 *
 * GATES:
 *  Input Gate (i):      i_t = sigmoid(W_ii * x_t + b_ii + W_hi * h_t-1 + b_hi)
 *  Forget Gate (f):     f_t = sigmoid(W_if * x_t + b_if + W_hf * h_t-1 + b_hf)
 *  Cell Gate (g):       g_t = tanh(W_ig * x_t + b_ig + W_hg * h_t-1 + b_hg)
 *  Output Gate (o):     o_t = sigmoid(W_io * x_t + b_io + W_ho * h_t-1 + b_ho)
 *
 * STATES:
 *  Cell State (c):      c_t = f_t * c_t-1 + i_t * g_t
 *  Hidden State (h):    h_t = o_t * tanh(c_t)
 */
public class LSTMCell {

    private final int inputSize;
    private final int hiddenSize;

    // Input weights
    private final Tensor wIi; // Weight matrix for input to input gate
    private final Tensor wIf; // Weight matrix for input to forget gate
    private final Tensor wIg; // Weight matrix for input to cell gate
    private final Tensor wIo; // Weight matrix for input to output gate

    // Hidden weights
    private final Tensor wHi; // Weight matrix for hidden to input gate
    private final Tensor wHf; // Weight matrix for hidden to forget gate
    private final Tensor wHg; // Weight matrix for hidden to cell gate
    private final Tensor wHo; // Weight matrix for hidden to output gate

    // Biases
    private final Tensor bIi; // Bias for input to input gate
    private final Tensor bIf; // Bias for input to forget gate
    private final Tensor bIg; // Bias for input to cell gate
    private final Tensor bIo; // Bias for input to output gate
    private final Tensor bHi; // Bias for hidden to input gate
    private final Tensor bHf; // Bias for hidden to forget gate
    private final Tensor bHg; // Bias for hidden to cell gate
    private final Tensor bHo; // Bias for hidden to output gate

    // A more compact way to store biases - one bias per gate
    private final Tensor bI; // Bias for input gate (combines bIi and bHi if weights are concatenated)
    private final Tensor bF; // Bias for forget gate
    private final Tensor bG; // Bias for cell gate
    private final Tensor bO; // Bias for output gate


    public LSTMCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Initialize weights - Xavier or He initialization is common
        // For simplicity, using randn for now, assuming Tensor.randn exists
        // These can be initialized as a single large matrix W_ih and W_hh for efficiency
        // W_ih = [W_ii, W_if, W_ig, W_io] (hiddenSize * 4, inputSize)
        // W_hh = [W_hi, W_hf, W_hg, W_ho] (hiddenSize * 4, hiddenSize)

        this.wIi = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();
        this.wIf = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();
        this.wIg = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();
        this.wIo = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();

        this.wHi = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();
        this.wHf = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();
        this.wHg = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();
        this.wHo = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();

        // Initialize biases - often initialized to zero, or forget gate bias to 1
        // b = [b_i, b_f, b_g, b_o] (hiddenSize * 4)
        this.bI = Tensor.zeros(1, hiddenSize).grad(true).parameter();
        this.bF = Tensor.zeros(1, hiddenSize).grad(true).parameter(); // Forget gate bias often 1
        this.bG = Tensor.zeros(1, hiddenSize).grad(true).parameter();
        this.bO = Tensor.zeros(1, hiddenSize).grad(true).parameter();

        // The individual bias tensors might be redundant if we use the combined ones above.
        // For now, keeping the combined ones and removing the more granular ones to simplify.
        this.bIi = null; this.bIf = null; this.bIg = null; this.bIo = null;
        this.bHi = null; this.bHf = null; this.bHg = null; this.bHo = null;
    }

    /**
     * Represents the output of an LSTM cell for a single time step.
     */
    public static class LSTMState {
        public final Tensor hiddenState; // Ht
        public final Tensor cellState;   // Ct

        public LSTMState(Tensor hiddenState, Tensor cellState) {
            this.hiddenState = hiddenState;
            this.cellState = cellState;
        }
    }

    /**
     * Performs a single forward step of the LSTM cell.
     *
     * @param input      Xt (current input)
     * @param prevState  (Ht-1, Ct-1) (previous hidden state and cell state)
     * @return LSTMState (Ht, Ct) (new hidden state and cell state)
     */
    public LSTMState forward(Tensor input, LSTMState prevState) {
        Tensor prevHidden = prevState.hiddenState;
        Tensor prevCell = prevState.cellState;

        // Calculate gates
        // Input gate
        Tensor i_t_input = input.matmul(wIi);
        Tensor i_t_hidden = prevHidden.matmul(wHi);
        Tensor i_t = (i_t_input.add(i_t_hidden)).add(bI).sigmoid();

        // Forget gate
        Tensor f_t_input = input.matmul(wIf);
        Tensor f_t_hidden = prevHidden.matmul(wHf);
        Tensor f_t = (f_t_input.add(f_t_hidden)).add(bF).sigmoid();

        // Cell gate (candidate cell state)
        Tensor g_t_input = input.matmul(wIg);
        Tensor g_t_hidden = prevHidden.matmul(wHg);
        Tensor g_t = (g_t_input.add(g_t_hidden)).add(bG).tanh();

        // Output gate
        Tensor o_t_input = input.matmul(wIo);
        Tensor o_t_hidden = prevHidden.matmul(wHo);
        Tensor o_t = (o_t_input.add(o_t_hidden)).add(bO).sigmoid();

        // Calculate new cell state
        // c_t = f_t * c_t-1 + i_t * g_t
        Tensor currentCell = (f_t.mul(prevCell)).add(i_t.mul(g_t));

        // Calculate new hidden state
        // h_t = o_t * tanh(c_t)
        Tensor currentHidden = o_t.mul(currentCell.tanh());

        return new LSTMState(currentHidden, currentCell);
    }

    /**
     * Returns a list of all learnable parameters in this cell.
     * Useful for passing to an optimizer.
     */
    public List<Tensor> parameters() {
        return Arrays.asList(
            wIi, wIf, wIg, wIo,
            wHi, wHf, wHg, wHo,
            bI, bF, bG, bO
        );
    }
}
