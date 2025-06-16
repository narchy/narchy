package jcog.tensor.model;

import jcog.tensor.Tensor;
// Assuming TensorOp might be needed for custom gradient definitions later, if not, it can be removed.
// import jcog.tensor.TensorOp;
// import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.List;

/**
 * GRU Cell (Gated Recurrent Unit)
 *
 * Takes input Xt and previous hidden state Ht-1
 * Outputs hidden state Ht
 *
 * GATES:
 *  Reset Gate (r):   r_t = sigmoid(W_ir * x_t + b_ir + W_hr * h_t-1 + b_hr)
 *  Update Gate (z):  z_t = sigmoid(W_iz * x_t + b_iz + W_hz * h_t-1 + b_hz)
 *
 * CANDIDATE:
 *  Candidate (n):    n_t = tanh(W_in * x_t + b_in + r_t * (W_hn * h_t-1 + b_hn))
 *
 * OUTPUT:
 *  Hidden State (h): h_t = (1 - z_t) * n_t + z_t * h_t-1
 */
public class GRUCell {

    private final int inputSize;
    private final int hiddenSize;

    // Input weights
    private final Tensor wIr; // Weight matrix for input to reset gate
    private final Tensor wIz; // Weight matrix for input to update gate
    private final Tensor wIn; // Weight matrix for input to candidate state

    // Hidden weights
    private final Tensor wHr; // Weight matrix for hidden to reset gate
    private final Tensor wHz; // Weight matrix for hidden to update gate
    private final Tensor wHn; // Weight matrix for hidden to candidate state

    // Biases (one bias tensor per gate/candidate calculation for simplicity)
    // bIr and bHr are effectively combined into bR, etc.
    private final Tensor bR; // Bias for reset gate
    private final Tensor bZ; // Bias for update gate
    private final Tensor bN; // Bias for candidate state (often split for input and hidden parts)
                             // For W_in*x_t + b_in and r_t * (W_hn*h_t-1 + b_hn)
                             // Let's use separate bIn and bHn for now for clarity with the formula.
    private final Tensor bIn; // Bias for input part of candidate
    private final Tensor bHn; // Bias for hidden part of candidate


    public GRUCell(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        // Initialize weights (e.g., Xavier or He initialization)
        // Using randGaussian as a placeholder from Tensor class
        this.wIr = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();
        this.wIz = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();
        this.wIn = Tensor.randGaussian(inputSize, hiddenSize).grad(true).parameter();

        this.wHr = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();
        this.wHz = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();
        this.wHn = Tensor.randGaussian(hiddenSize, hiddenSize).grad(true).parameter();

        // Initialize biases (often to zero)
        this.bIn = Tensor.zeros(1, hiddenSize).grad(true).parameter();
        this.bHn = Tensor.zeros(1, hiddenSize).grad(true).parameter();

        // These are alternative ways to structure biases.
        // If bIn and bHn are used, these might not be.
        // For now, nullify bR, bZ, bN if bIn/bHn are primary.
        // Or, use bR for (W_ir*x_t + W_hr*h_t-1) + bR
        // Let's use bR, bZ for the gates, and bIn, bHn for the candidate.
        this.bR = Tensor.zeros(1, hiddenSize).grad(true).parameter();
        this.bZ = Tensor.zeros(1, hiddenSize).grad(true).parameter();
        this.bN = null; // Not used if bIn and bHn are used for candidate.
    }

    /**
     * Performs a single forward step of the GRU cell.
     *
     * @param input      Xt (current input)
     * @param prevHidden Ht-1 (previous hidden state)
     * @return Tensor Ht (new hidden state)
     */
    public Tensor forward(Tensor input, Tensor prevHidden) {
        // Reset gate
        // r_t = sigmoid( (W_ir * x_t) + (W_hr * h_t-1) + bR )
        Tensor r_t_input = input.matmul(wIr);
        Tensor r_t_hidden = prevHidden.matmul(wHr);
        Tensor r_t = (r_t_input.add(r_t_hidden)).add(bR).sigmoid();

        // Update gate
        // z_t = sigmoid( (W_iz * x_t) + (W_hz * h_t-1) + bZ )
        Tensor z_t_input = input.matmul(wIz);
        Tensor z_t_hidden = prevHidden.matmul(wHz);
        Tensor z_t = (z_t_input.add(z_t_hidden)).add(bZ).sigmoid();

        // Candidate hidden state
        // n_t = tanh( (W_in * x_t + bIn) + r_t * (W_hn * h_t-1 + bHn) )
        Tensor n_t_input_part = input.matmul(wIn).add(bIn);
        Tensor n_t_hidden_part = prevHidden.matmul(wHn).add(bHn);
        Tensor n_t_reset_hidden = r_t.mul(n_t_hidden_part); // Element-wise multiplication
        Tensor n_t = (n_t_input_part.add(n_t_reset_hidden)).tanh();

        // New hidden state
        // h_t = (1 - z_t) * n_t + z_t * h_t-1
        // Assuming hiddenSize is available and batch size can be inferred or is 1
        int batchSize = input.rows(); // Or prevHidden.rows()
        Tensor ones = Tensor.ones(batchSize, hiddenSize); // Assuming hiddenSize is known here
        Tensor one_minus_z_t = ones.sub(z_t);

        Tensor h_t_part1 = one_minus_z_t.mul(n_t);
        Tensor h_t_part2 = z_t.mul(prevHidden);
        Tensor currentHidden = h_t_part1.add(h_t_part2);

        return currentHidden;
    }

    /**
     * Returns a list of all learnable parameters in this cell.
     */
    public List<Tensor> parameters() {
        return Arrays.asList(
            wIr, wIz, wIn,
            wHr, wHz, wHn,
            bR, bZ, bIn, bHn
        );
    }
}
