package jcog.tensor;

import java.util.stream.IntStream;

/**
 * Implements Rotary Positional Encoding (RoPE), a method for injecting relative positional
 * information into Transformer models.
 * <p>
 * RoPE works by rotating pairs of input features in the query (Q) and key (K) vectors
 * based on their absolute position. This effectively allows the attention mechanism to
 * capture relative positional dependencies.
 * <p>
 * It is often preferred in modern Transformer architectures (e.g., LLaMA, PaLM) due to its
 * strong performance, ability to handle variable sequence lengths naturally, and tendency
 * to provide good long-range dependency modeling.
 * <p>
 * This component is designed to be used typically within the attention mechanism, applied
 * to Q and K tensors before attention scores are computed.
 *
 * @see jcog.tensor.PositionalEncoding for additive absolute positional encodings.
 * @see jcog.tensor.MultiHeadAttention for an example of its integration.
 */
public class RotaryPositionalEncoding {

    private final int dim;
    private final int max_seq_len;
    private final double theta_base;
    private final Tensor inv_freq; // Shape [1, dim / 2]

    /**
     * Initializes Rotary Positional Encoding.
     *
     * @param dim          Dimension of the features (e.g., d_k head dimension). Must be even.
     * @param max_seq_len  Maximum sequence length for precomputing frequencies.
     * @param theta_base   Base for calculating theta values (e.g., 10000.0).
     */
    public RotaryPositionalEncoding(int dim, int max_seq_len, double theta_base) {
        if (dim % 2 != 0) {
            throw new IllegalArgumentException("Dimension must be even.");
        }
        this.dim = dim;
        this.max_seq_len = max_seq_len; // Not directly used in inv_freq precomputation here, but good to store
        this.theta_base = theta_base;

        int half_dim = dim / 2;
        double[] inv_freq_data = new double[half_dim];
        for (int i = 0; i < half_dim; i++) {
            inv_freq_data[i] = 1.0 / Math.pow(theta_base, (double) (2 * i) / dim);
        }
        // Assuming Tensor can be created from a flat array and reshaped,
        // or directly from a 2D array structure if supported.
        // For [1, half_dim] tensor:
        this.inv_freq = new Tensor(inv_freq_data, 1, half_dim, false);
    }

    /**
     * Applies Rotary Positional Embedding to the input tensor.
     *
     * @param q_or_k          Input tensor (Query or Key) of shape [seq_len_actual, dim].
     * @param seq_len         Actual sequence length of q_or_k.
     * @param position_offset Offset for positions (useful for caching in generation).
     * @return Tensor with rotary embeddings applied, shape [seq_len_actual, dim].
     */
    public Tensor apply_rotary_pos_emb(Tensor q_or_k, int seq_len, int position_offset) {
        if (q_or_k.cols() != this.dim) {
            throw new IllegalArgumentException("Input tensor dimension (" + q_or_k.cols() +
                                               ") does not match RoPE dimension (" + this.dim + ").");
        }
        if (q_or_k.rows() != seq_len) {
            throw new IllegalArgumentException("Input tensor sequence length (" + q_or_k.rows() +
                                               ") does not match provided seq_len (" + seq_len + ").");
        }

        // a. Create a position tensor `t`
        double[] positions_data = IntStream.range(position_offset, position_offset + seq_len)
                                           .asDoubleStream().toArray();
        Tensor t = new Tensor(positions_data, seq_len, 1, false); // Shape [seq_len, 1]

        // b. Calculate `freqs = t.matmul(this.inv_freq)`
        // This results in a [seq_len, dim / 2] tensor.
        Tensor freqs = t.matmul(this.inv_freq);

        // c. Concatenate `freqs` with itself column-wise to get `emb`
        // Shape [seq_len, dim]. This `emb` tensor now holds `m * theta_j` for each position `m` and dimension `j`.
        // The Tensor.concat method concatenates by columns by default if they are row vectors,
        // or more generally, it should handle this case if freqs are [seq_len, dim/2].
        // If Tensor.concat(Tensor...) concatenates horizontally:
        Tensor emb = Tensor.concat(freqs, freqs);
        if (emb.cols() != this.dim) {
            // Fallback or error if concat doesn't behave as expected for [N, C] + [N, C] -> [N, 2C]
            // This might require manual construction if Tensor.concat isn't suitable.
            // For now, assume Tensor.concat(A, B) when A and B are [N, C1] and [N, C2] results in [N, C1+C2].
            // So if A and B are both [seq_len, dim/2], result is [seq_len, dim].
             System.err.println("Warning: emb.cols() is " + emb.cols() + " but this.dim is " + this.dim + ". Review Tensor.concat behavior.");
        }


        // d, e, f, g: Apply rotations using the column-wise iteration approach
        // as direct reshaping to [seq_len, dim/2, 2] and slicing might be complex.

        Tensor[] rotated_cols = new Tensor[this.dim];

        for (int j = 0; j < this.dim; j += 2) {
            // Slice out the j-th and (j+1)-th columns from the input q_or_k
            Tensor x_j = q_or_k.slice(0, seq_len, j, j + 1);         // Shape [seq_len, 1]
            Tensor x_j_plus_1 = q_or_k.slice(0, seq_len, j + 1, j + 2); // Shape [seq_len, 1]

            // Slice out the j-th column from `emb`.
            // Since `emb` was `concat(freqs, freqs)`, `emb[:, j]` corresponds to `freqs[:, j/2]`
            // and `emb[:, j+1]` also corresponds to `freqs[:, j/2]`.
            // So, we effectively need `freqs.slice(0, seq_len, j/2, j/2 + 1)`
            Tensor emb_val_for_pair = freqs.slice(0, seq_len, j / 2, j / 2 + 1); // Shape [seq_len, 1]

            Tensor cos_val = emb_val_for_pair.cos(); // Shape [seq_len, 1]
            Tensor sin_val = emb_val_for_pair.sin(); // Shape [seq_len, 1]

            // out_col_j = x_j * cos_val - x_j_plus_1 * sin_val
            rotated_cols[j] = x_j.mul(cos_val).sub(x_j_plus_1.mul(sin_val));

            // out_col_j_plus_1 = x_j * sin_val + x_j_plus_1 * cos_val
            rotated_cols[j+1] = x_j.mul(sin_val).add(x_j_plus_1.mul(cos_val));
        }

        // Concatenate all rotated columns to form the output tensor
        // This assumes Tensor.concat can take an array of Tensors (all [seq_len, 1])
        // and join them into a single [seq_len, dim] tensor.
        return Tensor.concat(rotated_cols);
    }

    // Optional: A main method for basic testing if needed
    public static void main(String[] args) {
        int dim = 4; // e.g., d_k = 64 / num_heads = 16 -> dim = 16
        int max_seq_len = 20;
        double theta_base = 10000.0;

        RotaryPositionalEncoding rope = new RotaryPositionalEncoding(dim, max_seq_len, theta_base);
        System.out.println("inv_freq: " + rope.inv_freq.shapeStr());
        // rope.inv_freq.print();


        int seq_len_actual = 3;
        // Create a dummy tensor for q_or_k: [seq_len_actual, dim]
        // Example: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        Tensor q_or_k_dummy = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12
        }, seq_len_actual, dim, false);

        System.out.println("\nOriginal q_or_k:");
        q_or_k_dummy.print();

        Tensor rotated_q_or_k = rope.apply_rotary_pos_emb(q_or_k_dummy, seq_len_actual, 0);
        System.out.println("\nRotated q_or_k (position_offset=0):");
        rotated_q_or_k.print();

        Tensor rotated_q_or_k_offset_1 = rope.apply_rotary_pos_emb(q_or_k_dummy, seq_len_actual, 1);
        System.out.println("\nRotated q_or_k (position_offset=1):");
        rotated_q_or_k_offset_1.print();


        // Test with dim=2, seq_len=1, pos=0
        RotaryPositionalEncoding rope2 = new RotaryPositionalEncoding(2, 10, 10000.0);
        Tensor qk_simple = new Tensor(new double[]{1,2}, 1, 2, false);
        System.out.println("\nOriginal simple qk:");
        qk_simple.print();
        Tensor rotated_simple = rope2.apply_rotary_pos_emb(qk_simple, 1,0);
        System.out.println("\nRotated simple qk (pos=0):");
        rotated_simple.print();
        // Expected: For pos=0, freqs = 0. cos(0)=1, sin(0)=0.
        // out1 = x1*1 - x2*0 = x1
        // out2 = x1*0 + x2*1 = x2
        // So, output should be same as input.

        Tensor rotated_simple_pos1 = rope2.apply_rotary_pos_emb(qk_simple, 1,1);
        System.out.println("\nRotated simple qk (pos=1):");
        rotated_simple_pos1.print();
        // For pos=1, inv_freq[0] = 1.0 / (10000^(0/2)) = 1.0
        // freqs = [1] * [1.0] = [1.0]
        // emb = [1.0, 1.0]
        // cos(1)=0.5403, sin(1)=0.8414
        // x1=1, x2=2
        // out1 = 1*0.5403 - 2*0.8414 = 0.5403 - 1.6828 = -1.1425
        // out2 = 1*0.8414 + 2*0.5403 = 0.8414 + 1.0806 = 1.9220
    }
}
