package jcog.tensor;

import org.jetbrains.annotations.Nullable;
import jcog.tensor.Models.Linear; // Using Models.Linear
import jcog.tensor.RotaryPositionalEncoding;

public class MultiHeadAttention {

    public final int dModel;
    public final int numHeads;
    public final int d_k; // Dimension of key/query per head
    public final Linear wq, wk, wv; // Input projection layers
    public final Linear wo;         // Output projection layer
    private final boolean debugPrinting;
    private final RotaryPositionalEncoding rope;

    public MultiHeadAttention(int dModel, int numHeads, boolean biasForProjections, boolean requiresGrad, boolean debugPrinting, @Nullable RotaryPositionalEncoding rope) {
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ").");
        }
        if (rope != null && rope.dim != (dModel/numHeads)) {
            throw new IllegalArgumentException("RoPE dimension (" + rope.dim + ") must match head dimension d_k (" + (dModel/numHeads) + ").");
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.d_k = dModel / numHeads;
        this.debugPrinting = debugPrinting;
        this.rope = rope;

        // Initialize projection layers
        // Models.Linear constructor is (inFeatures: int, outFeatures: int, activation: UnaryOperator<Tensor>, bias: boolean)
        // Passing null for activation means no activation function will be applied, which is the desired behavior here.
        wq = new Linear(dModel, dModel, null, biasForProjections);
        wk = new Linear(dModel, dModel, null, biasForProjections);
        wv = new Linear(dModel, dModel, null, biasForProjections);
        wo = new Linear(dModel, dModel, null, biasForProjections);

        // The .weight and .bias fields are directly accessible and grad can be set.
        // Models.Linear already calls .grad(true).parameter() on weight and bias internally if bias is enabled.
        // However, explicitly setting them here ensures the intent if requiresGrad is true,
        // and handles the case where internal initialization might change.
        // For Models.Linear, weight is initialized with .grad(true).parameter()
        // and bias (if true) is also initialized with .grad(true).parameter().
        // So, these lines are somewhat redundant if requiresGrad is always true when layers are made,
        // but they don't hurt and make the requiresGrad logic explicit.
        if (requiresGrad) {
            wq.weight.grad(true).parameter();
            wk.weight.grad(true).parameter();
            wv.weight.grad(true).parameter();
            wo.weight.grad(true).parameter();
            if (biasForProjections) {
                wq.bias.grad(true).parameter();
                wk.bias.grad(true).parameter();
                wv.bias.grad(true).parameter();
                wo.bias.grad(true).parameter();
            }
        }
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value, @Nullable Tensor attentionMask) {
        int seq_len_q = query.rows();
        int seq_len_k = key.rows();
        int seq_len_v = value.rows();

        if (debugPrinting) {
            System.out.println("MHA: Initial Query shape: " + query.shapeStr());
            System.out.println("MHA: Initial Key shape: " + key.shapeStr());
            System.out.println("MHA: Initial Value shape: " + value.shapeStr());
            if (attentionMask != null) {
                System.out.println("MHA: Initial Attention Mask shape: " + attentionMask.shapeStr());
            }
        }

        Tensor q_projected = wq.apply(query); // Shape [seq_len_q, dModel]
        Tensor k_projected = wk.apply(key);   // Shape [seq_len_k, dModel]
        Tensor v_projected = wv.apply(value); // Shape [seq_len_v, dModel]

        if (debugPrinting) {
            System.out.println("MHA: Projected Q shape: " + q_projected.shapeStr());
            System.out.println("MHA: Projected K shape: " + k_projected.shapeStr());
            System.out.println("MHA: Projected V shape: " + v_projected.shapeStr());
        }

        Tensor[] head_outputs = new Tensor[numHeads];

        for (int h = 0; h < numHeads; h++) {
            int startCol = h * d_k;
            int endCol = (h + 1) * d_k;

            // Slice to get per-head Q, K, V
            // slice(rowStart, rowEnd, colStart, colEnd) - rowEnd and colEnd are exclusive
            Tensor q_h = q_projected.slice(0, seq_len_q, startCol, endCol); // Shape [seq_len_q, d_k]
            Tensor k_h = k_projected.slice(0, seq_len_k, startCol, endCol); // Shape [seq_len_k, d_k]
            Tensor v_h = v_projected.slice(0, seq_len_v, startCol, endCol); // Shape [seq_len_v, d_k] (as d_v per head is d_k)

            if (this.rope != null) {
                int seq_len_q_actual = q_h.rows(); // Assuming q_h is [seq_len_q_actual, d_k]
                int seq_len_k_actual = k_h.rows(); // Assuming k_h is [seq_len_k_actual, d_k]

                // Apply RoPE. position_offset = 0 for standard application.
                // RoPE expects dim to be d_k, which was checked in the constructor.
                q_h = this.rope.apply_rotary_pos_emb(q_h, seq_len_q_actual, 0);
                k_h = this.rope.apply_rotary_pos_emb(k_h, seq_len_k_actual, 0);

                if (debugPrinting) {
                    System.out.println("MHA Head " + h + ": q_h shape after RoPE: " + q_h.shapeStr());
                    System.out.println("MHA Head " + h + ": k_h shape after RoPE: " + k_h.shapeStr());
                }
            }

            if (debugPrinting) {
                System.out.println("MHA Head " + h + ": q_h shape: " + q_h.shapeStr());
                System.out.println("MHA Head " + h + ": k_h shape: " + k_h.shapeStr());
                System.out.println("MHA Head " + h + ": v_h shape: " + v_h.shapeStr());
            }

            // The attentionMask (if provided) should be [seq_len_q, seq_len_k]
            // It is applied within scaledDotProductAttention identically for each head.
            Tensor context_h = AttentionMechanisms.scaledDotProductAttention(q_h, k_h, v_h, attentionMask, debugPrinting); // Shape [seq_len_q, d_k]
            head_outputs[h] = context_h;
        }

        // Concatenate head outputs
        // Tensor.concat(Tensor...) concatenates them by columns, which is what we need.
        // If each head_output[h] is [seq_len_q, d_k], then concat will produce [seq_len_q, numHeads * d_k] = [seq_len_q, dModel]
        Tensor concatenated_heads = Tensor.concat(head_outputs);
        if (debugPrinting) {
            System.out.println("MHA: Concatenated heads shape: " + concatenated_heads.shapeStr());
        }

        // Final linear projection
        Tensor output = wo.apply(concatenated_heads); // Shape [seq_len_q, dModel]
        if (debugPrinting) {
            System.out.println("MHA: Output shape: " + output.shapeStr());
        }

        return output;
    }
}
