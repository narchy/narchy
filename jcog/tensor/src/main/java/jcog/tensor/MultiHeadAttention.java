package jcog.tensor;

import org.jetbrains.annotations.Nullable;
import jcog.linear.Linear; // Assuming Models.Linear is jcog.linear.Linear

public class MultiHeadAttention {

    public final int dModel;
    public final int numHeads;
    public final int d_k; // Dimension of key/query per head
    public final Linear wq, wk, wv; // Input projection layers
    public final Linear wo;         // Output projection layer
    private final boolean debugPrinting;

    public MultiHeadAttention(int dModel, int numHeads, boolean biasForProjections, boolean requiresGrad, boolean debugPrinting) {
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ").");
        }
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.d_k = dModel / numHeads;
        this.debugPrinting = debugPrinting;

        // Initialize projection layers
        // Assuming Models.Linear is equivalent to jcog.linear.Linear
        // The constructor for Linear is (in: int, out: int, init: Tensor, bias: boolean)
        // We'll pass null for init tensor, which means it will be initialized internally (e.g. randGaussian)
        wq = new Linear(dModel, dModel, null, biasForProjections);
        wk = new Linear(dModel, dModel, null, biasForProjections);
        wv = new Linear(dModel, dModel, null, biasForProjections);
        wo = new Linear(dModel, dModel, null, biasForProjections);

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
