package jcog.tensor.model;

import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

public class AttentionMechanisms {

    /**
     * Computes Scaled Dot-Product Attention.
     *
     * @param query         Tensor of shape [seq_len_q, d_k]
     * @param key           Tensor of shape [seq_len_k, d_k]
     * @param value         Tensor of shape [seq_len_v, d_v] (where seq_len_k == seq_len_v)
     * @param attentionMask Optional tensor of shape [seq_len_q, seq_len_k].
     *                      Contains 0.0 for positions to attend to, and a large negative number (e.g., -1e9) for masked positions.
     * @param debugPrinting If true, print shapes of intermediate tensors.
     * @return Tensor of shape [seq_len_q, d_v]
     */
    public static Tensor scaledDotProductAttention(Tensor query, Tensor key, Tensor value, @Nullable Tensor attentionMask, boolean debugPrinting) {
        if (key.rows() != value.rows()) {
            throw new IllegalArgumentException("seq_len_k (" + key.rows() + ") must be equal to seq_len_v (" + value.rows() + ").");
        }
        if (query.cols() != key.cols()) {
            throw new IllegalArgumentException("d_k of query (" + query.cols() + ") must be equal to d_k of key (" + key.cols() + ").");
        }

        int d_k_val = query.cols();

        if (debugPrinting) {
            System.out.println("SDPA: Query shape: " + query.shapeStr());
            System.out.println("SDPA: Key shape: " + key.shapeStr());
            System.out.println("SDPA: Value shape: " + value.shapeStr());
        }

        Tensor scores = query.matmul(key.transpose()); // Shape [seq_len_q, seq_len_k]
        if (debugPrinting) System.out.println("SDPA: Scores shape: " + scores.shapeStr());

        Tensor scaledScores = scores.div(Math.sqrt(d_k_val));
        if (debugPrinting) System.out.println("SDPA: Scaled Scores shape: " + scaledScores.shapeStr());

        if (attentionMask != null) {
            if (attentionMask.rows() != scaledScores.rows() || attentionMask.cols() != scaledScores.cols()) {
                throw new IllegalArgumentException("Attention mask shape " + attentionMask.shapeStr() +
                                                   " does not match scaled scores shape " + scaledScores.shapeStr());
            }
            scaledScores = scaledScores.add(attentionMask);
            if (debugPrinting) System.out.println("SDPA: Scaled Scores after mask shape: " + scaledScores.shapeStr());
        }

        Tensor attentionWeights = scaledScores.softmax(); // Softmax applies row-wise
        if (debugPrinting) System.out.println("SDPA: Attention Weights shape: " + attentionWeights.shapeStr());

        Tensor context = attentionWeights.matmul(value); // Shape [seq_len_q, d_v]
        if (debugPrinting) System.out.println("SDPA: Context shape: " + context.shapeStr());

        return context;
    }
}
