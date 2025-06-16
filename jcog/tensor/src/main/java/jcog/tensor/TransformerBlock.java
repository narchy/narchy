package jcog.tensor;

import jcog.model.LayerNorm; // Assuming Models.LayerNorm -> jcog.model.LayerNorm
import org.jetbrains.annotations.Nullable;
import java.util.function.UnaryOperator;

public class TransformerBlock {

    public final MultiHeadAttention mha;
    public final FeedForwardNetwork ffn;
    public final LayerNorm norm1;
    public final LayerNorm norm2;
    public final int dModel;

    /**
     * Initializes a Transformer Block.
     *
     * @param dModel             The dimensionality of the input/output embeddings.
     * @param numHeads           Number of heads for the Multi-Head Attention.
     * @param dff                Inner dimension for the Feed-Forward Network.
     * @param ffnActivation      Activation function for the FFN (e.g., Tensor::relu).
     * @param attentionDropoutRate Dropout rate for attention mechanism (currently not directly used by MHA).
     * @param ffnDropoutRate     Dropout rate for the Feed-Forward Network.
     * @param biasForProjections If bias should be used in MHA and FFN linear layers.
     * @param requiresGrad       If the weights of MHA and FFN should require gradients.
     */
    public TransformerBlock(int dModel, int numHeads, int dff,
                            UnaryOperator<Tensor> ffnActivation,
                            @SuppressWarnings("unused") float attentionDropoutRate, // Marked as unused for now
                            float ffnDropoutRate,
                            boolean biasForProjections,
                            boolean requiresGrad) {
        this.dModel = dModel;

        // Initialize Multi-Head Attention
        // The 'debugPrinting' argument for MHA is set to false by default here.
        this.mha = new MultiHeadAttention(dModel, numHeads, biasForProjections, requiresGrad, false);

        // Initialize Feed-Forward Network
        this.ffn = new FeedForwardNetwork(dModel, dff, ffnActivation, ffnDropoutRate, biasForProjections, requiresGrad);

        // Initialize Layer Normalization layers
        // LayerNorm(dim, epsilon)
        // Epsilon is a small value to prevent division by zero.
        // Gamma and Beta are initialized and handled within LayerNorm.
        // If requiresGrad was true for the block, LayerNorm parameters should also require grad.
        // The LayerNorm class in jcog.model.LayerNorm handles its own parameters' grad status.
        this.norm1 = new LayerNorm(dModel, 1e-5);
        this.norm2 = new LayerNorm(dModel, 1e-5);

        if (requiresGrad) {
            // LayerNorm parameters (gamma and beta) are typically made parameters by default if grad is true.
            // Assuming jcog.model.LayerNorm handles this internally based on its own logic or a global setting.
            // If explicit parameter marking is needed:
            // this.norm1.gamma.grad(true).parameter();
            // this.norm1.beta.grad(true).parameter();
            // this.norm2.gamma.grad(true).parameter();
            // this.norm2.beta.grad(true).parameter();
            // For now, we rely on LayerNorm's internal handling.
        }
    }

    /**
     * Forward pass for the Transformer Block.
     *
     * @param x             Input Tensor of shape [sequence_length, dModel].
     * @param attentionMask Optional mask for the self-attention mechanism. Shape [sequence_length, sequence_length].
     * @return Output Tensor of shape [sequence_length, dModel].
     */
    public Tensor forward(Tensor x, @Nullable Tensor attentionMask) {
        // Self-Attention sub-layer
        // Query, Key, and Value are all 'x' for self-attention
        Tensor attention_output = mha.forward(x, x, x, attentionMask);

        // Residual connection and Layer Normalization
        Tensor sublayer1_input = x.add(attention_output);
        Tensor sublayer1_output = norm1.apply(sublayer1_input);

        // Feed-Forward Network sub-layer
        Tensor ffn_output = ffn.forward(sublayer1_output);

        // Residual connection and Layer Normalization
        Tensor sublayer2_input = sublayer1_output.add(ffn_output);
        Tensor output = norm2.apply(sublayer2_input);

        return output;
    }

    /**
     * Sets the training mode for components within the Transformer Block that have distinct behaviors
     * during training vs. inference (e.g., Dropout in FFN).
     *
     * @param training true if the model is in training mode, false otherwise.
     */
    public void train(boolean training) {
        ffn.train(training);
        // If MultiHeadAttention had its own dropout/mode switch, it would be called here:
        // mha.train(training);
    }
}
