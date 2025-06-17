package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.tensor.MultiHeadAttention;
import jcog.tensor.FeedForwardNetwork;
import jcog.model.LayerNorm; // Corrected import path
import org.jetbrains.annotations.Nullable;

import java.util.function.UnaryOperator;

public class TransformerDecoderBlock {

    private final MultiHeadAttention maskedSelfAttention;
    private final MultiHeadAttention crossAttention;
    private final FeedForwardNetwork ffn;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LayerNorm norm3;

    /**
     * Constructor for TransformerDecoderBlock.
     * @param dModel Dimensionality of the input/output embeddings.
     * @param numHeads Number of attention heads for both self-attention and cross-attention.
     * @param dff Inner dimension for the Feed-Forward Network.
     * @param ffnActivation Activation function for the FFN.
     * @param dropoutRate Dropout rate for FFN. MultiHeadAttention handles its own dropout if any.
     * @param biasForProjections If bias should be used in MHA and FFN linear layers.
     * @param requiresGrad If the weights of MHA and FFN should require gradients.
     */
    public TransformerDecoderBlock(int dModel, int numHeads, int dff,
                                   UnaryOperator<Tensor> ffnActivation,
                                   float dropoutRate,
                                   boolean biasForProjections,
                                   boolean requiresGrad) {

        this.maskedSelfAttention = new MultiHeadAttention(dModel, numHeads, biasForProjections, requiresGrad, false);
        this.crossAttention = new MultiHeadAttention(dModel, numHeads, biasForProjections, requiresGrad, false);
        this.ffn = new FeedForwardNetwork(dModel, dff, ffnActivation, dropoutRate, biasForProjections, requiresGrad);

        this.norm1 = new LayerNorm(dModel, 1e-5);
        this.norm2 = new LayerNorm(dModel, 1e-5);
        this.norm3 = new LayerNorm(dModel, 1e-5);

        // Assuming LayerNorm handles its parameter 'requiresGrad' status internally
        // or through its own constructor/methods if necessary.
    }

    /**
     * Forward pass for the Transformer Decoder Block.
     * @param targetInput Tensor from the previous decoder layer or target embedding.
     *                    Shape: [target_sequence_length, dModel].
     * @param encoderOutput Output from the Transformer Encoder.
     *                      Shape: [source_sequence_length, dModel].
     * @param selfAttentionMask Mask for the self-attention mechanism (look-ahead mask).
     *                          Typically, shape: [target_sequence_length, target_sequence_length].
     * @param crossAttentionMask Mask for the cross-attention mechanism (source padding mask).
     *                           Typically, shape: [target_sequence_length, source_sequence_length].
     * @return Output Tensor of shape [target_sequence_length, dModel].
     */
    public Tensor forward(Tensor targetInput, Tensor encoderOutput,
                          @Nullable Tensor selfAttentionMask, @Nullable Tensor crossAttentionMask) {

        // 1. Masked Self-Attention sub-layer
        Tensor selfAttentionOutput = this.maskedSelfAttention.forward(
            targetInput, targetInput, targetInput, selfAttentionMask
        );
        Tensor sublayer1Input = targetInput.add(selfAttentionOutput); // Residual connection
        Tensor sublayer1Output = this.norm1.apply(sublayer1Input);    // Layer normalization

        // 2. Cross-Attention (Encoder-Decoder Attention) sub-layer
        Tensor crossAttentionOutput = this.crossAttention.forward(
            sublayer1Output, encoderOutput, encoderOutput, crossAttentionMask
        );
        Tensor sublayer2Input = sublayer1Output.add(crossAttentionOutput); // Residual connection
        Tensor sublayer2Output = this.norm2.apply(sublayer2Input);       // Layer normalization

        // 3. Feed-Forward Network sub-layer
        Tensor ffnOutput = this.ffn.forward(sublayer2Output);
        Tensor sublayer3Input = sublayer2Output.add(ffnOutput); // Residual connection
        Tensor output = this.norm3.apply(sublayer3Input);      // Layer normalization

        return output;
    }

    /**
     * Sets the training mode for components within this block (e.g., Dropout in FFN).
     * @param training true if in training mode, false otherwise.
     */
    public void train(boolean training) {
        this.ffn.train(training); // Assuming FFN has a 'train' method for its dropout
        // If MultiHeadAttention instances also have dropout or other mode-specific behavior:
        // this.maskedSelfAttention.train(training); // Assuming MHA has such a method
        // this.crossAttention.train(training);    // Assuming MHA has such a method
    }
}
