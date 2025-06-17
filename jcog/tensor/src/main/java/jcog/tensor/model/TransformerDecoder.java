package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.tensor.PositionalEncoding;
// Assuming TokenEmbedding might be used here, or handled outside.
// import jcog.tensor.TokenEmbedding;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

public class TransformerDecoder {

    // private final TokenEmbedding targetTokenEmbedding; // If decoder handles its own embeddings
    private final PositionalEncoding positionalEncoding;
    private final List<TransformerDecoderBlock> decoderBlocks;
    // Potentially a final LayerNorm if the architecture applies one after all blocks
    // private final jcog.model.LayerNorm finalNorm;

    /**
     * Constructor for TransformerDecoder.
     * @param embeddingDim Dimensionality of the target embeddings (d_model).
     * @param numLayers Number of TransformerDecoderBlocks.
     * @param numHeads Number of attention heads in each block.
     * @param ffDim Dimensionality of the feed-forward network in each block.
     * @param ffnActivation Activation function for FFNs (e.g., Tensor::relu).
     * @param dropoutRate Dropout rate used in FFNs and potentially other parts.
     * @param maxSequenceLength Maximum length for positional encoding of target sequences.
     * @param biasForProjections If bias should be used in MHA and FFN linear layers.
     * @param requiresGrad If weights within the decoder blocks should require gradients.
     */
    public TransformerDecoder(
            // TokenEmbedding targetTokenEmbedding, // Pass if embedding is done here
            int embeddingDim,
            int numLayers,
            int numHeads,
            int ffDim,
            UnaryOperator<Tensor> ffnActivation,
            double dropoutRate,
            int maxSequenceLength,
            boolean biasForProjections,
            boolean requiresGrad) {

        // this.targetTokenEmbedding = targetTokenEmbedding; // Store if used
        // For decoder, PE is typically not learned for fixed sinusoidal
        this.positionalEncoding = new PositionalEncoding(maxSequenceLength, embeddingDim, false, false);

        this.decoderBlocks = new ArrayList<>(numLayers);
        for (int i = 0; i < numLayers; i++) {
            this.decoderBlocks.add(new TransformerDecoderBlock(
                embeddingDim,
                numHeads,
                ffDim,
                ffnActivation,
                (float) dropoutRate,
                biasForProjections,
                requiresGrad
            ));
        }
        // if (applyFinalNorm) { this.finalNorm = new jcog.model.LayerNorm(embeddingDim, 1e-5); }
    }

    /**
     * Forward pass for the Transformer Decoder.
     * @param targetInputEmbeddings Embedded target sequence.
     *                              Shape: [target_sequence_length, embeddingDim].
     *                              (Assumes token IDs have already been converted to embeddings).
     * @param encoderOutput Output from the Transformer Encoder.
     *                      Shape: [source_sequence_length, embeddingDim].
     * @param selfAttentionMask Mask for the self-attention mechanism in decoder blocks (look-ahead mask).
     * @param crossAttentionMask Mask for the cross-attention mechanism in decoder blocks (source padding mask).
     * @return Output Tensor from the decoder. Shape: [target_sequence_length, embeddingDim].
     */
    public Tensor forward(Tensor targetInputEmbeddings, Tensor encoderOutput,
                          @Nullable Tensor selfAttentionMask, @Nullable Tensor crossAttentionMask) {

        // 1. Apply positional encoding to target input embeddings
        Tensor decoderInput = this.positionalEncoding.forward(targetInputEmbeddings);

        // 2. Pass through the stack of TransformerDecoderBlocks
        Tensor currentOutput = decoderInput;
        for (TransformerDecoderBlock block : this.decoderBlocks) {
            currentOutput = block.forward(
                currentOutput,
                encoderOutput,
                selfAttentionMask,
                crossAttentionMask
            );
        }

        // 3. Optional: Apply a final LayerNorm
        // if (this.finalNorm != null) {
        //     currentOutput = this.finalNorm.apply(currentOutput);
        // }

        return currentOutput;
    }

    /**
     * Sets the training mode for components within the decoder (e.g., Dropout in FFNs).
     * @param training true if in training mode, false otherwise.
     */
    public void train(boolean training) {
        for (TransformerDecoderBlock block : this.decoderBlocks) {
            block.train(training); // Propagate training mode to each decoder block
        }
    }
}
