package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.tensor.TokenEmbedding;
import jcog.tensor.LMHead;
// No static import for Tensor.RELU, using Tensor::relu directly

import java.util.function.UnaryOperator;
import org.jetbrains.annotations.Nullable;

public class TransformerModel {

    private final TokenEmbedding sourceTokenEmbedding;
    private final TokenEmbedding targetTokenEmbedding;
    private final TransformerEncoder encoder;
    private final TransformerDecoder decoder;
    private final LMHead outputProjection;

    // Activation function for FFNs, typically ReLU
    private static final UnaryOperator<Tensor> FFN_ACTIVATION = Tensor::relu;

    /**
     * Constructor for the full Transformer model.
     * @param sourceVocabSize Size of the source vocabulary.
     * @param targetVocabSize Size of the target vocabulary.
     * @param embeddingDim Dimensionality of embeddings and model (d_model).
     * @param numLayers Number of layers for both encoder and decoder.
     * @param numHeads Number of attention heads.
     * @param ffDim Dimensionality of the feed-forward network inner layer.
     * @param dropoutRate Dropout rate.
     * @param maxSequenceLength Maximum sequence length for positional encodings.
     * @param shareTargetEmbeddingAndLMHeadWeights Whether to share weights (conceptual, not fully implemented).
     * @param biasForProjections If bias should be used in MHA and FFN linear layers.
     * @param requiresGrad If model weights should require gradients.
     */
    public TransformerModel(int sourceVocabSize, int targetVocabSize, int embeddingDim,
                            int numLayers, int numHeads, int ffDim, double dropoutRate,
                            int maxSequenceLength, boolean shareTargetEmbeddingAndLMHeadWeights,
                            boolean biasForProjections, boolean requiresGrad) {

        this.sourceTokenEmbedding = new TokenEmbedding(sourceVocabSize, embeddingDim, requiresGrad);
        this.targetTokenEmbedding = new TokenEmbedding(targetVocabSize, embeddingDim, requiresGrad);

        // TransformerEncoder constructor is now:
        // (embeddingDim, numLayers, numHeads, ffDim, ffnActivation, dropoutRate, maxSequenceLength, biasForProjections, requiresGrad)
        this.encoder = new TransformerEncoder(embeddingDim, numLayers, numHeads, ffDim,
                                              FFN_ACTIVATION, dropoutRate, maxSequenceLength,
                                              biasForProjections, requiresGrad);

        // TransformerDecoder constructor is:
        // (embeddingDim, numLayers, numHeads, ffDim, ffnActivation, dropoutRate, maxSequenceLength, biasForProjections, requiresGrad)
        this.decoder = new TransformerDecoder(embeddingDim, numLayers, numHeads, ffDim,
                                              FFN_ACTIVATION, dropoutRate, maxSequenceLength,
                                              biasForProjections, requiresGrad);

        // LMHead constructor is: (dModel, vocabSize, bias, requiresGrad)
        this.outputProjection = new LMHead(embeddingDim, targetVocabSize, biasForProjections, requiresGrad);

        if (shareTargetEmbeddingAndLMHeadWeights) {
            // This is a placeholder comment as true weight sharing (where gradients to one affect the other directly
            // because they are the same Tensor object or share the same underlying data, possibly transposed)
            // is complex and depends on the capabilities of the Linear and Tensor classes (e.g., using a transposed view).
            // The current TokenEmbedding and LMHead (via Linear) create independent weight Tensors.
            // For true sharing, LMHead's Linear layer would need to be initialized using targetTokenEmbedding.weight (potentially transposed).
            System.out.println("Note: 'shareTargetEmbeddingAndLMHeadWeights' is true, but true weight sharing " +
                               "requires specific API support in Linear/LMHead to use the TokenEmbedding's weight tensor. " +
                               "Currently, weights remain independent.");
        }
    }

    /**
     * Forward pass for the Transformer model.
     * @param sourceTokenIds Tensor of source token IDs. Expected to be 1D (e.g., shape [source_sequence_length]).
     *                       The TokenEmbedding layer handles converting this to [source_sequence_length, embeddingDim].
     * @param targetTokenIds Tensor of target token IDs (for training/teacher forcing). Expected to be 1D.
     * @param sourceMask Optional mask for source sequence padding to be used in encoder self-attention
     *                   and decoder cross-attention. Shape depends on MultiHeadAttention.
     * @param targetMask Optional mask for target sequence (look-ahead and padding) for decoder self-attention.
     * @param crossMask Optional mask specifically for decoder cross-attention (if different from sourceMask or needs specific shape).
     *                  Often, sourceMask is reused or adapted for this. If null, decoder might use its internally generated cross-attention mask based on encoderOutput.
     * @return Output logits from the model. Shape: [target_sequence_length, targetVocabSize].
     */
    public Tensor forward(Tensor sourceTokenIds, Tensor targetTokenIds,
                          @Nullable Tensor sourceMask, @Nullable Tensor targetMask, @Nullable Tensor crossMask) {

        // TokenEmbedding.forward expects a 1D tensor of token IDs and returns 2D tensor (numTokens, embeddingDim).
        // sourceTokenIds and targetTokenIds are assumed to be 1D here (e.g., from flattening a [N,1] or [1,N] tensor).
        Tensor sourceEmbeddings = this.sourceTokenEmbedding.forward(sourceTokenIds.flattenRow());
        Tensor targetEmbeddings = this.targetTokenEmbedding.forward(targetTokenIds.flattenRow());

        // Encoder expects (sequenceLength, embeddingDim) and an optional mask.
        Tensor encoderOutput = this.encoder.apply(sourceEmbeddings, sourceMask);

        // Decoder expects (target_sequence_length, embeddingDim) for targetEmbeddings,
        // encoderOutput, selfAttentionMask (targetMask), and crossAttentionMask.
        Tensor decoderOutput = this.decoder.forward(targetEmbeddings, encoderOutput, targetMask, crossMask);

        // LMHead projects decoder output to vocabulary logits.
        Tensor logits = this.outputProjection.forward(decoderOutput);

        return logits;
    }

    /**
     * Sets the training mode for all trainable components of the model (like Dropout layers).
     * @param training true for training mode, false for evaluation mode.
     */
    public void train(boolean training) {
        // Embeddings typically do not have dropout, so no train() method.
        // If TokenEmbedding were to include dropout, it would need a train() method.
        this.encoder.setTraining(training); // This calls train() on internal TransformerBlocks
        this.decoder.train(training);       // This calls train() on internal TransformerDecoderBlocks
        // LMHead (Linear projection) also typically does not have dropout.
    }
}
