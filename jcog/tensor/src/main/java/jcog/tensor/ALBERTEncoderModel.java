package jcog.tensor;

import jcog.model.LayerNorm;
import org.jetbrains.annotations.Nullable;
import java.util.function.UnaryOperator;

/**
 * ALBERT Encoder Model.
 * Features factorized token embeddings and shared Transformer layers.
 */
public class ALBERTEncoderModel {

    public final FactorizedTokenEmbedding factorizedTokenEmbedding;
    public final PositionalEncoding positionalEncoding;
    public final TransformerBlock sharedTransformerBlock;
    public final LayerNorm normLayer;
    public final int dModel;
    public final int vocabSize;
    public final int maxSequenceLength;
    public final int numLayers;
    private final RotaryPositionalEncoding rope; // Optional RoPE

    /**
     * Constructor for ALBERTEncoderModel.
     *
     * @param vocabSize                Size of the vocabulary.
     * @param embeddingDimE            Dimension of the intermediate factorized embedding (E).
     * @param dModel                   Dimension of the model's hidden states (H, also output of embedding).
     * @param maxSequenceLength        Maximum sequence length the model can handle.
     * @param numLayers                Number of times the shared TransformerBlock is applied.
     * @param numHeads                 Number of attention heads in the shared TransformerBlock.
     * @param dff                      Dimension of the feed-forward network's inner layer in TransformerBlock.
     * @param ffnActivation            Activation function for the FFN in TransformerBlock (e.g., Tensor::gelu).
     * @param ffnDropoutRate           Dropout rate for the FFN in TransformerBlock.
     * @param learnedPositionalEncoding If true, positional encodings are learned; otherwise, sinusoidal.
     * @param biasInProjections        If true, bias is used in MultiHeadAttention and FFN linear layers.
     * @param addFinalNormLayer        If true, a LayerNorm is applied after the final Transformer layer.
     * @param requiresGrad             If true, model parameters will track gradients.
     * @param rope                     Optional RotaryPositionalEncoding instance.
     */
    public ALBERTEncoderModel(
            int vocabSize,
            int embeddingDimE,
            int dModel,
            int maxSequenceLength,
            int numLayers,
            int numHeads,
            int dff,
            UnaryOperator<Tensor> ffnActivation,
            float ffnDropoutRate,
            boolean learnedPositionalEncoding,
            boolean biasInProjections,
            boolean addFinalNormLayer,
            boolean requiresGrad,
            @Nullable RotaryPositionalEncoding rope
    ) {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.maxSequenceLength = maxSequenceLength;
        this.numLayers = numLayers;
        this.rope = rope; // Store RoPE instance

        this.factorizedTokenEmbedding = new FactorizedTokenEmbedding(vocabSize, embeddingDimE, dModel, requiresGrad);
        this.positionalEncoding = new PositionalEncoding(maxSequenceLength, dModel, learnedPositionalEncoding, requiresGrad);

        // Assuming TransformerBlock constructor is updated to accept 'rope'
        // and pass it to its MultiHeadAttention instance.
        // The MHA dropout rate is set to 0.0f as it's often not used directly in MHA,
        // FFN dropout is handled by ffnDropoutRate.
        this.sharedTransformerBlock = new TransformerBlock(
                dModel,
                numHeads,
                dff,
                ffnActivation,
                0.0f, // mhaDropoutRate - typically 0 for ALBERT/BERT MHA layers themselves
                ffnDropoutRate,
                biasInProjections,
                requiresGrad,
                this.rope // Pass the rope instance to the TransformerBlock
        );

        this.normLayer = addFinalNormLayer ? new LayerNorm(dModel, 1e-5) : null;
        // LayerNorm typically handles its own requiresGrad internally based on its parameters.
        // If 'requiresGrad' is true for the model, LayerNorm's gamma/beta should also be learnable.
        // The LayerNorm constructor initializes gamma/beta with .grad(true).parameter().
    }

    /**
     * Forward pass for the ALBERT encoder model.
     *
     * @param inputIds      Tensor of input token IDs. Expected shape by FactorizedTokenEmbedding e.g., [sequenceLength, 1] or [sequenceLength].
     * @param attentionMask Optional attention mask (e.g., for padding). Shape typically [sequenceLength, sequenceLength] or broadcastable.
     * @return Output tensor from the final Transformer layer, shape [sequenceLength, dModel].
     */
    public Tensor forward(Tensor inputIds, @Nullable Tensor attentionMask) {
        // Determine sequence length. FactorizedTokenEmbedding handles various input shapes.
        // For simplicity, let's assume inputIds.rows() is a reliable way if it's [seq_len, 1]
        // or inputIds.volume() if it's a flat vector that TokenEmbedding_E can handle.
        // FactorizedTokenEmbedding itself should validate/interpret inputIds shape.
        int sequenceLength = (inputIds.cols() == 1) ? inputIds.rows() : inputIds.volume();


        if (sequenceLength == 0) {
            // Or throw an error, or return an empty tensor of appropriate shape.
            return Tensor.zeros(0, this.dModel);
        }
        if (sequenceLength > this.maxSequenceLength) {
            // Or truncate, or throw error. For now, let positional encoding handle it (it might truncate or error).
            System.err.println("Warning: Input sequence length (" + sequenceLength +
                               ") exceeds model's maxSequenceLength (" + this.maxSequenceLength + "). Behavior depends on PositionalEncoding.");
        }

        // 1. Token Embeddings
        Tensor embeddings = this.factorizedTokenEmbedding.forward(inputIds); // Shape [sequenceLength, dModel]

        // 2. Scale embeddings (common practice, e.g., in original Transformer)
        // This scaling factor might be embeddingDimE if dModel was different from FactorizedOutputDim
        // But here, factorizedTokenEmbedding outputs dModel directly.
        embeddings = embeddings.mul(Math.sqrt((double) this.dModel));

        // 3. Add Positional Encodings
        // Note: If RoPE is used, traditional positional encoding might be skipped or modified,
        // but ALBERT typically uses standard positional encodings. RoPE is applied per-head in MHA.
        embeddings = this.positionalEncoding.forward(embeddings); // Shape [sequenceLength, dModel]

        // 4. Pass through shared Transformer layers
        Tensor hiddenState = embeddings;
        for (int i = 0; i < this.numLayers; i++) {
            // System.out.println("ALBERT Layer: " + i); // for debugging
            hiddenState = this.sharedTransformerBlock.forward(hiddenState, attentionMask);
        }

        // 5. Optional final Layer Normalization
        if (this.normLayer != null) {
            hiddenState = this.normLayer.apply(hiddenState);
        }

        return hiddenState;
    }

    /**
     * Sets the model (specifically, components like Dropout in FFN) to training or evaluation mode.
     *
     * @param training If true, sets to training mode; otherwise, evaluation mode.
     */
    public void train(boolean training) {
        // Propagate training mode to components that have different behavior during train/eval (e.g., Dropout)
        this.sharedTransformerBlock.train(training);
        // FactorizedTokenEmbedding and PositionalEncoding as implemented currently don't have dropout
        // or other training-mode-specific behaviors. If they did, they'd need a train() method too.
        // LayerNorm also typically doesn't change behavior, but if it did (e.g. Batch Norm like behavior), it would need it.
    }

    // Example main for conceptual testing (actual compilation/running depends on full environment)
    public static void main(String[] args) {
        int vocabSize = 1000;
        int embeddingDimE = 128; // ALBERT's factorized embedding dimension
        int dModel = 768;        // ALBERT's hidden dimension (H)
        int maxSeqLen = 64;
        int numLayers = 3; // Number of times to apply the shared block
        int numHeads = 12;
        int dff = dModel * 4; // Standard FFN expansion

        // RoPE setup (optional)
        RotaryPositionalEncoding ropeInstance = null; // No RoPE for this example
        // To enable RoPE:
        // int d_k = dModel / numHeads;
        // if (d_k % 2 == 0) { // RoPE dim must be even
        //     ropeInstance = new RotaryPositionalEncoding(d_k, maxSeqLen, 10000.0);
        // } else {
        //     System.err.println("Cannot enable RoPE as d_k (" + d_k + ") is not even.");
        // }


        ALBERTEncoderModel albert = new ALBERTEncoderModel(
                vocabSize, embeddingDimE, dModel, maxSeqLen, numLayers, numHeads, dff,
                Tensor::gelu, // FFN activation
                0.1f,      // FFN dropout
                false,     // Learned positional encoding (false for sinusoidal)
                true,      // Bias in projections
                true,      // Add final norm layer
                true,      // Requires grad
                ropeInstance    // RoPE instance
        );

        System.out.println("ALBERT Model Initialized.");
        System.out.println("Factorized Embedding E->H: " + albert.factorizedTokenEmbedding.embeddingDimE + " -> " + albert.factorizedTokenEmbedding.hiddenDimH);
        System.out.println("Positional Encoding Max Len: " + albert.positionalEncoding.maxSequenceLength);
        System.out.println("Shared Transformer Block: dModel=" + albert.sharedTransformerBlock.dModel + ", Heads=" + albert.sharedTransformerBlock.mha.numHeads);
        if (albert.rope != null) {
            System.out.println("RoPE enabled with dim: " + albert.rope.dim);
        } else {
            System.out.println("RoPE not enabled.");
        }


        // Dummy input
        Tensor inputIds = new Tensor(new double[][]{{10}, {20}, {30}, {0}}); // Batch 1, Seq 4
        System.out.println("\nInput IDs shape: " + inputIds.shapeStr());

        // Dummy attention mask (e.g., no padding)
        // Tensor attentionMask = Tensor.ones(inputIds.rows(), inputIds.rows()); // No padding mask
        Tensor attentionMask = null; // No mask

        albert.train(false); // Set to eval mode for this forward pass
        Tensor output = albert.forward(inputIds, attentionMask);

        System.out.println("\nALBERT Output shape: " + output.shapeStr()); // Expected: [4, 768]
        System.out.println("Output sample (first 5 values of first vector):");
        for(int i=0; i<Math.min(5, output.cols()); i++) {
            System.out.printf("%.4f ", output.getAsDouble(0,i));
        }
        System.out.println();
    }
}
