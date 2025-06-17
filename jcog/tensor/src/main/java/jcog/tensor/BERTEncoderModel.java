package jcog.tensor;

import jcog.model.Layers;     // Assuming Models.Layers -> jcog.model.Layers
import jcog.model.LayerNorm; // Assuming Models.LayerNorm -> jcog.model.LayerNorm
import org.jetbrains.annotations.Nullable;
import java.util.function.UnaryOperator;

public class BERTEncoderModel {

    public final TokenEmbedding tokenEmbedding;
    public final TokenEmbedding segmentEmbedding; // Can be null
    public final PositionalEncoding positionalEncoding;
    public final Layers transformerBlocks;
    public final LayerNorm normLayer; // Optional final normalization layer

    public final int dModel;
    public final int vocabSize;
    public final int maxSequenceLength;
    public final int numSegments;

    public BERTEncoderModel(int vocabSize, int dModel, int maxSequenceLength,
                            int numLayers, int numHeads, int dff,
                            UnaryOperator<Tensor> ffnActivation,
                            float ffnDropoutRate,
                            boolean learnedPositionalEncoding,
                            int numSegments, // e.g., 2 for sentence A/B
                            boolean biasInProjections,
                            boolean addFinalNormLayer,
                            boolean requiresGrad) {

        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.maxSequenceLength = maxSequenceLength;
        this.numSegments = numSegments;

        this.tokenEmbedding = new TokenEmbedding(vocabSize, dModel, requiresGrad);

        if (numSegments > 0) {
            this.segmentEmbedding = new TokenEmbedding(numSegments, dModel, requiresGrad);
        } else {
            this.segmentEmbedding = null;
        }

        this.positionalEncoding = new PositionalEncoding(maxSequenceLength, dModel, learnedPositionalEncoding, requiresGrad);

        this.transformerBlocks = new Layers();
        for (int i = 0; i < numLayers; i++) {
            // attentionDropoutRate is 0.0f as MHA doesn't take it directly yet
            TransformerBlock block = new TransformerBlock(
                dModel,
                numHeads,
                dff,
                ffnActivation,
                0.0f, // MHA dropout rate - currently not used by MHA constructor
                ffnDropoutRate,
                biasInProjections,
                requiresGrad,
                null // Pass null for RoPE
            );
            this.transformerBlocks.layer.add(block);
        }

        if (addFinalNormLayer) {
            this.normLayer = new LayerNorm(dModel, 1e-5);
            // Assuming LayerNorm handles its own parameters' grad status
            // If explicit parameter marking is needed and requiresGrad is true:
            // if (requiresGrad && this.normLayer.gamma != null) this.normLayer.gamma.grad(true).parameter();
            // if (requiresGrad && this.normLayer.beta != null) this.normLayer.beta.grad(true).parameter();
        } else {
            this.normLayer = null;
        }
    }

    public Tensor forward(Tensor inputIds, @Nullable Tensor attentionMask, @Nullable Tensor segmentIds) {
        int sequenceLength = inputIds.volume();

        if (sequenceLength == 0) {
            return Tensor.zeros(0, dModel); // Return empty embeddings
        }

        if (sequenceLength > maxSequenceLength) {
            throw new IllegalArgumentException("Input sequence length (" + sequenceLength +
                                               ") exceeds model's max sequence length (" + maxSequenceLength + ").");
        }

        Tensor embeddings = tokenEmbedding.forward(inputIds); // Shape: [sequenceLength, dModel]

        // Optional scaling, common practice
        embeddings = embeddings.mul(Math.sqrt((double) dModel));

        if (segmentEmbedding != null) {
            if (segmentIds == null) {
                throw new IllegalArgumentException("segmentIds are required when numSegments > 0.");
            }
            if (segmentIds.volume() != sequenceLength) {
                throw new IllegalArgumentException("segmentIds volume (" + segmentIds.volume() +
                                                   ") must match inputIds volume (" + sequenceLength + ").");
            }
            Tensor segEmbeds = segmentEmbedding.forward(segmentIds);
            embeddings = embeddings.add(segEmbeds);
        } else if (segmentIds != null) {
            // Warn or error if segment IDs provided but model doesn't support them
            System.err.println("Warning: segmentIds provided but the model was initialized with numSegments=0. Segment IDs will be ignored.");
        }


        embeddings = positionalEncoding.forward(embeddings); // Apply positional encoding

        Tensor hiddenState = embeddings;
        for (Object blockObj : transformerBlocks.layer) { // Models.Layers stores layers in an ArrayList<Object>
            if (blockObj instanceof TransformerBlock) {
                TransformerBlock block = (TransformerBlock) blockObj;
                // Pass the (non-causal) attentionMask to the TransformerBlock
                hiddenState = block.forward(hiddenState, attentionMask);
            } else {
                throw new IllegalStateException("TransformerBlocks layer list contains an unexpected object type: " + blockObj.getClass().getName());
            }
        }

        if (normLayer != null) {
            hiddenState = normLayer.apply(hiddenState);
        }

        return hiddenState; // Shape: [sequenceLength, dModel]
    }

    /**
     * Sets the training mode for components within the BERT model that have distinct behaviors
     * during training vs. inference (e.g., Dropout in FFNs of TransformerBlocks).
     *
     * @param training true if the model is in training mode, false otherwise.
     */
    public void train(boolean training) {
        for (Object blockObj : transformerBlocks.layer) {
            if (blockObj instanceof TransformerBlock) {
                ((TransformerBlock) blockObj).train(training);
            }
        }
        // If tokenEmbedding, segmentEmbedding, or positionalEncoding (if learned and has dropout)
        // had a 'train' method, it would be called here. Currently, they don't have dropout.
    }
}
