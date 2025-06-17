package jcog.tensor;

import jcog.model.Layers; // Assuming Models.Layers -> jcog.model.Layers
import org.ejml.simple.SimpleMatrix;
import java.util.function.UnaryOperator;

public class GPTLanguageModel {

    public final TokenEmbedding tokenEmbedding;
    public final PositionalEncoding positionalEncoding;
    public final Layers transformerBlocks; // Using Models.Layers to manage a list of TransformerBlocks
    public final LMHead lmHead;

    public final int dModel;
    public final int vocabSize;
    public final int maxSequenceLength;

    public GPTLanguageModel(int vocabSize, int dModel, int maxSequenceLength,
                            int numLayers, int numHeads, int dff,
                            UnaryOperator<Tensor> ffnActivation,
                            float ffnDropoutRate,
                            boolean learnedPositionalEncoding,
                            boolean biasInProjections,
                            boolean requiresGrad) {

        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.maxSequenceLength = maxSequenceLength;

        this.tokenEmbedding = new TokenEmbedding(vocabSize, dModel, requiresGrad);
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
                requiresGrad
            );
            this.transformerBlocks.layer.add(block);
        }

        // Bias is commonly true for the LM head
        this.lmHead = new LMHead(dModel, vocabSize, true, requiresGrad);
    }

    private Tensor createCausalMask(int sequenceLength) {
        SimpleMatrix maskMatrix = new SimpleMatrix(sequenceLength, sequenceLength);
        // Initializes to all zeros by default with SimpleMatrix
        for (int i = 0; i < sequenceLength; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                if (j > i) { // Element (i,j): query i, key j. If key > query, mask.
                    maskMatrix.set(i, j, -1e9); // Large negative number to prevent attention
                } else {
                    maskMatrix.set(i, j, 0.0);  // Allow attention
                }
            }
        }
        return new Tensor(maskMatrix, false); // Mask does not require gradients
    }

    public Tensor forward(Tensor inputIds) {
        int sequenceLength = inputIds.volume(); // Works for [seq_len, 1] or [1, seq_len]

        if (sequenceLength == 0) {
             // Return empty logits or handle as an error
            return Tensor.zeros(0, vocabSize);
        }

        if (sequenceLength > maxSequenceLength) {
            throw new IllegalArgumentException("Input sequence length (" + sequenceLength +
                                               ") exceeds model's max sequence length (" + maxSequenceLength + ").");
        }

        Tensor embeddings = tokenEmbedding.forward(inputIds); // Shape: [sequenceLength, dModel]

        // Common practice: scale embeddings by sqrt(dModel)
        embeddings = embeddings.mul(Math.sqrt((double)dModel));

        embeddings = positionalEncoding.forward(embeddings); // Apply positional encoding

        Tensor causalMask = createCausalMask(sequenceLength);

        Tensor hiddenState = embeddings;
        for (Object blockObj : transformerBlocks.layer) { // Models.Layers stores layers in an ArrayList<Object>
            if (blockObj instanceof TransformerBlock) {
                TransformerBlock block = (TransformerBlock) blockObj;
                hiddenState = block.forward(hiddenState, causalMask);
            } else {
                // Handle case where an unexpected layer type is in transformerBlocks, though unlikely with current setup
                throw new IllegalStateException("TransformerBlocks layer list contains an unexpected object type: " + blockObj.getClass().getName());
            }
        }

        Tensor logits = lmHead.forward(hiddenState); // Shape: [sequenceLength, vocabSize]
        return logits;
    }

    /**
     * Sets the training mode for components within the GPT model that have distinct behaviors
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
        // If tokenEmbedding or positionalEncoding (if learned and has dropout) had a 'train' method:
        // tokenEmbedding.train(training);
        // positionalEncoding.train(training);
        // lmHead does not have dropout, so no train method needed for it.
    }
}
