package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.tensor.TransformerBlock;
import jcog.tensor.PositionalEncoding;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

public class TransformerEncoder implements UnaryOperator<Tensor> {

    private final List<TransformerBlock> layers;
    private final PositionalEncoding positionalEncoding;

    /**
     * Constructor for TransformerEncoder.
     * @param embeddingDim Dimensionality of the input embeddings (d_model).
     * @param numLayers Number of TransformerBlocks.
     * @param numHeads Number of attention heads in each TransformerBlock.
     * @param ffDim Dimensionality of the feed-forward network in each TransformerBlock.
     * @param ffnActivation Activation function for the FFN in TransformerBlocks.
     * @param dropoutRate Dropout rate to be used within TransformerBlocks.
     * @param maxSequenceLength Maximum length of the input sequences for positional encoding.
     * @param biasForProjections If bias should be used in MHA and FFN linear layers within blocks.
     * @param requiresGrad If weights within the blocks should require gradients.
     */
    public TransformerEncoder(int embeddingDim, int numLayers, int numHeads, int ffDim,
                              UnaryOperator<Tensor> ffnActivation, double dropoutRate,
                              int maxSequenceLength, boolean biasForProjections, boolean requiresGrad) {
        this.layers = new ArrayList<>(numLayers);
        // PositionalEncoding constructor: PositionalEncoding(maxSequenceLength, embeddingDim, learned, requiresGrad)
        this.positionalEncoding = new PositionalEncoding(maxSequenceLength, embeddingDim, false, false);

        for (int i = 0; i < numLayers; i++) {
            // TransformerBlock constructor: TransformerBlock(dModel, numHeads, dff, ffnActivation,
            // attentionDropoutRate, ffnDropoutRate, biasForProjections, requiresGrad)
            this.layers.add(new TransformerBlock(
                embeddingDim,          // dModel
                numHeads,
                ffDim,                 // dff
                ffnActivation,         // ffnActivation
                (float) dropoutRate,   // attentionDropoutRate
                (float) dropoutRate,   // ffnDropoutRate
                biasForProjections,    // biasForProjections
                requiresGrad           // requiresGrad
            ));
        }
    }

    /**
     * Applies the encoder to the input tensor without a mask.
     * @param input Tensor, typically (sequenceLength, embeddingDim).
     * @return Output tensor from the encoder.
     */
    @Override
    public Tensor apply(Tensor input) {
        // Call the main apply method with no mask.
        return apply(input, null);
    }

    /**
     * Applies the encoder to the input tensor with an optional attention mask.
     * @param input Tensor, typically (sequenceLength, embeddingDim).
     * @param mask Optional mask tensor for the self-attention mechanism in TransformerBlocks.
     * @return Output tensor from the encoder.
     */
    public Tensor apply(Tensor input, @Nullable Tensor mask) {
        // Add positional encodings using the PositionalEncoding class's forward method
        Tensor output = this.positionalEncoding.forward(input);

        // Pass through each TransformerBlock
        for (TransformerBlock layer : this.layers) {
            // TransformerBlock.forward takes (Tensor x, @Nullable Tensor attentionMask)
            output = layer.forward(output, mask);
        }
        return output;
    }

    /**
     * Sets the training mode for components like Dropout within the TransformerBlocks.
     * @param training true for training mode, false for evaluation mode.
     */
    public void setTraining(boolean training) {
       for (TransformerBlock block : this.layers) {
           // Call train method on each block
           block.train(training);
       }
    }
}
