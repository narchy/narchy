package jcog.tensor.model;

import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.NotNull;

/**
 * Provides absolute positional encoding for sequences, either fixed (sinusoidal) or learned.
 * <p>
 * These encodings are typically added directly to the input token embeddings to provide the model
 * with information about the position of tokens in the sequence.
 * <p>
 * This method is common in classic Transformer architectures like BERT and the original GPT.
 * For newer Transformer models, {@link RotaryPositionalEncoding} (RoPE)
 * is often preferred as it applies relative positional information by modifying query and key vectors.
 *
 * @see RotaryPositionalEncoding
 */
public class PositionalEncoding {

    private final int maxSequenceLength;
    private final int embeddingDim;
    private final boolean learned;
    private final Tensor encodings; // Either learned positionalEmbeddings or fixed sinusoidalEncodings

    // Static helper method for generating sinusoidal encodings
    private static SimpleMatrix generateSinusoidalMatrix(int maxSeqLen, int embDim) {
        SimpleMatrix m = new SimpleMatrix(maxSeqLen, embDim);
        double[] data = Tensor.array(m);
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < embDim; i++) {
                // (2*(i/2)) ensures pairs use same divTerm for sin and cos
                double divTerm = Math.pow(10000.0, (double)(2 * (i / 2)) / embDim);
                double angle = (double)pos / divTerm;
                int offset = pos * embDim + i;
                if (i % 2 == 0) { // Even index: sin
                    data[offset] = Math.sin(angle);
                } else { // Odd index: cos
                    data[offset] = Math.cos(angle);
                }
            }
        }
        return m;
    }

    public PositionalEncoding(int maxSequenceLength, int embeddingDim, boolean learned, boolean requiresGrad) {
        this.maxSequenceLength = maxSequenceLength;
        this.embeddingDim = embeddingDim;
        this.learned = learned;

        if (learned) {
            this.encodings = Tensor.randGaussian(maxSequenceLength, embeddingDim, 0.02);
            if (requiresGrad) {
                this.encodings.grad(true).parameter();
            }
        } else {
            // Pre-calculate sinusoidal encodings using the static helper method
            this.encodings = new Tensor(generateSinusoidalMatrix(maxSequenceLength, embeddingDim), false); // Not trainable, no grad
        }
    }

    public Tensor forward(@NotNull Tensor inputEmbeddings) {
        int sequenceLength = inputEmbeddings.rows();
        if (inputEmbeddings.cols() != this.embeddingDim) {
            throw new IllegalArgumentException("Input embedding dimension (" + inputEmbeddings.cols() +
                                               ") does not match PositionalEncoding dimension (" + this.embeddingDim + ").");
        }
        if (sequenceLength > this.maxSequenceLength) {
            throw new IllegalArgumentException("Input sequence length (" + sequenceLength +
                                               ") exceeds maximum sequence length (" + this.maxSequenceLength + ").");
        }

        // Slice the required part of positional encodings
        // Tensor.slice(rowStart, rowEnd, colStart, colEnd) -> parameters are exclusive for end
        Tensor positionalEncodingSlice = this.encodings.slice(0, sequenceLength, 0, this.embeddingDim);

        // The add operation should handle gradients correctly.
        // If this.encodings is learned and requires grad, its gradients will be updated.
        // Gradients will also flow through to inputEmbeddings.
        return inputEmbeddings.add(positionalEncodingSlice);
    }
}
