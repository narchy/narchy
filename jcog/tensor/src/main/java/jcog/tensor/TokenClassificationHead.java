package jcog.tensor;

import jcog.model.Dropout; // Assuming Models.Dropout -> jcog.model.Dropout
import jcog.linear.Linear;   // Assuming Models.Linear -> jcog.linear.Linear

public class TokenClassificationHead {

    public final Dropout dropout; // Optional dropout layer
    public final Linear linear;   // The classification layer
    public final int numClasses;

    /**
     * Initializes a Token Classification Head.
     *
     * @param dModel       Dimension of the input token embeddings (e.g., BERT's hidden size).
     * @param numClasses   Number of output classes for each token.
     * @param dropoutRate  Dropout rate to apply before the linear layer. If 0, dropout is not applied.
     * @param bias         Whether the linear layer should have a bias.
     * @param requiresGrad If the linear layer's weights require gradients.
     */
    public TokenClassificationHead(int dModel, int numClasses, float dropoutRate,
                                   boolean bias, boolean requiresGrad) {
        this.numClasses = numClasses;

        if (dropoutRate > 0) {
            this.dropout = new Dropout(dropoutRate);
        } else {
            this.dropout = null;
        }

        // Initialize the linear classification layer. Pass null for internal activation.
        this.linear = new Linear(dModel, numClasses, null, bias);

        if (requiresGrad) {
            this.linear.weight.grad(true).parameter();
            if (bias && this.linear.bias != null) {
                this.linear.bias.grad(true).parameter();
            }
        }
    }

    /**
     * Forward pass for token classification.
     *
     * @param tokenEmbeddings Tensor of contextualized embeddings for each token in the sequence.
     *                        Expected shape [sequence_length, dModel].
     * @return Logits of shape [sequence_length, numClasses].
     */
    public Tensor forward(Tensor tokenEmbeddings) {
        if (tokenEmbeddings.cols() != this.linear.weight.cols()) { // linear.weight is [out, in] so cols is input_dim
             throw new IllegalArgumentException("Input tokenEmbeddings dimension (" + tokenEmbeddings.cols() +
                                               ") does not match linear layer input dimension (" + this.linear.weight.cols() + ").");
        }

        Tensor embeddingsToClassify = tokenEmbeddings;

        if (dropout != null) {
            // Dropout is applied on the [sequence_length, dModel] tensor
            embeddingsToClassify = dropout.apply(embeddingsToClassify);
        }

        // Linear layer applies row-wise if input is [N, M] and weight is [K, M] (transposed internally to [M, K])
        // Resulting in [N, K] which is [sequence_length, numClasses]
        Tensor logits = linear.apply(embeddingsToClassify);
        return logits;
    }

    /**
     * Sets the training mode for the dropout layer.
     *
     * @param training true for training (dropout applied), false for inference (dropout bypassed).
     */
    public void train(boolean training) {
        if (dropout != null) {
            dropout.training = training;
        }
    }
}
