package jcog.tensor;

import jcog.model.Dropout; // Assuming Models.Dropout -> jcog.model.Dropout
import jcog.linear.Linear;   // Assuming Models.Linear -> jcog.linear.Linear

public class SequenceClassificationHead {

    public final Dropout dropout; // Optional dropout layer
    public final Linear linear;   // The classification layer
    public final int numClasses;

    /**
     * Initializes a Sequence Classification Head.
     *
     * @param dModel       Dimension of the input sequence representation (e.g., BERT's hidden size).
     * @param numClasses   Number of output classes.
     * @param dropoutRate  Dropout rate to apply before the linear layer. If 0, dropout is not applied.
     * @param bias         Whether the linear layer should have a bias.
     * @param requiresGrad If the linear layer's weights require gradients.
     */
    public SequenceClassificationHead(int dModel, int numClasses, float dropoutRate,
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
     * Forward pass for sequence classification.
     *
     * @param sequenceRepresentation Tensor representing the entire sequence, typically the embedding of the [CLS] token.
     *                               Expected shape [1, dModel].
     * @return Logits of shape [1, numClasses].
     */
    public Tensor forward(Tensor sequenceRepresentation) {
        if (sequenceRepresentation.rows() != 1) {
            throw new IllegalArgumentException("Input sequenceRepresentation must have exactly one row " +
                                               "(e.g., [CLS] token embedding). Found shape: " +
                                               sequenceRepresentation.shapeStr());
        }
        if (sequenceRepresentation.cols() != this.linear.weight.cols()) { // linear.weight is [out, in] so cols is input_dim
             throw new IllegalArgumentException("Input sequenceRepresentation dimension (" + sequenceRepresentation.cols() +
                                               ") does not match linear layer input dimension (" + this.linear.weight.cols() + ").");
        }


        Tensor representationToClassify = sequenceRepresentation;

        if (dropout != null) {
            // Dropout is applied on the [1, dModel] tensor
            representationToClassify = dropout.apply(representationToClassify);
        }

        Tensor logits = linear.apply(representationToClassify); // Output shape [1, numClasses]
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
