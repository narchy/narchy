package jcog.tensor;

import jcog.linear.Linear; // Assuming Models.Linear -> jcog.linear.Linear

public class LMHead {

    public final Linear projection;

    /**
     * Initializes the Language Modeling Head.
     *
     * @param dModel       Dimension of the Transformer output.
     * @param vocabSize    Size of the vocabulary.
     * @param bias         Whether the linear layer should have a bias.
     * @param requiresGrad If the projection weights require gradients.
     */
    public LMHead(int dModel, int vocabSize, boolean bias, boolean requiresGrad) {
        // Pass null for internal activation function, as it's a simple linear projection.
        this.projection = new Linear(dModel, vocabSize, null, bias);

        if (requiresGrad) {
            this.projection.weight.grad(true).parameter();
            if (bias && this.projection.bias != null) {
                this.projection.bias.grad(true).parameter();
            }
        }
    }

    /**
     * Projects the Transformer output to vocabulary logits.
     *
     * @param transformerOutput Output from the last Transformer block, shape [sequence_length, dModel].
     * @return Logits of shape [sequence_length, vocabSize].
     */
    public Tensor forward(Tensor transformerOutput) {
        return projection.apply(transformerOutput);
    }
}
