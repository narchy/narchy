package jcog.tensor;

import jcog.linear.Linear; // Assuming Models.Linear -> jcog.linear.Linear
import jcog.model.Dropout; // Assuming Models.Dropout -> jcog.model.Dropout
import java.util.function.UnaryOperator;

public class FeedForwardNetwork {

    public final Linear linear1;
    public final UnaryOperator<Tensor> activation;
    public final Linear linear2;
    public final Dropout dropout; // Can be null if dropoutRate is 0

    /**
     * Initializes a position-wise Feed-Forward Network.
     *
     * @param dModel             Input and output dimension.
     * @param dff                Inner dimension (dimension of the hidden layer).
     * @param activationFunction The activation function to use (e.g., Tensor::relu).
     * @param dropoutRate        Dropout rate. If 0, dropout is not applied.
     * @param biasLinear         Boolean indicating if bias should be used in linear layers.
     * @param requiresGrad       If true, linear layer weights should be parameters and require gradients.
     */
    public FeedForwardNetwork(int dModel, int dff, UnaryOperator<Tensor> activationFunction,
                              float dropoutRate, boolean biasLinear, boolean requiresGrad) {

        // Initialize linear1: dModel -> dff. Pass null for internal activation.
        this.linear1 = new Linear(dModel, dff, null, biasLinear);

        this.activation = activationFunction;

        // Initialize linear2: dff -> dModel. Pass null for internal activation.
        this.linear2 = new Linear(dff, dModel, null, biasLinear);

        if (dropoutRate > 0) {
            this.dropout = new Dropout(dropoutRate);
        } else {
            this.dropout = null;
        }

        if (requiresGrad) {
            this.linear1.weight.grad(true).parameter();
            if (biasLinear && this.linear1.bias != null) {
                this.linear1.bias.grad(true).parameter();
            }

            this.linear2.weight.grad(true).parameter();
            if (biasLinear && this.linear2.bias != null) {
                this.linear2.bias.grad(true).parameter();
            }
        }
    }

    /**
     * Forward pass for the Feed-Forward Network.
     *
     * @param x Input Tensor of shape [sequence_length, dModel].
     * @return Output Tensor of shape [sequence_length, dModel].
     */
    public Tensor forward(Tensor x) {
        Tensor hidden = linear1.apply(x);
        Tensor activated = activation.apply(hidden);

        Tensor postDropout = activated;
        if (dropout != null) {
            postDropout = dropout.apply(activated); // apply() uses the 'training' field internally
        }

        Tensor output = linear2.apply(postDropout);
        return output;
    }

    /**
     * Sets the training mode for the dropout layer.
     * @param training true for training (dropout applied), false for inference (dropout bypassed).
     */
    public void train(boolean training) {
        if (dropout != null) {
            dropout.training = training;
        }
    }
}
