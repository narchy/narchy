package jcog.tensor.rl.pg3.networks;

import jcog.tensor.Tensor;
import jcog.tensor.Mod;
import jcog.tensor.Models;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import org.jetbrains.annotations.NotNull;

import java.util.Objects;

/**
 * A Q-value network (Critic) for algorithms like DDPG and SAC.
 * It estimates the expected return (Q-value) for a given state-action pair.
 *
 * <p>The network architecture is defined by a {@link NetworkConfig}.
 * The input to this network is typically a concatenation of state and action tensors.
 * The output is a single scalar value representing the Q-value.
 */
public class QValueNet extends Mod {

    private final Models.Layers layers;
    private final int stateDim;
    private final int actionDim;

    /**
     * Constructs a Q-Value Network.
     *
     * @param config    The network configuration.
     * @param stateDim  The dimensionality of the state space.
     * @param actionDim The dimensionality of the action space.
     */
    public QValueNet(@NotNull NetworkConfig config, int stateDim, int actionDim) {
        Objects.requireNonNull(config, "NetworkConfig must not be null");
        if (stateDim <= 0 || actionDim <= 0) {
            throw new IllegalArgumentException("State and action dimensions must be positive.");
        }
        this.stateDim = stateDim;
        this.actionDim = actionDim;

        // The input to the first layer is stateDim + actionDim
        int inputDim = stateDim + actionDim;
        int outputDim = 1; // Q-value is a single scalar

        this.layers = new Models.Layers(
            config.hiddenActivation(),
            config.outputActivation(), // Typically null for Q-value regression, or linear
            config.biasInLastLayer(),
            config.layers(inputDim, outputDim) // layers method from config should handle input/output dims
        );
        // Initialization is handled by Models.Layers constructor
    }

    /**
     * Performs a forward pass through the network to compute the Q-value.
     *
     * @param state  The input state tensor. Shape: (batch_size, state_dim)
     * @param action The input action tensor. Shape: (batch_size, action_dim)
     * @return The output Q-value tensor. Shape: (batch_size, 1)
     */
    public Tensor forward(Tensor state, Tensor action) {
        if (state.cols() != stateDim) {
            throw new IllegalArgumentException("State tensor has incorrect column dimension. Expected " + stateDim + ", got " + state.cols());
        }
        if (action.cols() != actionDim) {
            throw new IllegalArgumentException("Action tensor has incorrect column dimension. Expected " + actionDim + ", got " + action.cols());
        }
        if (state.rows() != action.rows()) {
            throw new IllegalArgumentException("State and action tensors must have the same number of rows (batch size).");
        }

        // Concatenate state and action tensors along the column dimension
        Tensor input = state.concat(action, 1); // dim=1 for column-wise concatenation
        return this.layers.forward(input);
    }

    /**
     * A convenience override for Mod's forward that expects a single Tensor.
     * This QValueNet primarily uses `forward(Tensor state, Tensor action)`.
     * If called with a single tensor, it assumes this tensor is already the concatenated state-action.
     *
     * @param concatenatedStateAction The input tensor, assumed to be state and action concatenated.
     *                               Shape: (batch_size, state_dim + action_dim)
     * @return The output Q-value tensor. Shape: (batch_size, 1)
     */
    @Override
    public Tensor forward(Tensor concatenatedStateAction) {
        if (concatenatedStateAction.cols() != stateDim + actionDim) {
            System.err.println("Warning: QValueNet.forward(Tensor) called with tensor of unexpected column dimension. " +
                               "Expected " + (stateDim + actionDim) + " (stateDim + actionDim), got " +
                               concatenatedStateAction.cols() + ". Ensure input is correctly concatenated state-action.");
        }
        return this.layers.forward(concatenatedStateAction);
    }


    /**
     * Sets the training mode for the underlying layers.
     *
     * @param training True if the network should be in training mode, false otherwise.
     */
    @Override
    public void train(boolean training) {
        this.layers.train(training);
        super.train(training);
    }

    /**
     * Returns the underlying {@link Models.Layers} instance.
     *
     * @return The internal layers model.
     */
    public Models.Layers getLayers() {
        return layers;
    }

    @Override
    public String toString() {
        return "QValueNet(" + layers.toString() +
               ", stateDim=" + stateDim +
               ", actionDim=" + actionDim +
               ')';
    }
}
