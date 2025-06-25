package jcog.tensor.rl.pg3.networks;

import jcog.tensor.Tensor;
import jcog.tensor.Mod;
import jcog.tensor.Models;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import org.jetbrains.annotations.NotNull;

import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * A deterministic policy network (Actor) for algorithms like DDPG.
 * It maps states directly to actions, rather than to parameters of a probability distribution.
 *
 * <p>The network architecture is defined by a {@link NetworkConfig}.
 * The output activation function is crucial for constraining actions to a valid range
 * (e.g., using {@code Tensor::tanh} or {@code Tensor::clipUnitPolar} to map outputs to [-1, 1]).
 */
public class DeterministicPolicyNet extends Mod {

    private final Models.Layers layers;
    private final UnaryOperator<Tensor> outputActivation;

    public DeterministicPolicyNet(@NotNull NetworkConfig config, int stateDim, int actionDim) {
        Objects.requireNonNull(config, "NetworkConfig must not be null");
        if (stateDim <= 0 || actionDim <= 0) {
            throw new IllegalArgumentException("State and action dimensions must be positive.");
        }

        this.outputActivation = config.outputActivation(); // May be null if last layer handles it

        this.layers = new Models.Layers(
            config.hiddenActivation(),
            this.outputActivation, // Output activation for the final layer
            config.biasInLastLayer(),
            config.layers(stateDim, actionDim)
        );

        // Initialize parameters (e.g., weights and biases)
        // The Models.Layers constructor typically handles initialization.
        // Specific initializations like Xavier or He can be configured within NetworkConfig if needed
        // or applied here if Models.Layers supports post-initialization hooks.
        // For now, relying on default initialization of Models.Layers.
    }

    /**
     * Performs a forward pass through the network to compute an action from a given state.
     *
     * @param state The input state tensor.
     * @return The output action tensor. The values are determined by the network's layers
     *         and the specified output activation function.
     */
    @Override
    public Tensor forward(Tensor state) {
        return this.layers.forward(state);
    }

    /**
     * Sets the training mode for the underlying layers.
     *
     * @param training True if the network should be in training mode (e.g., for dropout), false otherwise.
     */
    @Override
    public void train(boolean training) {
        this.layers.train(training);
        super.train(training);
    }

    /**
     * Returns the underlying {@link Models.Layers} instance.
     * This can be useful for accessing parameters or applying custom operations.
     *
     * @return The internal layers model.
     */
    public Models.Layers getLayers() {
        return layers;
    }

    @Override
    public String toString() {
        return "DeterministicPolicyNet(" + layers.toString() +
               (outputActivation != null ? ", outputActivation=" + outputActivation.getClass().getSimpleName() : "") +
               ')';
    }
}
