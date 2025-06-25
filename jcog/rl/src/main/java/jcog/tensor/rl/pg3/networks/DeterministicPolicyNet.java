package jcog.tensor.rl.pg3.networks;

import jcog.tensor.Tensor;
import jcog.tensor.apadani.MetaWeights;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import jcog.tensor.rl.util.RLNetworkUtils;

import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * A neural network model that outputs a deterministic action given a state.
 * This is typically used as the actor network in DDPG algorithms.
 * The network architecture consists of a series of hidden layers followed by an output layer
 * that produces the action. The output activation is typically Tanh to constrain actions
 * within a specific range (e.g., [-1, 1]).
 */
public class DeterministicPolicyNet extends MetaWeights<Tensor, Tensor> {

    private final NetworkConfig config;
    private final UnaryOperator<Tensor> network;
    private final int actionDim;

    /**
     * Constructs a DeterministicPolicyNet.
     *
     * @param config    The network configuration specifying learning rate, hidden layer sizes, etc.
     * @param stateDim  The dimensionality of the state input.
     * @param actionDim The dimensionality of the action output.
     */
    public DeterministicPolicyNet(NetworkConfig config, int stateDim, int actionDim) {
        Objects.requireNonNull(config, "Network config cannot be null");
        if (stateDim <= 0) throw new IllegalArgumentException("State dimension must be positive.");
        if (actionDim <= 0) throw new IllegalArgumentException("Action dimension must be positive.");

        this.config = config;
        this.actionDim = actionDim;

        List<Integer> layerSizes = RLNetworkUtils.buildLayerSizes(stateDim, config.hiddenLayerSizes(), actionDim);

        // For DDPG, the actor's output activation is often tanh to bound actions (e.g., to [-1, 1]).
        // The layers leading up to it can use ReLU or other activations.
        this.network = RLNetworkUtils.buildSequentialNetwork(
            layerSizes,
            config.hiddenActivation(),
            config.outputActivation() != null ? config.outputActivation() : Tensor::tanh, // Default to tanh for action output
            config.useLayerNorm(),
            config.useBiasInLastLayer()
        );

        // Initialize weights if a specific initializer is configured
        if (config.weightInitializer() != null) {
            RLNetworkUtils.initializeWeights(this.network, config.weightInitializer());
        }

        // Collect parameters from the network
        getWeights().addAll(Tensor.params(this.network));
        //train(true); // Set to training mode by default, can be changed by agent
    }

    /**
     * Performs a forward pass through the network to get a deterministic action for the given state.
     *
     * @param state The input state tensor.
     * @return A tensor representing the deterministic action.
     */
    @Override
    public Tensor apply(Tensor state) {
        return this.network.apply(state);
    }

    /**
     * Gets the configuration of this network.
     *
     * @return The {@link NetworkConfig} instance.
     */
    public NetworkConfig getConfig() {
        return this.config;
    }

    /**
     * Gets the dimensionality of the action output by this network.
     * @return The action dimension.
     */
    public int getActionDim() {
        return this.actionDim;
    }

    /**
     * Sets the training mode for the network.
     * This typically propagates to layers like Dropout or BatchNorm if they are used.
     *
     * @param training True to set to training mode, false for evaluation mode.
     */
    public void train(boolean training) {
        RLNetworkUtils.setNetworkTrainMode(this.network, training);
    }
}
