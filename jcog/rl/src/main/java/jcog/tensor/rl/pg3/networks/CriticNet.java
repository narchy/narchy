package jcog.tensor.rl.pg3.networks;

import jcog.tensor.Tensor;
import jcog.tensor.apadani.MetaWeights;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import jcog.tensor.rl.util.RLNetworkUtils;

import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * A neural network model that estimates the Q-value (state-action value).
 * This is typically used as the critic network in Actor-Critic algorithms like DDPG and SAC.
 * The network architecture usually processes the state through some layers, then concatenates
 * the action, and passes this combined representation through further layers to output a single Q-value.
 */
public class CriticNet extends MetaWeights<Tensor[], Tensor> { // Input is Tensor[]{state, action}

    private final NetworkConfig config;
    private final UnaryOperator<Tensor> network;
    private final int stateDim;
    private final int actionDim;

    /**
     * Constructs a CriticNet.
     *
     * @param config    The network configuration.
     * @param stateDim  The dimensionality of the state input.
     * @param actionDim The dimensionality of the action input.
     */
    public CriticNet(NetworkConfig config, int stateDim, int actionDim) {
        Objects.requireNonNull(config, "Network config cannot be null");
        if (stateDim <= 0) throw new IllegalArgumentException("State dimension must be positive.");
        if (actionDim <= 0) throw new IllegalArgumentException("Action dimension must be positive.");

        this.config = config;
        this.stateDim = stateDim;
        this.actionDim = actionDim;

        // Standard DDPG critic architecture:
        // Input state -> Layer1 -> Activation -> [Concat with Action] -> Layer2 -> Activation -> Output Q-value
        // We need to define how layers are structured around the action concatenation point.
        // A common approach is to have one or more layers process the state first,
        // then concatenate actions, then pass through more layers.

        // Example: state -> fc1_dims -> action_input_layer_dims -> fc2_dims -> 1 (Q-value)
        // Let's assume hiddenLayerSizes refers to layers after state processing and action concatenation.
        // If hiddenLayerSizes = [256, 256], then:
        // state -> L1 (e.g., 256) -> concat(action) -> L2 (256) -> L3 (256) -> Q-value
        // Or, state -> L1_s, action -> L1_a, concat(L1_s_out, L1_a_out) -> L2 -> Q-value (more complex)

        // Simpler approach: state -> fc1 -> relu -> concat(output_fc1, action) -> fc2 -> relu -> fc3 -> Q-value
        // Let config.hiddenLayerSizes() define the layers *after* state and action are combined.
        // The first hidden layer size in config could be the output size for the state's initial processing.

        if (config.hiddenLayerSizes() == null || config.hiddenLayerSizes().length < 1) {
            throw new IllegalArgumentException("CriticNet requires at least one hidden layer size in NetworkConfig for post-concatenation processing.");
        }

        // Layer to process state before action concatenation
        int firstStateLayerOutputDim = config.hiddenLayerSizes()[0]; // Use first hidden size for state pre-processing output
        UnaryOperator<Tensor> stateProcessingLayer = RLNetworkUtils.buildLinearLayer(
            stateDim, firstStateLayerOutputDim, config.hiddenActivation(), config.useLayerNorm(), true // bias
        );

        // Layers after state-action concatenation
        // Input to these layers will be firstStateLayerOutputDim + actionDim
        List<Integer> postConcatLayerSizes = RLNetworkUtils.buildLayerSizes(
            firstStateLayerOutputDim + actionDim,
            // Pass remaining hidden layer sizes, or just one if only one was specified for state processing
            config.hiddenLayerSizes().length > 1 ?
                java.util.Arrays.copyOfRange(config.hiddenLayerSizes(), 1, config.hiddenLayerSizes().length)
                : new int[]{}, // If only one hidden size, means it was for state, post-concat goes straight to output layer or one more hidden
            1 // Output dimension is 1 (the Q-value)
        );

        // If hiddenLayerSizes had only one entry, it was used for state processing.
        // Add another layer of similar size before output if postConcatLayerSizes is only [1]
        if (postConcatLayerSizes.size() == 1 && postConcatLayerSizes.get(0) == 1 && config.hiddenLayerSizes().length ==1) {
             postConcatLayerSizes = List.of(firstStateLayerOutputDim, 1);
        }


        UnaryOperator<Tensor> postConcatNetwork = RLNetworkUtils.buildSequentialNetwork(
            postConcatLayerSizes,
            config.hiddenActivation(),
            config.outputActivation() != null ? config.outputActivation() : Tensor.LINEAR, // Q-values are typically unbounded
            config.useLayerNorm(),
            config.useBiasInLastLayer()
        );

        this.network = stateAndAction -> {
            Tensor state = stateAndAction[0];
            Tensor action = stateAndAction[1];

            Tensor processedState = stateProcessingLayer.apply(state);
            Tensor concatenated = Tensor.concatColumns(processedState, action);
            return postConcatNetwork.apply(concatenated);
        };


        // Initialize weights if a specific initializer is configured
        // This needs to be applied carefully if network is composite.
        // Assuming stateProcessingLayer and postConcatNetwork handle their own params internally via RLNetworkUtils.
        if (config.weightInitializer() != null) {
            RLNetworkUtils.initializeWeights(stateProcessingLayer, config.weightInitializer());
            RLNetworkUtils.initializeWeights(postConcatNetwork, config.weightInitializer());
        }

        // Collect parameters
        getWeights().addAll(Tensor.params(stateProcessingLayer));
        getWeights().addAll(Tensor.params(postConcatNetwork));
        //train(true); // Set to training mode by default
    }

    /**
     * Performs a forward pass through the network to get the Q-value for the given state-action pair.
     *
     * @param stateAction A Tensor array where stateAction[0] is the state and stateAction[1] is the action.
     *                    Both state and action should be row vectors (batch_size x dim).
     * @return A tensor representing the Q-value(s), typically of shape (batch_size, 1).
     */
    @Override
    public Tensor apply(Tensor[] stateAction) {
        if (stateAction == null || stateAction.length != 2) {
            throw new IllegalArgumentException("Input must be a Tensor array of length 2: {state, action}");
        }
        Tensor state = stateAction[0];
        Tensor action = stateAction[1];

        if (state.cols() != this.stateDim) {
            throw new IllegalArgumentException("State tensor has incorrect column dimension. Expected " + this.stateDim + ", got " + state.cols());
        }
        if (action.cols() != this.actionDim) {
            throw new IllegalArgumentException("Action tensor has incorrect column dimension. Expected " + this.actionDim + ", got " + action.cols());
        }
        if (state.rows() != action.rows()) {
            throw new IllegalArgumentException("State and Action tensors must have the same number of rows (batch size).");
        }

        return this.network.apply(new Tensor[]{state, action});
    }

    /**
     * Gets the configuration of this network.
     * @return The {@link NetworkConfig} instance.
     */
    public NetworkConfig getConfig() {
        return this.config;
    }

    /**
     * Sets the training mode for the network.
     * @param training True to set to training mode, false for evaluation mode.
     */
    public void train(boolean training) {
        // This needs to be propagated to the underlying network components
        // if they have layers like Dropout or BatchNorm.
        // RLNetworkUtils.buildSequentialNetwork handles this for its layers.
        // If stateProcessingLayer is a complex model, it might need its own train(boolean) call.
        // For now, assuming RLNetworkUtils handles layers within the built components.
        if (this.network instanceof MetaWeights) { // Should not be the case here with current build
             ((MetaWeights<?,?>)this.network).getWeights().forEach(p -> { if (p instanceof MetaWeights) ((MetaWeights<?,?>)p).train(training);});
        }
        // More robustly, if stateProcessingLayer and postConcatNetwork are accessible and are MetaWeights based:
        // RLNetworkUtils.setNetworkTrainMode(stateProcessingLayer, training);
        // RLNetworkUtils.setNetworkTrainMode(postConcatNetwork, training);
        // For now, this is a simplified version. If specific layers (BatchNorm, Dropout) are directly
        // part of stateProcessingLayer or postConcatNetwork, their train() methods are called by
        // the RLNetworkUtils.setNetworkTrainMode if it's applied to them.
        // The MetaWeights.train() method itself doesn't exist.
        // We need to ensure that `RLNetworkUtils.setNetworkTrainMode` is called on the actual UnaryOperator<Tensor>
        // instances that make up the network if they contain trainable layers affected by mode.
        // The current `network` field is a lambda, so we'd need access to its components.
        // This detail is important if layer norm or dropout is used.
        // For simplicity, let's assume `RLNetworkUtils.buildSequentialNetwork` and `buildLinearLayer`
        // construct layers that correctly respond to `Tensor.setTrain(boolean)` if it's called on them.
        // The current Tensor framework might handle this via a global training flag or per-tensor graph.
        // Let's assume, for now, that individual layers used (Linear, etc.) correctly handle training mode
        // if their `train(boolean)` is called, or if Tensor library has a global mode.
        // The `MetaWeights` class itself does not have a `train` method.
        // The `train` method is typically on the model that *uses* the MetaWeights (like GaussianPolicyNet).

        // TODO: Clarify how training mode is propagated to layers created by RLNetworkUtils.
        // For now, this method is a placeholder if deeper propagation is needed.
        // If layers are simple Linear layers without Dropout/BatchNorm, this might not be critical.
    }
}
