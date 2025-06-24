package jcog.tensor.rl.agents.pg.networks;

import jcog.tensor.Models;
import jcog.tensor.Tensor;
import jcog.tensor.rl.agents.pg.configs.NetworkConfig; // Use new config
import jcog.tensor.rl.util.RLNetworkUtils; // For createMlp

import java.util.Objects;
import java.util.function.UnaryOperator;

// Renamed to ValueNet
public class ValueNet extends Models.Layers {
    public final NetworkConfig config;
    public final int stateDim;
    public final UnaryOperator<Tensor> network; // The actual MLP

    public ValueNet(NetworkConfig config, int stateDim) {
        super(null, null, false);
        this.config = Objects.requireNonNull(config, "NetworkConfig cannot be null");
        this.stateDim = stateDim;

        this.network = RLNetworkUtils.createMlp(stateDim, 1, config); // Output dimension is 1 for value

        // Add the created network to this.layer so its parameters are discoverable
        // and train() mode propagates. This assumes RLNetworkUtils.createMlp returns
        // an instance that can be added to Models.Layers's internal list,
        // ideally an instance of Models.Model or Models.Layers itself.
        if (this.network instanceof Models.Model) { // Models.Model is parent of Models.Layers
            this.layer.add((Models.Model) this.network);
        } else {
            // This case might indicate that parameters of 'network' are not automatically found by Tensor.parameters(thisValueNetInstance)
            // or that train() mode is not propagated correctly.
            // For a simple UnaryOperator<Tensor> that is not a Models.Model, special handling for parameters and train mode might be needed.
            // However, createMlp is expected to return a proper network structure.
            System.err.println("Warning: ValueNet's MLP (" + this.network.getClass().getName() + ") from RLNetworkUtils.createMlp might not be fully integrated for parameter discovery or train mode propagation if not a jcog.tensor.Models.Model instance and added to the internal layer list.");
        }
    }

    @Override
    public Tensor apply(Tensor state) {
        return this.network.apply(state);
    }

    @Override
    public void train(boolean training) {
        super.train(training); // Propagates to models/layers added to this.layer
        // If `network` is a Models.Layers (or Model) and was added to `this.layer`,
        // super.train(training) handles it.
        // If `network` has its own train method and wasn't added (or isn't a Model), it might need explicit call:
        // if (this.network instanceof Models.Layers) { // Defensive check
        //    ((Models.Layers) this.network).train(training);
        // }
        // However, the expectation is that adding to this.layer is sufficient if it's a Model.
    }
}
