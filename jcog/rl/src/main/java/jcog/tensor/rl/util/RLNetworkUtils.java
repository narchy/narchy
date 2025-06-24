package jcog.tensor.rl.util;

import jcog.tensor.Models;
import jcog.tensor.Models.Linear;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.PGBuilder.NetworkConfig;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

public final class RLNetworkUtils {

    private RLNetworkUtils() {
        // Utility class, not to be instantiated
    }

    /**
     * Creates a Multi-Layer Perceptron (MLP) with specific initialization and dropout settings
     * commonly used in RL.
     *
     * @param ins    Number of input units.
     * @param outs   Number of output units.
     * @param config Network configuration (hidden layers, activations, init, dropout).
     * @return A UnaryOperator representing the MLP.
     */
    public static UnaryOperator<Tensor> createMlp(int ins, int outs, NetworkConfig config) {
        Objects.requireNonNull(config, "NetworkConfig cannot be null for MLP creation");
        var layers = IntStream.concat(IntStream.of(ins), Arrays.stream(config.hiddenLayers())).toArray();
        layers = IntStream.concat(Arrays.stream(layers), IntStream.of(outs)).toArray();

        return new Models.Layers(config.activation(), config.outputActivation(), config.biasInLastLayer(), layers) {
            @Override
            protected Linear layer(int l, int maxLayers, int I, int O, UnaryOperator<Tensor> a, boolean bias) {
                var linearLayer = (Linear) super.layer(l, maxLayers, I, O, a, bias);
                if (config.orthogonalInit()) {
                    Tensor.orthoInit(linearLayer.weight, (l < maxLayers - 1) ? Math.sqrt(2.0) : 0.01);
                    if (bias && linearLayer.bias != null) linearLayer.bias.zero();
                }
                return linearLayer;
            }

            @Override
            protected void afterLayer(int O, int l, int maxLayers) {
                if (config.dropout() > 0 && l < maxLayers - 1) {
                    // Accessing the 'layer' field of the anonymous Models.Layers subclass
                    // This assumes 'layer' is accessible (e.g., protected or package-private in Models.Layers)
                    // and refers to the list of layers being built.
                    // If Models.Layers.layer is private, this pattern needs adjustment.
                    // Assuming 'this.layer' refers to the list of layers in Models.Layers:
                    super.layer.add(new Models.Dropout(config.dropout()));
                }
            }
        };
    }
}
