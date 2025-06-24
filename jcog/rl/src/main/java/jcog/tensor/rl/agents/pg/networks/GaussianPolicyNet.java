package jcog.tensor.rl.agents.pg.networks;

import jcog.tensor.Models;
import jcog.tensor.Models.Linear;
import jcog.tensor.Tensor;
import jcog.tensor.rl.agents.pg.configs.NetworkConfig; // Use new config
// Uses GaussianDistribution from the new util package
import jcog.tensor.rl.agents.pg.util.AgentUtils;

import java.util.Objects;
import java.util.function.UnaryOperator;

public class GaussianPolicyNet extends Models.Layers {
    public final NetworkConfig config;
    public final int inputs;
    public final int outputs;
    public final Models.Layers body;
    public final Linear muHead;
    public final Linear logSigmaHead;

    private static int getLastLayerSize(NetworkConfig config, int inputs) {
        if (config.hiddenLayers() == null || config.hiddenLayers().length == 0) {
            return inputs;
        }
        return config.hiddenLayers()[config.hiddenLayers().length - 1];
    }

    private static Linear createOrthogonalLinear(int I, int O, UnaryOperator<Tensor> activation, boolean bias, double gain) {
        var linearLayer = new Linear(I, O, activation, bias);
        Tensor.orthoInit(linearLayer.weight, gain);
        if (bias && linearLayer.bias != null) {
            linearLayer.bias.zero();
        }
        return linearLayer;
    }

    private static Models.Layers buildDefaultMlpBody(NetworkConfig config, int inputs) {
        Models.Layers newBody = new Models.Layers(config.activation(), config.activation(), true);
        int lastLayerSize = inputs;
        if (config.hiddenLayers() != null) {
            for (int hiddenSize : config.hiddenLayers()) {
                newBody.layer.add(createOrthogonalLinear(lastLayerSize, hiddenSize, config.activation(), true, Math.sqrt(2.0)));
                lastLayerSize = hiddenSize;
            }
        }
        return newBody;
    }

    public GaussianPolicyNet(NetworkConfig config, int inputs, int outputs) {
        this(config, inputs, outputs,
             buildDefaultMlpBody(config, inputs),
             createOrthogonalLinear(getLastLayerSize(config, inputs), outputs, null, config.biasInLastLayer(), 0.01),
             createOrthogonalLinear(getLastLayerSize(config, inputs), outputs, null, config.biasInLastLayer(), 0.01)
        );
    }

    public GaussianPolicyNet(NetworkConfig config, int inputs, int outputs, Models.Layers body, Linear muHead, Linear logSigmaHead) {
        super(null, null, false);
        this.config = Objects.requireNonNull(config, "NetworkConfig cannot be null");
        this.inputs = inputs;
        this.outputs = outputs;
        this.body = Objects.requireNonNull(body, "Body cannot be null");
        this.muHead = Objects.requireNonNull(muHead, "muHead cannot be null");
        this.logSigmaHead = Objects.requireNonNull(logSigmaHead, "logSigmaHead cannot be null");

        this.layer.add(this.body);
        this.layer.add(this.muHead);
        this.layer.add(this.logSigmaHead);
    }

    public AgentUtils.GaussianDistribution getDistribution(Tensor state, float sigmaMin, float sigmaMax) {
        var bodyOutput = this.body.apply(state);
        var mu = this.muHead.apply(bodyOutput);
        var sigma = this.logSigmaHead.apply(bodyOutput).exp().clip(sigmaMin, sigmaMax);
        // sigmaMin and sigmaMax are floats passed in. These would typically come from ActionConfig's FloatRange.floatValue().
        return new AgentUtils.GaussianDistribution(mu, sigma);
    }

    @Override
    public Tensor apply(Tensor t) {
        // Defaulting to returning mu. getDistribution() is preferred for policy evaluation.
        var bodyOutput = this.body.apply(t);
        return this.muHead.apply(bodyOutput);
    }

    @Override
    public void train(boolean training) {
        super.train(training);
        // body, muHead, logSigmaHead are added to this.layer, so super.train() handles them.
    }
}
