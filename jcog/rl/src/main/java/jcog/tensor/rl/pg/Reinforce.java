package jcog.tensor.rl.pg;

import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg2.PGBuilder;

import java.util.function.UnaryOperator;

/**
 * This class represents an older REINFORCE implementation.
 * It has a non-standard policy network structure and is superseded by the `pg3` API.
 *
 * @deprecated This class is part of an older API. Prefer using {@link jcog.tensor.rl.pg3.ReinforceAgent}
 *             from the `pg3` package for new development, which offers a more standardized and configurable approach.
 */
@Deprecated
public class Reinforce extends AbstractReinforce {

    final UnaryOperator<Tensor> policyActivation =
        Tensor.RELU;
        //Tensor.RELU_LEAKY;
        //Tensor::tanh;
        //Tensor.SIGMOID;
        //Tensor::swish;
        //Tensor::elu;
        //Tensor::tanh;
        //Tensor::siglinear;

    final Optimizers.ParamNoise paramNoiseStep;

    public Reinforce(int inputs, int outputs, int hiddenPolicy, int episodeLen) {
        this(inputs, outputs, hiddenPolicy, episodeLen, 0, 0);
    }

    protected Reinforce(int inputs, int outputs, int hiddenPolicy, int episodeLen, int policyInputsAux, int policyOutputsAux) {
        super(inputs, outputs);
        this.episode.set(episodeLen);

        boolean biasInPolicyLastLayer = true;

        UnaryOperator<Tensor> pre = null;
        UnaryOperator<Tensor> p;

        var policyLayerCount = 4;
        p = new Models.Layers(policyActivation, null, biasInPolicyLastLayer,
                Models.Layers.layerLerp(inputs + policyInputsAux, hiddenPolicy, (outputs*2) + policyOutputsAux, policyLayerCount)) {

            @Override
            protected UnaryOperator<Tensor> layer(int l, int maxLayers, int I, int O, UnaryOperator<Tensor> a, boolean bias) {
                //return new ModelsExperimental.FastSlowNetwork(I, O, a, bias);
                //return new Models.ULT(I, O, I+O,  4, a, bias);

                if (l == 0) {
                    //return new Models.ULT(I, O, I+O,  2, a, bias);
                    //return new ModelsExperimental.NormalizedULT(I, O, I+O,  2, a, bias);
                    //var x = new ODELayer(I, O, I, true, () -> 3); return (z) -> a.apply(x.apply(z));
                }

//                if (l < maxLayers-1) {
//                    //return new Models.WinogradLinear(I, O, a, bias);
//                    //return new Models.MixtureOfExperts(I, O, 2, false, a, bias);
//                    //return new ModelsExperimental.PredictiveCodingLinearUnit(I, I+O, O, a, bias);
//                    var b = a != null || bias ? new Models.BiasActivation(O, a, bias) : null;
//                    var mem =
//                            //(I * O)/1;
//                            //I;
//                            I/4;
//                    return Tensor.compose(new HyperNeuralTuringMachine(mem, I, O, 3,
//                            HyperNeuralTuringMachine.Strategy.FULL_BACKPROP
//                            //HyperNeuralTuringMachine.Strategy.EVOLUTIONARY
//                            //HyperNeuralTuringMachine.Strategy.CONTINUOUS_APPROX
//                            //HyperNeuralTuringMachine.Strategy.STRAIGHT_THROUGH
//                            , 0.001f, 8), b);
//                }
//                return new Models.ULT(I, O, Math.max(I,O),  2, a, bias);
                return super.layer(l, maxLayers, I, O, a, bias);

            }

            @Override
            protected void beforeLayer(int I, int l, int maxLayers) {
                //layer.add(new Models.LayerNorm(I, layerNormRate));

                //layer.add(t -> t.clipGrad(-1, +1));
            }

            @Override
            protected void afterLayer(int O, int l, int maxLayers) {
                if (l < maxLayers-1 && O > 10)
                    layer.add(new Models.Dropout(0.1f));

                //layer.add(t -> t.clipGrad(-1, +1));

                //layer.add(t -> t.gradDebug());
            }
        };

        this.policy = p;
        train(policy, false); //start in inference mode

        paramNoiseStep = new Optimizers.ParamNoise(paramNoise);
        policyOpt.step.add(paramNoiseStep);
    }

    @Override
    protected double[] _action(Tensor currentState) {
        return sampleAction(policy.apply(currentState).detach());
    }

    protected Tensor policyLoss(Tensor advantages, int i) {
        var actionProb = policy.apply(states.get(i));
        var policyGradient = logProb(actions.get(i), actionProb)
                .mul(advantages.data(i)).sum();
        return policyGradient.add(exploreBonus(actionProb));
    }

    @Override
    protected final double minimizePolicy(Tensor advantages, int i) {
        return loss(policyLoss(advantages, i).neg()).minimize(ctx).scalar();
    }

    /** auxiliary loss function, for subclasses */
    protected Tensor loss(Tensor l) {
        return l;
    }


}
