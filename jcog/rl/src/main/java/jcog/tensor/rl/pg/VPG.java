package jcog.tensor.rl.pg;

import jcog.math.FloatSupplier;
import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg2.PGBuilder;
import jcog.tensor.util.TensorUtil;

import java.util.function.UnaryOperator;

import static jcog.tensor.Models.Layers.layerLerp;

/**
 * Vanilla Policy Gradient
 * @deprecated This class represents an older VPG (Vanilla Policy Gradient / REINFORCE with a baseline) implementation.
 *             For REINFORCE without a baseline, please use {@link PGBuilder.ReinforceStrategy}.
 *             For VPG (REINFORCE with a baseline) or more advanced Actor-Critic methods like A2C/A3C,
 *             it's recommended to use {@link PGBuilder.PPOStrategy} and configure its hyperparameters
 *             (e.g., gamma, lambda for GAE which can approximate VPG advantages, ppoClip can be made very large)
 *             or await/implement a dedicated ActorCriticStrategy using the new framework components
 *             ({@link PGBuilder.GaussianPolicy}, {@link PGBuilder.ValueNetwork}).
 *             The policy network in this old VPG has a non-standard structure.
 */
@Deprecated
public class VPG extends Reinforce {

    public final UnaryOperator<Tensor> value;

    final FloatSupplier valueLearning = ()->policyLearning.floatValue()*(10/3f);

    final Tensor.Optimizer valueOpt =
        new Optimizers.ADAM(valueLearning).get();
        //new Optimizers.SGD(valueLearning).get();
        //new Optimizers.LION(valueLearning).get(5);
        //new Optimizers.Ranger(valueLearning).get(256);
        //new Optimizers.SGDMomentum(valueLearning, 0.9f).get(256);

    private static final Tensor.Loss valueLossFn =
        Tensor.Loss.MeanSquared
        //Tensor.Loss.Huber
        ;

    public double valueLoss;

    UnaryOperator<Tensor> valueActivation =
        policyActivation;

    public VPG(int inputs, int outputs, int hiddenPolicy, int hiddenValue, int episodeLen) {
        super(inputs, outputs, hiddenPolicy, episodeLen);

        var valueLayerCount = 2;

        this.value = new Models.Layers(valueActivation, null,
                layerLerp(inputs, hiddenValue, 1, valueLayerCount));

        train(value, false); //start in inference mode

        valueOpt.step.add(paramNoiseStep);
    }

    /**
     * Reflects how well the value network is predicting the value of each state.
     * A high or increasing value loss indicates that the value predictions are inaccurate,
     * signaling issues in learning.
     */
    @Override protected final void updateValue(Tensor returns) {
        train(value, ()-> {
            var s = states.size();
            this.valueLoss = TensorUtil.sumParallel(s, i -> Math.abs(minimizeValue(returns, i))) / s;
            ctx.div(s).optimize(valueOpt);
        });
    }

    double minimizeValue(Tensor returns, int i) {
        return value(states.get(i))
            .loss(returns.slice(i), valueLossFn)
            .minimize(ctx)
            .scalar();
    }

    protected Tensor value(Tensor s) {
        return value.apply(s);
    }

    double baselineValue(int i) {
        return value(states.get(i)).scalar();
    }

    @Override protected double[] values(int r) {
        var vals = new double[r + 1];
        TensorUtil.runParallel(0, r, (s, e)->{
            for (var i = s; i < e; i++)
                vals[i] = baselineValue(i);
        });
        return vals;
    }

}