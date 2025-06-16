package jcog.tensor.rl.pg;

import jcog.signal.FloatRange;
import jcog.tensor.Tensor;
import jcog.tensor.util.TensorUtil;

public class A2C extends VPG {

    /**
     * Purpose: Balances the importance of the value function loss relative to the policy loss.
     * Typical range: 0.5 to 1.0
     * Effect: A higher value puts more emphasis on accurately predicting state values, which can lead to better advantage estimates but might slow down policy improvement.
     */
    public final FloatRange valueCoef = new FloatRange(0.5f, 0, 1f);
    
    private double lastValue = Double.NaN;

    public A2C(int inputs, int outputs, int hiddenPolicy, int hiddenValue, int episodeLen) {
        super(inputs, outputs, hiddenPolicy, hiddenValue, episodeLen);
    }

    @Override
    protected void update(Tensor returns, Tensor advantages) {
//        if (states.size() < episode.intValue())
//            return;

        updateValue(returns);

        entropyNext = 0;

        train(policy, () -> {
            var s = states.size();

            double policyCoef = 1 - valueCoef.floatValue();
            this.policyLoss = TensorUtil.sumParallel(s, i -> Math.abs(
                policyLoss(advantages, i).neg()
                    .mul(policyCoef)
                    .minimize(ctx).scalar()
            ))/s;

            this.entropyCurrent = entropyNext / (s);

            ctx.div(s).optimize(policyOpt);
        });
    }


    @Override
    protected double minimizeValue(Tensor returns, int i) {
        return value(states.get(i))
            .loss(returns.slice(i),
                Tensor.Loss.MeanSquared
                //Tensor.Loss.Huber
            )
            .mul(valueCoef.floatValue())
            .minimize(ctx)
            .scalar();
    }
}