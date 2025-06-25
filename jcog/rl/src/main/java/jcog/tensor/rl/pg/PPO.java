package jcog.tensor.rl.pg;

import jcog.data.list.Lst;
import jcog.signal.FloatRange;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg2.PGBuilder;

import java.util.List;

/**
 * @deprecated This class represents an older PPO implementation.
 *             Use {@link jcog.tensor.rl.pg3.PPOAgent} instead.
 *             The new PPOAgent offers a more standard and flexible PPO implementation, including GAE.
 */
@Deprecated
public class PPO extends VPG {

    public final Clipping clipping =
        Clipping.HARD;
        //Clipping.TANH;
        //Clipping.SIGMOID;

    /** "epsilon" */
    public final FloatRange proximal = new FloatRange(0.2f, 1e-6f, 1); // PPO epsilon

    public PPO(int inputs, int outputs, int hiddenPolicy, int hiddenValue, int episodeLen) {
        super(inputs, outputs, hiddenPolicy, hiddenValue, episodeLen);
    }

    private Tensor logProb(int i) {
        var actionProb = policy.apply(states.get(i)).detach();
        return logProb(actions.get(i), actionProb).detach();
    }

    final List<Tensor> oldLogProbs = new Lst<>();

    @Override
    protected void remember(double reward, Tensor currentState, double[] action) {
        super.remember(reward, currentState, action);
        //oldLogProbs.add(logProb(states.size()-1));
        oldLogProbs.add(null); //placeholder
    }

    @Override
    protected void reset() {
        super.reset();
        oldLogProbs.clear();
    }

    @Override
    protected void forget() {
        super.forget();
        oldLogProbs.removeFirst();
    }

    @Override
    protected Tensor policyLoss(Tensor advantages, int i) {
        var actionProb = policy.apply(states.get(i));
        var newLogProb = logProb(actions.get(i), actionProb);

        var oldLogProb = this.oldLogProbs.get(i);
        if (oldLogProb == null)
            oldLogProb = newLogProb; //no old one to compare to, ratio=1

        if (oldLogProbs.get(i)==null)
            oldLogProbs.set(i, newLogProb.detachCopy());
        else
            oldLogProbs.get(i).setData(newLogProb);

        var ratio = newLogProb.sub(oldLogProb).exp(); //exp(log(new/old)) = exp(log(new)-log(old))
        var ratioClipped   = clip(ratio, proximal.getAsDouble(), clipping);

        var a = advantages.data(i);

        var policyGradient = Tensor.min(ratio.mul(a), ratioClipped.mul(a));

        return policyGradient.sum().add(exploreBonus(actionProb));
    }

    public enum Clipping {
        HARD, SIGMOID, TANH
        //, SOFTPLUS, COMBINED
    }

    /** epsilon = (0, 1] */
    @Deprecated private Tensor clip(Tensor x, double epsilon, Clipping c) {
        double min = 1 - epsilon, max = 1 + epsilon;
        return switch (c) {
            case HARD    -> x.clip(min, max);
            case SIGMOID -> x.clipSigmoid(min, max);
            case TANH    -> x.clipTanh(min, max);

//            case SOFTPLUS:
//                double beta = 1;
//                var upperBound = log1p(exp(mult(sub(x, 1 + epsilon), beta)));
//                var lowerBound = log1p(exp(mult(sub(x, 1 - epsilon), beta)));
//                return add(sub(add(upperBound, lowerBound), Math.log(2)), 1);

//            case COMBINED:
//                alpha = 5.0; // Controls the steepness of the transition
//                Tensor tanhTerm = add(mult(tanh(mult(sub(ratio, 1), alpha)), epsilon), 1);
//                Tensor softplusTerm = mult(softplus(Tensor.abs(sub(ratio, 1))), 1e-3);
//                return add(tanhTerm, softplusTerm);
        };
    }
}
