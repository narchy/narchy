package jcog.tensor.rl.pg;

import jcog.data.list.Lst;
import jcog.signal.FloatRange;
import jcog.signal.IntRange;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.util.TensorUtil;
import jcog.util.ArrayUtil;

import java.util.List;
import java.util.function.UnaryOperator;

public abstract class AbstractReinforce extends AbstractPG {

    public UnaryOperator<Tensor> policy;
    public final FloatRange gamma = new FloatRange(0.9f, 0, 0.99f);
    public final IntRange episode = new IntRange(3, 1, 128);
    public final IntRange epochs = new IntRange(1, 1, 8);
    public final FloatRange actionNoise = new FloatRange(0, 0, 1);
    public final FloatRange paramNoise = new FloatRange(0, 0, 0.0001f);

    /**
     * GAE Lambda
     * Higher (closer to 1): Considers more future rewards, reducing bias but increasing variance.
     * Lower (closer to 0): Focuses more on immediate rewards, increasing bias but reducing variance.
     */
    public final FloatRange lambda = new FloatRange(0.5f /*0.95f*/, 0, 1);

    /**
     * Entropy Regulation Coefficient (entropyCoef):
     * Purpose: Encourages exploration by adding an entropy bonus to the policy loss.
     * Typical range: 0.01 to 0.1
     * Effect: A higher value encourages more random (exploratory) behavior, which can help prevent premature convergence to suboptimal policies. However, too high a value can make learning unstable.
     */
    public final FloatRange exploreBonus = new FloatRange(0.01f, -0.5f, +0.5f);

    /**
     * policy learning rate
     */
    public final FloatRange policyLearning = new FloatRange(
            0.00003f
            //0.0003f
            //0.003f
            //0.000003f
            , 1e-9f, 5e-1f);

    final Tensor.Optimizer policyOpt =
        //new Optimizers.SGD(policyLearning).get();
        new Optimizers.ADAM(policyLearning).get();
        //new Optimizers.SGDMomentum(policyLearning, 0.9f).get();
        //new Optimizers.Ranger(policyLearning).get(5);
        //new Optimizers.LION(policyLearning).get(5);
        //new Optimizers.LIONAdaptive(policyLearning).get(5);
        //new Optimizers.LIONAdaptivePercent(policyLearning, 0.1f).get(5);


    protected final List<Tensor> states = new Lst<>();
    protected final List<Tensor> actions = new Lst<>();
    final List<Double> rewards = new Lst<>();
    final Tensor.GradQueue ctx = new Tensor.GradQueue();
    private final IntRange frameSkip = new IntRange(0, 0, 8);

    public double policyLoss;

    /** mean entropy, across actions */
    public double entropyCurrent;
    protected double entropyNext;

    int iter;
    private int _frameSkip = frameSkip.intValue();

    public AbstractReinforce(int inputs, int outputs) {
        super(inputs, outputs);
    }


    @Override
    public double[] act(double[] input, double reward) {

        var state = Tensor.row(input);

        var action = action(state);

        remember(reward(reward), state, action);

        if (_frameSkip <= 0) {
            update();
            iter++;
            _frameSkip = frameSkip.intValue();
        } else {
            _frameSkip--;
        }

        return action;
    }

    /** to implement reward shaping in subclasses */
    protected double reward(double reward) {
        return reward;
    }

    protected void remember(double reward, Tensor currentState, double[] action) {
        var e = episode.intValue();
        while (states.size() >= e)
            forget();

        rewards.add(reward);
        states.add(currentState);
        actions.add(Tensor.row(action));
    }

    protected void forget() {
        states.removeFirst();
        rewards.removeFirst();
        actions.removeFirst();
    }

    protected void reset() {
        states.clear();
        actions.clear();
        rewards.clear();
    }

    protected final void update() {
        int epochs = this.epochs.intValue();
        for (int i = 0; i < epochs; i++) {
            var r = returnsAndAdvantages(rewards, gamma.asFloat(), lambda.asFloat());
            update(r[0], r[1]);
        }
    }

    protected void update(Tensor returns, Tensor advantages) {
        updateValue(returns);
        updatePolicy(advantages);
    }

    protected void updateValue(Tensor tensor) {
        //NOP
    }

    /**
     * Indicates the performance of the policy gradient update.
     * It's derived from the PPO clipped surrogate objective.
     * This loss helps determine how well the policy network
     * is learning to improve its action selections.
     */
    protected final void updatePolicy(Tensor advantages) {
        train(policy, () -> {
            this.entropyNext = 0;
            var s = states.size();
            var lossSum = TensorUtil.sumParallel(s, i -> Math.abs(
                minimizePolicy(advantages, i)));
            this.ctx.div(s).optimize(policyOpt);
            this.policyLoss = lossSum / (s * outputs);
            this.entropyCurrent = entropyNext / (s);
        });
    }

    /** returns loss */
    protected abstract double minimizePolicy(Tensor advantages, int i);

    protected final double[] sampleAction(Tensor actionProb) {
        return sampleAction(mu(actionProb), sigma(actionProb), actionNoise);
    }

    private Tensor[] returnsAndAdvantages(List<Double> rewards, double gamma, double lambda) {
        return returnsAndAdvantages(rewards, gamma, lambda, values(rewards.size()));
    }

    protected double[] values(int r) {
        var vals = new double[r + 1];
        for (var i = 0; i < r; i++)
            vals[i] = baselineReward(i);
        return vals;
    }

    protected double baselineReward(int r) {
        return rewards.get(r);
    }

    @Override
    protected void reviseAction(double[] actionPrev) {
        if (!actions.isEmpty()) {
            //copy environment-provided actionPrev to allow the environment to override with what action actually happened.
            //System.out.println(n2(actionPrevDesired) + "\t" + n2(actionPrev));
            ArrayUtil.copy(actionPrev, actions.getLast().array());
        }
    }

    protected final Tensor exploreBonus(Tensor actionProb) {
        return entropy(sigma(actionProb)).mul(exploreBonus.getAsDouble());
    }

    @Override
    @Deprecated protected final Tensor entropy(Tensor sigma) {
        var e = super.entropy(sigma);
        var es = e.scalar();
        synchronized(this) {
            this.entropyNext += es;
        }
        return e;
    }

}
