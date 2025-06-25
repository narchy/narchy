package jcog.tensor.rl.pg;

import jcog.signal.FloatRange;
import jcog.signal.IntRange;
import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.Tensor.GradQueue;
import jcog.tensor.util.TensorUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * @deprecated Use {@link jcog.tensor.rl.pg3.SACAgent} instead.
 */
@Deprecated
public class SAC extends AbstractPG {

    public final IntRange episode = new IntRange(3, 1, 128);
    public final FloatRange entropy = new FloatRange(0.01f, -0.15f, 0.15f);

    public final Models.Layers policy, q1, q2;

    private final List<Tensor> states = new ArrayList<>();
    private final List<Tensor> actions = new ArrayList<>();
    private final List<Double> rewards = new ArrayList<>();

    public final FloatRange gamma = new FloatRange(0.9f, 0, 0.9999f);
    public final FloatRange lambda = new FloatRange(0.95f, 0, 1);

    private final Tensor.Optimizer policyOpt, q1Opt, q2Opt;

    public double policyLoss, q1Loss, q2Loss;
    boolean baselineValueOrReward = true;

    static final private UnaryOperator<Tensor> activation = Tensor.RELU_LEAKY;

    public final FloatRange policyLearn = new FloatRange(0.00003f, 1e-9f, 5e-1f);
    public final FloatRange valueLearn = new FloatRange(0.0001f, 1e-9f, 5e-1f);

    public final FloatRange actionNoise = new FloatRange(0, 0, 0.5f);

    public SAC(int inputs, int outputs, int hiddenPolicy, int hiddenValue, int episodeLen) {
        super(inputs, outputs);

        episode.set(episodeLen);

        UnaryOperator<Tensor> policyOut = Tensor::clipUnitPolar;

        policy = new Models.Layers(activation, policyOut, true,
                inputs,
                //hiddenPolicy,
                hiddenPolicy, hiddenPolicy/2,
                //hiddenPolicy, hiddenPolicy/2, hiddenPolicy/3,
                //hiddenPolicy, hiddenPolicy/2, hiddenPolicy/3, hiddenPolicy/4,
                outputs * 2
        );
        q1 = qNetwork(inputs, hiddenValue, outputs, 0);
        q2 = qNetwork(inputs, hiddenValue, outputs, 1);

        policy.train(false); q1.train(false); q2.train(false);

//        policyOpt = new Optimizers.SGDMomentum(policyLearn, 0.99f).get(1024);
//        q1Opt = new Optimizers.SGDMomentum(valueLearn, 0.99f).get(1024);
//        q2Opt = new Optimizers.SGDMomentum(valueLearn, 0.99f).get(1024);

//        policyOpt = new Optimizers.SGD(policyLearn).get(1024);
//        q1Opt = new Optimizers.SGD(valueLearn).get(1024);
//        q2Opt = new Optimizers.SGD(valueLearn).get(1024);

        policyOpt = new Optimizers.ADAM(policyLearn).get();
        q1Opt = new Optimizers.ADAM(valueLearn).get();
        q2Opt = new Optimizers.ADAM(valueLearn).get();
    }

    private Models.Layers qNetwork(int inputSize, int hidden, int actions, int n) {
        int[] layerSizes = {
            inputSize + actions,
            //hidden,
            //hidden, hidden/2,
            hidden, hidden/2, hidden/3,
            1
        };
        return new Models.Layers(activation, null, true,
            layerSizes
        )/* {
            @Override
            protected UnaryOperator<Tensor> layer(int I, int O, UnaryOperator<Tensor> a, boolean bias) {
                return // switch (n) {
//                    case 0 ->
                        new Models.ULT(I, O, (int) Util.mean(I,O)/2,  2, activation, bias);
//                    default -> new Models.MixtureOfExperts(I, O, 2, activation, bias);
//                };
            }
        }*/;
    }

    @Override
    public double[] act(double[] input, double reward) {
        var currentState = Tensor.row(input);
        var action = action(currentState);

        remember(reward, currentState, Tensor.row(action));

        update();

        return action;
    }

    protected void remember(double reward, Tensor currentState, Tensor action) {
        int episodeLength = this.episode.intValue();
        while (states.size() >= episodeLength) {
            forget();
        }
        states.add(currentState.detach());
        actions.add(action.detach());
        rewards.add(reward);
    }

    protected void forget() {
        rewards.removeFirst(); states.removeFirst(); actions.removeFirst();
    }

    protected void update() {
        if (states.isEmpty()) return;

        var returnsAndAdvantages = returnsAndAdvantagesGAE();
        var returns = returnsAndAdvantages[0];
        var advantage = returnsAndAdvantages[1];

        updatePolicy();
        updateValue(returns);
    }

    private Tensor[] returnsAndAdvantagesGAE() {
        int n = states.size();
        double[] values = new double[n + 1];

//        for (int i = 0; i < n; i++)
//            values[i] = baselineValue(i);
//        values[n] = values[n-1]; // bootstrap from last state

        if (baselineValueOrReward) {
            //TODO parallelize, see VPG's baselineValueOrReward
            for (int i = 0; i < n; i++)
                values[i] = value(states.get(i), actions.get(i));
        } else {
            for (int i = 0; i < n; i++)
                values[i] = rewards.get(i);
        }

        values[n] = values[n-1]; // Bootstrap from last state-action pair

        return returnsAndAdvantages(rewards, gamma.getAsDouble(), lambda.getAsDouble(), values);
    }

    private double value(Tensor state, Tensor action) {
        var sa = state.concat(action);
        var q1 = this.q1.apply(sa).scalar();
        var q2 = this.q2.apply(sa).scalar();
        return Math.min(q1, q2);
    }

//    @Deprecated private Tensor [] returnsAndAdvantages(int n, double[] values) {
//        double[] advantages = new double[n];
//        double[] returns = new double[n];
//        double lastGaeLam = 0, lastReturn = values[n];
//
//        float gamma = this.gamma.floatValue();
//        float lambda = 0.95f; // You might want to make this a configurable parameter
//
//        for (int t = n - 1; t >= 0; t--) {
//            var rt = rewards.get(t);
//            double delta = rt + gamma * values[t + 1] - values[t];
//            advantages[t] = (lastGaeLam = delta + gamma * lambda * lastGaeLam);
//            returns[t] = (lastReturn = rt + gamma * lastReturn);
//        }
//
//        return new Tensor[]{Tensor.row(returns), Tensor.row(advantages)};
//    }

    /** simple Monte Carlo discounted returns */
    private Tensor returns() {
        var r = rewards.size();
        var returns = new double[r];
        double runningReturn = 0;
        for (int i = r - 1; i >= 0; i--) {
            runningReturn = rewards.get(i) + gamma.floatValue() * runningReturn;
            returns[i] = runningReturn;
        }
        return Tensor.row(returns);
    }

    protected void updatePolicy() {
        train(policy, ()->{
            var s = states.size();
            double lossSum = TensorUtil.sumParallel(s, i -> Math.abs(minimizePolicy(i)));
            policyCtx.div(s).optimize(policyOpt);
            this.policyLoss = lossSum / s;
        });
    }

    private double minimizePolicy(int i) {
        var state = states.get(i);
        var action = Tensor.row(_action(state));
        var logSigma = logSigma(state, policy, action);

        var sa = state.concat(action);
        var q1v = q1.apply(sa);
        var q2v = q2.apply(sa);
        var policyGradient = Tensor.min(q1v, q2v);
        var entropyBonus = entropy(logSigma.exp()).mul(entropy.getAsDouble());

        return policyGradient.add(entropyBonus).neg()
                .minimize(policyCtx).scalar();
    }

//    private Tensor entropyBonus(Tensor logSigma) {
//        throw new TODO(); //ensure whether entropy takes and is given either sigma/logSigma
//        //return entropy(logSigma).mul(entropy.getAsDouble());
//    }

    private final GradQueue policyCtx = new GradQueue(), q1ctx = new GradQueue(), q2ctx = new GradQueue();

    protected void updateValue(Tensor returns) {
        q1.train(true);
        q2.train(true);

        var N = states.size();

        final double[] q1LossSum = {0};
        final double[] q2LossSum = { 0 };

        TensorUtil.runParallel(0, N, (s, e) -> {
            for (int i = s; i < e; i++) {
                var reward = rewards.get(i);
                var statePrev = states.get(i);
                var actPrev = actions.get(i);

                var stateNext = i < s - 1 ? states.get(i + 1) : statePrev;
                var actNext = Tensor.row(_action(stateNext));
                var saNext = stateNext.concat(actNext);
                var nextQ1 = q1.apply(saNext);//.detach();
                var nextQ2 = q2.apply(saNext);//.detach();
                var nextQ = Tensor.min(nextQ1, nextQ2);
                var logSigmaNext = logSigma(stateNext, policy, actNext);//.detach();
                var target = Tensor.scalar(reward)
                        .add(nextQ.mul(gamma.floatValue()))
                        .sub(entropy(logSigmaNext.exp()).mul(entropy.getAsDouble()));//.detach();

                var saPrev = statePrev.concat(actPrev);
                var d1 = q1.apply(saPrev).mse(target).minimize(q1ctx).scalar();
                var d2 = q2.apply(saPrev).mse(target).minimize(q2ctx).scalar();

                synchronized(q1LossSum) {
                    q1LossSum[0] += d1;
                    q2LossSum[0] += d2;
                }
            }
        });

        q1Loss = q1LossSum[0] / N;
        q2Loss = q2LossSum[0] / N;

        q1ctx.div(N).optimize(q1Opt);
        q2ctx.div(N).optimize(q2Opt);

        q1.train(false);
        q2.train(false);
    }

//    private static final float sigmaMin =
//            VPG.sigmaMin;
//
//    public final FloatRange sigmaMax = new FloatRange(
//            //1
//            9/10f, sigmaMin, 2);
//    protected Tensor sigma(Tensor a) {
//        return logSigma(a).exp(sigmaMin, sigmaMax.getAsDouble());
//    }

    protected double[] _action(Tensor state) {
        var actionProb = policy.apply(state).detach();
        return sampleAction(
            mu(actionProb),
            sigma(actionProb),
            actionNoise);
    }

    private Tensor logSigma(Tensor state, UnaryOperator<Tensor> policy, Tensor meanTarget) {
        var act = policy.apply(state);
        Tensor mean = mu(act), sigma = sigma(act);

        var normalDist = meanTarget.sub(mean).div(sigma);
        return normalDist.pow(2).add(2 * Math.PI).log().add(sigma.log()).mul(-0.5f);
    }
}