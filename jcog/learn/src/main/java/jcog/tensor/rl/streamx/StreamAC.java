package jcog.tensor.rl.streamx;

import jcog.signal.FloatRange;
import jcog.signal.IntRange;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import jcog.tensor.Tensor.Optimizer;
import jcog.tensor.rl.pg.AbstractPG;
import jcog.util.ArrayUtil;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.DoubleSupplier;
import java.util.function.UnaryOperator;

import static jcog.tensor.Tensor.scalar;
import static jcog.tensor.rl.streamx.StreamXUtil.*;


/**
 * Stream Actor-Critic(λ) implementation with separate policy and value networks
 */
public class StreamAC extends AbstractPG {

    public final FloatRange policyLr = new FloatRange(1, 0, 4);
    public final FloatRange valueLr = new FloatRange(1, 0, 4);

    /** paper: 0.99 */
    public final FloatRange gamma = FloatRange.unit(0.5f);

    /** paper: 0.8 */
    public final FloatRange lambda = FloatRange.unit(0.8f);

    private final double entropyBonus =
        0.01; //PAPER
        //0.001f;
        //0.0001f;
        //0;

    public final FloatRange actionNoise = new FloatRange(0, 0, 1);

    public final IntRange epochs = new IntRange(1, 1, 8);

    /** queue length per input dimension */
    public static final int NORM_HISTORY = 128;

    private static final float sparsity =
        //.9f; //PAPER
        //0.75f;
        //0.5f;
        0.25f;

    private static final float sparseNoise =
        0; //PAPER
        //1E-8f;

    /** 0 to disable */
    private static final float weightClip =
        //5;
        //8;
        0; //DISABLED

    //double kappaPolicy = 3, kappaValue = 2; //PAPER
    double kappaPolicy = 1, kappaValue = 1;

    public final UnaryOperator<Tensor> policy, value;
    public final EligibilityTraces policyTraces, valueTraces;
    private final Optimizer policyOptimizer, valueOptimizer;

    private final ObservationNormalizerRollingWelford inputNorm, rewardNorm;
    public final AtomicBoolean inputScaling = new AtomicBoolean(false), rewardScaling = new AtomicBoolean(false);

    public double policyLoss, valueLoss, tdErr, entropy;
    private double tdErrCurrent;

    // Store the last action taken for computing log probabilities
    private double[] actionPrev;
    private Tensor inputPrev;

    boolean detachValuePrev = false;
    boolean detachValueCur = false;
    boolean detachTdError = false;

    private final DoubleSupplier TD_ERR = ()->this.tdErrCurrent;

    public StreamAC(int inputs, int outputs, int policyHidden, int valueHidden) {
        super(inputs, outputs);
        
        this.inputNorm =
                //new ObservationNormalizer(inputs, obsNormRate);
                new ObservationNormalizerRollingWelford(inputs, NORM_HISTORY);
        this.rewardNorm =
                //new ObservationNormalizer(1, obsNormRate);
                new ObservationNormalizerRollingWelford(1, NORM_HISTORY);

        this.policy = new LayerNormNetwork(inputs, policyHidden, outputs*2, sparsity, sparseNoise);

//        this.policy = new Models.Layers(
//                Tensor::reluLeaky, null, true,
//                Models.Layers.layerLerp(inputs, policyHidden, outputs*2, 4)
//        );
//        this.value = new Models.Layers(
//                Tensor::reluLeaky, null, true,
//                Models.Layers.layerLerp(inputs, valueHidden, 1, 5)
//        );

//        var m1 = new AdaptiveMemoryEfficientKernelLayer(inputs, inputs, 16);
////        var m1 = new AGLU(inputs, inputs, 2, null);
////        var m1 = DynamicAdaptiveFilter.model(
////                inputs, inputs, 32,
////                0.001f, 0.95f, 256);
////        var m1 = new CellularAutomataLinear(
////                inputs, inputs, 2,
////                4,
////                4,
////                ()->1E-4f);
//
////        int chunkSize = 64;
////        int tableSize = 16;
////        double learningRate = 0.01;
////        int numHashes = 2;
////        var m1 = new AdaptiveCacheAwareMemoryLayer(
////                numHashes, chunkSize, outputs, tableSize,
////                new AdaptiveCacheAwareMemoryLayer.SimpleHasher(),
////                new AdaptiveCacheAwareMemoryLayer.LinearInterpolator(),
////                new AdaptiveCacheAwareMemoryLayer.LRUEvictionPolicy(tableSize),1, 1);
//        var m2 = new Models.Linear(inputs, outputs*2, null, true);
//        var m = Tensor.compose(Tensor.compose(m1, Tensor::tanh), m2);
//        this.policy = m;

        this.value = new LayerNormNetwork(inputs, valueHidden, 1, sparsity, sparseNoise);
//        this.value = new Models.Layers(
//                Tensor::reluLeaky, null, true,
//                Models.Layers.layerLerp(inputs, valueHidden, 1, 3)
//        );

        this.policyTraces = new EligibilityTraces(gamma, lambda);
        this.valueTraces = new EligibilityTraces(gamma, lambda);

        var weightClip = this.weightClip!=0 ? new Optimizers.WeightClip(this.weightClip) : null;

        this.policyOptimizer = new ObGD(policyLr, kappaPolicy, policyTraces, TD_ERR, weightClip).optimizer();
        this.valueOptimizer = new ObGD(valueLr, kappaValue, valueTraces, TD_ERR, weightClip).optimizer();
//        this.policyOptimizer = new ObGDWithMomentum(policyLr, kappaPolicy, policyTraces, TD_ERR, weightClip, 0.5f).optimizer();
//        this.valueOptimizer = new ObGDWithMomentum(valueLr, kappaValue, valueTraces, TD_ERR, weightClip, 0.5f).optimizer();
        trainMode(false);
    }

    @Override
    protected double[] _action(Tensor state) {
        var actionProb = policy.apply(state);

        var action = sampleAction(mu(actionProb), sigma(actionProb), actionNoise);

        reviseAction(action);

        return action;
    }

    @Override
    protected void reviseAction(double[] actionPrev) {
        if (this.actionPrev==null)
            this.actionPrev = actionPrev.clone();
        else
            ArrayUtil.copy(actionPrev, this.actionPrev);
    }


    @Override
    public double[] act(double[] input, double reward) {
        var ii = Tensor.row(input);
        var inputCur = inputScaling.get() ? inputNorm.update(ii) : ii;

        if (rewardScaling.get())
            reward = rewardNorm.update(scalar(reward)).scalar();

        // If there is a previous state, compute TD error and update networks
        if (inputPrev != null) {
            this.policyLoss = this.valueLoss = 0;
            this.entropy = 0;
            this.tdErr = 0;

            var epochs = this.epochs.intValue();
            for (var i = 0; i < epochs; i++) {
                // Get value estimates
                var valuePrev = value(inputPrev);
                var valueCur = value(inputCur);

                if (detachValuePrev)
                    valuePrev = valuePrev.detach();
                if (detachValueCur)
                    valueCur = valueCur.detach(); // Detach to prevent gradient flow?

                // Compute TD error: δ = r + γ * V(s') - V(s)
                var tdTarget = scalar(reward).add(valueCur.mul(scalar(gamma)));
                var tdError = tdTarget.sub(valuePrev);
                this.tdErrCurrent = tdError.scalar();
                this.tdErr +=
                        Math.abs(tdErrCurrent);
                    //tdError.scalar(); //signed

                if (detachTdError)
                    tdError = tdError.detach();

                update(tdError, valuePrev, valueCur, reward);
            }

            this.tdErr/=epochs;
            this.policyLoss/=(outputs*epochs); //TODO policyLossNext, entropyNext
            this.valueLoss/=(epochs); //TODO policyLossNext, entropyNext
            this.entropy/=(epochs);
        }


        // Sample new action based on current state
        var action = _action(inputCur);

        // Store current state and action for the next step
        inputPrev = inputCur;
        actionPrev = action;

        return action;
    }

    private Tensor value(Tensor x) {
        return value.apply(x);
        //return value.apply(x).tanh();
    }

    private void update(Tensor tdError, Tensor valuePrev, Tensor valueCur, double reward) {

        // Perform training updates
        train(value, () -> {

            var valueLoss =
                    valuePrev.neg(); // -V(S) //PAPER

                    //tdError.relu().sub(valuePrev);

                    //tdError.sub(valuePrev);

                    //valuePrev.neg().add(valueCur); // -V(S) + V(S')

                    //tdError.neg();
                    //valuePrev.neg().sub(tdError); // -V(S) - tdErr
                    //valueCur.neg(); // -V(S')
                    //valuePrev.neg().add(tdError); // -V(S) + tdErr
                    //valuePrev.neg().div(Tensor.max(valueCur.abs(),valuePrev.abs()).add(1));
                    //valuePrev.neg().div(valueCur.abs().add(valuePrev.abs()).add(1)); // -V(S) / (|V(S')| + |V(S)|)
                    //valuePrev.neg().sub(valueCur); // -V(S) - V(S')

                    //valuePrev.neg().div(tdError.abs().add(1));
                    //valuePrev.neg().mul(tdError.neg());
                    //valuePrev.neg().add(tdError.neg());

                    //tdError.abs();
                    //tdError.sqr(); //sqr(δ)

                    //tdError.add(valuePrev);

                    //valuePrev.sqr().add(valueCur.sqr()).add(tdError.abs());
                    //valuePrev.abs().add(tdError.abs());
                    //valuePrev.abs().mul(tdError.abs());
                    //valuePrev.abs().add(tdError.abs().detach());

                    //valuePrev.abs();
                    //valuePrev.sqr();
                    //valueCur.sqr();
                    //valueCur.neg();

                    //valuePrev;
            valueLoss.minimize(valueOptimizer, null);
            this.valueLoss += Math.abs(valueLoss.scalar());
        });

        var actionProb = policy.apply(inputPrev);

        train(policy, () -> {
            var sigma = sigma(actionProb);

            var entropy = entropy(sigma);
            this.entropy += entropy.scalar();

            var logProb = logProb(Tensor.row(actionPrev), actionProb).sum();

            var entropyBonus = entropy.mul(this.entropyBonus).mul(tdError.signum());
            var policyLoss = logProb.add(entropyBonus);

            policyLoss.minimize(policyOptimizer, null);
            this.policyLoss += Math.abs(policyLoss.scalar());
        });
    }

//    Tensor logProb(Tensor actions, Tensor actionProb) {
//        var mean = mu(actionProb);
//        var sigma = sigma(actionProb);
//        return actions.sub(mean).sqr().div(sigma.sqr().mul(2)).add(sigma.log()).add(Math.log(Math.sqrt(2 * Math.PI))).neg();
//    }

    private void trainMode(boolean training) {
        train(policy, training);
        train(value, training);
    }

    public void onEpisodeEnd() {
        policyTraces.reset();
        valueTraces.reset();
        actionPrev = null; // Clear the last action at episode end
    }

    public static class LayerNormNetwork implements UnaryOperator<Tensor> {
        //Tensor::swish;
        final UnaryOperator<Tensor> activation = Tensor::reluLeaky;
        public final LayerNormLinear l1, l2, l3;
        //public final Models.Linear l3;
        //public final Models.ULT l3;

        public LayerNormNetwork(int inputs, int hidden, int outputs, float sparsity, float noise) {
            var rng = ThreadLocalRandom.current();

            l1 = new LayerNormLinear(inputs, hidden, sparsity, noise);
            initSparse(l1.linear.weight, sparsity, sparseNoise, rng);

            l2 = new LayerNormLinear(hidden, hidden, sparsity, noise);
            initSparse(l2.linear.weight, sparsity, sparseNoise, rng);

            l3 = new LayerNormLinear(hidden, outputs, sparsity, noise);
//            l3 = new Models.Linear(hidden, outputs, null, biasInLastLayer);
            initSparse(l3.linear.weight, sparsity, sparseNoise, rng);

//            l3 = new Models.ULT(hidden, outputs, hidden, 4, null, biasInLastLayer);
        }

        @Override
        public Tensor apply(Tensor x) {
            return l3.apply(activation.apply(l2.apply(activation.apply(l1.apply(x)))));
        }
    }
}
