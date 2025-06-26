package jcog.tensor.rl.pg;

import jcog.Fuzzy;
import jcog.Util;
import jcog.agent.Agent;
import jcog.math.normalize.FloatNormalizer;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.tensor.Models;
import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

import static java.lang.Math.PI;

public abstract class AbstractPG {

    protected final int inputs, outputs;

    private static final boolean normalizeReturns = false;
    private static final boolean normalizeAdvantages = true;

    private static final boolean detachEntropySigma = false;
    private static final boolean muClipSmooth = false;

    public static final float sigmaMin = 2e-1f; //?
    public final FloatRange sigmaMax = new FloatRange(4 /*6*/, sigmaMin*2, 8);

    protected RandomBits rng = new RandomBits(new XoRoShiRo128PlusRandom());


    protected AbstractPG(int inputs, int outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    /** action post-processor */
    public Consumer<double[]> actionFilter = (a) -> { };

    protected final double[] action(Tensor currentState) {
        var a = _action(currentState);
        actionFilter.accept(a);
        return a;
    }

    abstract protected double[] _action(Tensor state);

    protected static Tensor[] returnsAndAdvantages(List<Double> rewards, double gamma, double lambda, double[] values) {
        var n = rewards.size();

        // Estimate for terminal state
        values[n] =
            values[n - 1];
            //0;

        var deltas = new double[n];
        var advantages = new double[n];
        var returns = new double[n];

        // Calculate deltas and returns
        for (var i = n - 1; i >= 0; i--) {
            var r = rewards.get(i);
            deltas[i] = r + gamma * values[i + 1] - values[i];
            double rr;
            if (i < n - 1) {
                advantages[i] = deltas[i] + gamma * lambda * advantages[i + 1];
                rr = returns[i + 1];
            } else {
                advantages[i] = deltas[i];
                rr = values[i + 1];
            }
            returns[i] = r + gamma * rr;
        }


        if (n > 1) {
            if (normalizeAdvantages)
                normalize(advantages);
            if (normalizeReturns)
                normalize(returns);
        }

        return new Tensor[]{Tensor.row(returns), Tensor.row(advantages)};
    }

    protected static void normalize(double[] x) {
        var meanVar = Util.variance(x);
        double mean = meanVar[0], stddev = Math.sqrt(meanVar[1]);

        var r = x.length;
        for (var i = 0; i < r; i++)
            x[i] = (x[i] - mean) / (stddev + EPSILON);
    }

    public abstract double[] act(double[] input, double reward);

    public PGAgent agent() {
        return new PGAgent(this);
    }

    static protected void train(UnaryOperator<Tensor> value, Runnable r) {
        train(value, true);
        try {
            r.run();
        } finally {
            train(value, false);
        }
    }

    @Deprecated protected static void train(UnaryOperator<Tensor> value, boolean training) {
        if (value instanceof Models.Layers vl) vl.train(training);
    }

    public static class PGAgent extends Agent {

        public final AbstractPG pg;

        /** probability of action revision */
        final FloatRange actionRevise = FloatRange.unit(1/*0.995f*/);

        public boolean rewardNormalize;

        public boolean inputPolarize;
        public boolean rewardPolarize;

        public long cycleTimeNS;

        private final FloatNormalizer rewardNorm = rewardNormalize ?
                new FloatNormalizer(2, 1000) : null;

        public PGAgent(AbstractPG pg) {
            super(pg.inputs, pg.outputs);
            this.pg = pg;
        }

        @Override
        public String toString() {
            return super.toString() + "(" + pg.getClass() + ")";
        }

        @Override
        public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
            //cleanup any NaN's
            Util.replaceNaNwithRandom(inputPrev, pg.rng);
            Util.replaceNaNwithRandom(actionPrev, pg.rng);
            Util.replaceNaNwithRandom(input, pg.rng);
            Util.replaceNaNwithRandom(actionNext, pg.rng);
            if (reward!=reward) reward = pg.rng.nextFloat();

            Fuzzy.polarize(actionPrev);

            if (inputPolarize)
                Fuzzy.polarize(input);

            if (pg.rng.nextBoolean(actionRevise.asFloat()))
                pg.reviseAction(actionPrev);

            var r = reward;
            if (rewardNormalize)
                r = rewardNorm.valueOf(r);

            if (rewardPolarize) {
                r = Fuzzy.polarize(r); //convert to -1..+1
                //r = r - 1; //convert to -1..0 "it's all bad"
                //r = Util.lerp(r, -1, -0.1f); //convert to -1..-0.1 "it's all bad"
            }

            var start = System.nanoTime();

            var a = pg.act(input, r);

            var end = System.nanoTime();

            cycleTimeNS = end - start;

            Fuzzy.unpolarize(a);
            System.arraycopy(a, 0, actionNext, 0, actionNext.length);
        }

    }


    /* Constrains mu to [-1,1]  */
    protected Tensor mu(Tensor actionProb) {
//        if (actionProb.cols()!=outputs*2)
//            throw new WTF();
        var mu = actionProb.slice(0, outputs);
        return muClipSmooth ? mu.clipTanh() : mu.clipUnitPolar();
    }

    protected Tensor logSigma(Tensor actionProb) {
//        if (actionProb.cols()!=outputs*2)
//            throw new WTF();
        return actionProb.slice(outputs, outputs * 2);
    }

    /** subclasses may implement to revise the actual action taken in the last step */
    protected void reviseAction(double[] actionPrev) {

    }

    protected double[] sampleAction(Tensor mean, Tensor sigma, FloatRange _actionNoise) {
        var actions = mean.volume();

        var a = new double[actions];
        actionGaussian(mean, sigma, a);
        //actionGaussianTruncated(mean, sigma, a);
        //actionOU(mean, sigma, a);
        //actionPerlin(mean, sigma, a);

        var actionNoise = _actionNoise.asFloat();
        if (actionNoise > 0)
            for (var i = 0; i < outputs; i++) {
                var noise =
                    rng.nextFloat() * actionNoise * 2 /* x2 for -1..+1 */;
                    //rng.nextGaussian() * actionNoise;
                a[i] += noise;
            }

        Util.clampSafe(a, -1, +1);

        return a;
    }


    /**
     * Computes the entropy of a Gaussian policy.
     * H = 0.5 + 0.5 * ln(2 * PI) + ln(sigma)
     */
    protected Tensor entropy(Tensor sigma) {
        if (detachEntropySigma) sigma = sigma.detachCopy();

        sigma = Tensor.max(sigma, Tensor.zerosShaped(sigma).fill(sigmaMin).grad(sigma.hasGrad() /* HACK */)); //ensure non-negative
        //sigma = Tensor.max(sigma, Tensor.zerosShaped(sigma).fill(0.242 * sigmaMin)); //ensure non-negative

        //return sigma.log().mean().add((Math.log(2 * Math.PI) + 1)/2);
        return Tensor.max(Tensor.scalar(0), sigma.log().mean().add((Math.log(2 * PI) + 1)/2));
        //return sigma.log().add(Math.log(Math.sqrt(2 * PI * Math.E)) / 2).mean();
        //return sigma.log().mean().add(-Math.log(sigmaMin) + (Math.log(2 * Math.PI) + 1)/2);
        //return sigma.log().add(-Math.log(sigmaMin) + (Math.log(2 * Math.PI) + 1)/2).mean();
        //return sigma.mean();
        //return sigma.mean().sqrt();
        //return sigma.log().add((Math.log(2 * Math.PI) + 1)/2).mean();
        //return sigma.log().add(-Math.log(sigmaMin) + (Math.log(2 * Math.PI) + 1)/2).mean();
        //return sigma.log().mean().add(-Math.log(sigmaMin) + (Math.log(2 * Math.PI) + 1)/2);

        //return logSigma.sum().add(-Math.log(sigmaMin) + (Math.log(2 * Math.PI) + 1) / 2);
    }

    protected void actionGaussian(Tensor mean, Tensor sigma, double[] a) {
        for (var i = 0; i < outputs; i++)
            a[i] = rng.nextGaussian(mean.data(i), sigma.data(i));
    }

    enum SigmaMode {
        Exp,
        ExpClip,
        ExpClipSmooth,
        SoftPlus
    }
    SigmaMode sigmaMode =
        SigmaMode.Exp
        //SigmaMode.ExpClipSmooth
        //SigmaMode.ExpClip
        //SigmaMode.SoftPlus
    ;

    protected final Tensor sigma(Tensor actionProb) {
        var logSigma = logSigma(actionProb);

        final var sigmaMax = this.sigmaMax.asFloat();
        return switch (sigmaMode) {
            case Exp ->
                    logSigma.exp().clip(sigmaMin, sigmaMax);
            case ExpClip ->
                    logSigma.exp(sigmaMin, sigmaMax);
                    //logSigma.exp().clip(sigmaMin, sigmaMax);
            case ExpClipSmooth ->
                    logSigma.clipSigmoid(Math.log(sigmaMin), Math.log(sigmaMax)).exp();
                    //logSigma.exp().clipSigmoid(sigmaMin, sigmaMax);
                    //logSigma.exp().sigmoid().mul(sigmaMax-sigmaMin).add(sigmaMin);
            case SoftPlus ->
                    logSigma.softplus().add(sigmaMin).clip(sigmaMin, sigmaMax);
        };
//            logSigma.softplus()
//                //.clip(sigmaMin, sigmaMax.asFloat());
//                .add(sigmaMin).clip(sigmaMin, sigmaMax.asFloat());
//                //.add(sigmaMin);
//                //.clipTanh(sigmaMin, sigmaMax.asFloat());
//                //.clipSigmoid(sigmaMin, sigmaMax.asFloat());

    }


    protected final Tensor logProb(Tensor actionPrev, Tensor actionProb) {
        //var logSigma = logSigma(actionProb)
        var sigma = sigma(actionProb);
        var logSigma = sigma.log();

        // Calculate log probability components
        var variance = sigma.sqr().add(EPSILON);
        var diffSq = actionPrev.sub(mu(actionProb)).sqr();

        // Compute log probability: -0.5 * ((a - μ)^2 / σ^2 + 2 * log σ + log(2π))
        return diffSq.div(variance)
                .add(variance.mul(2*PI).log())
                .mul(-0.5);
    }

    /** 1e-8 to 1e-6 for most applications. */
    private static final double EPSILON =
            //1e-6;
            //1e-3;
            //1e-8;
            1e-5;

}
