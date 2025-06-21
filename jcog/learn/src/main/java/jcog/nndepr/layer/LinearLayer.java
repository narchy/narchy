package jcog.nndepr.layer;

import jcog.Util;
import jcog.activation.DiffableFunction;
import jcog.activation.ReluActivation;
import jcog.data.bit.MetalBitSet;
import jcog.nndepr.optimizer.WeightUpdater;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Random;
import java.util.random.RandomGenerator;

import static java.lang.Math.sqrt;
import static jcog.Util.fma;

/** TODO https://ai.stackexchange.com/questions/17584/why-does-the-bias-need-to-be-a-vector-in-a-neural-network */
public class LinearLayer extends AbstractLayer {

    private static final float DROPOUT_DEFAULT =
            0;
            //0.01f;
            //0.05f;
            //0.1f;
            //0.2f;
            //0;

    /** delta */
    public final double[] delta;

    /** weights */
    public final double[] W;

    /**
     * delta-weights, gradient
     */
    @Deprecated public final double[] dW;

    /** previous gradient */
    @Deprecated public final double[] dWPrev;

    /**
     * per weight, holds enabled / disabled state for dropout
     */
    public final MetalBitSet enabled;
    public final DiffableFunction activation;
    private final boolean bias;

    /**
     * https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
     */
    public double dropout;

    public boolean dropping;


    public LinearLayer(int inputSize, int outputSize, DiffableFunction activation, boolean bias) {
        super(inputSize + (bias ? 1 : 0), outputSize);
        if (inputSize < 1) throw new UnsupportedOperationException();
        if (outputSize < 1) throw new UnsupportedOperationException();
        this.bias = bias;
        delta = new double[in.length];
        W = new double[outputSize * in.length];
        dW = new double[W.length];
        dWPrev = new double[W.length];
        enabled = MetalBitSet.bits(W.length);
        dropout = DROPOUT_DEFAULT > 0 &&
            inputSize > (1 / (DROPOUT_DEFAULT)) ?
            DROPOUT_DEFAULT : 0;
        this.activation = activation;
        startNext();
    }

    //    /**
//     * gradient post-processing
//     * https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
//     */
//    public static double gradClamp(double x) {
//        return clampSafe(x, -1, +1);
//        //return x;
//    }

    /**
     * https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
     * https://intoli.com/blog/neural-network-initialization/
     */
    @Override
    public void initialize(@Nullable Random r) {
        if (r == null)
            initZero();
        else {
            if (activation == ReluActivation.the)
                randomizeHe(r);
            else
                randomizeXavier(r);
        }
    }

    /** init all weights to 0 */
    private void initZero() {
        Arrays.fill(W, 0);
    }

    /** https://paperswithcode.com/method/xavier-initialization
     *  https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
     * */
    private void randomizeXavier(RandomGenerator r) {
        float x = (float)ins();
        randomizeGaussian(sqrt(2.0 / Util.mean(x, (float) outs())), r);
    }

    private void randomizeHe(Random r) {
        randomizeGaussian(sqrt(2.0 / ins()), r);
    }

    private void randomizeGaussian(double stddev, RandomGenerator r) {
        for (int w = 0; w < W.length; w++)
            W[w] = isBias(w) ? 0 : r.nextGaussian() * stddev;
    }



    /**
     * forward prop
     */
    @Override
    public double[] forward(double[] x, RandomGenerator rng) {
        double[] in = input(x);

        double[] W = this.W, out = this.out;
        int O = out.length;
        int io = 0;

        double dropIn = 1 - dropout;
        int I = in.length;

        int n = W.length;
        DiffableFunction a = this.activation;

        for (int o = 0; o < O; o++) {
            double y = 0;
            for (int i = 0; i < I; i++, io++) {

                double ii = in[i];
                if (
                    ii == ii //not NaN, but possibly non-finite
                    //Double.isFinite(ii)
                )
                    y = fma(ii * dropIn, W[io], y);

            }

            out[o] = a!=null ? a.valueOf(y) : y /* linear */;
        }
        return out;
    }

    private double[] input(double[] x) {
        double[] in = this.in;
        System.arraycopy(x, 0, in, 0, x.length);
        if (bias) in[in.length - 1] = 1;
        return in;
    }

    @Override
    public double[] delta(WeightUpdater updater, double[] dx) {
        updater.update(this, dx);
        return delta;
    }

    public void startNext() {
        updateDropout();
        Arrays.fill(delta, 0);
    }

    private void updateDropout() {

        double dropout = this.dropout;
        if (dropout <= Float.MIN_NORMAL) return;

        double dropIn = 1 - dropout;

        int n = W.length;
        RandomBits rng = new RandomBits(
            new XoRoShiRo128PlusRandom()
            //ThreadLocalRandom.current()
        );
        boolean invert = dropout > 0.5f;

        if (dropping)
            enabled.setAll(!invert);

        boolean dropping;

        int d = rng.nextFloor((float)((invert ? (1-dropout) : dropout) * n));
        if (d > 0) {
            dropping = true;
            for (int i = 0; i < d; i++) {
                int w = rng.nextInt(n);
                if (!isBias(w))
                    enabled.set(w, invert);
            }
        } else
            dropping = invert;

//        int maxSkip = Math.max(1, Math.round(n * dropout));
//        int nextDropOut = rng.nextInt(maxSkip*2);
//        for (int io = nextDropOut; io < n; ) {
//            enabled.set(io, false);
//            enabledAll = false;
//            io += rng.nextInt(maxSkip*2);
//        }
        this.dropping = dropping;

    }

    private boolean isBias(int w) {
        return bias && (w % in.length==in.length-1);
    }

    @Override
    public void copyLerp(AbstractLayer l, float rate) {
        LinearLayer d = (LinearLayer)l;
        var y = d.W;
        var x = this.W;
        for (int i = 0; i < W.length; i++)
            x[i] = Util.lerpSafe(rate, x[i], y[i]);
    }
}