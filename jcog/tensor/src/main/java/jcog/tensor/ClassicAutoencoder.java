package jcog.tensor;

import jcog.Util;
import jcog.activation.DiffableFunction;
import jcog.activation.SigmoidActivation;
import jcog.math.FloatSupplier;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import org.jetbrains.annotations.Nullable;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;

import static java.util.Arrays.fill;
import static jcog.Util.fma;

/**
 * Denoising Autoencoder (from DeepLearning.net)
 */
public class ClassicAutoencoder extends AbstractAutoencoder {
    public final FloatSupplier alpha;

    @Deprecated protected final AtomicBoolean normalize = new AtomicBoolean(false);

    /**
     * encoded vector
     */
    public final double[] y;
    /**
     * decoded vector
     */
    public final double[] z;
    public final double[][] W;
    @Deprecated
    public final FloatRange noise = new FloatRange(0, 0, 1);
    public final FloatRange corruption = new FloatRange(0, 0, 1);
    private final double[] hbias, vbias;
    private final double[] delta /*"L_vbias"*/;

    /**
     * input vector after preprocessing (noise, corruption, etc..)
     */
    public double[] x;
    /**
     * domain and range must be non-negative
     */
    public DiffableFunction activationEnc =
        SigmoidActivation.the;
        //SigLinearActivation.the;

    public DiffableFunction activationDec =
        activationEnc;

    private boolean clipGrad = true;

    public ClassicAutoencoder(int ins, int outs, FloatSupplier learningRate) {
        this(ins, outs, learningRate, null);
    }

    public ClassicAutoencoder(int ins, int outs, FloatSupplier learningRate, @Nullable Random rng) {

        this.alpha = learningRate;
        x = new double[ins];
        z = new double[ins];
        delta = new double[ins];
        vbias = new double[ins];

        W = new double[outs][ins];

        hbias = new double[outs];
//        L_hbias = new double[outs];
        y = new double[outs];

        clear(rng);
    }

    public void clear() {
        clear(this.rng.rng);
    }

    @Override
    public void clear(Random rng) {

        if (rng == null)
            rng = new XoRoShiRo128PlusRandom();

        if (!(rng instanceof ThreadLocalRandom))
            this.rng = new RandomBits(rng);

        fill(hbias, 0); fill(vbias, 0);
        //clear(hbias, 1f/hbias.length); clear(vbias, 1f/vbias.length);

        fill(delta, 0);

        var a =
            //(float)(2/Math.sqrt(W[0].length)); //He initializatoin
            (float)(2/Math.sqrt(Math.max(W.length, W[0].length))); //He initializatoin

        for (var wi : W)
            clear(wi, a);
    }

    @Override
    public final double[] get(double[] x) {
        return get(x, noise.doubleValue(), corruption.floatValue(), normalize.getOpaque(), this.y);
    }

    @Deprecated
    private double[] get(double[] x, double noise, float corruption, @Deprecated boolean normalize, double[] y) {

        this.x = x;

        preprocess(x, noise, corruption);

        int ins = x.length, outs = y.length;

        feedForward(x, y);

        activationEnc.applyTo(y, 0, outs);

        if (normalize) {
            Util.normalize(y, y.length);
            //Util.normalizeCartesian(y, NORMALIZATION_EPSILON);
        }

        return y;
    }

    private void feedForward(double[] x, double[] y) {
        var W = this.W;
        var hbias = this.hbias;

        int ins = x.length, outs = y.length;
        for (var o = 0; o < outs; o++) {
            var yo = hbias[o];
            var wi = W[o];
            for (var i = 0; i < ins; i++)
                yo = fma(wi[i], x[i], yo);

            if (yo != yo)
                yo = 0; //HACK corrupted hbias?

            y[o] = yo;
        }
    }


    public final double[] decode(double[] y) {
        return decode(y, this.z);
    }

    @Deprecated
    public double[] decode(double[] y, double[] z) {

        int ii = z.length, oo = y.length;

        for (var i = 0; i < ii; i++) {
            var zi = vbias[i];
            for (var o = 0; o < oo; o++)
                zi = fma(W[o][i], y[o], zi);  //zi += w[o][i] * y[o];

            z[i] = activationDec.valueOf(zi);
        }

        return z;
    }


    public int outputs() {
        return y.length;
    }

    public double[] output() {
        return y;
    }

    @Override
    @Deprecated
    public void put(double[] x) {

        var y = get(x);

        var z = decode(y, this.z);

        if (latent != null)
            latent.latent(x, y, z);

        var ii = this.inputs();
        var delta = this.delta;
        for (var i = 0; i < ii; i++)
            delta[i] = x[i] - z[i];

        learn(alpha.asFloat());
    }

    private void learn(double lr) {
        //forget(lr*lr);

        int ii = inputs(), oo = outputs();


        // Decoder part (output layer to hidden layer)
        var aDec = this.activationDec;
        for (var i = 0; i < ii; i++) {
            var d = lr * grad(delta[i] * aDec.derivative(z[i]));
            vbias[i] += d;
            for (var o = 0; o < oo; o++)
                W[o][i] = fma(d, y[o], W[o][i]);  //W[o][i] += d * y[o];
        }


        // Encoder part (hidden layer to input layer)
        var aEnc = this.activationEnc;
        for (var o = 0; o < oo; o++) {
            var wo = this.W[o];

            double lbi = 0;
            for (var i = 0; i < ii; i++)
                lbi = fma(wo[i], delta[i], lbi);

            var g = grad(aEnc.derivative(y[o]) * lbi);

            this.hbias[o] += lr * g;
            for (var i = 0; i < ii; i++)
                wo[i] = fma(lr, fma(delta[i], y[o], g * x[i]), wo[i]);
        }
    }

    public double[] reconstruct(double[] x) {
        return decode(get(x, 0, 0, false, new double[this.y.length]), z);
    }

    public int inputs() {
        return x.length;
    }

    /**
     * TODO some or all of the bias vectors may need modified too here
     */
    public void forget(double rate) {
        var mult = 1 - rate;
        for (var doubles : this.W)
            Util.mul(doubles, mult);
        Util.mul(hbias, mult);
        Util.mul(vbias, mult);
    }

    private double grad(double gradient) {
        return clipGrad ? Util.clampSafe(gradient, -1, +1) : gradient;

        // Or implement gradient normalization
        // double norm = Math.abs(gradient);
        // return norm > 1.0 ? gradient / norm : gradient;
    }
}