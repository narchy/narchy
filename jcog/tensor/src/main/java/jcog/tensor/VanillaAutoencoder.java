package jcog.tensor;

import jcog.math.FloatSupplier;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.util.ArrayUtil;
import org.jetbrains.annotations.Nullable;

import java.util.Random;
import java.util.function.UnaryOperator;

public class VanillaAutoencoder extends AbstractAutoencoder {

    private static final UnaryOperator<Tensor> encActivation = Tensor.SIGMOID;

    public final Models.Layers encoder, decoder;
    private final Models.Dropout corruption;

    /** 0 disables weight sharing between encoder and decoder networks, 1 = instantaneous sharing */
    private final FloatRange weightSharing = FloatRange.unit(0);

    public final FloatRange sparse = new FloatRange(/*0*/ 0.005f, 0, 0.25f);

    public final FloatRange corrupt;

    private final int inputSize, encodedSize;

    private final Tensor.Optimizer optimizer;

    @Nullable public Tensor.GradQueue queue = null;

    private static final boolean
        biasInitZero = true,
        biasInEncoderOutput = true, biasInDecoderOutput = false;

    public VanillaAutoencoder(int inputSize, int encodedSize, FloatSupplier lr) {
        this(inputSize, ArrayUtil.EMPTY_INT_ARRAY, encodedSize, lr);
    }

    public VanillaAutoencoder(int inputSize, int[] hiddenSizes, int encodedSize, FloatSupplier lr) {
        this(inputSize, hiddenSizes, encodedSize,
            new Optimizers.ADAM(lr).get(),
            //new Optimizers.LION(lr).get(2),
            //new Optimizers.SGD(lr).getFast(),
            0.0001f,
            Tensor.RELU, Tensor.SIGMOID
            //Tensor.RELU_LEAKY, Tensor.SIGMOID
            //Tensor.SIGMOID, Tensor.SIGMOID
            //Tensor.RELU_LEAKY, Tensor::siglinear
            //Tensor::relu, Tensor::sigmoid
            //Tensor::swish, Tensor::sigmoid
            //Tensor::relu, Tensor::siglinear
            //Tensor::tanh, Tensor::sigmoid
            //Tensor::relu, Tensor::sigmoid
            //null, Tensor::sigmoid
            //Tensor::relu, Tensor::relu
        );
    }

    public VanillaAutoencoder(int inputSize, int[] hiddenSizes, int encodedSize,
                              Tensor.Optimizer o,
                              float corruptionRate,
                              UnaryOperator<Tensor> hiddenActivation,
                              UnaryOperator<Tensor> outputActivation) {
        this.inputSize = inputSize;
        this.encodedSize = encodedSize;

        this.corruption = new Models.Dropout(corruptionRate);
        this.corrupt = corruption.dropoutRate;

        // Create encoder layers
        var encoderSizes = new int[hiddenSizes.length + 2];
        encoderSizes[0] = inputSize;
        System.arraycopy(hiddenSizes, 0, encoderSizes, 1, hiddenSizes.length);
        encoderSizes[encoderSizes.length - 1] = encodedSize;

        // Create decoder layers (reverse of encoder)
        var decoderSizes = new int[hiddenSizes.length + 2];
        decoderSizes[0] = encodedSize;
        for (var i = 0; i < hiddenSizes.length; i++)
            decoderSizes[i + 1] = hiddenSizes[hiddenSizes.length - 1 - i];
        decoderSizes[decoderSizes.length - 1] = inputSize;

        this.encoder = new Models.Layers(hiddenActivation, encActivation, biasInEncoderOutput, encoderSizes);
        this.decoder = new Models.Layers(hiddenActivation, outputActivation, biasInDecoderOutput, decoderSizes);

        this.rng = new jcog.random.RandomBits(new XoRoShiRo128PlusRandom());

        this.optimizer = o;
        clear(rng.rng);
    }

    @Override
    public void clear(Random rng) {
        if (!biasInitZero) {
            //randomize biases (weights are already randomized, but it's re-randomized)
            clear(encoder);
            clear(decoder);
        }

//        //copy initial weights
//        var n = encoder.layer.size();
//        int p = 0;
//        for (var i = 0; i < n; i++) {
//            UnaryOperator<Tensor> encoderLayer = encoder.layer.get(i), decoderLayer = decoder.layer.get(n - 1 - i);
//            if (encoderLayer instanceof Models.Linear el && decoderLayer instanceof Models.Linear dl)
//                el.weight.setData(dl.weight.transpose().data);
//        }
    }

    private void clear(Models.Layers x) {
        for (var xl : x.layer) {
            Tensor.parameters(xl).forEach(t -> t.setData(
                Tensor.randGaussian(t.rows(), t.cols(), Math.sqrt(2.0 / t.cols())).data)
            );
        }
    }

    @Override
    public double[] get(double[] x) {
        return encoder.apply(input(x)).array();
    }

    @Override
    public void put(double[] x) {
        var input = input(x);
        var encoded = encoder.apply(input);
        var decoded = decoder.apply(encoded);

        if (latent != null)
            latent.latent(x, encoded.array(), decoded.array());

        var decodeLoss = decoded.
            sub(input).sqr(); //MSE without mean
            //mse(input);
            //mae(input);

        var loss = decodeLoss;

        var sparse = this.sparse.floatValue();
        if (sparse > 0)
            loss = loss.addMul(sparsityLoss(encoded), sparse);

        var weightSharing = this.weightSharing.floatValue();
        if (weightSharing > 0)
            loss = loss.addMul(weightSharingLoss(), weightSharing);

        loss.minimize(optimizer, queue);
    }

    private Tensor weightSharingLoss() {
        var loss = Tensor.scalar(0);

        var n = encoder.layer.size();
        int p = 0;
        for (var i = 0; i < n; i++) {
            var encoderLayer = encoder.layer.get(i);
            var decoderLayer = decoder.layer.get(n - 1 - i);

            if (encoderLayer instanceof Models.Linear e && decoderLayer instanceof Models.Linear d) {

                // Calculate distance between encoder weights and transposed decoder weights
                var ee = e.weight;
                var dd = d.weight.transpose();
                if (!ee.sameShape(dd))
                    throw new UnsupportedOperationException();

                var weightDiff = ee.subAbs(dd).sum();
                loss = loss.add(weightDiff);
                p += ee.volume();

                //bias sharing: probably not helpful as bias shapes seem to almost never match
//                if (e.bias!=null && d.bias!=null) {
//                    if (e.bias.sameShape(d.bias)) {
//                        var biasDiff = e.bias.subAbs(d.bias).sum();
//                        loss = loss.add(biasDiff);
//                        p += e.bias.volume();
//                    }
//                }
            }
        }

        var l = p > 0 ? loss.div(p) : Tensor.scalar(0);
        //System.out.println(n4(l.scalar()));
        return l;
    }

    /**
     * Goal: Most neurons in the encoded layer should be close to zero for any given input.
     * Implementation: Often uses L1 regularization on the activations of the encoded layer.
     * Effect: Forces the network to learn more distinctive and interpretable features.
     * Benefit: Improves feature disentanglement and can lead to better generalization.
     */
    private static Tensor sparsityLoss(Tensor encoded) {
        return
            encoded.lenL2().subAbs(1); //prefer L2 dist == 1
            //encoded.lenL1().subAbs(1); //prefer L1 dist == 1
            //encoded.lenL1().sub(1).relu(); //prefer L1 dist <= 1
            //encoded.lenL2().sub(1).relu(); //prefer L2 dist <= 1
            //encoded.lenL1().mul(1 - 1.0/encoded.volume());
            //encoded.lenL1().subAbs(Tensor.scalar(1)).mul(normalize.floatValue()); //~=1 //normalization regularization: length diff between encoded and 1
            //encoded.lenL1().sub(1).relu().subAbs(Tensor.scalar(0)).mul(normalize.floatValue()); //<=1 (at most 1)
    }


    public void commit() {
        if (queue == null) throw new UnsupportedOperationException();
        queue.optimize(optimizer);
    }

    /**
     * preprocess input
     */
    private Tensor input(double[] x) {
        return corruption.apply(Tensor.row(x));
    }
}