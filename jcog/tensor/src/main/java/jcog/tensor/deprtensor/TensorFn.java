package jcog.tensor.deprtensor;

import jcog.Is;
import jcog.Util;
import jcog.data.list.Lst;
import jcog.tensor.Predictor;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * Differentiable Tensor - inspired by PyTorch
 */
@Is("Automatic_differentiation")
public abstract class TensorFn extends Tens0r {

    Tens0r[] x;

    protected TensorFn() {
        this(null);
    }

    TensorFn(Tens0r... x) {
        super();
        this.x = x;
    }

    @Override
    public void zeroGrad() {
        super.zeroGrad();
        if (x!=null)
            for (var xx : x)
                xx.zeroGrad();
    }

    public static TensorFn div(Tens0r a, Tens0r b) {
        return new TensorOpFn(TensorOp.DIV, a, b);
    }

    public static TensorOpFn pow(Tens0r a, double b) {
        return pow(a, scalar(b));
    }

    public static TensorOpFn pow(Tens0r a, Tens0r b) {
        return new TensorOpFn(TensorOp.POW, a, b);
    }

    public static TensorFn sqr(Tens0r a) {
        //return pow(a, 2);
        return multEW(a, a); //TODO more efficient sqr?
    }

    public static TensorFn sqrt(Tens0r a) {
        return pow(a, 0.5f);
    }

    public static TensorFn exp(Tens0r a) {
        return new TensorOpFn(TensorOp.EXP, a);
    }

    public static TensorFn exp(Tens0r a, double clipMin, double clipMax) {
        return exp(clip(a, Math.log(clipMin), Math.log(clipMax)));
    }

    public static TensorFn slice(Tens0r x, int index) {
        return slice(x, index, index + 1);
    }

    public static TensorFn slice(Tens0r x, int from, int to) {
        return new SliceTensor(x, from, to);
    }

    public static Tens0r clipTanh(Tens0r t, double yMin, double yMax) {
        return clipTanh(t, -1, +1, yMin, yMax);
    }
    public static Tens0r clipTanh(Tens0r t, double xMin, double xMax, double yMin, double yMax) {
        return clipTanh(t, xMin, xMax, yMin, yMax, 1);
    }
    public static Tens0r clipTanh(Tens0r t, double xMin, double xMax, double yMin, double yMax, double alpha) {
        // Scale input to [-2, 2] for the tanh function
        var xScaled = sub(mult(sub(t, xMin), 4 / (xMax - xMin)), 2);
        var yScaled = tanh(mult(xScaled, alpha));
        var y = add(mult(add(yScaled, 1), (yMax - yMin) / 2), yMin);
        return y;
    }

    public static TensorFn clip(Tens0r x, double min, double max) {
        return new ClipTensor(x, min, max);
    }
    public static TensorFn clipUnitPolar(Tens0r x) {
        return clip(x, -1, +1);
    }

    public static Tens0r sum(Tens0r x) {
        return x.isScalar() ? x : new TensorOpFn(TensorOp.SUM, x);
    }

    public static Tens0r mean(Tens0r x) {
        return x.isScalar() ? x : div(sum(x), scalar(x.volume()));
    }

    public static TensorFn log(Tens0r x) {
        return new TensorOpFn(TensorOp.LOG, x);
    }
    public static TensorFn log1p(Tens0r x) {
        return new TensorOpFn(TensorOp.LOG1P, x);
    }
    public static TensorFn softplus(Tens0r x) {
        return new TensorOpFn(TensorOp.SOFTPLUS, x);
    }

    public static TensorFn minEle(Tens0r x, Tens0r y) {
        return new TensorOpFn(TensorOp.MIN_ELE, x, y);
    }

    public static TensorFn add(Tens0r a, Tens0r b, Tens0r c) {
        return add(add(a, b), c);
    }

    public static TensorFn add(Tens0r a, Tens0r b) {
        if (b.isScalar())
            return new TensorOpFn(TensorOp.ADD_SCALAR, a, b);
        else if (a.isScalar())
            return new TensorOpFn(TensorOp.ADD_SCALAR, b, a);
        else
            return new TensorOpFn(TensorOp.ADD, a, b);
    }

    public static Tens0r add(Tens0r a, double b) {
        return b == 0 ? a : new TensorOpFn(TensorOp.ADD_SCALAR, a, scalar(b));
    }

    public static TensorFn neg(Tens0r a) {
        return new TensorOpFn(TensorOp.NEG, a);
    }

    public static Tens0r sub(Tens0r a, double scalar) {
        return add(a, -scalar);
    }

    public static TensorFn sub(Tens0r a, Tens0r b) {
        return add(a, neg(b));
    }

    public static TensorFn relu(Tens0r a) {
        return new TensorOpFn(TensorOp.RELU, a);
    }

    public static TensorFn leakyRelu(Tens0r a) {
        return new TensorOpFn(TensorOp.LEAKY_RELU, a);
    }

    public static TensorFn elephant(Tens0r a) {
        return new TensorOpFn(TensorOp.ELEPHANT, a);
    }

    public static TensorFn sigmoid(Tens0r a) {
        return new TensorOpFn(TensorOp.SIGMOID, a);
    }

    public static TensorFn tanh(Tens0r a) {
        return new TensorOpFn(TensorOp.TANH, a);
    }

    public static Tens0r linear(Tens0r a) {
        return a;
    }

    public static TensorFn mse(Tens0r yPred, double yTrue) {
        return mse(yPred, scalar(yTrue));
    }

    public static TensorFn mse(Tens0r yPred, Tens0r yTrue) {
        return new TensorOpFn(TensorOp.MSE, yPred, yTrue);
    }

    public static TensorFn mmult(Tens0r a, Tens0r b) {
        return new TensorOpFn(TensorOp.M_MULT, a, b);
    }

    public static TensorFn multEW(Tens0r a, Tens0r b) {
        return new TensorOpFn(TensorOp.EW_MULT, a, b);
    }

    public static TensorFn multScalar(Tens0r x, Tens0r scalar) {
        if (!scalar.isScalar())
            throw new UnsupportedOperationException();
        return new TensorOpFn(TensorOp.MULT_SCALAR, x, scalar);
    }

    public static Tens0r mult(double a, Tens0r b) {
        return mult(b, a);
    }

    public static Tens0r mult(Tens0r a, double b) {
        if (b == 1)
            return a;
        if (b == -1)
            return neg(a);
        return multScalar(a, scalar(b));
    }

    public static TensorFn min(Tens0r a, Tens0r b) {
        return new TensorOpFn(TensorOp.MIN, a, b);
    }

    /**
     * effectively 'constant' (no backward pass),
     * but sets its value dynamically in the forward pass
     */
    public static TensorFn scalar(DoubleSupplier d) {
        return new LambdaScalarTensor(d);
    }

    public static TensorFn concat(Tens0r a, Tens0r b) {
        return new TensorOpFn(TensorOp.CONCAT, a, b);
    }

    @Override
    public Tens0r detach() {
        x = null;
        //grad = null; //TODO
        return this;
    }

    abstract public void forward();

    @Override
    public void backward() {
        if (x != null) {
            for (Tens0r i : x)
                i.backward();
        }
    }

    public double minimize(Layers model, Optimize o) {
        double l = minimize();
        o.run(model);
        return l;
    }

    public enum Activate {
        Relu(TensorFn::relu, Init.He),
        LeakyRelu(TensorFn::leakyRelu, Init.HeLeaky),
        Elephant(TensorFn::elephant, Init.He),
        Sigmoid(TensorFn::sigmoid, Init.Xavier),
        TanH(TensorFn::tanh, Init.LeCun),
        Linear(TensorFn::linear, Init.LeCun),
        LinearClipUnit(TensorFn::clipUnitPolar, Init.LeCun);

        final Init init;
        final Function<Tens0r, Tens0r> fn;

        Activate(Function<Tens0r, Tens0r> fn, Init init) {
            this.fn = fn;
            this.init = init;
        }
    }


    public enum Init {
        He {
            @Override
            public void apply(Tens0r t, RandomGenerator rng) {
                int o = t.h();
                t.setGaussian(Math.sqrt(2.0 / o), rng);
            }
        },
        HeLeaky {
            @Override
            public void apply(Tens0r t, RandomGenerator rng) {
                double a = 0.01;//Op.LEAKY_RELU.alpha;
                t.setGaussian(Math.sqrt(2.0 / ((1 + a * a) * t.h())), rng);
            }
        },
        Xavier {
            @Override
            public void apply(Tens0r t, RandomGenerator rng) {
                t.setGaussian(Math.sqrt(2.0 / Util.mean((float) t.w(), (float) t.h())), rng);
            }
        },
        LeCun {
            @Override
            public void apply(Tens0r t, RandomGenerator rng) {
                t.setGaussian(Math.sqrt(1.0 / t.h()), rng);
            }
        };

        abstract public void apply(Tens0r weights, RandomGenerator rng);

        public void apply(Tens0r weights) {
            apply(weights, ThreadLocalRandom.current());
        }
    }

    @Deprecated
    static class TensorOpFn extends TensorFn {

        final TensorOp op;

        private TensorOpFn(TensorOp op, Tens0r... x) {
            super();
            this.op = op;
            this.x = x;
            setData(this.op.allocate(this.x));
            forward();
        }

        @Deprecated
        TensorOpFn(double[][] data, TensorOp op, Tens0r... x) {
            this.data = data;
            this.grad = new double[data.length][data[0].length];
            this.op = op;
            this.x = x;
        }

        @Override
        public void backward() {
            if (Optimize.clipGradBeforeBackward)
                Optimize.clip(grad);

            if (this.x!=null) {
                this.op.backward(this.x, grad);
            }

            super.backward();
        }

        @Override
        public void forward() {
            op.forward(x, data);
        }

        @Override
        public String toString() {
            return (op != null ? op : "tensor") + super.toString();
        }
    }

    @Deprecated
    public static abstract class AbstractLayer {

        abstract public Tens0r forward(Tens0r input);

        abstract public void backward(Tens0r input);

        abstract public void zeroGrad();

        abstract public AbstractLayer clone();

        abstract public void reset();

        public void addParams(Lst<Tens0r> target) {

        }
    }

    public static class AutoEncoder extends AbstractLayer {
        private final int inputSize, hiddenSize;
        private final Activate activationHidden, activationOutput;

        private Optimize.DenseLayer encoder, decoder;

        public AutoEncoder(int inputSize, int hiddenSize) {
            this(inputSize, hiddenSize, Activate.Relu, Activate.Sigmoid);
        }

        public AutoEncoder(int inputSize, int hiddenSize, Activate activationHidden, Activate activationOutput) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.activationHidden = activationHidden;
            this.activationOutput = activationOutput;

            // Encoder transforms input to hidden representation
            this.encoder = new Optimize.DenseLayer(inputSize, hiddenSize, activationHidden);

            // Decoder transforms hidden representation back to output
            this.decoder = new Optimize.DenseLayer(hiddenSize, inputSize, activationOutput);
        }

        @Override
        public void addParams(Lst<Tens0r> target) {
            encoder.addParams(target);
            decoder.addParams(target);
        }

        @Override
        public Tens0r forward(Tens0r input) {
            // Pass input through encoder, then pass the result through decoder
            Tens0r encoded = encoder.forward(input);
            Tens0r decoded = decoder.forward(encoded);

            var err = mse(input, decoded);
            //System.out.println(err.scalar());
            err.minimize();

            return decoded;
        }

        @Override
        public void backward(Tens0r outputGradient) {
            throw new UnsupportedOperationException();

//            // Backpropagate through decoder first
//            Tensor gradDecoder = decoder.backward(outputGradient);
//            // Backpropagate through encoder
//            return encoder.backward(gradDecoder);
        }

        @Override
        public void zeroGrad() {
        }

        @Override
        public AbstractLayer clone() {
            return new AutoEncoder(inputSize, hiddenSize, activationHidden, activationOutput);
        }

        @Override
        public void reset() {
            encoder.reset();
            decoder.reset();
        }
    }


    /**
     * untested
     */
    public static class LayerNormalization extends AbstractLayer {
        final double epsilon = 1e-5;
        final Tens0r gamma, beta;

        transient private Tens0r input;
        private Tens0r output;


        public LayerNormalization(LayerNormalization l) {
            this.gamma = newTensorSizeOf(l.gamma);
            this.beta = newTensorSizeOf(l.beta);
            reset();
        }

        public LayerNormalization(int features) {
            this.gamma = new Tens0r(new double[1][features]);
            this.beta = new Tens0r(new double[1][features]);
            reset();
        }

        @Override
        public void addParams(Lst<Tens0r> target) {
            target.add(beta);
            target.add(gamma);
        }

        @Override
        public Tens0r forward(Tens0r input) {
            this.input = input;
            double[] mean = new double[input.data[0].length];
            double[] variance = new double[input.data[0].length];

            if (output == null)
                /* TODO share the common code with DenseLayer */
                output = new TensorOpFn(new double[input.data.length][input.data[0].length], TensorOp.DENSE, input) {

                    @Override
                    public void backward() {
                        if (x == null)
                            return;

                        LayerNormalization.this.backward(x[0]);

                        x[0].backward();
                    }

                };

            double[][] output = this.output.data;
            // Compute mean and variance
            for (int j = 0; j < input.data[0].length; j++) {
                for (double[] row : input.data) {
                    mean[j] += row[j];
                }
                mean[j] /= input.data.length;
            }
            for (int j = 0; j < input.data[0].length; j++) {
                for (double[] row : input.data) {
                    variance[j] += Math.pow(row[j] - mean[j], 2);
                }
                variance[j] /= input.data.length;
            }

            // Normalize and scale-shift
            for (int i = 0; i < input.data.length; i++) {
                for (int j = 0; j < input.data[0].length; j++) {
                    output[i][j] = (input.data[i][j] - mean[j]) / Math.sqrt(variance[j] + epsilon);
                    output[i][j] = output[i][j] * gamma.data[0][j] + beta.data[0][j];
                }
            }

            return this.output;
        }

        @Override public void backward(Tens0r output) {
            double[][] gradGamma = gamma.grad;
            double[][] gradBeta = beta.grad;

            double[][] gradOutput = output.grad;
            double[][] normalizedData = output.data; //A.data;
            int batch_size = input.data.length;

            for (int j = 0; j < gamma.data.length; j++) {
                for (int i = 0; i < batch_size; i++) {
                    gradGamma[0][j] += gradOutput[i][j] * normalizedData[i][j];
                    gradBeta[0][j] += gradOutput[i][j];
                }
            }

            double[][] gradInput = input.grad;
            for (int i = 0; i < batch_size; i++) {
                for (int j = 0; j < gamma.data.length; j++) {
                    gradInput[i][j] = gradOutput[i][j] * gamma.data[0][j];
                }
            }

//                        this.gamma.grad = new Tensor(gradGamma);
//                        this.beta.grad = new Tensor(gradBeta);
//                        this.input.grad = new Tensor(gradInput); // Assuming input.grad is needed elsewhere
        }

        @Override
        public void zeroGrad() {
            output.zeroGrad();
        }

        @Override
        public LayerNormalization clone() {
            return new LayerNormalization(this);
        }

        @Override
        public void reset() {
            int features = gamma.w();
            for (int i = 0; i < features; i++) {
                gamma.data[0][i] = 1.0; // Initialize gamma to 1
                beta.data[0][i] = 0.0;  // Initialize beta to 0
            }
        }
    }

    abstract public static class LayersBuilder {
        abstract int inputs();

        abstract int outputs();

        abstract List<AbstractLayer> layers();

        public static class SimpleLayersBuilder extends LayersBuilder {
            final List<AbstractLayer> layers = new Lst();
            private final int inputs, outputs;

            public float dropoutRate;
            int dropoutMinInputs = 16;
            boolean layerNorm;

            public SimpleLayersBuilder(int[] layerSizes, Activate internal) {
                this(layerSizes, internal, Activate.Linear);
            }

            public SimpleLayersBuilder(int[] layerSizes, Activate internal, Activate output) {
                this(layerSizes, internal, output, 1);
            }

            public SimpleLayersBuilder(int[] layerSizes, Activate internal, Activate output, float inputCompression) {

                if (inputCompression!=1) {
                    int newLayer0Size = Math.round(layerSizes[0] * inputCompression);
                    layers.add(new AutoEncoder(layerSizes[0], newLayer0Size));
                    layerSizes[0] = newLayer0Size;
                }

                int i = layerSizes[0], o = -1;
                this.inputs = i;
                for (int l = 0; l < layerSizes.length - 1; l++) {
                    o = layerSizes[l + 1];
                    Optimize.DenseLayer x = new Optimize.DenseLayer(i, o,
                            l < layerSizes.length - 2 ?
                                    internal :
                                    output);

                    if (l > 0 && i >= dropoutMinInputs) /* && l < layerSizes.length - 2*/
                        x.setDropoutRate(dropoutRate);

                    layers.add(x);

                    if (layerNorm && l > 0 && l < layerSizes.length - 2)
                        layers.add(new LayerNormalization(o));

                    i = o;
                }
                this.outputs = o;
            }

            @Override
            int inputs() {
                return inputs;
            }

            @Override
            int outputs() {
                return outputs;
            }

            @Override
            List<AbstractLayer> layers() {
                return layers;
            }
        }
    }

    /**
     * multi-layer perceptron (MLP) model
     */
    @Deprecated
    public static class Layers extends Parameterized {
        final List<AbstractLayer> layers;
        private final int inputs, outputs;

        public Layers(Layers copyArchFrom) {
            this.inputs = copyArchFrom.inputs;
            this.outputs = copyArchFrom.outputs;
            this.layers = new Lst<>(copyArchFrom.layers.size());
            for (var l : copyArchFrom.layers)
                layers.add(l.clone());
            addParams();
        }

        public Layers(LayersBuilder b) {
            this.inputs = b.inputs();
            this.outputs = b.outputs();
            this.layers = b.layers();

            addParams();
        }

        @Deprecated
        private void addParams() {
            for (AbstractLayer layer : layers)
                layer.addParams(params);
        }

        @Override
        public Tens0r forward(Tens0r x) {
            for (var l : layers)
                x = l.forward(x);
            return x;
        }

        @Override
        public void zeroGrad() {
            for (var l : layers)
                l.zeroGrad();
        }

        @Override
        public String toString() {
            return layers.toString();
        }

        public Predictor predict(Optimize o) {
            return new Predictor() {

                @Override
                public String toString() {
                    return Layers.this + "," + o;
                }

                @Override
                public double[] put(double[] x, double[] y, @Deprecated float pri) {
                    Tens0r yActual = forward(new Tens0r(x));
                    double loss = mse(yActual, new Tens0r(y)).minimize(Layers.this, o);
                    return yActual.data[0];
                }

                @Override
                public double[] get(double[] x) {
                    return forward(new Tens0r(x)).data[0];
                }

                @Override
                public void clear(@Deprecated Random rng) {
                    for (var l : layers)
                        l.reset();
                }

            };
        }
    }

    /**
     * TODO this is intended to replace LayersBuilder
     */
    static class NetworkBuilder {

        public static Tens0r createDenseLayer(Tens0r input, Tens0r weights, Tens0r biases, Function<Tens0r, Tens0r> activationFn) {
            // Matrix multiplication between input and weights
            Tens0r preActivation = mmult(input, weights);

            // Adding biases
            Tens0r biased = add(preActivation, biases);

            // Applying activation function
            return activationFn.apply(biased);
        }


        public static List<Tens0r> buildNetwork(Tens0r inputTensor, int[] layerSizes, Tens0r[] weights, Tens0r[] biases, Function<Tens0r, Tens0r> activationFunction) {
            List<Tens0r> layers = new Lst<>();
            Tens0r currentLayerInput = inputTensor; // Input tensor to the network

            for (int i = 0; i < layerSizes.length - 1; i++) {
                currentLayerInput = createDenseLayer(
                        currentLayerInput,
                        weights[i],
                        biases[i],
                        activationFunction
                );
                layers.add(currentLayerInput);
            }

            return layers;
        }
    }

}
