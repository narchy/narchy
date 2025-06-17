package jcog.tensor;

import jcog.data.bit.MetalBitSet;
import jcog.model.LayerNorm;
import jcog.data.list.Lst;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.function.UnaryOperator;

import static jcog.Util.lerp;
import static jcog.tensor.Tensor.*;

public class Models {

    public static class BiasActivation implements UnaryOperator<Tensor> {
        @Nullable
        public final Tensor bias;
        @Nullable
        final UnaryOperator<Tensor> activation;

        public final int outputs;

        public BiasActivation(int O, @Nullable UnaryOperator<Tensor> activation, boolean bias) {
            this.outputs = O;
            this.activation = activation;
            this.bias = bias(bias, O);
        }

        @Override
        public Tensor apply(Tensor x) {
            return biasActivation(x);
        }

        protected Tensor biasActivation(Tensor x) {
            if (this.bias != null)
                x = x.add(this.bias);
            if (this.activation != null)
                x = activation.apply(x);
            return x;
        }
    }

    public static class Linear extends BiasActivation {
        public final Tensor weight;

        public Linear(int inFeatures, int outFeatures, @Nullable UnaryOperator<Tensor> activation, boolean bias) {
            super(outFeatures, activation, bias);

            var init = activation == RELU || activation == RELU_LEAKY ? randHe(inFeatures, outFeatures) : randXavier(inFeatures, outFeatures);

            this.weight = init.grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            var inner =
                    inner(x);
            return outer(inner);
        }

        private Tensor inner(Tensor x) {
            //return super.apply(x.matmul(this.weight));
            var y = bias != null ?
                x.matmulAdd(this.weight, bias) :
                x.matmul(this.weight);

            return activation != null ? activation.apply(y) : y;
        }

        private Tensor outer(Tensor x) {
            return x;
            //return x.clipGrad(-1, +1);
        }
    }


    private static @Nullable Tensor bias(boolean bias, int outFeatures) {
        return bias ? zeros(1, outFeatures).grad(true).parameter() : null;
    }



    public static class Layers implements UnaryOperator<Tensor> {
        public final List<UnaryOperator<Tensor>> layer = new Lst<>();

        /**
         * 0 to disable
         */
        public float dropout =
            0; //disabled
            //0.1f;
            //1/4f;

        public boolean layerNorm;
        public double layerNormRate = 1e-5f;

        public Layers() {
        }

        @Deprecated
        public Layers(UnaryOperator<Tensor> activationHidden, @Nullable UnaryOperator<Tensor> activationOutput, int... layerSizes) {
            this(activationHidden, activationOutput, true, layerSizes);
        }

        @Deprecated
        public Layers(UnaryOperator<Tensor> activationHidden, @Nullable UnaryOperator<Tensor> activationOutput, boolean biasInLastLayer, int... layerSizes) {

            var L = layerSizes.length - 1;
            for (var l = 0; l < L; l++) {
                var innerLayer = l < L - 1;
                var I = layerSizes[l];
                var O = layerSizes[l + 1];
                var a = innerLayer ? activationHidden : activationOutput;
                var bias = innerLayer || biasInLastLayer;

//                if (innerLayer) {
//                    if (i > 0 && I>4) //HACK
//                        layer.add(
//                            //new ULT(I, I, I%2==0 ? I : I+1,  2)
//                        );
//                }

                beforeLayer(I, l, L);
                layer.add(layer(l, L,  I, O, a, bias));
                afterLayer(O, l, L);
            }

            train(true);
        }

        protected void afterLayer(int O, int l, int maxLayers) {
            //                if (innerLayer) {
//                    layer.add(
//                        new MixtureOfExperts(O, O, 2, null, false)
//                        //new FastSlowNetwork(O, O, a, bias)
//                    );
//                }
            var innerLayer = l < maxLayers - 1;
            if (innerLayer) {
                if (dropout > 0 && O > 1)
                    layer.add(new Dropout(dropout));

                if (layerNorm && O > 2)
                    layer.add(
                            new jcog.model.LayerNorm(O, layerNormRate)
                            //new PowerNorm(O, layerNormRate)
                    );

                //layer.add(new PowerNorm(O));
            }
        }

        protected void beforeLayer(int I, int l, int maxLayers) {

        }

        public static int[] layerLerp(int in, int hidden, int out, int layers) {
            return layerLerp(in, hidden, out, layers, 1);
        }

        public static int[] layerLerp(int in, int hidden, int out, int layers, float power) {
            var y = new IntArrayList(layers + 2);
            y.add(in);
            for (var l = 0; l < layers; l++)
                y.add(Math.round(lerp((float) Math.pow((float) l / layers, power), hidden, out)));
            y.add(out);
            return y.toArray();
        }

        protected UnaryOperator<Tensor> layer(int l, int maxLayers, int I, int O, UnaryOperator<Tensor> a, boolean bias) {
            return new Linear(I, O, a, bias);

//            if (I > 32)
//                return new LowRankLinear(I, O, a, bias,
//                        //(int)Util.sqrt(I)
//                        (int)Math.pow(I, 1/3f)
//                );

            //return new CirculantLinear(I, O, a, bias);
            //return new WinogradLinear(I, O, a, bias);

            //return new ULT(I, O, I,  4, a, bias);
            //return new NormalizedULT(I, O, I,  4, a, bias);
            //return new MixtureOfExperts(I, O, 2, a, bias);
            //return new FastSlowNetwork(I, O, a, bias);
        }

        /**
         * switch between training and inference
         * TODO do this recursively, maybe add boolean to TensorOp
         */
        @Deprecated
        public void train(boolean training) {
            for (var l : layer) {
                if (l instanceof Dropout d)
                    d.training = training;
            }
        }

        /**
         * forward pass
         */
        @Override
        public Tensor apply(Tensor x) {
            var y = x;
            for (int i = 0, size = layer.size(); i < size; i++)
                y = layer.get(i).apply(y);
            return y;
        }

    }

    public static class Dropout implements UnaryOperator<Tensor> {
        public final FloatRange dropoutRate;
        private boolean training = true;

        public Dropout(float dropoutRate) {
            this.dropoutRate = FloatRange.unit(dropoutRate);
        }

        @Override
        public Tensor apply(Tensor input) {
            if (!training)
                return input; // In inference mode, just return the input tensor as-is

            var dropoutRate = this.dropoutRate.floatValue();
            if (dropoutRate <= 0)
                return input;

            var yy = input.array().clone();

            var n = yy.length;

            var random = new RandomBits(new XoRoShiRo128PlusRandom());

            var nDropped = random.nextFloor(dropoutRate * n);
            var zeros = MetalBitSet.bits(n);
            for (var i = 0; i < nDropped; i++) {
                var e = random.nextInt(n);
                zeros.set(e);
                yy[e] = 0;
            }

            var y = new Tensor(yy, input.rows(), input.cols(), input.hasGrad());

            if (y.hasGrad()) {
                y.op = new TensorOp(input) {
                    @Override
                    public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                        // During backpropagation, scale the gradients by the dropout mask
                        double[] o = array(gradOut[0]), g = array(grad);
                        var e = o.length;
                        var dropoutFactor = 1.0 / (1 - dropoutRate);
                        for (var i = 0; i < e; i++)
                            o[i] = zeros.test(i) ? 0 : g[i] * dropoutFactor;
                    }
                };
            }
            return y;
        }
    }

    /**
     * Universal Linear Transformer
     * @deprecated This class is deprecated. Use {@link jcog.tensor.MultiHeadAttention}
     * (in conjunction with {@link jcog.tensor.AttentionMechanisms#scaledDotProductAttention}
     * and {@link jcog.tensor.Models.Linear} for projections) as the preferred, more modular,
     * and flexible alternative for implementing multi-head attention mechanisms.
     * {@code MultiHeadAttention} offers better support for features like attention masking.
     */
    @Deprecated
    public static class ULT extends BiasActivation {
        public final Tensor Wq, Wk, Wv, Wo;
        protected final int dHead;
        private final int heads;

        public ULT(int inputDim, int outputDim, int dModel, int heads, @Nullable UnaryOperator<Tensor> activation, boolean bias) {
            super(outputDim, activation, bias);
            this.heads = heads;
            if (dModel % heads != 0)
                dModel += heads - dModel % heads; //perfectly divisible

            this.dHead = dModel / heads;
            if (dHead <= 0)
                throw new IllegalArgumentException();

            this.Wq = randHe(inputDim, dModel).grad(true).parameter();
            this.Wk = randHe(inputDim, dModel).grad(true).parameter();
            this.Wv = randHe(inputDim, dModel).grad(true).parameter();
            this.Wo = randHe(dModel, outputDim).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            // Linear projections
            var Q = shape(x.matmul(Wq));
            var K = shape(x.matmul(Wk));
            var V = shape(x.matmul(Wv));

            // Unnormalized attention scores
            var scores = Q.matmulTranspose(K);

            // Normalized attention scores
            var scoresNormalized = scores.div(Math.sqrt(dHead));

            // attention weights via softmax
            var attentionWeights = scoresNormalized.softmax();

            // apply attention
            var output = attentionWeights.matmul(V);

            // Final linear projection
            var y = unshape(output).matmul(Wo);

            return super.apply(y);
        }

        /**
         * shape for multihead attention
         */
        protected Tensor shape(Tensor x) {
            return x.reshape(
                    x.rows() * heads,
                    x.cols() / heads);
        }

        protected Tensor unshape(Tensor x) {
            return x.reshape(
                    x.rows() / heads,
                    x.cols() * heads);
        }

    }


    /**
     * Benefits of using LayerNorm:
     * Improved training stability: By normalizing the inputs to each layer, we reduce the internal covariate shift, which can lead to more stable gradients during backpropagation.
     * Faster convergence: LayerNorm can help the network converge faster, especially in deep architectures.
     * Reduced sensitivity to initialization: The normalization helps mitigate issues that can arise from poor weight initialization.
     * Works well with different activation functions: LayerNorm is compatible with various activation functions and can be used in different types of neural network architectures.
     */
    /**
     * @deprecated This class is deprecated. Use {@link jcog.model.LayerNorm} instead.
     */
    @Deprecated
    public static class LayerNorm implements UnaryOperator<Tensor> {
        protected final double epsilon;

        public final Tensor gamma, beta;

        public LayerNorm(int dim, double epsilon) {
            this.epsilon = epsilon;
            this.gamma = ones(1, dim).grad(true).parameter();
            this.beta = zeros(1, dim).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            var mean = x.mean();
            var var = x.mse(mean);
            var std = var.add(epsilon).pow(power());
            var normalized = x.sub(mean).div(std);
            return normalized.mul(gamma).add(beta);
        }

        private static final Tensor HALF = scalar(0.5f);

        protected Tensor power() {
            return HALF;
        }
    }

    /**
     * Implements a Mixture of Experts (MoE) layer.
     * <p>
     * This layer uses multiple "expert" sub-networks and a gating network to determine
     * how to combine the outputs of these experts.
     * <p>
     * This implementation uses a "dense" approach where all experts are evaluated for each input,
     * and their outputs are combined using weights from a gating network. The gating network
     * uses a softmax activation to produce a probability distribution over the experts for each input token.
     * It does not currently implement sparse top-K routing (where only a subset of experts are chosen).
     * </p>
     * <p>
     * The 'gatingSimple' constructor parameter is currently a no-op and input-dependent gating is always used.
     * The main activation function and bias (if any) are applied after mixing the expert outputs.
     * </p>
     */
    public static class MixtureOfExperts extends BiasActivation {
        public final List<UnaryOperator<Tensor>> experts;
        public final Linear gating;
        private final int inputs, outputs;

        // Use softmax for gating to get a probability distribution over experts.
        private final static UnaryOperator<Tensor> gatingActivaton =
                Tensor::softmax;

        private final boolean gatingSimple;

        public MixtureOfExperts(int I, int O, int experts, boolean gatingSimple, UnaryOperator<Tensor> activation, boolean bias) {
            super(O, activation, bias);
            this.inputs = I;
            this.outputs = O;
            this.gatingSimple = gatingSimple;

            this.experts = new Lst<>(experts);
            for (var i = 0; i < experts; i++)
                this.experts.add(new Linear(I, O, null, false));

            this.gating = new Linear(this.gatingSimple ? 1 : I, experts, gatingActivaton,
                    false
            );
        }

        @Override
        public Tensor apply(Tensor x) {
            return super.apply(mix(x, gating(x)));
        }

        private Tensor gating(Tensor x) {
            // TODO: Review the 'gatingSimple' flag. For now, ensuring input-dependent gating.
            // if (gatingSimple) {
            //     return gating.apply(gating.weight); // Original problematic path
            // } else {
            //     return gating.apply(x);
            // }
            return gating.apply(x); // Always use input-dependent gating
        }

        /**
         * Combines expert outputs
         */
        private Tensor mix(Tensor x, Tensor gate) {
            Tensor y = null;
            var n = experts.size();
            for (var i = 0; i < n; i++) {
                var yi = experts.get(i).apply(x).mul(gate.slice(i));
                y = y != null ? y.add(yi) : yi;
            }
            return y;
            //return y.div(gate.sum()); //normalize to weighted sum
        }
    }


    /**
     * Winograd's minimal filtering algorithm for reduced multiply-adds.
     * @deprecated Suggest using {@link jcog.tensor.Models.Linear} for general linear transformations.
     * The benefits of Winograd algorithm are typically for convolutional layers and may not be generally applicable here.
     */
    @Deprecated
    public static class WinogradLinear extends BiasActivation {
        private final Tensor W1, W2, W3;
        private final int actualOutputs;

        public WinogradLinear(int inFeatures, int outputs, UnaryOperator<Tensor> activation, boolean bias) {
            super(outputs, activation, bias);
            this.actualOutputs = outputs;
            int paddedOutputs = (outputs + 1) & ~1; // Round up to next even number
            int halfOut = paddedOutputs / 2;

            this.W1 = randHe(inFeatures, halfOut).grad(true).parameter();
            this.W2 = randHe(inFeatures, halfOut).grad(true).parameter();
            this.W3 = randHe(inFeatures, halfOut).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            var m1 = x.matmul(W1);
            var m2 = x.matmul(W2);
            var m3 = x.matmul(W3);

            var left = m1.add(m2);
            var right = m2.add(m3);
            var full = left.concat(right);

            return super.apply(actualOutputs < full.volume() ? full.slice(0, actualOutputs) : full);
        }
    }


    /**
     * Low rank matrix approximation via UV decomposition. Memory efficient for redundant features.
     * @deprecated Suggest using {@link jcog.tensor.Models.Linear}. While parameter efficiency is important,
     * it can often be achieved via architectural choices (e.g., parameter sharing, factorized embeddings)
     * or model compression techniques applied to standard layers.
     */
    @Deprecated
    public static class LowRankLinear extends BiasActivation {
        public final Tensor U, V;

        public LowRankLinear(int inFeatures, int rank, int outFeatures, UnaryOperator<Tensor> activation, boolean bias) {
            super(outFeatures, activation, bias);
            var scale = Math.sqrt(2.0 / (inFeatures + outFeatures));
            this.U = randGaussian(inFeatures, rank, scale).grad(true).parameter();
            this.V = randGaussian(rank, outFeatures, scale).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            return super.apply(x.matmul(U).matmul(V));
        }
    }


    /**
     * Diagonal-only weights for efficient feature-wise scaling.
     * @deprecated Suggest using {@link jcog.tensor.Models.Linear} for general linear operations.
     * For learnable feature-wise scaling, {@link jcog.tensor.Models.LayerNorm} (specifically its {@code gamma} parameter)
     * provides this, or direct element-wise multiplication with a learnable parameter tensor can be used.
     */
    @Deprecated
    public static class DiagonalLinear extends BiasActivation {
        private final Tensor diagonal;

        public DiagonalLinear(int features, UnaryOperator<Tensor> activation, boolean bias) {
            super(features, activation, bias);
            this.diagonal = randGaussian(1, features, Math.sqrt(2.0 / features))
                    .grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            return super.apply(x.mul(diagonal));
        }
    }



}