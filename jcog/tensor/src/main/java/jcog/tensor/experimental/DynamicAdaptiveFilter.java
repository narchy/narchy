package jcog.tensor.experimental;

import jcog.TODO;
import jcog.Util;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import org.eclipse.collections.api.tuple.Pair;
import org.eclipse.collections.impl.tuple.Tuples;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

import static jcog.Str.n2;
import static jcog.tensor.Tensor.array;

/**
 * DynamicAdaptiveFilter (DAF) is a layer designed to replace standard linear layers,
 * offering improved performance, learning capabilities, and adaptivity while maintaining
 * a fixed memory footprint. It employs adaptive filters, an attention mechanism, and
 * dynamic learning rate adjustments to facilitate continuous learning in non-stationary
 * environments.
 */
public class DynamicAdaptiveFilter {
    // Layer dimensions and hyperparameters
    private final int inputSize;
    private final int outputSize;
    private final int numFilters;
    private final float initialLearningRate;
    private final float forgettingFactor;
    private final int bufferSize;

    // Weight matrices
    private final float[] weights; // [numFilters * inputSize]
    private final float[] attentionWeights; // [numFilters]
    private final float[] outputWeights; // [numFilters * outputSize]

    // Buffers for moving averages (for dynamic learning rate)
    private final float[] gradientMovingAverage; // [numFilters * inputSize + numFilters + numFilters * outputSize]
    private final float[] gradientVariance; // [numFilters * inputSize + numFilters + numFilters * outputSize]

//    // Activation buffers
//    private final float[] filterOutputs; // [numFilters]
//    private final float[] attentionedOutputs; // [numFilters]
//    private final float[] layerOutput; // [outputSize]

    // Ring buffers for recent inputs and outputs
    private final float[] recentInputs; // [bufferSize * inputSize]
    private final float[] recentOutputs; // [bufferSize * outputSize]
    private int bufferIndex;

    /**
     * Constructs a DynamicAdaptiveFilter layer with the specified parameters.
     *
     * @param inputSize        Number of input features.
     * @param outputSize       Number of output features.
     * @param numFilters       Number of adaptive filters.
     * @param learningRate     Initial learning rate.
     * @param forgettingFactor Forgetting factor for moving averages.
     * @param bufferSize       Size of the ring buffers for recent inputs and outputs.
     */
    public DynamicAdaptiveFilter(int inputSize, int outputSize, int numFilters,
                                 float learningRate, float forgettingFactor, int bufferSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.numFilters = numFilters;
        this.initialLearningRate = learningRate;
        this.forgettingFactor = forgettingFactor;
        this.bufferSize = bufferSize;

        // Initialize weight matrices
        this.weights = new float[numFilters * inputSize];
        this.attentionWeights = new float[numFilters];
        this.outputWeights = new float[numFilters * outputSize];

        // Initialize gradient accumulators for dynamic learning rate
        var totalParams = (numFilters * inputSize) + numFilters + (numFilters * outputSize);
        this.gradientMovingAverage = new float[totalParams];
        this.gradientVariance = new float[totalParams];
        Arrays.fill(this.gradientMovingAverage, 0.0f);
        Arrays.fill(this.gradientVariance, 1.0f); // Initialize variance to 1 to prevent division by zero

        // Initialize ring buffers
        this.recentInputs = new float[bufferSize * inputSize];
        this.recentOutputs = new float[bufferSize * outputSize];
        this.bufferIndex = 0;

        // Initialize weights with small random values
        initializeWeights();
    }

    /**
     * Initializes the weight matrices with small random values and normalizes attention weights.
     */
    private void initializeWeights() {
        // Initialize adaptive filter weights with Xavier/Glorot initialization for better convergence
        var limit = (float) Math.sqrt(2.0/*6.0*/ / (inputSize + 1)); // Assuming activation function is tanh or similar
        for (var i = 0; i < weights.length; i++)
            weights[i] = (float) ((Math.random() * 2 * limit) - limit); // Initialized between -limit and limit

        // Initialize attention weights uniformly
        Arrays.fill(attentionWeights, 1.0f / numFilters);

        // Initialize output weights with Xavier/Glorot initialization
        var limitOutput = (float) Math.sqrt(2.0/*6.0*/ / (numFilters + outputSize));
        for (var i = 0; i < outputWeights.length; i++)
            outputWeights[i] = (float) ((Math.random() * 2 * limitOutput) - limitOutput);
    }

    /**
     * Applies the ReLU activation function element-wise to the input array.
     *
     * @param x Input array.
     * @return Activated array.
     */
    private float[] relu(float[] x) {
        var activated = new float[x.length];
        for (var i = 0; i < x.length; i++) activated[i] = Math.max((float) 0, x[i]);
        return activated;
    }

    /**
     * Applies a numerically stable softmax function to the input array.
     *
     * @param x Input array.
     * @return Softmax probabilities.
     */
    private float[] softmax(float[] x) {
        var max = Float.NEGATIVE_INFINITY;
        for (var val : x) if (val > max) max = val;

        var sum = 0.0f;
        var exp = new float[x.length];
        for (var i = 0; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
            sum += exp[i];
        }

        var softmax = new float[x.length];
        for (var i = 0; i < x.length; i++) softmax[i] = exp[i] / sum;
        return softmax;
    }

    /**
     * Performs the forward pass of the DAF layer.
     *
     * @param input        Input activations from the previous layer.
     */
    public Pair<float[],Consumer<float[]>> forward(float[] input) {
        if (input.length != inputSize)
            throw new IllegalArgumentException("Input length mismatch. Expected: " + inputSize + ", Received: " + input.length);

        float[] output = new float[outputSize];
        float[] filterOutputs = new float[numFilters];
        double[] attentionedOutputs = new double[numFilters];

        synchronized (this) {
            // Update ring buffers with current input and previous output
            System.arraycopy(input, 0, recentInputs, bufferIndex * inputSize, inputSize);
            System.arraycopy(output, 0, recentOutputs, bufferIndex * outputSize, outputSize);
            bufferIndex = (bufferIndex + 1) % bufferSize;

            // Compute filter outputs: filterOutputs = weights * input
            computeFilterOutputs(input, filterOutputs);

            // Apply activation function
            var activatedFilters = relu(filterOutputs);

            // Compute attention weights using softmax
            var normalizedAttentionWeights = softmax(activatedFilters);
            System.arraycopy(normalizedAttentionWeights, 0, attentionWeights, 0, numFilters);

            // Weight filter outputs with attention weights
            for (var f = 0; f < numFilters; f++) attentionedOutputs[f] = activatedFilters[f] * attentionWeights[f];

            // Compute layer output: layerOutput = outputWeights^T * attentionedOutputs
            computeLayerOutput(attentionedOutputs, output);

            return Tuples.pair(output, grad ->
                    backpropagate(input, grad, activatedFilters, normalizedAttentionWeights, attentionedOutputs));
        }
    }

    /**
     * Computes the outputs of the adaptive filters.
     *
     * @param input Input activations.
     */
    private void computeFilterOutputs(float[] input, float[] filterOutputs) {
        for (var f = 0; f < numFilters; f++) {
            var sum = 0.0f;
            var filterOffset = f * inputSize;
            for (var i = 0; i < inputSize; i++) sum += weights[filterOffset + i] * input[i];
            filterOutputs[f] = sum;
        }
    }

    /**
     * Computes the final layer output by applying output weights to attentioned filter outputs.
     */
    private void computeLayerOutput(double[] attentionedOutputs, float[] layerOutput) {
        for (var o = 0; o < outputSize; o++) {
            var sum = 0.0f;
            var outputWeightOffset = o * numFilters;
            for (var f = 0; f < numFilters; f++)
                sum += outputWeights[outputWeightOffset + f] * attentionedOutputs[f];
            layerOutput[o] = sum;
        }
    }

    /**
     * Performs the backward pass, computing gradients and updating weights.
     *
     * @param input                      Input activations.
     * @param targetOutput               Target output activations.
     * @param activatedFilters           Activated filter outputs after ReLU.
     * @param normalizedAttentionWeights Normalized attention weights after softmax.
     */
    private void backpropagate(float[] input, float[] grad,
                               float[] activatedFilters, float[] normalizedAttentionWeights, double[] attentionedOutputs) {

        synchronized (this) {
            // Compute gradients for output weights: dL/dW_output = error * attentionedOutputs
            var gradOutputWeights = new float[numFilters * outputSize];
            for (var o = 0; o < outputSize; o++) {
                var outputWeightOffset = o * numFilters;
                for (var f = 0; f < numFilters; f++)
                    gradOutputWeights[outputWeightOffset + f] = (float) (grad[o] * attentionedOutputs[f]);
            }

            // Compute gradients for attention weights: dL/dW_attention = sum(error * W_output * activatedFilters)
            var gradAttentionWeights = new float[numFilters];
            for (var f = 0; f < numFilters; f++) {
                var g = 0.0f;
                for (var o = 0; o < outputSize; o++)
                    g += grad[o] * outputWeights[o * numFilters + f] * activatedFilters[f];
                gradAttentionWeights[f] = g;
            }

            // Compute gradients for adaptive filters: dL/dW_filters = sum(error * W_output * attentionWeights * input)
            var gradWeights = new float[numFilters * inputSize];
            for (var f = 0; f < numFilters; f++)
                for (var i = 0; i < inputSize; i++)
                    for (var o = 0; o < outputSize; o++)
                        gradWeights[f * inputSize + i] += grad[o] * outputWeights[o * numFilters + f]
                                * normalizedAttentionWeights[f] * input[i];

            // Update weights with gradients using correct mapping
            updateWeights(gradWeights, 0, numFilters * inputSize); // Adaptive filter weights
            updateWeights(gradAttentionWeights, numFilters * inputSize, numFilters * inputSize + numFilters); // Attention weights
            updateWeights(gradOutputWeights, numFilters * inputSize + numFilters, (numFilters * inputSize) + numFilters + (numFilters * outputSize)); // Output weights
        }
    }

    /**
     * Updates the weights using the computed gradients, dynamic learning rate, and forgetting factor.
     *
     * @param gradients  Array of gradients for the weights.
     * @param startIndex Starting index in the moving averages and variances.
     * @param endIndex   Ending index (exclusive) in the moving averages and variances.
     */
    private void updateWeights(float[] gradients, int startIndex, int endIndex) {
        for (int idx = startIndex, g = 0; idx < endIndex; idx++, g++) {
            if (g >= gradients.length)
                throw new IllegalArgumentException("Gradient array is shorter than the specified range. " +
                        "Gradients length: " + gradients.length + ", Expected: " + (endIndex - startIndex));
            // Update moving average of gradients
            gradientMovingAverage[idx] = forgettingFactor * gradientMovingAverage[idx] + (1 - forgettingFactor) * gradients[g];
            // Update moving average of squared gradients
            gradientVariance[idx] = forgettingFactor * gradientVariance[idx] + (1 - forgettingFactor) * gradients[g] * gradients[g];
            // Compute gradient variance (E[g^2] - (E[g])^2)
            var variance = gradientVariance[idx] - (gradientMovingAverage[idx] * gradientMovingAverage[idx]);
            // Ensure variance is non-negative
            variance = Math.max(variance, 1e-8f);
            // Compute adaptive learning rate
            var adaptiveLR = initialLearningRate / (float) Math.sqrt(variance);
            // Update weight
            // Adaptive filter weights
            if (idx < numFilters * inputSize)
                weights[idx] += adaptiveLR * gradients[g];
            else // Output weights
                // Attention weights
                if (idx < numFilters * inputSize + numFilters)
                    attentionWeights[idx - numFilters * inputSize] += adaptiveLR * gradients[g];
                else
                    outputWeights[idx - numFilters * inputSize - numFilters] += adaptiveLR * gradients[g];
        }
    }

    /**
     * Clears the recent inputs and outputs buffers.
     */
    public synchronized void resetBuffers() {
        synchronized(this) {
            Arrays.fill(recentInputs, 0.0f);
            Arrays.fill(recentOutputs, 0.0f);
            bufferIndex = 0;
        }
    }

    /**
     * Retrieves the recent inputs stored in the ring buffer.
     *
     * @return Array of recent inputs.
     */
    public synchronized float[] getRecentInputs() {
        return recentInputs.clone();
    }

    /**
     * Retrieves the recent outputs stored in the ring buffer.
     *
     * @return Array of recent outputs.
     */
    public synchronized float[] getRecentOutputs() {
        return recentOutputs.clone();
    }

    public static UnaryOperator<Tensor> model(int inputSize, int outputSize, int numFilters,
                                              float learningRate, float forgettingFactor, int bufferSize) {

        var daf = new DynamicAdaptiveFilter(inputSize, outputSize, numFilters,
                learningRate, forgettingFactor, bufferSize);

        boolean gradNorm = true;

        return (x) -> {
            var o_b = daf.forward(Util.toFloat(x.array()));
            var output = Util.toDouble(o_b.getOne());
            var back = o_b.getTwo();
            var t = new Tensor(output, 1, output.length, true);
            if (t.hasGrad()) {
                t.op = new Tensor.TensorOp(x) {

                    @Override
                    public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                        var g = array(grad).clone();

                        if (gradNorm)
                            Optimizers.GradNormL2.normL2(g);

                        System.out.println(n2(g));
                        back.accept(Util.toFloat(g));
                        if (gradOut[0]!=null) {
                            throw new TODO();
                            //gradOut[0].set(...);
                        }
                    }
                };

            }
            return t;
        };
    };

    /**
     * Main method for demonstrating the usage of the DynamicAdaptiveFilter.
     *
     * @param args Command-line arguments (not used).
     */
    public static void main(String[] args) {
        // Hyperparameters
        var inputSize = 10;
        var outputSize = 5;
        var numFilters = 8;
        var learningRate = 0.1f;
        var forgettingFactor = 0.95f;
        var bufferSize = 100;

        // Initialize DAF layer
        var daf = new DynamicAdaptiveFilter(inputSize, outputSize, numFilters,
                learningRate, forgettingFactor, bufferSize);

        // Example training loop
        var epochs = 1000;
        for (var epoch = 0; epoch < epochs; epoch++) {
            // Generate random input and target output for demonstration
            var input = new float[inputSize];
            var grad = new float[outputSize];
            for (var i = 0; i < inputSize; i++) input[i] = (float) Math.random()*2-1;
            for (var i = 0; i < outputSize; i++) grad[i] = (float) Math.random()*2-1;

            synchronized(daf) {
                daf.forward(input).getTwo().accept(grad);
            }
        }
    }
}
