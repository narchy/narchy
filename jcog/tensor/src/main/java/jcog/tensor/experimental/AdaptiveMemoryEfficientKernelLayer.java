package jcog.tensor.experimental;

import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;

import java.util.function.UnaryOperator;

import static jcog.tensor.Tensor.array;

/**
 * A compact, self-contained Java implementation of a multi-dimensional
 * Adaptive Memory-Efficient Kernel Layer (AMEKL) with RBF kernels.
 *
 * Key Points:
 * - Each neuron uses fixed-size memory buffers for storing recent activations/gradients.
 * - RBF kernel parameters include amplitude, sigma, and center (with dimension = inputSize).
 * - Backprop uses mathematically correct partial derivatives for both parameters and inputs.
 * - Demonstrates forward pass, backward pass, and parameter update (SGD).
 */
public class AdaptiveMemoryEfficientKernelLayer implements UnaryOperator<Tensor> {

    private final Neuron[] neuron;
    private final int in;

    double epsilon      =
            //1e-8;
            1e-0;

    /**
     * Constructs an AMEKL with the specified input dimension and number of neurons.
     *
     * @param in            Dimensionality of the input (number of input features).
     * @param out           Number of neurons in this layer.
     * @param memorySize    Fixed size of each neuron's memory buffer.
     */
    public AdaptiveMemoryEfficientKernelLayer(int in, int out, int memorySize) {
        this.in = in;
        this.neuron = new Neuron[out];

        for (var i = 0; i < out; i++)
            // For each neuron, initialize RBF parameters randomly:
            // Param layout: [ amplitude, sigma, center_0, center_1, ..., center_(inputSize-1) ]
            neuron[i] = new Neuron(in, rbfRandom(in), new Memory(memorySize));
    }

    @Override
    public Tensor apply(Tensor X) {
        var x = X.array();
        var y = forward(x);

        var Y = Tensor.row(y).grad(true);
        Y.op = new Tensor.TensorOp(X) {
            @Override
            public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                float learningRate = 1e-2f;
                var gradIn = AdaptiveMemoryEfficientKernelLayer.this.backward(x, array(grad));
                update(learningRate);
                if (gradOut[0] != null)
                    System.arraycopy(gradIn, 0, array(gradOut[0]), 0, gradIn.length);
            }
        };
        return Y;
    }

    // -------------------------------------------------------------------------
    //                                PUBLIC METHODS
    // -------------------------------------------------------------------------

    /**
     * Forward pass: Computes the output of each neuron given an input vector.
     *
     * @param x An array of length = inputSize.
     * @return The activation of each neuron in this layer.
     */
    public double[] forward(double[] x) {
        if (x.length != in)
            throw new IllegalArgumentException("Input dimension mismatch.");

        synchronized(this) {
            var y = new double[neuron.length];
            for (var i = 0; i < neuron.length; i++)
                y[i] = neuron[i].forward(x);
            return y;
        }
    }

    /**
     * Backward pass: Computes gradients w.r.t. input, and accumulates parameter gradients internally.
     *
     * @param input      The same input used in forward pass.
     * @param gradOutput The gradient from the subsequent layer (or final loss), length = numberOfNeurons.
     * @return Gradient of this layer w.r.t. its input (same dimension as input).
     */
    public double[] backward(double[] input, double[] gradOutput) {
        if (gradOutput.length != neuron.length)
            throw new IllegalArgumentException("gradOutput dimension mismatch.");

        synchronized(this) {
            var gradInput = new double[in];  // accumulate across neurons
            for (var i = 0; i < neuron.length; i++) {
                var localGrad = neuron[i].grad(input, gradOutput[i]);
                // Sum contributions to gradInput
                for (var j = 0; j < in; j++)
                    gradInput[j] += localGrad[j];
            }
            return gradInput;
        }
    }

    /**
     * Updates the neuron parameters using a simple (but memory-adaptive) SGD step.
     *
     * @param learningRate Base learning rate for this layer.
     */
    public void update(double learningRate) {
        for (var nrn : neuron)
            nrn.update(learningRate);
    }

    // -------------------------------------------------------------------------
    //                               HELPER METHOD
    // -------------------------------------------------------------------------

    /**
     * Random initialization for an RBF kernel's parameters.
     * Layout: param[0] = amplitude, param[1] = sigma,
     *         param[2..(1+inputSize)] = center coordinates.
     * amplitude ~ (0.5 to 1.5), sigma ~ (0.1 to 1.0), center ~ (-1 to 1)
     */
    private static double[] rbfRandom(int inputSize) {
        var params = new double[inputSize + 2];
        params[0] = 0.5 + Math.random();         // amplitude
        params[1] = 0.5 + 0.9 * Math.random();   // sigma
        for (var i = 2; i < 2 + inputSize; i++)
            params[i] = -1.0 + 2.0 * Math.random();
        return params;
    }

    /**
     * Represents a single neuron in the AMEKL layer, using a chosen KernelFunction.
     */
    private class Neuron {
        private final Memory memoryBuffer;
        private final int inputDim;

        // Trainable parameters: [ amplitude, sigma, center_0, ..., center_(inputDim-1) ]
        private final double[] kernelParams;

        // Accumulate gradients until update
        private final double[] pendingParamGrads;

        // Cached forward-pass values for efficient backprop
        private double cachedKernelValue;  // e^{-E} portion
        private double cachedAmplitude;
        private double cachedSigma;
        private double cachedSquaredDist;  // sum( (x_i - c_i)^2 )

        Neuron(int inputDim,
               double[] initialParams,
               Memory memory) {
            this.inputDim = inputDim;
            this.kernelParams = initialParams.clone();
            this.memoryBuffer = memory;
            this.pendingParamGrads = new double[initialParams.length];
        }

        /**
         * Computes forward activation: f(x) = amplitude * exp(-squaredDist / (2*sigma^2)),
         * where squaredDist = sum_i (x_i - center_i)^2.
         */
        double forward(double[] input) {
            var amplitude = kernelParams[0];
            var sigma     = kernelParams[1];
            // center starts from index 2
            var center    = kernelParams;

            // 1) Compute squared distance
            var squaredDist = 0.0;
            for (var i = 0; i < inputDim; i++) {
                var diff = input[i] - center[i + 2];
                squaredDist += diff * diff;
            }

            // 2) Compute exponent term
            var E       = squaredDist / (2.0 * sigma * sigma);
            var expTerm = Math.exp(-E);

            // 3) Activation = amplitude * expTerm
            var activation = amplitude * expTerm;

            // Cache for backprop
            this.cachedAmplitude   = amplitude;
            this.cachedSigma       = sigma;
            this.cachedSquaredDist = squaredDist;
            this.cachedKernelValue = expTerm; // e^{-E}

            // Memory usage: store the activation
            memoryBuffer.storeActivation(activation);

            return activation;
        }

        /**
         * Computes the gradients w.r.t. input and accumulates param gradients.
         *
         * @param x    The input vector from forward pass.
         * @param grad The upstream gradient (scalar) for this neuron.
         * @return The gradient of this neuron w.r.t. each input dimension.
         */
        double[] grad(double[] x, double grad) {
            // Retrieve cached values to avoid re-computing
            var amplitude   = this.cachedAmplitude;
            var sigma       = this.cachedSigma;
            var squaredDist = this.cachedSquaredDist;
            var expTerm     = this.cachedKernelValue; // e^{-E}

            // We'll need the center
            var center = kernelParams;

            // 1) Grad wrt input x_i
            //    f(x) = amplitude * e^{-E}
            //    E = squaredDist / (2*sigma^2)
            //    partial f/partial x_i = amplitude * e^{-E} * (center_i - x_i) / sigma^2
            var gradInput = new double[inputDim];
            for (var i = 0; i < inputDim; i++) {
                var dfdxi = amplitude * expTerm * (center[i + 2] - x[i]) / (sigma * sigma);
                gradInput[i] = grad * dfdxi;
            }

            // 2) Grad wrt amplitude: partial f/partial amplitude = e^{-E}
            var dfda = expTerm;
            pendingParamGrads[0] += grad * dfda;

            // 3) Grad wrt sigma:
            //    partial f/partial sigma = amplitude * e^{-E} * (squaredDist / sigma^3)
            var dfdsigma = amplitude * expTerm * (squaredDist / (sigma * sigma * sigma));
            pendingParamGrads[1] += grad * dfdsigma;

            // 4) Grad wrt center_i:
            //    partial f/partial center_i = amplitude * e^{-E} * ((x_i - center_i)/sigma^2)
            for (var i = 0; i < inputDim; i++) {
                var dfdc_i = amplitude * expTerm * (x[i] - center[i + 2]) / (sigma * sigma);
                pendingParamGrads[2 + i] += grad * dfdc_i;
            }

            // Store gradient w.r.t. input in memory buffer
            memoryBuffer.storeGradient(gradInput);

            return gradInput;
        }

        /**
         * Applies gradient updates to the parameters via a simple SGD step,
         * but scaled by the neuron-specific average gradient norm from memory.
         */
        void update(double baseLearningRate) {
            // 1) Compute the average gradient norm from memory
            double avgGradNorm = memoryBuffer.getAverageGradientNorm();

            // 2) Adapt the learning rate inversely to the gradient norm (simple heuristic).
            //    Avoid division by zero by adding a small epsilon.
            double adaptedLR    = (avgGradNorm > 0.0)
                    ? (baseLearningRate / (epsilon + avgGradNorm))
                    : baseLearningRate;

            // 3) Update parameters
            for (int i = 0; i < kernelParams.length; i++) {
                kernelParams[i] -= adaptedLR * pendingParamGrads[i];
                pendingParamGrads[i] = 0.0; // reset after update
            }
        }
    }

    /**
     * KernelFunction interface: you can implement RBF, polynomial, wavelet, etc.
     * Here, we implement a multi-dimensional RBF in an inner class below.
     */
    private interface KernelFunction {
        double apply(double[] input, double[] params);
        double[] gradient(double[] input, double[] params);
    }

    /**
     * A simple fixed-size circular buffer to store recent activations and gradients.
     */
    private class Memory {
        private final double[] activationBuffer;
        private final double[][] gradientBuffer;
        private int currentIndex = 0;

        Memory(int bufferSize) {
            this.activationBuffer = new double[bufferSize];
            this.gradientBuffer   = new double[bufferSize][];
        }

        void storeActivation(double activation) {
            activationBuffer[currentIndex] = activation;
            // Move index in circular fashion
            currentIndex = (currentIndex + 1) % activationBuffer.length;
        }

        void storeGradient(double[] grad) {
            // Store the gradient at the last activation’s index (circular offset)
            var gradIndex = (currentIndex == 0)
                    ? activationBuffer.length - 1
                    : currentIndex - 1;
            gradientBuffer[gradIndex] = grad.clone();
        }

        /**
         * Returns the average L2-norm of the stored gradient vectors (excluding null entries).
         */
        double getAverageGradientNorm() {
            double sum   = 0.0;
            int    count = 0;
            for (var g : gradientBuffer) {
                if (g == null) continue;
                // Compute L2 norm^2 of one gradient
                double norm2 = 0.0;
                for (double val : g) {
                    norm2 += val * val;
                }
                sum   += norm2;
                count += 1;
            }
            return (count > 0) ? Math.sqrt(sum / count) : 0.0;
        }
    }

    // -------------------------------------------------------------------------
    //                                DEMO MAIN
    // -------------------------------------------------------------------------
    public static void main(String[] args) {
        // Example: 2D input, 3 neurons, buffer size = 10
        var layer = new AdaptiveMemoryEfficientKernelLayer(2, 3, 10);

        // A sample input vector
        var input = new double[]{1.2, -0.7};

        // Forward pass
        var output = layer.forward(input);
        System.out.print("Forward output: ");
        for (var val : output) {
            System.out.printf("%.5f ", val);
        }
        System.out.println();

        // Suppose we get an upstream gradient from next layer or final loss
        var gradOutput = new double[]{0.1, -0.05, 0.02};

        // Backward pass
        var gradInput = layer.backward(input, gradOutput);
        System.out.print("Gradient w.r.t. input: ");
        for (var val : gradInput) {
            System.out.printf("%.5f ", val);
        }
        System.out.println();

        // Update parameters (memory-adaptive SGD)
        var learningRate = 0.01;
        layer.update(learningRate);
        System.out.println("Parameters updated with adaptive scaling, base LR = " + learningRate);
    }
}
