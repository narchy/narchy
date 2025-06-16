package jcog.tensor.experimental;

import jcog.Util;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.Random;
import java.util.function.UnaryOperator;

import static jcog.tensor.Tensor.array;

/**
 * HyperNeuralTuringMachine (HNTM) with:
 *  - Complex-Valued Memory (quantum-inspired): Arjovsky et al. (2016)
 *  - Hypernetwork-synthesized operations: Ha et al. (ICLR 2017)
 *  - Evolutionary + Gradient Hybrid (strategies):
 *      1) STRAIGHT_THROUGH
 *      2) CONTINUOUS_APPROX
 *      3) EVOLUTIONARY
 *      4) FULL_BACKPROP
 *  - Optional chain rule backpropagation to the input tensor.
 *  - Inspired by Neural Turing Machines (Graves et al., 2014).
 */
public final class HyperNeuralTuringMachine implements UnaryOperator<Tensor> {

    @Override
    public Tensor apply(Tensor x) {
        var Y = forward(x.array());
        filter(Y, true);

        var y = new Tensor(Y, true);
        if (y.hasGrad()) {
            y.op = new Tensor.TensorOp(x) {
                @Override public void backward(SimpleMatrix grad, SimpleMatrix[] gradParent) {
                    var parentGrad = gradParent[0] != null;
                    var gg = array(grad).clone();

                    gradClip(gg);

                    var g = HyperNeuralTuringMachine.this.backward(gg, parentGrad);
                    if (parentGrad) {
                        filter(g, false);
                        System.arraycopy(g, 0, array(gradParent[0]), 0, g.length);
                    }
                }

                private void gradClip(double[] g) {
                    for (int i = 0; i < g.length; i++)
                        g[i] = Util.clamp(g[i], -1, +1);
                    Util.normalizeCartesianIfMagGt1(g, g.length, Double.MIN_NORMAL);
                }
            };
        }
        return y;
    }

    private void filter(double[] Y, boolean rng) {
        for (int i = 0; i < Y.length; i++)
            if (!Double.isFinite(Y[i]))
                Y[i] = rng ? random.nextDouble(-1, +1) : 0;
    }

    /** Training strategy options. */
    public enum Strategy {
        STRAIGHT_THROUGH,
        CONTINUOUS_APPROX,
        EVOLUTIONARY,
        FULL_BACKPROP
    }

    // ------------------------
    // Internal Data Structures
    // ------------------------

    /** Complex cell storing (real, imag). */
    private static final class ComplexCell {
        double real;
        double imag;

        ComplexCell(double r, double i) {
            this.real = r;
            this.imag = i;
        }

        ComplexCell copy() {
            return new ComplexCell(real, imag);
        }
    }

    /**
     * Tiny HyperNetwork that synthesizes an operation for each cell index.
     * (In a real system, this could be an MLP, LSTM, etc.)
     */
    private static final class HyperNetwork {
        private final double[] weights;
        private final Random random;

        HyperNetwork(int numParams, Random random) {
            this.weights = new double[numParams];
            this.random = random;
            // Initialize weights (e.g., Xavier initialization could be used here)
            Arrays.fill(this.weights, 0.5);
        }

        /**
         * Applies a synthesized operation to two complex cells.
         *
         * @param a   ComplexCell A
         * @param b   ComplexCell B
         * @param idx Cell index (for weight selection)
         * @return    Synthesized ComplexCell output
         */
        ComplexCell apply(ComplexCell a, ComplexCell b, int idx) {
            // Weighted sum minus a cross-term for flavor
            double w = weights[idx % weights.length];
            double realOut = w * (a.real + b.real) - (1.0 - w) * (a.imag * b.imag);
            double imagOut = w * (a.imag + b.imag) + (1.0 - w) * (a.real * b.real);
            return new ComplexCell(realOut, imagOut);
        }

        /**
         * Updates the hypernetwork parameters based on the accumulated gradients.
         *
         * @param gradWeights Accumulated gradients for each weight
         * @param lr          Learning rate
         * @param strategy    Training strategy used
         */
        void update(double[] gradWeights, double lr, Strategy strategy) {
            switch (strategy) {
                case EVOLUTIONARY:
                    // Random mutation: add Gaussian noise scaled by learning rate
                    for (int i = 0; i < weights.length; i++) {
                        weights[i] += random.nextGaussian() * lr * 0.1;
                    }
                    break;
                case STRAIGHT_THROUGH:
                case CONTINUOUS_APPROX:
                case FULL_BACKPROP:
                    // Gradient-based update: weight -= lr * grad (assuming minimization)
                    for (int i = 0; i < weights.length; i++) {
                        weights[i] -= lr * gradWeights[i];
                    }
                    break;
            }
        }
    }

    // ------------------------
    // Fields
    // ------------------------

    private final int memorySize;
    private final int inputSize;
    private final int outputSize;
    private final int steps;

    private final Strategy strategy;      // One of the four
    private final double learningRate;
    private final Random random;

    // Complex memory
    private final ComplexCell[] memory;

    // For FULL_BACKPROP, store T+1 states
    private final ComplexCell[][] memoryHistory; // memoryHistory[t][i] => memory at time t, cell i
    private final ComplexCell[][] dMemory;       // partial derivatives wrt memory

    // The hypernetwork
    private final HyperNetwork hyperNet;
    private final double[] gradWeights;

    public HyperNeuralTuringMachine(
            int memorySize,
            int inputSize,
            int outputSize,
            int steps,
            Strategy strategy,
            double learningRate, int hypernetParams) {

        this.memorySize = memorySize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.steps = steps;
        this.strategy = strategy;
        this.learningRate = learningRate;
        this.random = new XoRoShiRo128PlusRandom();

        this.memory = new ComplexCell[memorySize];
        for (int i = 0; i < memorySize; i++)
            memory[i] = new ComplexCell(0.0, 0.0);


        // For FULL_BACKPROP, allocate history
        this.memoryHistory = new ComplexCell[steps + 1][memorySize];
        this.dMemory = new ComplexCell[steps + 1][memorySize];
        for (int t = 0; t < steps + 1; t++) {
            for (int i = 0; i < memorySize; i++) {
                memoryHistory[t][i] = new ComplexCell(0.0, 0.0);
                dMemory[t][i] = new ComplexCell(0.0, 0.0);
            }
        }

        // Initialize hypernetwork with, say, 8 parameters
        this.hyperNet = new HyperNetwork(hypernetParams, this.random);
        gradWeights = new double[hyperNet.weights.length];
    }

    // ------------------------
    // Forward
    // ------------------------

    /**
     * Forward pass:
     *  1) Initialize memory to 0
     *  2) Copy input (real-part)
     *  3) T steps of hypernetwork transform
     *  4) Return the magnitude of first outputSize cells
     *  5) Store memory states if FULL_BACKPROP
     *
     * @param input Real-valued input array
     * @return      Real-valued output array
     */
    public double[] forward(double[] input) {
        synchronized(this) {
            // Reset memory
            for (int i = 0; i < memorySize; i++) {
                memory[i].real = 0.0;
                memory[i].imag = 0.0;
            }
            // Copy input into real-part
            for (int i = 0; i < Math.min(inputSize, memorySize); i++) {
                memory[i].real = input[i];
            }

            // Store step 0
            for (int i = 0; i < memorySize; i++) {
                memoryHistory[0][i].real = memory[i].real;
                memoryHistory[0][i].imag = memory[i].imag;
            }

            // T steps
            for (int t = 0; t < steps; t++) {
                ComplexCell[] oldMem = copyMemory(memory);

                for (int i = 0; i < memorySize; i++) {
                    ComplexCell a = oldMem[i];
                    // Neighbor addressing
                    ComplexCell b = oldMem[(i + 1) % memorySize];
                    // Hypernetwork-synthesized operation
                    ComplexCell out = hyperNet.apply(a, b, i);
                    memory[i].real = out.real;
                    memory[i].imag = out.imag;
                }

                // Store state if FULL_BACKPROP
                if (strategy == Strategy.FULL_BACKPROP) {
                    for (int i = 0; i < memorySize; i++) {
                        memoryHistory[t + 1][i].real = memory[i].real;
                        memoryHistory[t + 1][i].imag = memory[i].imag;
                    }
                }
            }

            // Output: magnitude of first outputSize cells
            double[] output = new double[outputSize];
            for (int i = 0; i < outputSize; i++) {
                if (i < memorySize) {
                    double r = memory[i].real;
                    double im = memory[i].imag;
                    output[i] = Math.sqrt(r * r + im * im);
                }
            }
            return output;
        }
    }


    /**
     * Backward pass supporting four strategies.
     * Optionally returns gradient wrt the input tensor.
     *
     * @param gradOutput        Gradient wrt the output of size=outputSize
     * @param computeInputGrad  Whether to compute backprop gradient wrt input
     * @return                  If computeInputGrad=true, returns a double[] of length inputSize;
     *                          otherwise null
     */
    public double[] backward(double[] gradOutput, boolean computeInputGrad) {
        if (gradOutput.length!=outputSize)
            throw new IllegalArgumentException();

        synchronized(this) {
            Arrays.fill(gradWeights, 0); // Initialize gradient accumulators for hypernetwork weights

            double[] w = null;
            switch (strategy) {
                case STRAIGHT_THROUGH:
                    w = straightThroughUpdate(gradOutput);
                    break;
                case CONTINUOUS_APPROX:
                    w = continuousApproxUpdate(gradOutput);
                    break;
                case EVOLUTIONARY:
                    evolutionaryUpdate();
                    break;
                case FULL_BACKPROP:
                    w = fullBackpropUpdate(gradOutput);
                    break;
            }

            if (w!=null)
                hyperNet.update(w, learningRate, strategy);

            return !computeInputGrad ? null : computeInputGradient();
        }
    }

    // ------------------------
    // Strategies
    // ------------------------

    /**
     * STRAIGHT_THROUGH: Ignores true derivative, pushes a naive gradient.
     *
     * @param gradOutput Gradient wrt output
     * @return           Gradient wrt hypernetwork weights
     */
    private double[] straightThroughUpdate(double[] gradOutput) {
        // Summation as a naive "fake gradient"
        double gradSum = Util.sum(gradOutput);

        // Distribute the gradient sum equally to all weights
        double[] gradWeights = new double[hyperNet.weights.length];
        Arrays.fill(gradWeights, gradSum / hyperNet.weights.length);
        return gradWeights;
    }

    /**
     * CONTINUOUS_APPROX: One-step partial derivative approximation.
     *
     * @param gradOutput Gradient wrt output
     * @return           Gradient wrt hypernetwork weights
     */
    private double[] continuousApproxUpdate(double[] gradOutput) {
        // Summation as a naive "fake gradient"
        double gradSum = Util.sum(gradOutput);

        // Treat gradSum as the gradient for all weights
        double[] gradWeights = new double[hyperNet.weights.length];
        Arrays.fill(gradWeights, gradSum / hyperNet.weights.length);
        return gradWeights;
    }

    /**
     * EVOLUTIONARY: Randomly mutate the hypernetwork weights.
     * Does not use gradient information.
     */
    private void evolutionaryUpdate() {
        // Evolutionary strategy handled inside hyperNet.update with strategy EVOLUTIONARY
        // No gradient accumulation needed
    }

    /**
     * FULL_BACKPROP: Proper chain rule through all steps and cells.
     *
     * @param gradOutput Gradient wrt output
     * @return           Gradient wrt hypernetwork weights
     */
    private double[] fullBackpropUpdate(double[] gradOutput) {
        // 1. Initialize dMemory to zero
        for (int t = 0; t < steps + 1; t++) {
            var dmt = dMemory[t];
            for (int i = 0; i < memorySize; i++)
                dmt[i].real = dmt[i].imag = 0;
        }

        // 2. Distribute gradOutput to final memory cells
        for (int i = 0; i < outputSize; i++) {
            if (i < memorySize) {
                double g = gradOutput[i];
                if (g!=0) {
                    // Partial derivatives wrt memory[t][i].real and imag
                    var cell = dMemory[steps][i];
                    var h = memoryHistory[steps][i];
                    var d = Math.sqrt(Util.sqr(h.real) + Util.sqr(h.imag));
                    cell.real += h.real > 0 ? g * (h.real / d) : 0;
                    cell.imag += h.imag > 0 ? g * (h.imag / d) : 0;
                }
            }
        }

        // 3. Initialize gradient accumulator for hypernetwork weights
        double[] gradWeights = new double[hyperNet.weights.length];
        Arrays.fill(gradWeights, 0.0);

        // 4. Backpropagate through time
        for (int t = steps; t >= 1; t--) {
            for (int i = 0; i < memorySize; i++) {
                // Get a and b from memoryHistory at step t-1
                var hTmin1 = memoryHistory[t - 1];
                ComplexCell a = hTmin1[i];
                ComplexCell b = hTmin1[(i + 1) % memorySize];
                double w = hyperNet.weights[i % hyperNet.weights.length];

                // Compute partial derivatives
                // From the forward pass:
                // realOut = w*(a.real + b.real) - (1.0 - w)*(a.imag * b.imag)
                // imagOut = w*(a.imag + b.imag) + (1.0 - w)*(a.real * b.real)

                // Partial derivatives
                double dRealOut_dAReal = w;
                double dRealOut_dAImag = - (1.0 - w) * b.imag;
                double dRealOut_dBReal = w;
                double dRealOut_dBImag = - (1.0 - w) * a.imag;

                double dImagOut_dAReal = (1.0 - w) * b.real;
                double dImagOut_dAImag = w;
                double dImagOut_dBReal = (1.0 - w) * a.real;
                double dImagOut_dBImag = w;

                // Gradients from current memory cell
                double dRealOut = dMemory[t][i].real;
                double dImagOut = dMemory[t][i].imag;

                // Accumulate gradients wrt a and b
                dMemory[t - 1][i].real += dRealOut_dAReal * dRealOut + dImagOut_dAReal * dImagOut;
                dMemory[t - 1][i].imag += dRealOut_dAImag * dRealOut + dImagOut_dAImag * dImagOut;

                dMemory[t - 1][(i + 1) % memorySize].real += dRealOut_dBReal * dRealOut + dImagOut_dBReal * dImagOut;
                dMemory[t - 1][(i + 1) % memorySize].imag += dRealOut_dBImag * dRealOut + dImagOut_dBImag * dImagOut;

                // Gradients wrt hypernetwork weight w
                double dRealOut_dw = (a.real + b.real) + (a.imag * b.imag);
                double dImagOut_dw = (a.imag + b.imag) - (a.real * b.real);

                // Accumulate gradients for hypernetwork weights
                int wIdx = i % hyperNet.weights.length;
                gradWeights[wIdx] += dRealOut_dw * dRealOut + dImagOut_dw * dImagOut;
            }
        }

        return gradWeights;
    }

    // ------------------------
    // Gradient to Input
    // ------------------------

    /**
     * Computes the gradient with respect to the input tensor based on dMemory.
     *
     * @return Gradient wrt input as a double array
     */
    private double[] computeInputGradient() {
        double[] gradInput = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            if (i < memorySize) {
                // Considering only real part contribution
                gradInput[i] += dMemory[0][i].real;
                // Optionally, include imaginary part if input affects it
                // For this illustrative example, we consider only real parts
            }
        }
        return gradInput;
    }

    // ------------------------
    // Utility
    // ------------------------

    private static ComplexCell[] copyMemory(ComplexCell[] src) {
        ComplexCell[] ret = new ComplexCell[src.length];
        for (int i = 0; i < src.length; i++)
            ret[i] = src[i].copy();
        return ret;
    }

    // ------------------------
    // Example Usage
    // ------------------------

    public static void main(String[] args) {
        // Construct a HNTM with memorySize=16, inputSize=4, outputSize=2, steps=3
        // Using FULL_BACKPROP for demonstration, LR=0.01
        HyperNeuralTuringMachine hntm = new HyperNeuralTuringMachine(
                16, 4, 2, 3,
                Strategy.FULL_BACKPROP,
                0.01, 8
        );

        double[] input = {0.1, 0.7, 0.3, 0.9};

        // Forward
        double[] out1 = hntm.forward(input);
        System.out.println("Output (first pass): " + Arrays.toString(out1));

        // Some hypothetical gradient from the next layer
        double[] gradOut = {0.5, -0.2};

        // Backward, requesting gradient wrt input
        double[] gradInput = hntm.backward(gradOut, /*computeInputGrad=*/true);
        System.out.println("Gradient wrt input  : " + Arrays.toString(gradInput));

        // Forward again to see updated outputs
        double[] out2 = hntm.forward(input);
        System.out.println("Output (second pass): " + Arrays.toString(out2));
    }
}
