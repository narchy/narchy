package jcog.tensor.experimental;

import jcog.data.list.Lst;
import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

import static jcog.tensor.Tensor.row;

/**
 * Equilibrium Propagation (EqProp) implementation using the provided Tensor API.
 * This implementation includes input handling, a detailed energy function, and iterative equilibrium settling.
 */
public class EqProp {

    /**
     * Represents the entire EqProp neural network.
     */
    public static class EqPropNetwork {
        private final List<UnaryOperator<Tensor>> layers;
        private final int inputSize;
        private final int hiddenSize;
        private final int outputSize;
        private final double beta; // Nudging factor
        private final int maxIterations; // Maximum iterations for settling
        private final double tolerance; // Convergence tolerance

        private Tensor inputTensor;
        private Tensor targetTensor;

        /**
         * Initializes the EqProp network with specified parameters.
         *
         * @param inputSize    Number of input features
         * @param hiddenSize   Number of hidden neurons
         * @param outputSize   Number of output neurons
         * @param beta         Nudging factor for the nudged phase
         * @param maxIterations Maximum iterations for equilibrium settling
         * @param tolerance    Tolerance for convergence
         */
        public EqPropNetwork(int inputSize, int hiddenSize, int outputSize, double beta, int maxIterations, double tolerance) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.beta = beta;
            this.maxIterations = maxIterations;
            this.tolerance = tolerance;

            layers = new Lst<>();
            layers.add(new Models.Linear(inputSize, hiddenSize, Tensor::tanh, true));
            layers.add(new Models.Linear(hiddenSize, outputSize, Tensor.SIGMOID, true));
        }

        /**
         * Trains the network on a single example using EqProp.
         *
         * @param input  Input vector
         * @param target Target output vector
         */
        public synchronized void put(double[] input, double[] target) {
            // Input Handling: Set the input tensor
            inputTensor = row(input);
            targetTensor = row(target);

            // Free Phase: Let the network settle into equilibrium without nudging
            var freePhaseActivations = settleEquilibrium(inputTensor, null);

            // Nudged Phase: Introduce target into the output layer and settle again
            var nudgedPhaseActivations = settleEquilibrium(inputTensor, targetTensor);

            // Compute gradients based on the difference in activations
            updateWeights(freePhaseActivations, nudgedPhaseActivations);
        }

        /**
         * Settles the network into equilibrium through iterative activation updates.
         *
         * @param input  Input tensor
         * @param target Target tensor for nudging (nullable)
         * @return List of activations for each layer after settling
         */
        private synchronized List<Tensor> settleEquilibrium(Tensor input, @Nullable Tensor target) {
            List<Tensor> activations = new Lst<>();
            activations.add(input); // Activation of layer 0 (input)

            // Initialize activations for hidden and output layers
            Tensor current = input;
            var L = layers.size();
            for (int i = 0; i < L; i++) {
                UnaryOperator<Tensor> layer = layers.get(i);
                current = layer.apply(current);
                activations.add(current); // Activation of layer i+1
            }

            // Iteratively update activations to reach equilibrium
            for (int iteration = 0; iteration < maxIterations; iteration++) {
                double maxChange = 0.0;

                // Iterate through each layer to update activations
                for (int i = 0; i < L; i++) {
                    UnaryOperator<Tensor> layer = layers.get(i);
                    Tensor previous = activations.get(i); // Activation from previous layer
                    Tensor updated = layer.apply(previous); // New activation for current layer

                    // If nudged and it's the output layer, apply nudging
                    if (target != null && i == L - 1) {
                        updated = nudgedOutput(updated, target);
                    }

                    // Retrieve the old activation of the current layer
                    Tensor oldActivation = activations.get(i + 1);

                    // Calculate the change for convergence (same shape tensors)
                    double change = oldActivation.sub(updated).maxAbs();
                    maxChange = Math.max(maxChange, change);

                    // Update the activation of the current layer
                    activations.set(i + 1, updated);
                }

                // Check for convergence
                if (maxChange < tolerance) {
                    break;
                }
            }

            return activations;
        }

        /**
         * Applies nudging to the output tensor based on the target.
         *
         * @param output Current output tensor
         * @param target Target tensor
         * @return Nudged output tensor
         */
        private Tensor nudgedOutput(Tensor output, Tensor target) {
            // Apply nudging: y = y - beta * (y - t)
            return output.sub(output.sub(target).mul(beta));
        }

        /**
         * Computes the energy of the network based on activations.
         *
         * @param activations Activations of all layers
         * @param target      Target tensor (nullable)
         * @return Energy value
         */
        private double computeEnergy(List<Tensor> activations, @Nullable Tensor target) {
            double energy = 0.0;

            // Define energy as sum of squared activations minus interactions
            // E = 0.5 * ||h||^2 + 0.5 * ||y - target||^2 - h * W * y
            // Where h: hidden activations, y: output activations, W: weights

            // Hidden activations
            Tensor hidden = activations.get(1); // Assuming layer 0 is input, 1 is hidden
            energy += 0.5 * hidden.dot(hidden).scalar();

            // Output activations
            Tensor output = activations.get(2); // Assuming layer 2 is output
            if (target != null) {
                Tensor diff = output.sub(target);
                energy += 0.5 * diff.dot(diff).scalar();
            } else {
                energy += 0.5 * output.dot(output).scalar();
            }

            // Interaction term: h * W * y
            Models.Linear outputLayer = (Models.Linear) layers.get(1);
            Tensor weights = outputLayer.weight; // W hidden->output
            var interaction = hidden.matmul(weights).dot(output).scalar();
            energy -= interaction;

            return energy;
        }

        /**
         * Updates the network's weights based on the difference between free and nudged phases.
         *
         * @param freePhaseActivations   Activations during the free phase
         * @param nudgedPhaseActivations Activations during the nudged phase
         */
        private void updateWeights(List<Tensor> freePhaseActivations, List<Tensor> nudgedPhaseActivations) {
            // Compute the difference in activations between nudged and free phases
            Tensor deltaOutput = nudgedPhaseActivations.get(2).sub(freePhaseActivations.get(2)).mul(1.0 / beta);

            {
                // Update weights between hidden and output
                Models.Linear outputLayer = (Models.Linear) layers.get(1);
                Tensor hiddenActivations = freePhaseActivations.get(1);
                Tensor gradientOutput = hiddenActivations.transposeMatmul(deltaOutput);
                outputLayer.weight.setGrad(gradientOutput);

                // Update biases for the output layer
                if (outputLayer.bias != null) {
                    Tensor biasGradient = deltaOutput.sum(true);
                    outputLayer.bias.setGrad(biasGradient);
                }
            }

            {
                // Similarly, update weights between input and hidden
                Models.Linear hiddenLayer = (Models.Linear) layers.get(0);
                Tensor inputActivations = freePhaseActivations.get(0);
                Tensor deltaHidden = nudgedPhaseActivations.get(1).sub(freePhaseActivations.get(1)).mul(1.0 / beta);
                Tensor gradientHidden = inputActivations.transposeMatmul(deltaHidden);
                hiddenLayer.weight.setGrad(gradientHidden);

                // Update biases for the hidden layer
                if (hiddenLayer.bias != null) {
                    Tensor biasGradientHidden = deltaHidden.sum(true);
                    hiddenLayer.bias.setGrad(biasGradientHidden);
                }
            }
        }

        /**
         * Predicts the output for a given input without updating the network.
         *
         * @param input Input vector
         * @return Output vector
         */
        public double[] get(double[] input) {
            // Input Handling: Set the input tensor
            inputTensor = row(input);

            // Forward pass without nudging
            var activations = settleEquilibrium(inputTensor, null);
            var output = activations.get(2); // Assuming layer 2 is output

            // Extract the output values
            int n = output.cols();
            double[] outputValues = new double[n];
            for (int i = 0; i < n; i++)
                outputValues[i] = output.data(i);

            return outputValues;
        }

        /**
         * Applies the optimizer to update weights based on accumulated gradients.
         *
         * @param o Optimizer instance
         */
        public void optimize(Tensor.Optimizer o) {
            List<Tensor> p = new Lst<>();
            for (var l : layers) {
                p.add(((Models.Linear) l).weight);
                if (((Models.Linear) l).bias != null)
                    p.add(((Models.Linear) l).bias);
            }
            o.run(p.stream());
        }
    }

    public static void main(String[] args) {
        // Define network parameters
        int inputSize = 2;
        int hiddenSize = 3;
        int outputSize = 1;
        double beta = 0.25f;
        int maxIterations = 1000;
        int epochs = 1000;
        double tolerance = 1e-5;
        double learningRate = 0.001;

        EqPropNetwork eqProp = new EqPropNetwork(inputSize, hiddenSize, outputSize, beta, maxIterations, tolerance);

        // Example training data: XOR problem
        double[][] inputs = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        double[][] targets = {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };

        // Initialize optimizer
        var optimizer =
                new Optimizers.SGD(() -> (float) learningRate).get();
                //new Optimizers.ADAM(() -> (float) learningRate/10).get();

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                eqProp.put(inputs[i], targets[i]);
                eqProp.optimize(optimizer);
            }

            // Optionally, print progress
            if ((epoch + 1) % 100 == 0) {
                System.out.println("Epoch " + (epoch + 1));
                for (int i = 0; i < inputs.length; i++) {
                    double[] output = eqProp.get(inputs[i]);
                    System.out.printf("Input: %s, Predicted: %.4f, Target: %.1f%n",
                            Arrays.toString(inputs[i]), output[0], targets[i][0]);
                }
                System.out.println();
            }
        }

        // Final predictions
        System.out.println("Final Predictions:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = eqProp.get(inputs[i]);
            System.out.printf("Input: %s, Predicted: %.4f, Target: %.1f%n",
                    Arrays.toString(inputs[i]), output[0], targets[i][0]);
        }
    }
}
