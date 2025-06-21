package jcog.tensor.experimental;

import jcog.random.XoRoShiRo128PlusRandom;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/** TODO use Tensors */
public class RecurrentAutoencoder {
    //protected long timeStep;
    protected static final RandomGenerator RANDOM = new XoRoShiRo128PlusRandom();
    protected final int inputSize, hiddenSize, sequenceLength;
    protected final double[] weights, biases, adamM, adamV;
    protected final double learningRate, beta1, beta2, epsilon;

    public RecurrentAutoencoder(int inputSize, int hiddenSize, int sequenceLength, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.sequenceLength = sequenceLength;
        this.learningRate = learningRate;
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;

        int paramSize = hiddenSize * (inputSize + hiddenSize) + inputSize * hiddenSize;
        weights = initializeArray(paramSize, i -> RANDOM.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize)));
        biases = new double[hiddenSize + inputSize];
        adamM = new double[paramSize + biases.length];
        adamV = new double[paramSize + biases.length];
    }

    protected static void clipGradients(double[] dWeights, double[] dBiases, double range) {
        double gradientNorm = Math.sqrt(Arrays.stream(dWeights).map(x -> x * x).sum() + Arrays.stream(dBiases).map(x -> x * x).sum());
        if (gradientNorm > range) {
            double scale = range / gradientNorm;
            for (int i = 0; i < dWeights.length; i++) dWeights[i] *= scale;
            for (int i = 0; i < dBiases.length; i++) dBiases[i] *= scale;
        }
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double dot(double[] a, int aOffset, double[] b, int bOffset, int length) {
        double sum = 0;
        for (int i = 0; i < length; i++) {
            sum += a[aOffset + i] * b[bOffset + i];
        }
        return sum;
    }

    private static double[] initializeArray(int size, java.util.function.IntToDoubleFunction init) {
        double[] array = new double[size];
        for (int i = 0; i < size; i++) {
            array[i] = init.applyAsDouble(i);
        }
        return array;
    }

    public double[] get(double[] input) {
        validateInput(input);
        double[] h = newH();
        double[] output = new double[input.length];

        for (int t = 0; t < sequenceLength; t++) {
            int hOffset = hiddenSize * (t + 1), inOutOffset = inputSize * t;
            computeHiddenLayer(input, h, t, hOffset, inOutOffset);
            computeOutputLayer(h, output, hOffset, inOutOffset);
        }

        return output;
    }

    public void put(double[] input, double[] target) {
        validateInput(input);
        if (target.length != input.length) throw new IllegalArgumentException("Target size mismatch");

        double[] h = newH();
        double[] y = new double[input.length];

        // Forward pass
        for (int t = 0; t < sequenceLength; t++) {
            int hOffset = hiddenSize * (t + 1), inOutOffset = inputSize * t;
            computeHiddenLayer(input, h, t, hOffset, inOutOffset);
            computeOutputLayer(h, y, hOffset, inOutOffset);
        }

        // Backward pass
        double[] dWeights = new double[weights.length];
        double[] dBiases = new double[biases.length];
        double[] dh = new double[h.length];

        for (int t = sequenceLength - 1; t >= 0; t--) {
            int hOffset = hiddenSize * (t + 1), inOutOffset = inputSize * t;
            backpropOutputToHidden(y, target, h, dWeights, dBiases, dh, hOffset, inOutOffset);
            backpropHiddenToHidden(input, h, dWeights, dBiases, dh, t, hOffset, inOutOffset);
        }

        clipGradients(dWeights, dBiases, 1);
        updateParameters(dWeights, dBiases);
    }

    private double[] newH() {
        return new double[hiddenSize * (sequenceLength + 1)];
    }

    protected void computeHiddenLayer(double[] input, double[] h, int t, int hOffset, int inOutOffset) {
        for (int i = 0; i < hiddenSize; i++) {
            double sum = biases[i] + dot(weights, i * inputSize, input, inOutOffset, inputSize)
                    + dot(weights, hiddenSize * inputSize + i * hiddenSize, h, hiddenSize * t, hiddenSize);
            h[hOffset + i] = Math.tanh(sum);
        }
    }

    protected void computeOutputLayer(double[] h, double[] output, int hOffset, int inOutOffset) {
        for (int i = 0; i < inputSize; i++) {
            output[inOutOffset + i] = sigmoid(biases[hiddenSize + i]
                    + dot(weights, hiddenSize * (inputSize + hiddenSize) + i * hiddenSize, h, hOffset, hiddenSize));
        }
    }

    protected void backpropOutputToHidden(double[] y, double[] target, double[] h, double[] dWeights, double[] dBiases, double[] dh, int hOffset, int inOutOffset) {
        for (int i = 0; i < inputSize; i++) {
            double d = (y[inOutOffset + i] - target[inOutOffset + i]) * y[inOutOffset + i] * (1 - y[inOutOffset + i]);
            dBiases[hiddenSize + i] += d;
            int weightsOffset = hiddenSize * (inputSize + hiddenSize) + i * hiddenSize;
            for (int j = 0; j < hiddenSize; j++) {
                dWeights[weightsOffset + j] += d * h[hOffset + j];
                dh[hOffset + j] += weights[weightsOffset + j] * d;
            }
        }
    }

    protected void backpropHiddenToHidden(double[] input, double[] h, double[] dWeights, double[] dBiases, double[] dh, int t, int hOffset, int inOutOffset) {
        for (int i = 0; i < hiddenSize; i++) {
            double d = (1 - h[hOffset + i] * h[hOffset + i]) * dh[hOffset + i];
            dBiases[i] += d;
            for (int j = 0; j < inputSize; j++) {
                dWeights[i * inputSize + j] += d * input[inOutOffset + j];
            }
            for (int j = 0; j < hiddenSize; j++) {
                dWeights[hiddenSize * inputSize + i * hiddenSize + j] += d * h[hiddenSize * t + j];
            }
            if (t > 0) dh[hiddenSize * t + i] += d;
        }
    }

    protected void updateParameters(double[] dWeights, double[] dBiases) {
        //timeStep++;
        //timeStep = 1;
        double alpha = learningRate;// * Math.sqrt(1 - Math.pow(beta2, timeStep)) / (1 - Math.pow(beta1, timeStep));
        updateParametersAdam(weights, dWeights, 0, alpha);
        updateParametersAdam(biases, dBiases, weights.length, alpha);
    }

    protected void updateParametersAdam(double[] params, double[] gradients, int offset, double alpha) {
        for (int i = 0; i < params.length; i++) {
            int idx = offset + i;
            adamM[idx] = beta1 * adamM[idx] + (1 - beta1) * gradients[i];
            adamV[idx] = beta2 * adamV[idx] + (1 - beta2) * gradients[i] * gradients[i];
            params[i] -= alpha * adamM[idx] / (Math.sqrt(adamV[idx]) + epsilon);
        }
    }

    public double loss(double[] input, double[] target) {
        double[] output = get(input);
        double loss = 0;
        for (int i = 0; i < output.length; i++) {
            loss += Math.pow(output[i] - target[i], 2);
        }
        return loss / output.length;
    }

    private void validateInput(double[] input) {
        if (input.length != inputSize * sequenceLength) {
            throw new IllegalArgumentException("Input size mismatch");
        }
    }

    public static class LSTMRecurrentAutoencoder extends RecurrentAutoencoder {
        private final double[] cellState, weightsInputGate, weightsForgetGate, weightsOutputGate;
        private final double[] biasesInputGate, biasesForgetGate, biasesOutputGate;
        private final double[] dWeightsInputGate, dWeightsForgetGate, dWeightsOutputGate;
        private final double[] dBiasesInputGate, dBiasesForgetGate, dBiasesOutputGate;
        private final double[] adamMInputGate, adamVInputGate, adamMForgetGate, adamVForgetGate, adamMOutputGate, adamVOutputGate;
        private final double[] adamMBiasesInputGate, adamVBiasesInputGate, adamMBiasesForgetGate, adamVBiasesForgetGate, adamMBiasesOutputGate, adamVBiasesOutputGate;

        public LSTMRecurrentAutoencoder(int inputSize, int hiddenSize, int sequenceLength, double learningRate) {
            super(inputSize, hiddenSize, sequenceLength, learningRate);

            int gateParamSize = hiddenSize * (inputSize + hiddenSize);
            weightsInputGate = initializeArray(gateParamSize, i -> RANDOM.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize)));
            weightsForgetGate = initializeArray(gateParamSize, i -> RANDOM.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize)));
            weightsOutputGate = initializeArray(gateParamSize, i -> RANDOM.nextGaussian() * Math.sqrt(2.0 / (inputSize + hiddenSize)));

            biasesInputGate = new double[hiddenSize];
            biasesForgetGate = new double[hiddenSize];
            biasesOutputGate = new double[hiddenSize];

            dWeightsInputGate = new double[gateParamSize];
            dWeightsForgetGate = new double[gateParamSize];
            dWeightsOutputGate = new double[gateParamSize];

            dBiasesInputGate = new double[hiddenSize];
            dBiasesForgetGate = new double[hiddenSize];
            dBiasesOutputGate = new double[hiddenSize];

            adamMInputGate = new double[gateParamSize];
            adamVInputGate = new double[gateParamSize];
            adamMForgetGate = new double[gateParamSize];
            adamVForgetGate = new double[gateParamSize];
            adamMOutputGate = new double[gateParamSize];
            adamVOutputGate = new double[gateParamSize];

            adamMBiasesInputGate = new double[hiddenSize];
            adamVBiasesInputGate = new double[hiddenSize];
            adamMBiasesForgetGate = new double[hiddenSize];
            adamVBiasesForgetGate = new double[hiddenSize];
            adamMBiasesOutputGate = new double[hiddenSize];
            adamVBiasesOutputGate = new double[hiddenSize];

            cellState = new double[hiddenSize * (sequenceLength + 1)];
        }

        @Override
        protected void computeHiddenLayer(double[] input, double[] h, int t, int hOffset, int inOutOffset) {
            double[] inputGate = new double[hiddenSize];
            double[] forgetGate = new double[hiddenSize];
            double[] outputGate = new double[hiddenSize];
            double[] cellInput = new double[hiddenSize];

            for (int i = 0; i < hiddenSize; i++) {
                int inputOffset = i * inputSize;
                int hiddenOffset = hiddenSize * inputSize + i * hiddenSize;

                inputGate[i] = sigmoid(dot(weightsInputGate, inputOffset, input, inOutOffset, inputSize)
                        + dot(weightsInputGate, hiddenOffset, h, hiddenSize * t, hiddenSize) + biasesInputGate[i]);

                forgetGate[i] = sigmoid(dot(weightsForgetGate, inputOffset, input, inOutOffset, inputSize)
                        + dot(weightsForgetGate, hiddenOffset, h, hiddenSize * t, hiddenSize) + biasesForgetGate[i]);

                outputGate[i] = sigmoid(dot(weightsOutputGate, inputOffset, input, inOutOffset, inputSize)
                        + dot(weightsOutputGate, hiddenOffset, h, hiddenSize * t, hiddenSize) + biasesOutputGate[i]);

                cellInput[i] = Math.tanh(dot(weights, inputOffset, input, inOutOffset, inputSize)
                        + dot(weights, hiddenOffset, h, hiddenSize * t, hiddenSize) + biases[i]);
            }

            for (int i = 0; i < hiddenSize; i++) {
                int cellOffset = hiddenSize * t + i;
                cellState[hiddenSize * (t + 1) + i] = cellState[cellOffset] * forgetGate[i] + cellInput[i] * inputGate[i];
                h[hOffset + i] = Math.tanh(cellState[hiddenSize * (t + 1) + i]) * outputGate[i];
            }
        }

        @Override
        protected void backpropHiddenToHidden(double[] input, double[] h, double[] dWeights, double[] dBiases, double[] dh, int t, int hOffset, int inOutOffset) {
            double[] inputGate = new double[hiddenSize];
            double[] forgetGate = new double[hiddenSize];
            double[] outputGate = new double[hiddenSize];
            double[] cellInput = new double[hiddenSize];

            // Recompute gates for backpropagation
            for (int i = 0; i < hiddenSize; i++) {
                int inputOffset = i * inputSize;
                int hiddenOffset = hiddenSize * inputSize + i * hiddenSize;

                inputGate[i] = sigmoid(dot(weightsInputGate, inputOffset, input, inOutOffset, inputSize)
                        + dot(weightsInputGate, hiddenOffset, h, hiddenSize * t, hiddenSize) + biasesInputGate[i]);

                forgetGate[i] = sigmoid(dot(weightsForgetGate, inputOffset, input, inOutOffset, inputSize)
                        + dot(weightsForgetGate, hiddenOffset, h, hiddenSize * t, hiddenSize) + biasesForgetGate[i]);

                outputGate[i] = sigmoid(dot(weightsOutputGate, inputOffset, input, inOutOffset, inputSize)
                        + dot(weightsOutputGate, hiddenOffset, h, hiddenSize * t, hiddenSize) + biasesOutputGate[i]);

                cellInput[i] = Math.tanh(dot(weights, inputOffset, input, inOutOffset, inputSize)
                        + dot(weights, hiddenOffset, h, hiddenSize * t, hiddenSize) + biases[i]);
            }

            double[] dInputGate = new double[hiddenSize];
            double[] dForgetGate = new double[hiddenSize];
            double[] dOutputGate = new double[hiddenSize];
            double[] dCellInput = new double[hiddenSize];
            double[] dCellState = new double[hiddenSize];

            for (int i = 0; i < hiddenSize; i++) {
                int cellOffset = hiddenSize * t + i;
                int nextCellOffset = hiddenSize * (t + 1) + i;

                dCellState[i] = dh[nextCellOffset] * outputGate[i] * (1 - Math.tanh(cellState[nextCellOffset]) * Math.tanh(cellState[nextCellOffset]))
                        + dCellState[i] * forgetGate[i];

                dInputGate[i] = dCellState[i] * cellInput[i] * inputGate[i] * (1 - inputGate[i]);
                dForgetGate[i] = dCellState[i] * cellState[cellOffset] * forgetGate[i] * (1 - forgetGate[i]);
                dOutputGate[i] = dh[nextCellOffset] * Math.tanh(cellState[nextCellOffset]) * outputGate[i] * (1 - outputGate[i]);
                dCellInput[i] = dCellState[i] * inputGate[i] * (1 - cellInput[i] * cellInput[i]);

                dBiasesInputGate[i] += dInputGate[i];
                dBiasesForgetGate[i] += dForgetGate[i];
                dBiasesOutputGate[i] += dOutputGate[i];
                dBiases[i] += dCellInput[i];
            }

            for (int i = 0; i < hiddenSize; i++) {
                int inputOffset = i * inputSize;
                int hiddenOffset = hiddenSize * inputSize + i * hiddenSize;

                for (int j = 0; j < inputSize; j++) {
                    dWeightsInputGate[inputOffset + j] += dInputGate[i] * input[inOutOffset + j];
                    dWeightsForgetGate[inputOffset + j] += dForgetGate[i] * input[inOutOffset + j];
                    dWeightsOutputGate[inputOffset + j] += dOutputGate[i] * input[inOutOffset + j];
                    dWeights[hiddenOffset + j] += dCellInput[i] * input[inOutOffset + j];
                }

                for (int j = 0; j < hiddenSize; j++) {
                    dWeightsInputGate[hiddenOffset + j] += dInputGate[i] * h[hiddenSize * t + j];
                    dWeightsForgetGate[hiddenOffset + j] += dForgetGate[i] * h[hiddenSize * t + j];
                    dWeightsOutputGate[hiddenOffset + j] += dOutputGate[i] * h[hiddenSize * t + j];
                    dWeights[hiddenOffset + j] += dCellInput[i] * h[hiddenSize * t + j];
                }

                if (t > 0) {
                    dh[hiddenSize * t + i] += dCellState[i] * forgetGate[i];
                }
            }
        }

        @Override
        protected void updateParameters(double[] dWeights, double[] dBiases) {
            super.updateParameters(dWeights, dBiases);
            updateParametersAdam(weightsInputGate, dWeightsInputGate, adamMInputGate, adamVInputGate);
            updateParametersAdam(weightsForgetGate, dWeightsForgetGate, adamMForgetGate, adamVForgetGate);
            updateParametersAdam(weightsOutputGate, dWeightsOutputGate, adamMOutputGate, adamVOutputGate);
            updateParametersAdam(biasesInputGate, dBiasesInputGate, adamMBiasesInputGate, adamVBiasesInputGate);
            updateParametersAdam(biasesForgetGate, dBiasesForgetGate, adamMBiasesForgetGate, adamVBiasesForgetGate);
            updateParametersAdam(biasesOutputGate, dBiasesOutputGate, adamMBiasesOutputGate, adamVBiasesOutputGate);
        }

        private void updateParametersAdam(double[] params, double[] gradients, double[] adamM, double[] adamV) {
//            timeStep++;
            double alpha = learningRate;// * Math.sqrt(1 - Math.pow(beta2, timeStep)) / (1 - Math.pow(beta1, timeStep));
            for (int i = 0; i < params.length; i++) {
                adamM[i] = beta1 * adamM[i] + (1 - beta1) * gradients[i];
                adamV[i] = beta2 * adamV[i] + (1 - beta2) * gradients[i] * gradients[i];
                params[i] -= alpha * adamM[i] / (Math.sqrt(adamV[i]) + epsilon);
            }
        }
    }

}
