package jcog.nndepr;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.deprtensor.Tens0r;
import jcog.util.ArrayUtil;

import javax.swing.*;
import java.awt.*;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static jcog.tensor.Tensor.zero;


public class CellularRNN {

    public static void main(String[] args) {
        int inputSize = 5;
        int outputSize = 3;
        int hiddenWidth = 5;
        int hiddenHeight = 5;
        CellularRNN n = new CellularRNN(inputSize, outputSize, hiddenWidth, hiddenHeight);
        Optimizer optimizer =
            //new SGDOptimizer();
            new AdamOptimizer(hiddenHeight, hiddenWidth);

        var v = new NetworkVisualizer(n);
        v.setSize(1000, 600);

        double[][] in = {
                {1,0,0,0,0},
                {0,1,0,0,0},
                {0,0,1,0,0},
                {0,0,0,1,0},
                {0,0,0,0,1}
        };
        double[][] out = {
                {0,0,1},
                {0,1,0},
                {0,1,1},
                {1,0,0},
                {1,0,1}
        };
        int i = 0;
        while (true) {
            int p = (i++) % in.length;
            n.train(optimizer, in[p], out[p]);
            if (i%4843==0) {
                System.out.println("MSE: " + n.calculateMeanSquaredError());
                v.render();
                Util.sleepMS(2);
            }
        }
    }

    static final double LEARNING_RATE = 1E-5;

    int recurrent;

    int inputSize, outputSize;
    int hiddenWidth, hiddenHeight;

    double[] inputs, outputs, target;
    double[][] hidden, hiddenNext;
    byte  [][] activationFn; // 0 - tanh, 1 - sigmoid, 2 - relu, 3 = linear
    double[][] delta, deltaNext;
    double[][][] weights, biases;
    double[][][] wGrad, bGrad;

    private byte outputActivation = 2;
    private byte defultActivation = 2;

    public CellularRNN(int inputSize, int outputSize, int hiddenWidth, int hiddenHeight) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenWidth = hiddenWidth;
        this.hiddenHeight = hiddenHeight;

        this.recurrent = Math.max(hiddenHeight, hiddenWidth);

        inputs = new double[inputSize];
        outputs = new double[outputSize];
        hidden = new double[hiddenHeight][hiddenWidth];
        hiddenNext = hidden.clone();
        weights = new double[hiddenHeight][hiddenWidth][4]; // North, South, East, West
        biases = new double[hiddenHeight][hiddenWidth][4];
        activationFn = new byte[hiddenHeight][hiddenWidth];

        wGrad = new double[hiddenHeight][hiddenWidth][4];
        bGrad = new double[hiddenHeight][hiddenWidth][4];
        delta = new double[hiddenHeight][hiddenWidth];
        deltaNext = delta.clone();

        initializeWeightsAndBiases();
        initializeActivationFunctions();
    }

    private void initializeWeightsAndBiases() {
        Random rand = ThreadLocalRandom.current();
        for (int i = 0; i < hiddenHeight; i++) {
            for (int j = 0; j < hiddenWidth; j++) {
                for (int k = 0; k < 4; k++) {
                    weights[i][j][k] = 0.1 * (rand.nextDouble() * 2 - 1); // Random [-1, 1]
                    biases[i][j][k] = 0; //rand.nextDouble() * 2 - 1;
                }
            }
        }
    }

    private void initializeActivationFunctions() {
        Random rand = ThreadLocalRandom.current();
        for (int i = 0; i < hiddenHeight; i++) {
            for (int j = 0; j < hiddenWidth; j++) {
                byte a;
                if (isOutput(i, j))
                    a = outputActivation;
                else if (isInput(i, j))
                    a = 3;
                else {
                    a = defultActivation;
                }
                        //2;
                        //(byte) rand.nextInt(3); // 0 - tanh, 1 - sigmoid, 2 - relu

                activationFn[i][j] = a;
            }
        }
    }

    public double[] get(double[] input) {
        setInput(input);
        forward();
        return outputs;
    }
    public void put(double[] input, double[] target) {
        setInput(input);
        this.target = target.clone();
    }

    private void setInput(double[] input) {
        System.arraycopy(input, 0, inputs, 0, inputSize);
    }

    public void train(Optimizer optimizer, double[] input, double[] target) {
        clearGrad();
        put(input, target);

        forward();


        backward(optimizer);
    }
    private double calculateMeanSquaredError() {
        double sumSquaredError = 0;
        for (int i = 0; i < outputSize; i++) {
            double error = target[i] - outputs[i];
            sumSquaredError += error * error;
        }
        return sumSquaredError / outputSize;
    }
    private double output(int i, int j, int di, int dj) {
        int ni = wrap(i + di, hiddenHeight);
        int nj = wrap(j + dj, hiddenWidth);
        return hidden[ni][nj];
    }

    private double weight(int i, int j, int di, int dj) {
        return weights[wrap(i + di, hiddenHeight)]
                      [wrap(j + dj, hiddenWidth)]
                      [dir(di, dj)];
    }

    private double delta(int i, int j) {
        return delta[wrap(i, hiddenHeight)][wrap(j, hiddenWidth)];
    }

    private static int wrap(int index, int limit) {
        return (index + limit) % limit;
    }

    private int dir(int di, int dj) {
        if (di == -1 && dj == 0) return 0; // North
        if (di == 1 && dj == 0) return 1; // South
        if (di == 0 && dj == -1) return 2; // East
        if (di == 0 && dj == 1) return 3; // West
        throw new IllegalArgumentException("Invalid direction: " + di + ", " + dj);
    }

    private void applyIO() {
        zero(hidden);
        System.arraycopy(inputs, 0, hidden[0], 0, inputs.length);
        ArrayUtil.copy(hidden, hiddenNext);
    }

    private boolean isInput(int i, int j) {
        return i == 0 && j < inputSize;
    }
    private boolean isOutput(int i, int j) {
        return i == hiddenHeight-1 && j < outputSize;
    }

    protected void forward() {

        applyIO();

        for (int iter = 0; iter < recurrent; iter++) {
            for (int i = 0; i < hiddenHeight; i++) {
                for (int j = 0; j < hiddenWidth; j++) {
                    if (!isInput(i, j))
                        forward(i, j);
                }
            }
            ArrayUtil.copy(hiddenNext, hidden);
        }

        System.arraycopy(hidden[hiddenHeight - 1], 0, outputs, 0, outputSize);
    }

    private void forward(int i, int j) {
        double sum = 0;
        sum += output(i, j, 0, -1) * weight(i, j, 0, -1); // North
        sum += output(i, j, 0, +1) * weight(i, j, 0, 1); // South
        sum += output(i, j, -1, 0) * weight(i, j, -1, 0); // East
        sum += output(i, j, +1, 0) * weight(i, j, 1, 0); // West
        double[] b = biases[i][j];
        sum += b[0] + b[1] + b[2] + b[3];
        hiddenNext[i][j] = activationFn(activationFn[i][j], sum);
    }

    private void _backward(Optimizer optimizer) {
        // Backpropagate errors
        for (int iter = 0; iter < 1; iter++) {
            for (int i = 0; i < hiddenHeight; i++) {
                for (int j = 0; j < hiddenWidth; j++) {
                    if (isInput(i, j) || isOutput(i, j)) continue;
                    //if (isOutput(i, j)) continue;
                    double sum = 0.0;
                    sum += delta(i + 1, j) * weight(i + 1, j, 0, 1); // South
                    sum += delta(i - 1, j) * weight(i - 1, j, 0, -1); // North
                    sum += delta(i, j - 1) * weight(i, j - 1, -1, 0); // East
                    sum += delta(i, j + 1) * weight(i, j + 1, 1, 0); // West
                    deltaNext[i][j] = sum * derivative(activationFn[i][j], hidden[i][j]);
                }
            }
            ArrayUtil.copy(deltaNext, delta);
        }

        gradAccum();

        Tens0r.clamp3(wGrad, wGrad, -1, +1);
        Tens0r.clamp3(bGrad, bGrad, -1, +1);
        optimizer.updateWeightsAndBiases(weights, biases, wGrad, bGrad, LEARNING_RATE);

        double weightClip = 16;
        Tens0r.clamp3(weights, weights, -weightClip, +weightClip);
        Tens0r.clamp3(biases, biases, -weightClip, +weightClip);
    }

    private void gradAccum() {
        for (int i = 0; i < hiddenHeight; i++) {
            for (int j = 0; j < hiddenWidth; j++) {
                double d = delta[i][j];
                double[] wg = wGrad[i][j], bg = bGrad[i][j];
                wg[0] += d * output(i, j, 0, -1);
                wg[1] += d * output(i, j, 0, 1);
                wg[2] += d * output(i, j, -1, 0);
                wg[3] += d * output(i, j, 1, 0);
                bg[0] += d;
                bg[1] += d;
                bg[2] += d;
                bg[3] += d;
            }
        }
    }

    private double activationSum(int i, int j) {
        double sum = 0;
        double[] w = weights[i][j];
        sum += output(i, j, 0, -1) * w[0]; // North
        sum += output(i, j, 0, +1) * w[1]; // South
        sum += output(i, j, -1, 0) * w[2]; // East
        sum += output(i, j, +1, 0) * w[3]; // West
        double[] b = biases[i][j];
        sum += b[0] + b[1] + b[2] + b[3];
        return sum;
    }

    private double activationFn(byte activationFunction, double x) {
        return switch (activationFunction) {
            case 0 -> Math.tanh(x);
            case 1 -> Util.sigmoid(x);
            case 2 -> Math.max(0, x);
            case 3 -> x;
            default -> throw new IllegalArgumentException("Invalid activation function: " + activationFunction);
        };
    }


    private void clearGrad() {
        Tensor.zero(wGrad);
        Tensor.zero(bGrad);
    }

    private void backward(Optimizer optimizer) {

        zero(delta);
        zero(deltaNext);

        for (int i = 0; i < outputSize; i++) {
            double dx = target[i] - outputs[i];
            delta[hiddenHeight - 1][i] =
                deltaNext[hiddenHeight - 1][i] =
                    dx * derivative(activationFn[hiddenHeight-1][i], outputs[i]);
        }

        _backward(optimizer);
    }



    private double derivative(byte activationFunction, double value) {
        return switch (activationFunction) {
            case 0 -> 1 - value * value; // tanh
            case 1 -> value * (1 - value); // sigmoid
            case 2 -> value > 0 ? 1 : 0; // relu
            case 3 -> 1; //identity
            default -> throw new IllegalArgumentException("Invalid activation function: " + activationFunction);
        };
    }

    private interface Optimizer {
        void updateWeightsAndBiases(double[][][] weights, double[][][] biases,
                                    double[][][] weightGradients, double[][][] biasGradients,
                                    double learningRate);
    }

    private static class SGDOptimizer implements Optimizer {
        public void updateWeightsAndBiases(double[][][] weights, double[][][] biases,
                                           double[][][] weightGradients, double[][][] biasGradients,
                                           double learningRate) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        weights[i][j][k] += learningRate * weightGradients[i][j][k];
                        biases[i][j][k] += learningRate * biasGradients[i][j][k];
                    }
                }
            }
        }
    }

    private static class AdamOptimizer implements Optimizer {
        private static final double BETA1 = 0.9;
        private static final double BETA2 = 0.999;
        private static final double EPSILON = 1e-8;

        private double[][][] weightMoments, weightVelocities;
        private double[][][] biasMoments, biasVelocities;
        private int iteration;

        AdamOptimizer(int hiddenHeight, int hiddenWidth) {
            weightMoments = new double[hiddenHeight][hiddenWidth][4];
            weightVelocities = new double[hiddenHeight][hiddenWidth][4];
            biasMoments = new double[hiddenHeight][hiddenWidth][4];
            biasVelocities = new double[hiddenHeight][hiddenWidth][4];
            iteration = 0;
        }

        public void updateWeightsAndBiases(double[][][] weights, double[][][] biases,
                                           double[][][] weightGradients, double[][][] biasGradients,
                                           double learningRate) {
            iteration++;
            double beta1t = Math.pow(BETA1, iteration);
            double beta2t = Math.pow(BETA2, iteration);

            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    for (int k = 0; k < weights[i][j].length; k++) {
                        weightMoments[i][j][k] = BETA1 * weightMoments[i][j][k] + (1 - BETA1) * weightGradients[i][j][k];
                        weightVelocities[i][j][k] = BETA2 * weightVelocities[i][j][k] + (1 - BETA2) * weightGradients[i][j][k] * weightGradients[i][j][k];
                        double weightUpdate = learningRate * weightMoments[i][j][k] / (1 - beta1t) / (Math.sqrt(weightVelocities[i][j][k] / (1 - beta2t)) + EPSILON);
                        weights[i][j][k] += weightUpdate;

                        biasMoments[i][j][k] = BETA1 * biasMoments[i][j][k] + (1 - BETA1) * biasGradients[i][j][k];
                        biasVelocities[i][j][k] = BETA2 * biasVelocities[i][j][k] + (1 - BETA2) * biasGradients[i][j][k] * biasGradients[i][j][k];
                        double biasUpdate = learningRate * biasMoments[i][j][k] / (1 - beta1t) / (Math.sqrt(biasVelocities[i][j][k] / (1 - beta2t)) + EPSILON);
                        biases[i][j][k] += biasUpdate;
                    }
                }
            }
        }
    }
}
class NetworkVisualizer extends JFrame {
        private static final int CELL_SIZE = 20;
        private static final int PADDING = 10;private CellularRNN network;
    private ArrayCanvas inputCanvas, hiddenCanvas, outputCanvas;
    final ArrayCanvas[] w;

    NetworkVisualizer(CellularRNN network) {
        setIgnoreRepaint(true);

        this.network = network;

        setTitle("Cellular RNN Visualization");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLayout(new GridLayout(1, 3, PADDING, PADDING));

        inputCanvas = new ArrayCanvas(network.inputSize, 1);
        hiddenCanvas = new ArrayCanvas(network.hiddenHeight, network.hiddenWidth);
        outputCanvas = new ArrayCanvas(1, network.outputSize);

        add(inputCanvas);
        add(hiddenCanvas);
        add(outputCanvas);

        w = new ArrayCanvas[network.weights.length];
        for (int i = 0; i < w.length; i++) {
            add(w[i] = new ArrayCanvas(network.weights[0].length, network.weights[0][0].length));
        }

        pack();
//        setLocationRelativeTo(null);
        setVisible(true);
    }

    void render() {
        inputCanvas.renderArray(network.inputs);
        hiddenCanvas.renderArray(network.hidden);
        outputCanvas.renderArray(network.outputs);
        for (int i = 0; i < network.weights.length; i++) {
            w[i].renderArray(network.weights[i]);
        }
        invalidate();
        repaint();
    }
}
class ArrayCanvas extends JComponent {
    private int rows, cols;
    private double[][] values;

    ArrayCanvas(int rows, int cols) {
        setIgnoreRepaint(true);
        this.rows = rows;
        this.cols = cols;
        values = new double[rows][cols];
    }

    public void renderArray(double[] array) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                values[i][j] = (i * cols + j < array.length) ? array[i * cols + j] : 0;
            }
        }
    }

    public void renderArray(double[][] array) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                values[i][j] = (i < array.length && j < array[i].length) ? array[i][j] : 0;
            }
        }

    }

    @Override
    protected void paintComponent(Graphics g) {
        //super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        int width = getWidth() / cols, height = getHeight() / rows;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                g2d.setColor(getColorForValue(values[i][j]));
                g2d.fillRect(j * width, i * height, width, height);
            }
        }
    }

    private static Color getColorForValue(double value) {
        if (value >= 0)
            return new Color(0, (int)(255 * Math.min(1, value)), 0);
        else
            return new Color((int)(255 * Math.min(1, -value)), 0, 0);
        //return new Color((int) (127 * (value + 1)) << 16 | (int) (127 * (1 - Math.abs(value))) << 8);
    }
}