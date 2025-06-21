package jcog.nndepr;

import jcog.Util;
import jcog.activation.TanhActivation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EvolvingGraphNNTest {

    @Test
    void testXOR2() {
        var g = new EvolvingGraphNN();

//        var v = new EvolvingGraphNN.Visualizer(g);
//        v.setVisible(true);

        double learningRate =
            0.1f;

        int batchSize = 32;
        int iters = 0, maxIters = 4096;

        g.activationDefault = TanhActivation.the;
        g.addInputs(2);
        g.addOutputs(1);
        g.addLayersConnectFull(
                //2
                3
                //4
            //6, 3
            //    2,2
        );

        double lossSum, lossMean;
        double[] x = new double[2], y = new double[1];
        do {
            lossSum = 0;
            for (int b = 0; b < batchSize; b++) {
                //2-XOR
                switch (g.random.nextInt(4)) {
                    case 0 -> {
                        x[0] = -1;
                        x[1] = -1;
                        y[0] = -1;
                    }
                    case 1 -> {
                        x[0] = +1;
                        x[1] = -1;
                        y[0] = +1;
                    }
                    case 2 -> {
                        x[0] = -1;
                        x[1] = +1;
                        y[0] = +1;
                    }
                    case 3 -> {
                        x[0] = +1;
                        x[1] = +1;
                        y[0] = -1;
                    }
                }

                double[] yPred = g.put(x, y, learningRate);
                double[] delta = Util.sub(y, yPred);
                //System.out.println(n2(x) + "\t" + n2(y) + "\t" + n2(delta));
                double loss = Util.sumAbs(delta);
                lossSum += loss;

//                Util.sleepMS(100);
//                v.repaint();
//                System.out.println(loss);
            }
            lossMean = lossSum / batchSize;

            //System.out.println(lossMean);

            //g.evolve(lossMean/10);

        } while (lossMean > 0.02 && iters++ < maxIters);
        assertTrue(iters < maxIters);
    }

    private EvolvingGraphNN network;

    @BeforeEach
    void setUp() {
        network = new EvolvingGraphNN(3, 2); // 3 inputs, 2 outputs
    }

    @Test
    void testNetworkInitialization() {
        assertEquals(3, network.inputCount());
        assertEquals(2, network.outputCount());
        assertEquals(5, network.nodes().size()); // 3 inputs + 2 outputs
    }

    @Test
    void testForward() {
        double[] input = {1.0, 2.0, 3.0};
        double[] output = network.forward(input);
        assertEquals(2, output.length);
        // Add more specific assertions based on expected behavior
    }

    @Test
    void testBackpropagation() {
        double[] input = {1.0, 2.0, 3.0};
        double[] target = {0.5, 1.5};
        double learningRate = 0.1;

        double[] initialOutput = network.forward(input);
        network.put(input, target, learningRate);
        double[] updatedOutput = network.forward(input);

        // Check if the output has changed after training
        assertNotEquals(initialOutput[0], updatedOutput[0], 0.001);
        assertNotEquals(initialOutput[1], updatedOutput[1], 0.001);
    }

    @Test
    void testAddHiddenNode() {
        int initialNodeCount = network.nodes().size();
        network.newHiddenNode();
        assertEquals(initialNodeCount + 1, network.nodes().size());
    }

    @Test
    void testAddFullyConnectedLayer() {
        int hiddenLayerSize = 5;
        network.addLayersConnectFull(5);
        assertEquals(10, network.nodes().size()); // 3 inputs + 5 hidden + 2 outputs
        // Add assertions to check if connections are properly established
    }

//    @Test
//    void testBPTT() {
//        double[][] inputs = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
//        double[][] targets = {{0.5, 1.5}, {1.5, 2.5}, {2.5, 3.5}};
//        int bpttSteps = 2;
//        double learningRate = 0.1;
//
//        network.put(inputs, targets, bpttSteps, learningRate);
//        // Add assertions to check if the network has learned from the sequence
//    }


}