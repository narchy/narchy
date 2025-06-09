package jcog.tensor.spike;

import jcog.tensor.Models;
import jcog.tensor.Optimizers;
import jcog.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static jcog.tensor.Tensor.*;
import static org.junit.jupiter.api.Assertions.*;

class SpikeTensorTest {

    @Test
    public void testSpikeTensorInModelLayers() {
        // Define network parameters
        var inputSize = 5;
        var outputSize = 3;
        var hiddenSize = 10;
        var simulationTime = 1.0;
        var dt = 0.1;

        // Create a SpikingNetwork
        var snn = SpikingNetwork.random(inputSize, outputSize, inputSize + outputSize);

        // Create a SpikeTensor
        var spikeTensor = new SpikeTensor(snn, 0, dt, 4);

        // Create a Models.Layers with SpikeTensor and additional layers
        var layers = new Models.Layers();
        layers.layer.add(spikeTensor);
        layers.layer.add(new Models.Linear(outputSize, hiddenSize, Tensor::relu, true));
        layers.layer.add(new Models.Linear(hiddenSize, outputSize, Tensor::sigmoid, true));

        // Create an input tensor
        var input = randGaussian(inputSize, 1, 1);

        // Process the input through the layers
        var output = layers.apply(input);

        // Assertions
        assertNotNull(output, "Output should not be null");
        assertEquals(1, output.rows(), "Output should have 1 row");
        assertEquals(outputSize, output.cols(), "Output should have " + outputSize + " columns");

        // Check that output values are within the expected range (0 to 1 due to sigmoid activation)
        for (var i = 0; i < outputSize; i++) {
            var value = output.data(0, i);
            assertTrue(value >= 0 && value <= 1, "Output value should be between 0 and 1, but was " + value);
        }
    }

    @Test
    public void testSpikeTensorLearningXOR() {
        // Network parameters
        var inputSize = 2;
        var hiddenSize = 10;
        var outputSize = 1;
        var dt = 0.5;
        var steps = 2;

        // Create SpikingNetwork and wrap it in SpikeTensor
        var snn = SpikingNetwork.random(inputSize, outputSize /*hiddenSize*/, inputSize + hiddenSize);
        var spikeTensor = new SpikeTensor(snn, 0, dt, steps);

        // Create a Models.Layers with SpikeTensor and additional layers
        var network = new Models.Layers();
        network.layer.add(spikeTensor);
        //network.layer.add(new Models.Linear(hiddenSize, outputSize, Tensor::sigmoid, true));

        // Create optimizer
        var optimizer = new Optimizer(new Optimizers.SGD(()->0.001f));

        // XOR input-output pairs
        List<Tensor> inputs = new ArrayList<>();
        inputs.add(row(0, 0));
        inputs.add(row(0, 1));
        inputs.add(row(1, 0));
        inputs.add(row(1, 1));
        List<Tensor> targets = new ArrayList<>();
        targets.add(scalar(0));
        targets.add(scalar(1));
        targets.add(scalar(1));
        targets.add(scalar(0));

        // Training loop
        GradQueue g = new GradQueue();
        var epochs = 2000;
        double finalLoss = 0;
        int batch = inputs.size()*4;
        double lossSum = 0;
        for (var epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0;

            for (var i = 0; i < inputs.size(); i++) {
                var input = inputs.get(i).grad(true);
                var target = targets.get(i);

                var output = network.apply(input);

                var loss = output.mse(target);
                epochLoss += loss.scalar();

                loss.minimize(g);
            }
            g.optimize(optimizer);

            lossSum += epochLoss / inputs.size();

            if (epoch % batch == 0) {
                System.out.printf("Epoch %d, Loss: %.4f%n", epoch, finalLoss);
                finalLoss = lossSum/batch;
                lossSum = 0;
            }

        }

        // Test the trained network
        System.out.println("Final test results:");
        for (var i = 0; i < inputs.size(); i++) {
            var input = inputs.get(i).grad(false);
            var output = network.apply(input);
            System.out.printf("Input: %s, Output: %.4f, Target: %.0f%n",
                    input, output.scalar(), targets.get(i).scalar());
        }

        // Assert that the final loss is below a threshold
        assertTrue(finalLoss < 0.1, "Final loss should be below 0.1, but was " + finalLoss);

        // Assert that the network has learned XOR with some tolerance
        for (var i = 0; i < inputs.size(); i++) {
            var input = inputs.get(i);
            var output = network.apply(input);
            var expected = targets.get(i).scalar();
            var actual = output.scalar();
            assertTrue(Math.abs(expected - actual) < 0.2,
                    "Output for input " + input + " should be close to " + expected + ", but was " + actual);
        }
    }

}