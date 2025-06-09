package jcog.tensor.depr;

import jcog.tensor.deprtensor.*;
import org.eclipse.collections.impl.list.mutable.primitive.DoubleArrayList;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.DoubleSupplier;

import static jcog.Str.n4;
import static jcog.tensor.deprtensor.TensorFn.Activate.Relu;
import static jcog.tensor.deprtensor.TensorFn.Activate.Sigmoid;
import static jcog.tensor.deprtensor.TensorFn.*;
import static org.junit.jupiter.api.Assertions.*;

class Tens0rTest {

    // 2-XOR training data
    public static final double[][] xor2_inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
    };
    public static final double[][] xor2_targets = {
            {0},
            {1},
            {1},
            {0}
    };

    @Deprecated private static void xor2(Optimize o) {
        // Define the model architecture
        // Input layer, one hidden layer with 2 neurons, output layer with 1 neuron
        Layers l = new Layers(new LayersBuilder.SimpleLayersBuilder(
                new int[]{ 2, 2, 1},
                Relu, Sigmoid
                //Sigmoid, Sigmoid
                //Elephant, Elephant
                //Elephant, Sigmoid
        ));

        int epochs = 6_000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            DoubleArrayList losses = new DoubleArrayList();
            for (int i = 0; i < xor2_inputs.length; i++) {
                Tens0r y = l.forward(new Tens0r(xor2_inputs[i]));
                double loss = mse(y, new Tens0r(xor2_targets[i]))
                        .minimize(l, o);
                losses.add(loss);
            }
            System.out.println("loss: " + n4(losses.average()) + " " + losses);
        }

        float errTolerance = 0.1f;
        for (int i = 0; i < xor2_inputs.length; i++) {
            var output = l.forward(new Tens0r(new double[][]{xor2_inputs[i]}));
            double delta = xor2_targets[i][0] - output.scalar(); // Check if the output is close enough to the target
            assertEquals(0, delta, errTolerance);
        }
    }

    /**
     * TODO use a more complicated problem where the batch can sample parts of it
     */
    private static void xor2_batched(Optimize o) {
        // Define the model architecture
        int[] layerSizes = {2, 2, 1};  // Input layer, one hidden layer with 2 neurons, output layer with 1 neuron
        Layers model = new Layers(new LayersBuilder.SimpleLayersBuilder(layerSizes,
                Sigmoid, Sigmoid));

        int epochs = 10_000;
        int batchSize = xor2_inputs.length;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double lSum = 0;
            for (int i = 0; i < batchSize; i++) {
                Tens0r y = model.forward(new Tens0r(xor2_inputs[i]));
                double l = mse(y, new Tens0r(xor2_targets[i])).minimize();
                lSum += l;
            }
            //System.out.println(lSum/batchSize);
            //model.scaleGrad(1f / batchSize);
            o.run(model);
            //System.out.println("loss: " + n4(losses.average()) + " " + losses);
        }

        float errTolerance = 0.05f;
        for (int i = 0; i < batchSize; i++) {
            var output = model.forward(new Tens0r(new double[][]{xor2_inputs[i]}));
            double delta = output.scalar() - xor2_targets[i][0]; // Check if the output is close enough to the target
            assertEquals(0, delta, errTolerance);
        }
    }

    @Test
    public void testAdd() {
        Tens0r a = new Tens0r(new double[][]{{1, 2}, {3, 4}}),
                b = new Tens0r(new double[][]{{2, 3}, {4, 5}});

        Tens0r c = add(a, b);
        c.grad[0][0] = 1;
        c.grad[0][1] = 1;
        c.grad[1][0] = 1;
        c.grad[1][1] = 1;
        c.backward();

        double[][] expectedCData = {{3, 5}, {7, 9}};

        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }


        double[][] expectedAGrad = {{1, 1}, {1, 1}};
        double[][] expectedBGrad = {{1, 1}, {1, 1}};

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on b
        for (int i = 0; i < b.grad.length; i++) {
            for (int j = 0; j < b.grad[0].length; j++) {
                assertEquals(expectedBGrad[i][j], b.grad[i][j], 0.001);
            }
        }
    }

    @Test
    public void testNeg() {
        var a = new Tens0r(new double[][]{{1, -2}, {-3, 4}});

        Tens0r c = neg(a);
        c.grad[0][0] = 1;
        c.grad[0][1] = 1;
        c.grad[1][0] = 1;
        c.grad[1][1] = 1;
        c.backward();

        double[][] expectedCData = {{-1, 2}, {3, -4}};
        double[][] expectedAGrad = {{-1, -1}, {-1, -1}};

        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001);
            }
        }
    }

    @Test
    public void testElementwiseMult() {
        Tens0r a = new Tens0r(new double[][]{{1, 2}, {3, 4}});
        Tens0r b = new Tens0r(new double[][]{{2, 3}, {4, 5}});

        Tens0r c = multEW(a, b);
        c.grad[0][0] = 1;
        c.grad[0][1] = 1;
        c.grad[1][0] = 1;
        c.grad[1][1] = 1;
        c.backward();

        double[][] expectedCData = {{2, 6}, {12, 20}};
        double[][] expectedAGrad = {{2, 3}, {4, 5}};
        double[][] expectedBGrad = {{1, 2}, {3, 4}};

        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on b
        for (int i = 0; i < b.grad.length; i++) {
            for (int j = 0; j < b.grad[0].length; j++) {
                assertEquals(expectedBGrad[i][j], b.grad[i][j], 0.001);
            }
        }
    }

    @Test
    public void testScalarMult() {
        var a = new Tens0r(new double[][]{{1, 2}, {3, 4}});
        double scalar = 0.5;
        Tens0r c = multScalar(a, scalar(scalar));

        c.grad[0][0] = 1;
        c.grad[0][1] = 1;
        c.grad[1][0] = 1;
        c.grad[1][1] = 1;
        c.backward();

        double[][] expectedCData = {{0.5, 1}, {1.5, 2}};
        double[][] expectedAGrad = {{0.5, 0.5}, {0.5, 0.5}};

        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001);
            }
        }
    }

    @Test
    public void testReLU() {
        double[][] aData = {{-1, 2}, {-3, 4}};
        var a = new Tens0r(aData);
        Tens0r c = relu(a);
        c.grad[0][0] = 1;
        c.grad[0][1] = 1;
        c.grad[1][0] = 1;
        c.grad[1][1] = 1;
        c.backward();

        double[][] expectedCData = {{0, 2}, {0, 4}};
        double[][] expectedAGrad = {{0, 1}, {0, 1}};

        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001);
            }
        }
    }

    @Test
    public void testMSE() {
        double[][] yPredData = {{2, 3}, {5, 7}};
        double[][] yTrueData = {{1, 2}, {4, 6}};
        var yPred = new Tens0r(yPredData);
        var yTrue = new Tens0r(yTrueData);

        Tens0r mseTensor = mse(yPred, yTrue);
        mseTensor.scalarGrad(1); // Properly initializing gradient as a scalar
        mseTensor.backward();

        double expectedMSE = 1; // MSE calculation (1^2 + 1^2 + 1^2 + 2^2) / 4
        assertEquals(expectedMSE, mseTensor.scalar(), 0.001, "MSE computation");

        // Calculating expected gradients for predictions
        double[][] expectedPredGrad = {{0.5, 0.5}, {0.5, 0.5}};
        for (int i = 0; i < yPred.grad.length; i++) {
            for (int j = 0; j < yPred.grad[0].length; j++) {
                assertEquals(expectedPredGrad[i][j], yPred.grad[i][j], 0.001, "Gradient w.r.t. predictions");
            }
        }
    }

    @Test
    public void testMMULT() {
        double[][] aData = {{1, 2}, {3, 4}};
        double[][] bData = {{2, 0}, {1, 3}};
        var a = new Tens0r(aData);
        var b = new Tens0r(bData);

        Tens0r c = mmult(a, b);
        c.grad[0][0] = 1;
        c.grad[0][1] = 1;
        c.grad[1][0] = 1;
        c.grad[1][1] = 1;
        c.backward();

        double[][] expectedCData = {{4, 6}, {10, 12}}; // Corrected expected results

        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }


        // Check forward pass
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }

        double[][] expectedAGrad = {{2, 4}, {2, 4}};
        double[][] expectedBGrad = {{4, 4}, {6, 6}};

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                int I = i, J = j;
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001, () -> "backward pass for gradients of A: incorrect at i=" + I + ", j=" + J);
            }
        }

        // Check backward pass for gradients on b
        for (int i = 0; i < b.grad.length; i++) {
            for (int j = 0; j < b.grad[0].length; j++) {
                assertEquals(expectedBGrad[i][j], b.grad[i][j], 0.001);
            }
        }
    }

    @Test
    public void testChainedOperations() {
        // Initialize matrices
        double[][] aData = {{1, 2}, {3, 4}};
        double[][] bData = {{2, 0}, {0, 2}};
        double[][] targetData = {{3, 4}, {7, 8}};

        // Create Tensors
        var a = new Tens0r(aData);
        var b = new Tens0r(bData);
        var target = new Tens0r(targetData);

        // Perform matrix multiplication
        Tens0r mmultResult = mmult(a, b);

        // Apply ReLU
        Tens0r reluResult = relu(mmultResult);

        // Compute MSE with target
        Tens0r mseResult = mse(reluResult, target);

        // Initialize gradient for backpropagation (normally set to 1 for loss functions)
        mseResult.scalarGrad(1);

        // Perform a single backward pass
        mseResult.backward(); // This single call now propagates through all preceding tensors


        // Assertions to check intermediate data correctness
        double[][] expectedMMultData = {{2, 4}, {6, 8}};
        double[][] expectedReluData = {{2, 4}, {6, 8}};
        double expectedMSE = 0.5; // Simple MSE calculation based on the squared differences and mean

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(expectedMMultData[i][j], mmultResult.data[i][j], 0.001, "MMult data mismatch");
                assertEquals(expectedReluData[i][j], reluResult.data[i][j], 0.001, "ReLU data mismatch");
                // MSE is a single value, check if needed
            }
        }
        assertEquals(expectedMSE, mseResult.scalar(), 0.001,
                "expectedMSE is incorrect");

        // Assertions for backward pass gradients
        // Example checks assuming some gradient values have been calculated
        double[][] expectedAGrad = {{-1, 0}, {-1, 0}};
        double[][] expectedBGrad = {{-2, 0}, {-3, 0}};

        // Check gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                int I = i, J = j;
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001, () -> "gradients on a: i=" + I + ", j=" + J);
            }
        }

        // Check gradients on b
        for (int i = 0; i < b.grad.length; i++) {
            for (int j = 0; j < b.grad[0].length; j++) {
                int I = i, J = j;
                assertEquals(expectedBGrad[i][j], b.grad[i][j], 0.001, () -> "gradients on b: i=" + I + ", j=" + J);
            }
        }
    }

    @Test
    public void xorSimple() {
        // XOR inputs and expected outputs
        double[][] inputData = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] targetData = {
                {0},
                {1},
                {1},
                {0}
        };

        // Initialize weights for a single hidden layer and output layer
        // These weights would typically be initialized randomly
        double[][] hiddenWeightsData = {
                {0.5, -0.5},
                {-0.5, 0.5}
        };
        double[][] outputWeightsData = {
                {1},
                {1}
        };

        Tens0r inputs = new Tens0r(inputData);
        Tens0r target = new Tens0r(targetData);

        Tens0r hiddenWeights = new Tens0r(hiddenWeightsData);
        Tens0r outputWeights = new Tens0r(outputWeightsData);

        OptimizeSGD optimizer = new OptimizeSGD(() -> 0.05f);  // example learning rate

        DoubleArrayList errs = new DoubleArrayList();
        for (int epoch = 0; epoch < 10; epoch++) {
            Tens0r actual = mmult(relu(mmult(inputs, hiddenWeights)), outputWeights);
            Tens0r output = mse(actual, target);
            // Initialize gradient for backpropagation
            output.scalarGrad(1);  // Assuming batch size of 1 for simplicity
            output.backward();  // This will automatically propagate through all layers

            errs.add(output.scalar());

            // Update weights
            optimizer.run(List.of(hiddenWeights, outputWeights));
        }

        assertTrue(errs.getFirst() > errs.getLast());
        assertTrue(errs.getLast() < 0.1f);


//        // Check outputs to see if they are correct
//        System.out.println("Trained XOR outputs:");
//        for (int i = 0; i < inputData.length; i++) {
//            Tensor testInput = new Tensor(new double[][]{inputData[i]});
//            Tensor testHidden = Tensor.mmult(testInput, hiddenWeights);
//            Tensor testRelu = Tensor.relu(testHidden);
//            Tensor testOutput = Tensor.mmult(testRelu, outputWeights);
//            System.out.println("Input: " + n4(inputData[i]) + " Output: " + testOutput.scalar());
//        }
    }

    @Test
    public void trainSingleNeuronToConstantOutput() {

        double inputVal = 1.0, targetVal = 0.5;

        var input = scalar(inputVal);
        var target = scalar(targetVal);

        Optimize.DenseLayer layer = new Optimize.DenseLayer(1, 1, Sigmoid);  // One input and one output
        OptimizeSGD optimizer = new OptimizeSGD(() -> 0.01f);  // Learning rate
        int iterations = 220;  // Number of training iterations
        for (int i = 0; i < iterations; i++) {
            var output = layer.forward(input);

            Tens0r loss = mse(output, target);

            loss.scalarGrad(1);  // Starting gradient for MSE should be set based on the derivative of the loss
            loss.backward();

            optimizer.step(List.of(layer.weights, layer.bias));
            layer.zeroGrad();

            //System.out.println("Iteration " + i + " Output: " + output.scalar() + "\t" + loss.scalar());
//            System.out.println("Weight Gradient: " + Arrays.deepToString(layer.weights.grad));
//            System.out.println("Bias Gradient: " + Arrays.deepToString(layer.bias.grad));
        }

        assertEquals(0.5, layer.forward(input).scalar(), 0.1, "Neuron should learn to output 0.5");
    }

    @Disabled @Test
    public void xor2_sgd() {
        xor2(new OptimizeSGD(() -> 0.002f));
    }

    @Disabled @Test
    public void xor2_adam() {
        xor2(new OptimizeADAM(() -> 0.01f, 0.5f, 0.5f, 0, 1e-8));
    }

    @Disabled @Test
    public void xor2_sgd_batch() {
        xor2_batched(new OptimizeSGD(() -> 0.1f));
    }

    @Disabled @Test
    public void xor2_adam_batch() {
        xor2_batched(new OptimizeADAM(() -> 0.01f));
    }

    @Test
    public void testMin() {
        var a = new Tens0r(new double[][]{{1, 4}, {5, 2}});
        var b = new Tens0r(new double[][]{{3, 2}, {2, 6}});

        Tens0r c = min(a, b);
        c.grad = new double[][]{{1, 1}, {1, 1}};
        c.backward();

        double[][] expectedCData = {{1, 2}, {2, 2}};

        double[][] expectedAGrad = {{1, 0}, {0, 1}};
        double[][] expectedBGrad = {{0, 1}, {1, 0}};

        assertArrayEquals(expectedCData, c.data);
        assertArrayEquals(expectedAGrad, a.grad);
        assertArrayEquals(expectedBGrad, b.grad);
    }

    @Test
    public void testConcat() {
        var a = new Tens0r(new double[][]{{1, 2}});
        var b = new Tens0r(new double[][]{{3, 4}});

        Tens0r c = concat(a, b);
        c.grad[0][0] = 1;  // Set gradients for backward pass
        c.grad[0][1] = 1;
        c.grad[0][2] = 1;
        c.grad[0][3] = 1;
        c.backward();

        double[][] expectedCData = {{1, 2, 3, 4}};  // Expected result from concat operation

        // Check forward pass results
        for (int i = 0; i < c.data.length; i++) {
            for (int j = 0; j < c.data[0].length; j++) {
                assertEquals(expectedCData[i][j], c.data[i][j], 0.001);
            }
        }

        double[][] expectedAGrad = {{1, 1}};  // Gradients should be split back to the original tensors
        double[][] expectedBGrad = {{1, 1}};

        // Check backward pass for gradients on a
        for (int i = 0; i < a.grad.length; i++) {
            for (int j = 0; j < a.grad[0].length; j++) {
                assertEquals(expectedAGrad[i][j], a.grad[i][j], 0.001);
            }
        }

        // Check backward pass for gradients on b
        for (int i = 0; i < b.grad.length; i++) {
            for (int j = 0; j < b.grad[0].length; j++) {
                assertEquals(expectedBGrad[i][j], b.grad[i][j], 0.001);
            }
        }
    }


    @Test
    public void testLambdaScalar() {
        var regularTensor = new Tens0r(new double[][]{{2, 4}, {6, 8}});

        {
            // Initialize a regular tensor with arbitrary values

            // Initialize a dynamic constant tensor with a value supplier
            DoubleSupplier supplier = () -> 3.0;
            TensorFn constantTensor = scalar(supplier);

            // Perform element-wise multiplication between constant and regular tensor
            TensorFn resultTensor = multScalar(regularTensor, constantTensor);

            // Forward pass to compute element-wise multiplication
            constantTensor.forward();  // Update constant tensor value
            resultTensor.forward();    // Perform multiplication

            // Check results of the forward pass
            double[][] expectedData = {{6, 12}, {18, 24}};  // Expected results from multiplying by constant 3
            for (int i = 0; i < resultTensor.data.length; i++) {
                for (int j = 0; j < resultTensor.data[0].length; j++) {
                    assertEquals(expectedData[i][j], resultTensor.data[i][j], 0.001);
                }
            }

            // Initialize gradients for backpropagation
            resultTensor.grad = new double[][]{{1, 1}, {1, 1}};
            resultTensor.backward();

            // Since the constant tensor should not propagate gradients, its gradients should remain null or zero
            assertNull(constantTensor.grad, "Gradients for constant tensor should not exist or be zero");

            // Check gradients on the regular tensor
            double[][] expectedGrad = {{3, 3}, {3, 3}};  // Each element in grad should be multiplied by the constant
            for (int i = 0; i < regularTensor.grad.length; i++) {
                for (int j = 0; j < regularTensor.grad[0].length; j++) {
                    assertEquals(expectedGrad[i][j], regularTensor.grad[i][j], 0.001);
                }
            }
        }

        {
            // Change the value of the constant and repeat the forward pass to see updated effects
            TensorFn constantTensor = scalar(() -> 5.0);
            TensorFn resultTensor = multScalar(regularTensor, constantTensor);
            constantTensor.forward();
            resultTensor.forward();

            double[][] expectedData = {{10, 20}, {30, 40}};  // New expected results from multiplying by updated constant 5
            for (int i = 0; i < resultTensor.data.length; i++) {
                for (int j = 0; j < resultTensor.data[0].length; j++) {
                    assertEquals(expectedData[i][j], resultTensor.data[i][j], 0.001);
                }
            }
        }
    }


    @Test
    public void testSumForward() {
        // Create a tensor with predefined data
        var tensor = new Tens0r(new double[][]{{1, 2}, {3, 4}});
        Tens0r[] inputs = {tensor};

        // Instance of SUM
        double[][] output = TensorOp.SUM.allocate(inputs);

        // Execute forward operation
        TensorOp.SUM.forward(inputs, output);

        // Check if the sum is correctly calculated
        assertEquals(10.0, output[0][0], "The sum of the elements should be 10");
    }

    @Test
    public void testSumBackward() {
        // Create a tensor with predefined data
        Tens0r tensor = new Tens0r(new double[][]{{1, 2}, {3, 4}});
        tensor.grad = new double[tensor.data.length][tensor.data[0].length]; // Initialize gradient storage
        Tens0r[] inputs = {tensor};

        // Instance of SUM
        double[][] output = TensorOp.SUM.allocate(inputs);

        // Perform forward pass to compute the sum
        TensorOp.SUM.forward(inputs, output);

        // Assume a gradient of 1 from downstream
        double[][] upstreamGrad = {{1}};
        TensorOp.SUM.backward(inputs, upstreamGrad);

        // Check if the gradients are correctly distributed back to each element
        double expectedGrad = 1; // Each element should receive the full upstream gradient
        for (double[] row : tensor.grad) {
            for (double gradVal : row) {
                assertEquals(expectedGrad, gradVal, "Each element should have a gradient of 1");
            }
        }
    }


    @Test
    public void testPowOperation() {
        // Create tensor and scalar power
        Tens0r base = new Tens0r(new double[][]{{2, 3, 4}});
        double power = 2.0;  // We will square the tensor

        // Expected results of base^2
        double[][] expectedForward = {{4, 9, 16}};

        // Perform the power operation
        Tens0r result = pow(base, power);

        // Check the forward result
        assertArrayEquals(expectedForward, result.data, "Pow operation forward pass failed");

        // Set gradient to 1 for simplicity and backpropagate
        result.grad = new double[][]{{1, 1, 1}};
        result.backward();

        // Expected gradients for base tensor (2 * base[i][j])
        double[][] expectedGrad = {{4, 6, 8}};  // Derivative of x^2 is 2x

        // Check the backward gradients
        assertArrayEquals(expectedGrad, base.grad, "Pow operation backward pass failed");
    }

    @Test
    public void testExpOperation() {
        // Create tensor
        var base = new Tens0r(new double[][]{{0, 1, 2}});

        // Expected results of exp(base)
        double[][] expectedForward = {{Math.exp(0), Math.exp(1), Math.exp(2)}};

        // Perform the exp operation
        Tens0r result = exp(base);

        // Check the forward result
        assertArrayEquals(expectedForward, result.data, "Exp operation forward pass failed");

        // Set gradient to 1 for simplicity and backpropagate
        result.grad = new double[][]{{1, 1, 1}};
        result.backward();

        // Expected gradients for base tensor (exp(base[i][j]))
        double[][] expectedGrad = {{Math.exp(0), Math.exp(1), Math.exp(2)}};

        // Check the backward gradients
        assertArrayEquals(expectedGrad, base.grad, "Exp operation backward pass failed");
    }

    @Test
    public void testSliceOperation() {
        // Create a tensor
        var original = new Tens0r(new double[][]{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}});

        // Perform the slice operation to extract elements from index 3 to 7
        Tens0r sliced = slice(original, 3, 7); // Should slice {3, 4, 5, 6}

        // Expected results of the slice
        double[][] expectedSlice = {{3, 4, 5, 6}};

        // Check the forward result
        assertArrayEquals(expectedSlice, sliced.data, "Slice operation forward pass failed");

        // Set gradient for the sliced tensor and backpropagate
        sliced.grad = new double[][]{{1, 1, 1, 1}};  // Apply a gradient of 1 to each sliced element
        sliced.backward();

        // Expected gradients should reflect in the original tensor at the sliced positions
        double[][] expectedOriginalGrad = {{0, 0, 0, 1, 1, 1, 1, 0, 0, 0}};

        // Check the backward gradients in the original tensor
        assertArrayEquals(expectedOriginalGrad, original.grad, "Slice operation backward pass failed");
    }
}
