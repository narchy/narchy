package jcog.tensor;

import jcog.Util;
import jcog.math.FloatMeanWindow;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;

import static jcog.tensor.Tensor.scalar;
import static org.junit.jupiter.api.Assertions.*;

public class TensorTest {

    public static final double[][] xor2_inputs = {
            {-1, -1},
            {-1, +1},
            {+1, -1},
            {+1, +1}
    };
    public static final double[][] xor2_targets = {
            {0},
            {+1},
            {+1},
            {0}
    };
    private static final double EPSILON = 1e-6;
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

    private static void testMLP_xor2(Tensor.Optimizer optimizer) {
        var mlp = new Models.Layers(Tensor::tanh, Tensor::sigmoid, true, 2, 4, 1);

        var cases = xor2_inputs.length;
        var losses = new FloatMeanWindow(cases * 32);
        for (var epoch = 0; epoch < 9000; epoch++) {
            var c = epoch % cases;
            var yPredicted = mlp.apply(Tensor.row(xor2_inputs[c]));
            var yActual = Tensor.row(xor2_targets[c]);

            var loss =
                    yPredicted.mse(yActual);
            //yPredicted.huber(yActual);

            losses.accept((float) loss.scalar());


            loss.minimize(optimizer, null);

//            if ((epoch - 999) % 100 == 0)
//                System.out.println(epoch + "\tLoss: " + (losses.mean()));
        }

        assertTrue(losses.mean() < 0.02f, losses.mean() + " mean loss");
    }

    private static void assertEquals(double[][] a, double[][] b, double thresh) {
        assertEquals(a, new Tensor(b, false), thresh);
    }

    private static void assertEquals(double[][] a, Tensor c) {
        assertEquals(a, c, 0.01f);
    }

    private static void assertEquals(double[][] a, Tensor c, double thresh) {
        assertEquals(new Tensor(a, false), c, thresh);
    }

    private static void assertEquals(Tensor a, Tensor c, double thresh) {
        assertTrue(c.sub(a).sumAbs() < thresh);
    }

    private static void assertGradEquals(double[][] a, Tensor c) {
        assertEquals(a, c.grad);
    }

    private static void arrayEquals(double[] expectedCData, double[] y) {
        assertArrayEquals(expectedCData, y, 0.001);
    }

    @Test
    public void testAddGradients() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var b = new Tensor(new double[][]{{5, 6}, {7, 8}}, true);
        var c = a.add(b);

        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        // Print gradients for debugging
        System.out.println("a grad: " + a.grad);
        System.out.println("b grad: " + b.grad);

        // Assert gradients
        assertGradEquals(new double[][]{{1, 1}, {1, 1}}, a);
        assertGradEquals(new double[][]{{1, 1}, {1, 1}}, b);
    }

    @Test
    public void testAdd() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true),
                b = new Tensor(new double[][]{{2, 3}, {4, 5}}, true);

        var c = a.add(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{{3, 5}, {7, 9}};

        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++) {
            for (var j = 0; j < c.data.numCols(); j++) {
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);
            }
        }


        var expectedAGrad = new double[][]{{1, 1}, {1, 1}};
        var expectedBGrad = new double[][]{{1, 1}, {1, 1}};

        // Check backward pass for gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++) {
            for (var j = 0; j < a.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001);
            }
        }

        // Check backward pass for gradients on b
        for (var i = 0; i < b.grad.data.numRows(); i++) {
            for (var j = 0; j < b.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedBGrad[i][j], b.grad.data(i, j), 0.001);
            }
        }
    }

    @Test
    public void testNeg() {
        var a = new Tensor(new double[][]{{1, -2}, {-3, 4}}, true);

        var c = a.neg();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{{-1, 2}, {3, -4}};
        var expectedAGrad = new double[][]{{-1, -1}, {-1, -1}};

        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++) {
            for (var j = 0; j < c.data.numCols(); j++) {
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++) {
            for (var j = 0; j < a.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001);
            }
        }
    }

    @Test
    public void testElementwiseMult() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var b = new Tensor(new double[][]{{2, 3}, {4, 5}}, true);

        var c = a.mul(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{{2, 6}, {12, 20}};
        var expectedAGrad = new double[][]{{2, 3}, {4, 5}};
        var expectedBGrad = new double[][]{{1, 2}, {3, 4}};

        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++) {
            for (var j = 0; j < c.data.numCols(); j++) {
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++) {
            for (var j = 0; j < a.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001);
            }
        }

        // Check backward pass for gradients on b
        for (var i = 0; i < b.grad.data.numRows(); i++) {
            for (var j = 0; j < b.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedBGrad[i][j], b.grad.data(i, j), 0.001);
            }
        }
    }

    @Test
    public void testScalarMult() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var scalar = 0.5;
        var c = a.mul(scalar);

        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{{0.5, 1}, {1.5, 2}};
        var expectedAGrad = new double[][]{{0.5, 0.5}, {0.5, 0.5}};

        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++) {
            for (var j = 0; j < c.data.numCols(); j++) {
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++) {
            for (var j = 0; j < a.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001);
            }
        }
    }

    @Test
    public void testReLU() {
        var aData = new double[][]{{-1, 2}, {-3, 4}};
        var a = new Tensor(aData, true);
        var c = a.relu();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{{0, 2}, {0, 4}};
        var expectedAGrad = new double[][]{{0, 1}, {0, 1}};

        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++) {
            for (var j = 0; j < c.data.numCols(); j++) {
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);
            }
        }

        // Check backward pass for gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++) {
            for (var j = 0; j < a.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001);
            }
        }
    }

    @Test
    public void testMSE() {
        var yPredData = new double[][]{{2, 3}, {5, 7}};
        var yTrueData = new double[][]{{1, 2}, {4, 6}};
        var yPred = new Tensor(yPredData, true);
        var yTrue = new Tensor(yTrueData, true);

        var mseTensor = yPred.mse(yTrue);
        mseTensor.minimize();

        double expectedMSE = 1; // MSE calculation (1^2 + 1^2 + 1^2 + 2^2) / 4
        Assertions.assertEquals(expectedMSE, mseTensor.scalar(), 0.001, "MSE computation");

        // Calculating expected gradients for predictions
        var expectedPredGrad = new double[][]{{0.5, 0.5}, {0.5, 0.5}};
        for (var i = 0; i < yPred.grad.data.numRows(); i++) {
            for (var j = 0; j < yPred.grad.data.numCols(); j++) {
                Assertions.assertEquals(expectedPredGrad[i][j], yPred.grad.data(i, j), 0.001, "Gradient w.r.t. predictions");
            }
        }
    }

    @Test
    public void testMMULT() {
        var aData = new double[][]{{1, 2}, {3, 4}};
        var bData = new double[][]{{2, 0}, {1, 3}};
        var a = new Tensor(aData, true);
        var b = new Tensor(bData, true);

        var c = a.matmul(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{{4, 6}, {10, 12}}; // Corrected expected results

        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++)
            for (var j = 0; j < c.data.numCols(); j++)
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);


        // Check forward pass
        for (var i = 0; i < c.data.numRows(); i++)
            for (var j = 0; j < c.data.numCols(); j++)
                Assertions.assertEquals(expectedCData[i][j], c.data(i, j), 0.001);


        var expectedAGrad = new double[][]{{2, 4}, {2, 4}};
        var expectedBGrad = new double[][]{{4, 4}, {6, 6}};

        // Check backward pass for gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++)
            for (var j = 0; j < a.grad.data.numCols(); j++)
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001, "backward pass for gradients of A: incorrect at i=" + i + ", j=" + j);

        // Check backward pass for gradients on b
        for (var i = 0; i < b.grad.data.numRows(); i++)
            for (var j = 0; j < b.grad.data.numCols(); j++)
                Assertions.assertEquals(expectedBGrad[i][j], b.grad.data(i, j), 0.001);
    }

    @Test
    public void testChainedOperations() {
        // Initialize matrices
        var aData = new double[][]{{1, 2}, {3, 4}};
        var bData = new double[][]{{2, 0}, {0, 2}};
        var targetData = new double[][]{{3, 4}, {7, 8}};

        // Create Tensors
        var a = new Tensor(aData, true);
        var b = new Tensor(bData, true);
        var target = new Tensor(targetData, true);

        // Perform matrix multiplication
        var mmultResult = a.matmul(b);

        // Apply ReLU
        var reluResult = mmultResult.relu();

        // Compute MSE with target
        var mseResult = reluResult.mse(target);

        // Perform a single backward pass
        mseResult.minimize(); // This single call now propagates through all preceding tensors

        // Assertions to check intermediate data correctness
        var expectedMMultData = new double[][]{{2, 4}, {6, 8}};
        var expectedReluData = new double[][]{{2, 4}, {6, 8}};
        var expectedMSE = 0.5; // Simple MSE calculation based on the squared differences and mean

        for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
                Assertions.assertEquals(expectedMMultData[i][j], mmultResult.data(i, j), 0.001, "MMult data mismatch");
                Assertions.assertEquals(expectedReluData[i][j], reluResult.data(i, j), 0.001, "ReLU data mismatch");
                // MSE is a single value, check if needed
            }
        }
        Assertions.assertEquals(expectedMSE, mseResult.scalar(), 0.001,
                "expectedMSE is incorrect");

        // Assertions for backward pass gradients
        // Example checks assuming some gradient values have been calculated
        var expectedAGrad = new double[][]{{-1, 0}, {-1, 0}};
        var expectedBGrad = new double[][]{{-2, 0}, {-3, 0}};

        // Check gradients on a
        for (var i = 0; i < a.grad.data.numRows(); i++) {
            for (var j = 0; j < a.grad.data.numCols(); j++) {
                int I = i, J = j;
                Assertions.assertEquals(expectedAGrad[i][j], a.grad.data(i, j), 0.001, () -> "gradients on a: i=" + I + ", j=" + J);
            }
        }

        // Check gradients on b
        for (var i = 0; i < b.grad.data.numRows(); i++) {
            for (var j = 0; j < b.grad.data.numCols(); j++) {
                int I = i, J = j;
                Assertions.assertEquals(expectedBGrad[i][j], b.grad.data(i, j), 0.001, () -> "gradients on b: i=" + I + ", j=" + J);
            }
        }
    }

    /** TODO MLP should be set with sigmoid activation */
    @Test
    public void testMLP_xor2_SGD() {
        testMLP_xor2(new Optimizers.SGD(() -> 0.1f).get(2));
    }

    @Test
    public void testMLP_xor2_ADAM() {
        testMLP_xor2(new Optimizers.ADAM(() -> 0.01f).get());
    }

    @Test
    public void testLog1p() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var c = a.log1p();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{Util.log1p(1), Util.log1p(2)}, {Util.log1p(3), Util.log1p(4)}}, c);
        assertGradEquals(new double[][]{{1.0 / 2.0, 1.0 / 3.0}, {1.0 / 4.0, 1.0 / 5.0}}, a);
    }

    @Test
    public void testSoftplus() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var c = a.softplus();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{
                {Util.log1p(Math.exp(1)), Util.log1p(Math.exp(2))},
                {Util.log1p(Math.exp(3)), Util.log1p(Math.exp(4))}
        }, c);
        assertGradEquals(new double[][]{
                {1.0 / (1 + Math.exp(-1)), 1 / (1 + Math.exp(-2))},
                {1.0 / (1 + Math.exp(-3)), 1 / (1 + Math.exp(-4))}
        }, a);
    }

    @Test
    void testSoftmax() {
        // Create a sample input tensor
        var data = new double[][]{
                {1.0, 2.0, 3.0},
                {0.5, 1.5, 2.5}
        };

        var inputTensor = new Tensor(new SimpleMatrix(data), false);
        var softmaxOutput = inputTensor.softmax();

        // Verify each row sums to 1
        for (var i = 0; i < softmaxOutput.rows(); i++) {
            var rowSum = 0.0;
            for (var j = 0; j < softmaxOutput.cols(); j++) {
                var value = softmaxOutput.data(i, j);
                assertTrue(value >= 0 && value <= 1, "Softmax output should be in [0,1] range");
                rowSum += value;
            }
            Assertions.assertEquals(1.0, rowSum, 1e-6);
        }
    }

    @Test
    public void testClip() {
        var a = new Tensor(new double[][]{{-2, 0.5}, {3, 5}}, true);
        var c = a.clip(0, 2);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{0, 0.5}, {2, 2}}, c);
        assertGradEquals(new double[][]{{0, 1}, {0, 0}}, a);
    }

    @Test
    public void testMean() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var c = a.mean();
        c.minimize();

        Assertions.assertEquals(2.5, c.scalar(), 0.001);
        assertGradEquals(new double[][]{{0.25, 0.25}, {0.25, 0.25}}, a);
    }

    @Test
    public void testSetData() {
        var a = Tensor.matrix(new double[][]{{1, 2}, {3, 4}});

        a.setData(true);
        assertEquals(new double[][]{{1}}, a);

        a.setData(5.0);
        assertEquals(new double[][]{{5.0}}, a);

        a.setData(new double[]{6, 7});
        assertEquals(new double[][]{{6, 7}}, a);

        a.setData(new double[][]{{8, 9}, {10, 11}});
        assertEquals(new double[][]{{8, 9}, {10, 11}}, a);
    }

    @Test
    public void testSetGrad() {
        var a = Tensor.matrix(new double[][]{{1, 2}, {3, 4}});
        a.setGrad(new double[][]{{5, 6}, {7, 8}});
        assertGradEquals(new double[][]{{5, 6}, {7, 8}}, a);
    }

    @Test
    public void testGradSumAbs() {
        var a = Tensor.matrix(new double[][]{{1, -2}, {3, -4}});
        a.setGrad(new double[][]{{5, 6}, {7, 8}});
        Assertions.assertEquals(26, a.grad.sumAbs(), 0.001);
    }

    @Test
    public void testGradMaxAbs() {
        var a = Tensor.matrix(new double[][]{{1, -2}, {3, -4}});
        a.setGrad(new double[][]{{5, 6}, {7, 8}});
        Assertions.assertEquals(8, a.grad.maxAbs(), 0.001);
    }

    @Test
    public void testMaxAbs() {
        var a = Tensor.matrix(new double[][]{{1, -2}, {3, -4}});
        Assertions.assertEquals(4, a.maxAbs(), 0.001);
    }

    @Test
    public void testDiff() {
        var x = new Tensor(2.0, true);
        var y = scalar(3.0);

        var z = x.mul(y);
        var w = z.exp().sum();

//        System.out.println("w: " + w);
        w.minimize();

//        System.out.println("dw/dx: " + x.grad);
//        System.out.println("dw/dy: " + y.grad);

        // Compute second-order gradient
        //x.getGrad().backward();
        //System.out.println("d2w/dx2: " + x.getGrad());
    }

    @Test
    public void testPow() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var c = a.pow(3); // Cube each element
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{1, 8}, {27, 64}}, c);
        assertGradEquals(new double[][]{{3, 12}, {27, 48}}, a);
    }

    @Test
    public void testSigmoid() {
        var a = new Tensor(new double[][]{{-1, 0}, {1, 2}}, true);
        var c = a.sigmoid();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{
                {1.0 / (1 + Math.exp(1)), 1.0 / (1 + Math.exp(0))},
                {1.0 / (1 + Math.exp(-1)), 1.0 / (1 + Math.exp(-2))}
        };

        assertEquals(expectedCData, c);
        assertGradEquals(new double[][]{
                {expectedCData[0][0] * (1 - expectedCData[0][0]), expectedCData[0][1] * (1 - expectedCData[0][1])},
                {expectedCData[1][0] * (1 - expectedCData[1][0]), expectedCData[1][1] * (1 - expectedCData[1][1])}
        }, a);
    }

    @Test
    public void testTanh() {
        var a = new Tensor(new double[][]{{-1, 0}, {1, 2}}, true);
        var c = a.tanh();
        var d = c.sum(); // Added a sum operation after tanh
        d.minimize(); // Backpropagate t

        var expectedCData = new double[][]{
                {Math.tanh(-1), Math.tanh(0)},
                {Math.tanh(1), Math.tanh(2)}
        };
        var expectedAGrad = new double[][]{
                {(1 - Math.pow(expectedCData[0][0], 2)), (1 - Math.pow(expectedCData[0][1], 2))},
                {(1 - Math.pow(expectedCData[1][0], 2)), (1 - Math.pow(expectedCData[1][1], 2))}
        };

        assertEquals(expectedCData, c);
        assertGradEquals(expectedAGrad, a);
    }

    @Test
    public void testLog() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var c = a.log();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{Math.log(1), Math.log(2)}, {Math.log(3), Math.log(4)}}, c);
        assertGradEquals(new double[][]{{1.0, 1.0 / 2.0}, {1.0 / 3.0, 1.0 / 4.0}}, a);
    }

    @Test
    public void testDiv() {
        var a = new Tensor(new double[][]{{6, 4}, {12, 10}}, true);
        var b = new Tensor(new double[][]{{2, 2}, {2, 5}}, true);
        var c = a.div(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{3, 2}, {6, 2}}, c);
        assertGradEquals(new double[][]{{0.5, 0.5}, {0.5, 0.2}}, a);
        assertGradEquals(new double[][]{{-1.5, -1}, {-3, -0.4}}, b);
    }

    @Test
    public void testConcat() {
        var a = new Tensor(new double[][]{{1, 2}, {4, 5}}, true);
        var b = new Tensor(new double[][]{{3, 4}, {6, 7}}, true);

        var c = a.concat(b);
        c.setGrad(new double[][]{{1, 1, 1, 1}, {1, 1, 1, 1}});
        c.minimize();

        assertEquals(new double[][]{{1, 2, 3, 4}, {4, 5, 6, 7}}, c);

        assertGradEquals(new double[][]{{1, 1}, {1, 1}}, a);
        assertGradEquals(new double[][]{{1, 1}, {1, 1}}, b);
    }

    @Test
    public void testMin() {
        var a = new Tensor(new double[][]{{1, 4}, {5, 2}}, true);
        var b = new Tensor(new double[][]{{3, 2}, {2, 6}}, true);

        var c = Tensor.min(a, b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{1, 2}, {2, 2}}, c);
        assertGradEquals(new double[][]{{1, 0}, {0, 1}}, a);
        assertGradEquals(new double[][]{{0, 1}, {1, 0}}, b);
    }

    @Test
    public void testMinEle_TwoScalars() {
        var a = new Tensor(5.0, true);
        var b = new Tensor(3.0, true);

        var minTensor = Tensor.minEle(List.of(a, b));
        minTensor.minimize();

        Assertions.assertEquals(3.0, minTensor.scalar(), 0.001);
        Assertions.assertEquals(0.0, a.grad.scalar(), 0.001);
        Assertions.assertEquals(1.0, b.grad.scalar(), 0.001);
    }

    @Test
    public void testMinEle_MultipleScalars() {
        var a = new Tensor(5.0, true);
        var b = new Tensor(3.0, true);
        var c = new Tensor(7.0, true);
        var d = new Tensor(2.0, true);

        var minTensor = Tensor.minEle(List.of(a, b, c, d));
        minTensor.minimize();

        Assertions.assertEquals(2.0, minTensor.scalar(), 0.001);
        Assertions.assertEquals(0.0, a.grad.scalar(), 0.001);
        Assertions.assertEquals(0.0, b.grad.scalar(), 0.001);
        Assertions.assertEquals(0.0, c.grad.scalar(), 0.001);
        Assertions.assertEquals(1.0, d.grad.scalar(), 0.001);
    }

    @Test
    public void testExp() {
        var a = new Tensor(new double[][]{{0, 1}, {2, 3}}, true);
        var c = a.exp();
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        var expectedCData = new double[][]{
                {Math.exp(0), Math.exp(1)},
                {Math.exp(2), Math.exp(3)}
        };
        // For exp(x), the gradient is the same as the output
        assertEquals(expectedCData, c);
        assertGradEquals(expectedCData, a);
    }

    @Test
    public void testSlice() {
        var x = Tensor.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).grad(true);

        var slice = x.slice(1, 3).grad(true);

        Assertions.assertEquals(3, slice.rows());
        Assertions.assertEquals(2, slice.cols());
        assertArrayEquals(new double[]{2, 3, 5, 6, 8, 9}, slice.array());

        slice.minimize();

        Assertions.assertEquals(3, slice.grad.rows());
        Assertions.assertEquals(2, slice.grad.cols());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, slice.grad.array());


        Assertions.assertEquals(3, x.grad.rows());
        Assertions.assertEquals(3, x.grad.cols());
        assertArrayEquals(new double[]{0, 1, 1, 0, 1, 1, 0, 1, 1}, x.grad.array());
    }

//    @Disabled /* TODO check numbers being tested */ @Test
//    public void testClipTanh() {
//        var a = new Tensor(new double[][]{{-2, 0}, {1, 3}}, true);
//        var c = a.clipTanh(-0.5, 0.5);
//        c.setGrad(new double[][]{{1, 1}, {1, 1}});
//        c.minimize();
//
//        assertEquals(new double[][]{
//                {-0.462117, 0},
//                {0.327946, 0.486817}
//        }, c);
//        assertGradEquals(new double[][]{
//                {0.178046, 0.5},
//                {0.432332, 0.131686}
//        }, a);
//    }

    @Test
    public void testSubAbs() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var b = new Tensor(new double[][]{{2, 1}, {5, 3}}, true);

        var c = a.subAbs(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{1, 1}, {2, 1}}, c);
        assertGradEquals(new double[][]{{-1, 1}, {-1, 1}}, a);
        assertGradEquals(new double[][]{{1, -1}, {1, -1}}, b);
    }

    @Disabled /* TODO check numbers being tested */
    @Test
    public void testKLDivergence() {
        // Test case 1: Identical distributions
        var p1 = Tensor.row(0.5, 0.5);
        var q1 = Tensor.row(0.5, 0.5);
        var kl1 = p1.klDivergence(q1);
        Assertions.assertEquals(0.0, kl1.sum().scalar(), EPSILON, "KL divergence of identical distributions should be 0");

        var epsilon = 1e-6;

        var p = Tensor.row(0.75, 0.25);
        var q = Tensor.row(0.25, 0.75);
        var kl = p.klDivergence(q);

        // Calculate expected KL divergence
        var expectedKL = 0.75 * Math.log(0.75 / 0.25) + 0.25 * Math.log(0.25 / 0.75);

        Assertions.assertEquals(expectedKL, kl.sum().scalar(), epsilon, "KL divergence should match manual calculation");
    }

    @Disabled /* TODO check numbers being tested */
    @Test
    public void testMatmulTranspose() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var b = new Tensor(new double[][]{{5, 6}, {7, 8}}, true);

        var c = a.matmulTranspose(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{17, 23}, {39, 53}}, c);
        assertGradEquals(new double[][]{{11, 15}, {11, 15}}, a);
        assertGradEquals(new double[][]{{4, 4}, {6, 6}}, b);
    }

    @Disabled /* TODO check numbers being tested */
    @Test
    public void testTransposeMatmul() {
        var a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var b = new Tensor(new double[][]{{5, 6}, {7, 8}}, true);

        var c = a.transposeMatmul(b);
        c.setGrad(new double[][]{{1, 1}, {1, 1}});
        c.minimize();

        assertEquals(new double[][]{{23, 31}, {34, 46}}, c);
        assertGradEquals(new double[][]{{11, 11}, {15, 15}}, a);
        assertGradEquals(new double[][]{{3, 7}, {4, 8}}, b);
    }

    @Test
    void testPowTensor() {
        // Test scalar base with scalar exponent
        var base = new Tensor(2.0, true);
        var exp = new Tensor(3.0, true);
        Assertions.assertEquals(8.0, base.pow(exp).scalar(), EPSILON, "2^3 should be 8");

        // Test matrix base with scalar exponent
        var matrix = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        exp = new Tensor(2.0, true);
        var result = matrix.pow(exp);
        assertEquals(
                new double[][]{{1, 4}, {9, 16}},
                result,
                EPSILON
                //"Element-wise square should be correct"
        );

        // Test power of 0
        exp = new Tensor(0.0, true);
        result = matrix.pow(exp);
        assertEquals(
                new double[][]{{1, 1}, {1, 1}},
                result,
                EPSILON
                //"x^0 should be all ones"
        );

        // Test power of 1
        exp = new Tensor(1.0, true);
        result = matrix.pow(exp);
        assertEquals(
                new double[][]{{1, 2}, {3, 4}},
                result,
                EPSILON
                //"x^1 should equal x"
        );
    }

    @Test
    void testPowGradients() {
        // Test gradient computation for scalar case
        var x = new Tensor(2.0, true);
        var y = x.pow(scalar(3).grad(true) /* force grad to use the pow(tensor) */);
        y.grad = new Tensor(1.0, false);
        assertTrue(y.hasGrad(), "Result should have gradient");

        // For x^3, derivative is 3x^2
        var expectedGrad = 3 * Math.pow(2.0, 2);
        y.minimize();
        Assertions.assertEquals(expectedGrad, x.grad.scalar(), EPSILON, "Gradient should be 3x^2");

        // Test gradient computation for matrix case
        var matrix = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var exp = new Tensor(2.0, true);
        y = matrix.pow(exp);
        y.grad = new Tensor(new double[][]{{1, 1}, {1, 1}}, false);
        y.minimize();

        // x^2's derivative is 2x
        assertEquals(new double[][]{{2 * 1, 2 * 2}, {2 * 3, 2 * 4}}, matrix.grad, EPSILON);
    }

    @Test
    void testPowEdgeCases() {
        // Test with zero base
        var zero = new Tensor(0.0, true);
        Assertions.assertEquals(0.0, zero.pow(2).scalar(), EPSILON, "0^2 should be 0");
        Assertions.assertEquals(1.0, zero.pow(0).scalar(), EPSILON, "0^0 should be 1");

        // Test with negative base
        var negative = new Tensor(-2.0, true);
        Assertions.assertEquals(4.0, negative.pow(2).scalar(), EPSILON, "(-2)^2 should be 4");

        // Test very large powers
        assertFalse(Double.isInfinite(new Tensor(1.0001, true).pow(1000).scalar()),
                "Large powers should not overflow");
    }

    @Test
    void testPowDouble() {
        // Test scalar power
        var x = new Tensor(2.0, true);
        org.junit.jupiter.api.Assertions.assertEquals(4.0, x.pow(2).scalar(), () -> "2^2 should be 4");
        org.junit.jupiter.api.Assertions.assertEquals(8.0, x.pow(3).scalar(), () -> "2^3 should be 8");

        // Test power of 1 (should return same tensor)
        var original = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        var result = original.pow(1);
        assertEquals(
                new double[][]{{1, 2}, {3, 4}},
                result,
                EPSILON
                //"x^1 should equal x"
        );

        // Test power of 0 (should return ones)
        result = original.pow(0);
        assertEquals(
                new double[][]{{1, 1}, {1, 1}},
                result,
                EPSILON
                //"x^0 should be all ones"
        );

        // Test matrix power
        var matrix = new Tensor(new double[][]{{2, 3}, {4, 5}}, true);
        result = matrix.pow(2);
        assertEquals(
                new double[][]{{4, 9}, {16, 25}},
                result,
                EPSILON
                //"Element-wise square should be correct"
        );

    }

//    @Nested
//    class SumRowColTest {
//
//        private static final double TOLERANCE = 1e-3;
//
//        // Helper method for numerical gradient checking
//        private static void checkNumericalGradient(Tensor inputTensor, Function<Tensor, Tensor> operation, @Nullable Tensor upstreamGradient) {
//            // Ensure inputTensor requires gradients for this check to be meaningful
//            var originalRequiresGrad = inputTensor.hasGrad();
//            if (!originalRequiresGrad) {
//                inputTensor.grad(true); // Temporarily enable grad for checking
//            }
//
//            // 1. Compute analytical gradient
//            var outputTensor = operation.apply(inputTensor);
//            if (outputTensor.isScalar() && upstreamGradient == null) {
//                // If output is scalar and no upstream grad provided, assume upstream grad is 1.0
//                outputTensor.minimize();
//            } else if (upstreamGradient != null) {
//                if (!outputTensor.sameShape(upstreamGradient)) {
//                    throw new IllegalArgumentException("Upstream gradient shape " + upstreamGradient.shapeStr() +
//                            " does not match output tensor shape " + outputTensor.shapeStr());
//                }
//                outputTensor.setGrad(upstreamGradient); // Set the provided upstream gradient
//                outputTensor.minimize(); // Trigger backward pass
//            } else {
//                // For non-scalar output and no specific upstream grad, use ones.
//                var onesGrad = Tensor.ones(outputTensor.rows(), outputTensor.cols());
//                outputTensor.setGrad(onesGrad);
//                outputTensor.minimize();
//            }
//
//            var analyticalGradMatrix = inputTensor.grad.data.copy(); // Store analytical grad
//
//            // 2. Compute numerical gradient for each element of inputTensor
//            var epsilon = 1e-5;
//            var numericalGradMatrix = new SimpleMatrix(inputTensor.rows(), inputTensor.cols());
//
//            for (var r = 0; r < inputTensor.rows(); r++) {
//                for (var c = 0; c < inputTensor.cols(); c++) {
//                    var originalValue = inputTensor.data(r, c);
//
//                    // Calculate loss + epsilon
//                    inputTensor.data.set(r, c, originalValue + epsilon);
//                    var outputPlus = operation.apply(inputTensor);
//                    var lossPlus = upstreamGradient == null ? outputPlus.sum().scalar() : outputPlus.mul(upstreamGradient).sum().scalar();
//
//
//                    // Calculate loss - epsilon
//                    inputTensor.data.set(r, c, originalValue - epsilon);
//                    var outputMinus = operation.apply(inputTensor);
//                    var lossMinus = upstreamGradient == null ? outputMinus.sum().scalar() : outputMinus.mul(upstreamGradient).sum().scalar();
//
//                    // Compute numerical gradient
//                    var gradVal = (lossPlus - lossMinus) / (2 * epsilon);
//                    numericalGradMatrix.set(r, c, gradVal);
//
//                    // Reset input tensor value
//                    inputTensor.data.set(r, c, originalValue);
//                }
//            }
//
//            // 3. Compare analytical and numerical gradients
//            for (var r = 0; r < inputTensor.rows(); r++) {
//                for (var c = 0; c < inputTensor.cols(); c++) {
//                    var analytical = analyticalGradMatrix.get(r, c);
//                    var numerical = numericalGradMatrix.get(r, c);
//                    Assertions.assertEquals(analytical, numerical, 1e-4,
//                            "Gradient mismatch at (" + r + "," + c + "): Analytical=" + analytical + ", Numerical=" + numerical);
//                }
//            }
//
//            // Restore original grad state
//            if (!originalRequiresGrad) {
//                inputTensor.grad(false);
//            }
//            inputTensor.zeroGrad(); // Clean up grad for next potential use in other tests
//        }
//
//        // Overload for scalar output where upstream gradient is implicitly 1.0
//        private static void checkNumericalGradient(Tensor inputTensor, Function<Tensor, Tensor> operation) {
//            checkNumericalGradient(inputTensor, operation, null);
//        }
//
//        /**
//         * Test multiple sum operations in a chain and verify gradients.
//         */
//        @Test
//        public void testSumChainGradient() {
//            var data = new double[][]{
//                    {1.0, 2.0},
//                    {3.0, 4.0}
//            };
//            var x = new Tensor(data, true); // requiresGrad=true
//
//            // Sum along columns to get [4,6]
//            var sumCols = x.sum(true);
//
//            // Then sum along rows to get 10
//            var finalSum = sumCols.sum(false); // scalar tensor
//
//            var expectedGrad = new SimpleMatrix(2, 2);
//            // Gradients should propagate back to sumCols and then to tensor
//            // sumCols has a gradient of 1.0
//            // tensor should have gradient [[1,1],[1,1]]
//            expectedGrad.fill(1.0);
//
//
//            // Assume gradient from finalSum is 1
//            //SimpleMatrix gradOutput = new SimpleMatrix(1, 1);
//            //finalSum.op.backward(gradOutput, new SimpleMatrix[] { expectedGrad });
//            finalSum.minimize();
//
//
//            var actualGrad = x.grad.data;
//
//            for (var i = 0; i < 2; i++) {
//                for (var j = 0; j < 2; j++) {
//                    org.junit.jupiter.api.Assertions.assertEquals(expectedGrad.get(i, j), actualGrad.get(i, j), TOLERANCE,
//                            "Gradient at (" + i + "," + j + ") is incorrect in sum chain.");
//                }
//            }
//        }
//
//        /**
//         * Test gradient computation when summing along rows.
//         */
//        @Test
//        public void testSumAlongRowsGradient() {
//            var data = new double[][]{
//                    {1.0, 2.0},
//                    {3.0, 4.0}
//            };
//            var tensor = new Tensor(data, true); // requiresGrad=true
//            var sumRows = tensor.sum(false);
//
//            // Assume some loss function; for testing, set gradient manually
//            // Here, we'll assume the gradient coming from sumRows is [1, 1]
//            // Then, the gradient w.r. to tensor should be [[1,1],[1,1]]
//            //SimpleMatrix gradOutput = new SimpleMatrix(2, 1, true, new double[] {1.0, 1.0});
//            //sumRows.backward(gradOutput);
//            sumRows.minimize();
//
//            var expectedGrad = new SimpleMatrix(2, 2);
//            expectedGrad.set(0, 0, 1.0);
//            expectedGrad.set(0, 1, 1.0);
//            expectedGrad.set(1, 0, 1.0);
//            expectedGrad.set(1, 1, 1.0);
//
//            var actualGrad = tensor.grad.data;
//
//            for (var i = 0; i < 2; i++) {
//                for (var j = 0; j < 2; j++) {
//                    Assertions.assertEquals(expectedGrad.get(i, j), actualGrad.get(i, j), TOLERANCE,
//                            "Gradient at (" + i + "," + j + ") is incorrect when summing along rows.");
//                }
//            }
//        }
//
//        @Nested
//        class StdVarianceTests {
//
//            @Test
//            void testStd_overall() {
//                var t = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // R=2, C=3
//                // Data: 1, 2, 3, 4, 5, 6
//                // Mean = (1+2+3+4+5+6)/6 = 21/6 = 3.5
//                // Squared diffs: (1-3.5)^2=6.25, (2-3.5)^2=2.25, (3-3.5)^2=0.25, (4-3.5)^2=0.25, (5-3.5)^2=2.25, (6-3.5)^2=6.25
//                // Sum of sq diffs = 6.25+2.25+0.25+0.25+2.25+6.25 = 17.5
//                // Variance (unbiased, N-1) = 17.5 / 5 = 3.5
//                // Std dev = sqrt(3.5) approx 1.8708
//                var expectedStd = Math.sqrt(3.5);
//                Assertions.assertEquals(expectedStd, t.std().scalar(), 1e-4);
//
//                // Numerical gradient check
//                // Upstream gradient for std() (which is scalar) is implicitly 1.0
//                checkNumericalGradient(t, Tensor::std);
//            }
//
//            @Test
//            void testVariance_overall_unbiasedVsBiased() {
//                var t = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, false); // N=6
//                // Sum of sq diffs = 17.5 (from testStd_overall)
//                // Variance (unbiased, N-1) = 17.5 / 5 = 3.5
//                // Variance (biased, N) = 17.5 / 6 = 2.91666...
//                Assertions.assertEquals(3.5, t.variance().scalar(), 1e-4, "Unbiased variance overall");
//                // No direct public method for biased overall variance, skip or test via std(axis) if needed for that path
//            }
//
//
//            @Test
//            void testStd_axis0_unbiased() { // Standard deviation of columns
//                var t = new Tensor(new double[][]{{1, 10, 100}, {3, 20, 200}, {5, 30, 700}}, true); // R=3, C=3
//                // Col 0: [1, 3, 5], Mean=3. SqDiffs: (1-3)^2=4, (3-3)^2=0, (5-3)^2=4. SumSqDiff=8. Var = 8/2=4. Std=2.
//                // Col 1: [10,20,30], Mean=20. SqDiffs: (10-20)^2=100, (20-20)^2=0, (30-20)^2=100. SumSqDiff=200. Var = 200/2=100. Std=10.
//                // Col 2: [100,200,700], Mean=1000/3=333.33. SqDiffs: (-233.33)^2, (-133.33)^2, (366.67)^2 approx 54444, 17777, 134444
//                // SumSqDiff for Col2 = (100-1000.0/3)^2 + (200-1000.0/3)^2 + (700-1000.0/3)^2
//                // = (-233.333)^2 + (-133.333)^2 + (366.666)^2
//                // = 54444.44 + 17777.77 + 134444.44 = 206666.65
//                // Var Col2 = 206666.65 / 2 = 103333.325. Std Col2 = sqrt(103333.325) approx 321.455
//
//                var expectedStdAxis0 = new double[]{2.0, 10.0, Math.sqrt((Math.pow(100 - 1000.0 / 3, 2) + Math.pow(200 - 1000.0 / 3, 2) + Math.pow(700 - 1000.0 / 3, 2)) / 2.0)};
//                var stdAxis0 = t.std(0); // unbiased is default for std(axis)
//                Assertions.assertEquals(1, stdAxis0.rows());
//                Assertions.assertEquals(3, stdAxis0.cols());
//                assertArrayEquals(expectedStdAxis0, stdAxis0.array(), 1e-3);
//
//                checkNumericalGradient(t, x -> x.std(0), Tensor.ones(1, t.cols()));
//            }
//
//            @Test
//            void testStd_axis1_unbiased() { // Standard deviation of rows
//                var t = new Tensor(new double[][]{{1, 3, 5}, {10, 20, 30}, {100, 200, 700}}, true); // R=3, C=3
//                // Row 0: [1, 3, 5], Mean=3. Std=2.
//                // Row 1: [10,20,30], Mean=20. Std=10.
//                // Row 2: [100,200,700], Mean=1000/3. Std=321.455
//                var expectedStdAxis1_data = new double[]{2.0, 10.0, Math.sqrt((Math.pow(100 - 1000.0 / 3, 2) + Math.pow(200 - 1000.0 / 3, 2) + Math.pow(700 - 1000.0 / 3, 2)) / 2.0)};
//                var expectedStdAxis1 = new Tensor(expectedStdAxis1_data, 3, 1, false);
//
//                var stdAxis1 = t.std(1);
//                Assertions.assertEquals(3, stdAxis1.rows());
//                Assertions.assertEquals(1, stdAxis1.cols());
//                assertArrayEquals(expectedStdAxis1.array(), stdAxis1.array(), 1e-3);
//
//                checkNumericalGradient(t, x -> x.std(1), Tensor.ones(t.rows(), 1));
//            }
//
//            @Test
//            void testVariance_axis0_biased() {
//                var t = new Tensor(new double[][]{{1, 10}, {3, 20}, {5, 30}}, false); // R=3, C=2
//                // Col 0: [1, 3, 5], Mean=3. SqDiffs: 4,0,4. SumSqDiff=8. Var_biased = 8/3 = 2.666...
//                // Col 1: [10,20,30], Mean=20. SqDiffs: 100,0,100. SumSqDiff=200. Var_biased = 200/3 = 66.666...
//                var expectedVarAxis0 = new double[]{8.0 / 3.0, 200.0 / 3.0};
//                var varAxis0 = t.variance(0, false); // Biased
//                assertArrayEquals(expectedVarAxis0, varAxis0.array(), 1e-3);
//            }
//
//            @Test
//            void testVariance_axis1_biased() {
//                var t = new Tensor(new double[][]{{1, 3, 5}, {10, 20, 30}}, false); // R=2, C=3
//                // Row 0: [1,3,5], Mean=3. SqDiffs: 4,0,4. SumSqDiff=8. Var_biased = 8/3 = 2.666...
//                // Row 1: [10,20,30], Mean=20. SqDiffs: 100,0,100. SumSqDiff=200. Var_biased = 200/3 = 66.666...
//                var expectedVarAxis1 = new double[]{8.0 / 3.0, 200.0 / 3.0};
//                var varAxis1 = t.variance(1, false); // Biased
//                assertArrayEquals(expectedVarAxis1, varAxis1.array(), 1e-3);
//            }
//
//            @Test
//            void testStd_singleValueTensor() {
//                var t = new Tensor(new double[][]{{42.0}}, true);
//                Assertions.assertEquals(0.0, t.std().scalar(), 1e-4, "Std of single value should be 0 (unbiased variance is NaN, sqrt(NaN)=NaN, but variance() returns 0 for scalar)");
//                // For std(axis), N=1, unbiased variance is NaN, sqrt(NaN) = NaN
//                assertTrue(Double.isNaN(t.std(0).scalar()), "Std(axis=0) of single value (unbiased) should be NaN");
//                assertTrue(Double.isNaN(t.std(1).scalar()), "Std(axis=1) of single value (unbiased) should be NaN");
//
//                // Biased std
//                Assertions.assertEquals(0.0, t.variance(0, false).sqrt().scalar(), 1e-4);
//                Assertions.assertEquals(0.0, t.variance(1, false).sqrt().scalar(), 1e-4);
//            }
//
////        @Test
////        void testStd_emptyTensor() {
////            var t = new Tensor(new double[][]{}, false); // 0 rows, 0 cols
////            assertTrue(Double.isNaN(t.std().scalar()), "Std of empty tensor should be NaN");
////            // std(0) on (0,0) tensor -> variance(0,0) -> N=0 -> NaN output shape (1,0)
////            // std(1) on (0,0) tensor -> variance(0,0) -> N=0 -> NaN output shape (0,1)
////            Assertions.assertEquals(0, t.std(0).cols()); // Shape (1,0) means 0 elements
////            Assertions.assertEquals(0, t.std(1).rows()); // Shape (0,1) means 0 elements
////        }
//        }
//
//        @Nested
//        class ArgMaxMinTests {
//
//            @Test
//            void testArgmax_axis0() {
//                var t = new Tensor(new double[][]{{1, 50, 2}, {10, 20, 300}, {5, 60, 1}}, false);
//                var expected = new Tensor(new double[][]{{1, 2, 1}}, false); // Indices of max in each column
//                var actual = t.argmax(0);
//                assertEquals(expected, actual, 1e-9);
//
//                // Test with duplicate max values (should pick first occurrence)
//                var t2 = new Tensor(new double[][]{{300, 50}, {10, 300}, {300, 1}}, false);
//                var expected2 = new Tensor(new double[][]{{0, 1}}, false);
//                var actual2 = t2.argmax(0);
//                assertEquals(expected2, actual2, 1e-9);
//            }
//
//            @Test
//            void testArgmax_axis1() {
//                var t = new Tensor(new double[][]{{1, 10, 5}, {50, 20, 60}, {2, 300, 1}}, false);
//                var expected = new Tensor(new double[][]{{1}, {2}, {1}}, false); // Indices of max in each row
//                var actual = t.argmax(1);
//                assertEquals(expected, actual, 1e-9);
//
//                var t2 = new Tensor(new double[][]{{300, 50, 300}, {10, 300, 1}}, false);
//                var expected2 = new Tensor(new double[][]{{0}, {1}}, false);
//                var actual2 = t2.argmax(1);
//                assertEquals(expected2, actual2, 1e-9);
//            }
//
//            @Test
//            void testArgmin_axis0() {
//                var t = new Tensor(new double[][]{{10, 2, 300}, {1, 20, 30}, {5, 0, 100}}, false);
//                var expected = new Tensor(new double[][]{{1, 2, 1}}, false); // Indices of min in each column
//                var actual = t.argmin(0);
//                assertEquals(expected, actual, 1e-9);
//
//                var t2 = new Tensor(new double[][]{{1, 50}, {10, 1}, {1, 100}}, false);
//                var expected2 = new Tensor(new double[][]{{0, 1}}, false);
//                var actual2 = t2.argmin(0);
//                assertEquals(expected2, actual2, 1e-9);
//
//            }
//
//            @Test
//            void testArgmin_axis1() {
//                var t = new Tensor(new double[][]{{10, 1, 5}, {50, 2, 60}, {300, 30, 10}}, false);
//                var expected = new Tensor(new double[][]{{1}, {1}, {2}}, false); // Indices of min in each row
//                var actual = t.argmin(1);
//                assertEquals(expected, actual, 1e-9);
//
//                var t2 = new Tensor(new double[][]{{50, 1, 1}, {10, 300, 10}}, false);
//                var expected2 = new Tensor(new double[][]{{1}, {0}}, false);
//                var actual2 = t2.argmin(1);
//                assertEquals(expected2, actual2, 1e-9);
//            }
//
//            @Test
//            void testArgmax_gradientError() {
//                var t = new Tensor(new double[][]{{1, 2}, {3, 4}}, true); // requiresGrad = true
//                assertThrows(UnsupportedOperationException.class, () -> t.argmax(0));
//                assertThrows(UnsupportedOperationException.class, () -> t.argmax(1));
//            }
//
//            @Test
//            void testArgmin_gradientError() {
//                var t = new Tensor(new double[][]{{1, 2}, {3, 4}}, true); // requiresGrad = true
//                assertThrows(UnsupportedOperationException.class, () -> t.argmin(0));
//                assertThrows(UnsupportedOperationException.class, () -> t.argmin(1));
//            }
//
//            @Test
//            void testArgmax_invalidAxis() {
//                var t = new Tensor(new double[][]{{1, 2}, {3, 4}}, false);
//                assertThrows(IllegalArgumentException.class, () -> t.argmax(2));
//                assertThrows(IllegalArgumentException.class, () -> t.argmax(-1));
//            }
//
//            @Test
//            void testArgmin_invalidAxis() {
//                var t = new Tensor(new double[][]{{1, 2}, {3, 4}}, false);
//                assertThrows(IllegalArgumentException.class, () -> t.argmin(2));
//                assertThrows(IllegalArgumentException.class, () -> t.argmin(-1));
//            }
//
////        @Test
////        void testArgmax_emptyTensor() {
////            var t = new Tensor(new double[][]{}, false); // 0 rows, 0 cols
////            var argmax0 = t.argmax(0); // Expect (1,0)
////            Assertions.assertEquals(1, argmax0.rows());
////            Assertions.assertEquals(0, argmax0.cols());
////
////            var argmax1 = t.argmax(1); // Expect (0,1)
////            Assertions.assertEquals(0, argmax1.rows());
////            Assertions.assertEquals(1, argmax1.cols());
////
////            var t2 = new Tensor(0, 3, false); // 0 rows, 3 cols
////            var argmax0_t2 = t2.argmax(0); // Expect (1,3) filled with 0s
////            Assertions.assertEquals(1, argmax0_t2.rows());
////            Assertions.assertEquals(3, argmax0_t2.cols());
////            assertArrayEquals(new double[]{0,0,0}, argmax0_t2.array(), 1e-9);
////
////            var argmax1_t2 = t2.argmax(1); // Expect (0,1)
////            Assertions.assertEquals(0, argmax1_t2.rows());
////            Assertions.assertEquals(1, argmax1_t2.cols());
////
////
////            var t3 = new Tensor(3, 0, false); // 3 rows, 0 cols
////            var argmax0_t3 = t3.argmax(0); // Expect (1,0)
////            Assertions.assertEquals(1, argmax0_t3.rows());
////            Assertions.assertEquals(0, argmax0_t3.cols());
////
////            var argmax1_t3 = t3.argmax(1); // Expect (3,1) filled with 0s
////            Assertions.assertEquals(3, argmax1_t3.rows());
////            Assertions.assertEquals(1, argmax1_t3.cols());
////            assertArrayEquals(new double[]{0,0,0}, argmax1_t3.array(), 1e-9);
////        }
//        }
//
//        @Nested
//        class SumSubTests {
//
//            @Test
//            void testSum_noArgs() {
//                var t = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // Sum = 21
//                var sumResult = t.sum();
//                Assertions.assertEquals(21.0, sumResult.scalar(), 1e-4);
//                checkNumericalGradient(t, Tensor::sum);
//            }
//
//            @Test
//            void testSum_axis0() { // Sum over columns -> result is row vector
//                var t = new Tensor(new double[][]{{1, 10, 100}, {2, 20, 200}, {3, 30, 300}}, true);
//                var sumAxis0 = t.sum(0);
//                Assertions.assertEquals(1, sumAxis0.rows());
//                Assertions.assertEquals(3, sumAxis0.cols());
//                assertArrayEquals(new double[]{6, 60, 600}, sumAxis0.array(), 1e-4);
//                checkNumericalGradient(t, x -> x.sum(0), Tensor.ones(1, t.cols()));
//            }
//
//            @Test
//            void testSum_axis1() { // Sum over rows -> result is col vector
//                var t = new Tensor(new double[][]{{1, 2, 3}, {10, 20, 30}, {100, 200, 300}}, true);
//                var sumAxis1 = t.sum(1);
//                Assertions.assertEquals(3, sumAxis1.rows());
//                Assertions.assertEquals(1, sumAxis1.cols());
//                assertArrayEquals(new double[]{6, 60, 600}, sumAxis1.array(), 1e-4);
//                checkNumericalGradient(t, x -> x.sum(1), Tensor.ones(t.rows(), 1));
//            }
//
//            @Test
//            void testSum_deprecated() {
//                var t = new Tensor(new double[][]{{1, 10}, {2, 20}, {3, 30}}, false);
//                // sum(false) -> sumRows() -> sum over columns -> (1,C)
//                var sumFalse = t.sum(false);
//                assertArrayEquals(new double[]{6, 60}, sumFalse.array(), 1e-4);
//                assertEquals(t.sum(0), sumFalse, 1e-9);
//
//
//                // sum(true) -> sumCols() -> sum over rows -> (R,1)
//                var sumTrue = t.sum(true);
//                assertArrayEquals(new double[]{11, 22, 33}, sumTrue.array(), 1e-4);
//                assertEquals(t.sum(1), sumTrue, 1e-9);
//            }
//
//            @Test
//            void testSum_invalidAxis() {
//                var t = new Tensor(new double[][]{{1, 2}, {3, 4}}, false);
//                assertThrows(IllegalArgumentException.class, () -> t.sum(2));
//                assertThrows(IllegalArgumentException.class, () -> t.sum(-1));
//            }
//
//            @Test
//            void testSub_gradient() {
//                var a = new Tensor(new double[][]{{10, 20}, {30, 40}}, true);
//                var b = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
//
//                // Test a - b
//                checkNumericalGradient(a, x -> x.sub(b), Tensor.ones(a.rows(), a.cols()));
//                // Check gradient for b: need to wrap it to make b the first argument to checkNumericalGradient
//                var initialBGrad = Tensor.zerosShaped(b); // placeholder for analytical
//                var bGradCopy = b.grad(true); // ensure b requires grad
//
//                Function<Tensor, Tensor> subOperationB = a::sub;
//
//                // We need to re-calculate analytical gradient for 'b' specifically
//                // The checkNumericalGradient function calculates grad for its first Tensor arg.
//                // So we need to structure the call carefully.
//                // Option 1: Manually compute analytical for b and compare.
//                // Option 2: Create a wrapper where b is the first arg.
//
//                // Let's do a combined check.
//                // Create a "meta" tensor whose elements are a and b, then define an operation. This is too complex.
//
//                // Simpler: checkNumericalGradient primarily tests the first argument's gradient.
//                // For 'b', we can set 'a' to not require gradients, then check 'b'.
//                a.grad(false); // Turn off grad for 'a'
//                b.grad(true);  // Ensure 'b' has grad
//                checkNumericalGradient(b, a::sub, Tensor.ones(a.rows(), a.cols()));
//
//                a.grad(true); // Restore grad for 'a'
//            }
//
//            @Test
//            void testBroadcastSub_gradient_row() { // (R,C) - (1,C)
//                var a = new Tensor(new double[][]{{10, 20, 30}, {40, 50, 60}}, true); // 2x3
//                var b_row = new Tensor(new double[][]{{1, 2, 3}}, true); // 1x3
//
//                // Test gradient for 'a'
//                checkNumericalGradient(a, x -> x.sub(b_row), Tensor.ones(a.rows(), a.cols()));
//
//                // Test gradient for 'b_row'
//                a.grad(false); // Turn off grad for 'a' to isolate 'b_row' grad check
//                checkNumericalGradient(b_row, a::sub, Tensor.ones(a.rows(), a.cols()));
//                a.grad(true);
//            }
//
//            @Test
//            void testBroadcastSub_gradient_col() { // (R,C) - (R,1)
//                var a = new Tensor(new double[][]{{10, 20, 30}, {40, 50, 60}}, true); // 2x3
//                var b_col = new Tensor(new double[][]{{1}, {2}}, true); // 2x1
//
//                // Test gradient for 'a'
//                checkNumericalGradient(a, x -> x.sub(b_col), Tensor.ones(a.rows(), a.cols()));
//
//                // Test gradient for 'b_col'
//                a.grad(false);
//                checkNumericalGradient(b_col, a::sub, Tensor.ones(a.rows(), a.cols()));
//                a.grad(true);
//            }
//        }
//    }
}


