package jcog.tensor;

import jcog.Util;
import jcog.math.FloatMeanWindow;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Nested;
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
    @Test public void testMLP_xor2_SGD() {
        testMLP_xor2(new Optimizers.SGD(()->0.1f).get(2));
    }

    @Test public void testMLP_xor2_ADAM() {
        testMLP_xor2(new Optimizers.ADAM(()->0.01f).get());
    }

    private static void testMLP_xor2(Tensor.Optimizer optimizer) {
        var mlp = new Models.Layers(Tensor::tanh, Tensor::sigmoid,true, 2, 4, 1);

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
        double[][] data = {
                {1.0, 2.0, 3.0},
                {0.5, 1.5, 2.5}
        };

        Tensor inputTensor = new Tensor(new SimpleMatrix(data), false);
        Tensor softmaxOutput = inputTensor.softmax();

        // Verify each row sums to 1
        for (int i = 0; i < softmaxOutput.rows(); i++) {
            double rowSum = 0.0;
            for (int j = 0; j < softmaxOutput.cols(); j++) {
                double value = softmaxOutput.data(i, j);
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
        assertArrayEquals(new double[] { 1, 1, 1, 1, 1, 1 }, slice.grad.array());


        Assertions.assertEquals(3, x.grad.rows());
        Assertions.assertEquals(3, x.grad.cols());
        assertArrayEquals(new double[]{0, 1, 1, 0, 1, 1, 0, 1, 1}, x.grad.array());
    }
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

    private static final double EPSILON = 1e-6;

    @Disabled /* TODO check numbers being tested */ @Test
    public void testKLDivergence() {
        // Test case 1: Identical distributions
        Tensor p1 = Tensor.row(0.5, 0.5);
        Tensor q1 = Tensor.row(0.5, 0.5);
        Tensor kl1 = p1.klDivergence(q1);
        Assertions.assertEquals(0.0, kl1.sum().scalar(), EPSILON, "KL divergence of identical distributions should be 0");

        double epsilon = 1e-6;

        Tensor p = Tensor.row(0.75, 0.25);
        Tensor q = Tensor.row(0.25, 0.75);
        Tensor kl = p.klDivergence(q);

        // Calculate expected KL divergence
        double expectedKL = 0.75 * Math.log(0.75/0.25) + 0.25 * Math.log(0.25/0.75);

        Assertions.assertEquals(expectedKL, kl.sum().scalar(), epsilon, "KL divergence should match manual calculation");
    }

    @Disabled /* TODO check numbers being tested */ @Test
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

    @Disabled /* TODO check numbers being tested */ @Test
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
        Tensor base = new Tensor(2.0, true);
        Tensor exp = new Tensor(3.0, true);
        Assertions.assertEquals(8.0, base.pow(exp).scalar(), EPSILON, "2^3 should be 8");

        // Test matrix base with scalar exponent
        Tensor matrix = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        exp = new Tensor(2.0, true);
        Tensor result = matrix.pow(exp);
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
        Tensor x = new Tensor(2.0, true);
        Tensor y = x.pow(scalar(3).grad(true) /* force grad to use the pow(tensor) */);
        y.grad = new Tensor(1.0, false);
        assertTrue(y.hasGrad(), "Result should have gradient");

        // For x^3, derivative is 3x^2
        double expectedGrad = 3 * Math.pow(2.0, 2);
        y.minimize();
        Assertions.assertEquals(expectedGrad, x.grad.scalar(), EPSILON, "Gradient should be 3x^2");

        // Test gradient computation for matrix case
        Tensor matrix = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor exp = new Tensor(2.0, true);
        y = matrix.pow(exp);
        y.grad = new Tensor(new double[][]{{1, 1}, {1, 1}}, false);
        y.minimize();

        // x^2's derivative is 2x
        assertEquals(new double[][]{{2*1, 2*2}, {2*3, 2*4}}, matrix.grad, EPSILON);
    }

    @Test
    void testPowEdgeCases() {
        // Test with zero base
        Tensor zero = new Tensor(0.0, true);
        Assertions.assertEquals(0.0, zero.pow(2).scalar(), EPSILON, "0^2 should be 0");
        Assertions.assertEquals(1.0, zero.pow(0).scalar(), EPSILON, "0^0 should be 1");

        // Test with negative base
        Tensor negative = new Tensor(-2.0, true);
        Assertions.assertEquals(4.0, negative.pow(2).scalar(), EPSILON, "(-2)^2 should be 4");

        // Test very large powers
        assertFalse(Double.isInfinite(new Tensor(1.0001, true).pow(1000).scalar()),
                "Large powers should not overflow");
    }

    @Test
    void testPowDouble() {
        // Test scalar power
        Tensor x = new Tensor(2.0, true);
        org.junit.jupiter.api.Assertions.assertEquals(4.0, x.pow(2).scalar(), ()-> "2^2 should be 4");
        org.junit.jupiter.api.Assertions.assertEquals(8.0, x.pow(3).scalar(), ()-> "2^3 should be 8");

        // Test power of 1 (should return same tensor)
        Tensor original = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor result = original.pow(1);
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
        Tensor matrix = new Tensor(new double[][]{{2, 3}, {4, 5}}, true);
        result = matrix.pow(2);
        assertEquals(
                new double[][]{{4, 9}, {16, 25}},
                result,
                EPSILON
                //"Element-wise square should be correct"
        );

    }

    @Nested
    class SumRowColTest {

        private static final double TOLERANCE = 1e-3;

        /**
         * Test multiple sum operations in a chain and verify gradients.
         */
        @Test
        public void testSumChainGradient() {
            double[][] data = {
                {1.0, 2.0},
                {3.0, 4.0}
            };
            Tensor x = new Tensor(data, true); // requiresGrad=true

            // Sum along columns to get [4,6]
            Tensor sumCols = x.sum(true);

            // Then sum along rows to get 10
            var finalSum = sumCols.sum(false); // scalar tensor

            SimpleMatrix expectedGrad = new SimpleMatrix(2, 2);
            // Gradients should propagate back to sumCols and then to tensor
            // sumCols has a gradient of 1.0
            // tensor should have gradient [[1,1],[1,1]]
            expectedGrad.fill(1.0);


            // Assume gradient from finalSum is 1
            //SimpleMatrix gradOutput = new SimpleMatrix(1, 1);
            //finalSum.op.backward(gradOutput, new SimpleMatrix[] { expectedGrad });
            finalSum.minimize();


            var actualGrad = x.grad.data;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    org.junit.jupiter.api.Assertions.assertEquals(expectedGrad.get(i, j), actualGrad.get(i, j), TOLERANCE,
                    "Gradient at (" + i + "," + j + ") is incorrect in sum chain.");
                }
            }
        }


        /**
         * Test gradient computation when summing along rows.
         */
        @Test
        public void testSumAlongRowsGradient() {
            double[][] data = {
                    {1.0, 2.0},
                    {3.0, 4.0}
            };
            Tensor tensor = new Tensor(data, true); // requiresGrad=true
            Tensor sumRows = tensor.sum(false);

            // Assume some loss function; for testing, set gradient manually
            // Here, we'll assume the gradient coming from sumRows is [1, 1]
            // Then, the gradient w.r. to tensor should be [[1,1],[1,1]]
            //SimpleMatrix gradOutput = new SimpleMatrix(2, 1, true, new double[] {1.0, 1.0});
            //sumRows.backward(gradOutput);
            sumRows.minimize();

            SimpleMatrix expectedGrad = new SimpleMatrix(2, 2);
            expectedGrad.set(0, 0, 1.0);
            expectedGrad.set(0, 1, 1.0);
            expectedGrad.set(1, 0, 1.0);
            expectedGrad.set(1, 1, 1.0);

            SimpleMatrix actualGrad = tensor.grad.data;

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    Assertions.assertEquals(expectedGrad.get(i, j), actualGrad.get(i, j), TOLERANCE,
                            "Gradient at (" + i + "," + j + ") is incorrect when summing along rows.");
                }
            }
        }

    }
}


