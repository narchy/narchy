package jcog.tensor;

import org.ejml.simple.SimpleMatrix; // Added for manualBackward
import org.junit.jupiter.api.Test;

import java.util.Comparator; // Added for PriorityQueue
import java.util.IdentityHashMap;
import java.util.PriorityQueue;

import static org.junit.jupiter.api.Assertions.*;


public class TensorTest {

    @Test
    void testScalarConstructor() {
        Tensor t = new Tensor(5.0, false);
        assertEquals(1, t.rows());
        assertEquals(1, t.cols());
        assertEquals(5.0, t.scalar(), 1e-6);
        assertFalse(t.hasGrad());
    }

    @Test
    void testArrayConstructor() {
        double[] data = {1.0, 2.0, 3.0, 4.0};
        Tensor t = new Tensor(data, 2, 2, true);
        assertEquals(2, t.rows());
        assertEquals(2, t.cols());
        assertArrayEquals(data, t.array(), 1e-6);
        assertTrue(t.hasGrad());
        assertNotNull(t.grad);
    }

    @Test
    void test2DArrayConstructor() {
        double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
        Tensor t = new Tensor(data, false);
        assertEquals(2, t.rows());
        assertEquals(2, t.cols());
        assertEquals(1.0, t.data(0, 0), 1e-6);
        assertEquals(2.0, t.data(0, 1), 1e-6);
        assertEquals(3.0, t.data(1, 0), 1e-6);
        assertEquals(4.0, t.data(1, 1), 1e-6);
        assertFalse(t.hasGrad());
    }

    @Test
    void testZerosFactory() {
        Tensor t = Tensor.zeros(2, 3);
        assertEquals(2, t.rows());
        assertEquals(3, t.cols());
        for (double val : t.array()) {
            assertEquals(0.0, val, 1e-6);
        }
        assertFalse(t.hasGrad());
    }

    @Test
    void testOnesFactory() {
        Tensor t = Tensor.ones(3, 2);
        assertEquals(3, t.rows());
        assertEquals(2, t.cols()); // Corrected
        for (double val : t.array()) {
            assertEquals(1.0, val, 1e-6);
        }
        assertFalse(t.hasGrad());
    }

    @Test
    void testRandGaussianFactory() {
        Tensor t = Tensor.randGaussian(2, 2, 1.0);
        assertEquals(2, t.rows());
        assertEquals(2, t.cols());
        // Hard to test exact values, but check shape and that values are populated
        assertNotNull(t.array());
        assertFalse(t.hasGrad());
    }

    @Test
    void testScalarStaticFactory() {
        Tensor t = Tensor.scalar(10.0);
        assertEquals(1, t.rows());
        assertEquals(1, t.cols());
        assertEquals(10.0, t.scalar(), 1e-6);
        assertFalse(t.hasGrad());
    }

    @Test
    void testRowStaticFactory() {
        Tensor t = Tensor.row(1.0, 2.0, 3.0);
        assertEquals(1, t.rows());
        assertEquals(3, t.cols());
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, t.array(), 1e-6);
        assertFalse(t.hasGrad());
    }

    @Test
    void testMatrixStaticFactory() {
        double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
        Tensor t = Tensor.matrix(data);
        assertEquals(2, t.rows());
        assertEquals(2, t.cols());
        assertEquals(1.0, t.data(0, 0), 1e-6);
        assertEquals(4.0, t.data(1, 1), 1e-6);
        assertFalse(t.hasGrad());
    }

    // Helper to check if all elements in a tensor are close to a specific value
    private void assertTensorEqualsValue(Tensor t, double value, double delta) {
        for (double val : t.array()) {
            assertEquals(value, val, delta);
        }
    }

    // Helper method to manually trigger backward pass for testing
    private void manualBackward(Tensor resultTensor, double rootGradValue) {
        if (!resultTensor.hasGrad() && resultTensor.op == null) {
            return;
        }
        if (resultTensor.grad == null) {
            resultTensor.grad = Tensor.zerosShaped(resultTensor);
        }
        resultTensor.grad.fill(rootGradValue);

        // The following is a simplified direct invocation for testing purposes,
        // assuming Tensor.op and Tensor.op.parents are accessible (e.g., package-private).
        // This does NOT replicate the full behavior of a graph-based backward pass
        // with topological sort as would happen with minimize() or a full .backward() call.
        if (resultTensor.op != null) {
            SimpleMatrix[] parentGrads = resultTensor.op.allocateParentGrads();
            resultTensor.op.backward(resultTensor.grad.data, parentGrads);
            for (int i = 0; i < resultTensor.op.parents.length; i++) {
                Tensor parent = resultTensor.op.parents[i];
                if (parentGrads[i] != null && parent.hasGrad()) {
                    if (parent.grad == null) {
                        parent.grad = Tensor.zerosShaped(parent);
                    }
                    // Accumulate gradients
                    double[] pg_arr = parent.grad.array();
                    double[] add_grad_arr = Tensor.array(parentGrads[i]); // Ensure this is not null
                    if (add_grad_arr != null) {
                         for(int j=0; j<pg_arr.length; j++) {
                            pg_arr[j] += add_grad_arr[j];
                        }
                    }
                }
            }
        } else if (resultTensor.parameter && resultTensor.grad != null) {
            // It's a parameter, its own grad was set, nothing else to do
        }
    }


    @Test
    void testAddScalar() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor result = a.add(5.0);
        assertArrayEquals(new double[]{6, 7, 8, 9}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1}, a.grad.array(), 1e-6);
    }

    @Test
    void testAddTensor() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor b = new Tensor(new double[][]{{5, 6}, {7, 8}}, true);
        Tensor result = a.add(b);
        assertArrayEquals(new double[]{6, 8, 10, 12}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertNotNull(b.grad, "Gradient for 'b' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1}, a.grad.array(), 1e-6);
        assertArrayEquals(new double[]{1, 1, 1, 1}, b.grad.array(), 1e-6);
    }

    @Test
    void testSubScalar() {
        Tensor a = new Tensor(new double[][]{{10, 20}, {30, 40}}, true);
        Tensor result = a.sub(5.0);
        assertArrayEquals(new double[]{5, 15, 25, 35}, result.array(), 1e-6);

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1}, a.grad.array(), 1e-6);
    }

    @Test
    void testSubTensor() {
        Tensor a = new Tensor(new double[][]{{10, 20}, {30, 40}}, true);
        Tensor b = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor result = a.sub(b);
        assertArrayEquals(new double[]{9, 18, 27, 36}, result.array(), 1e-6);

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertNotNull(b.grad, "Gradient for 'b' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1}, a.grad.array(), 1e-6);
        assertArrayEquals(new double[]{-1, -1, -1, -1}, b.grad.array(), 1e-6);
    }

    @Test
    void testMulScalar() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor result = a.mul(3.0);
        assertArrayEquals(new double[]{3, 6, 9, 12}, result.array(), 1e-6);

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{3, 3, 3, 3}, a.grad.array(), 1e-6);
    }

    @Test
    void testMulTensorElementWise() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true);
        Tensor b = new Tensor(new double[][]{{5, 6}, {7, 8}}, true);
        Tensor result = a.mul(b); // Element-wise multiplication
        assertArrayEquals(new double[]{5, 12, 21, 32}, result.array(), 1e-6);

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertNotNull(b.grad, "Gradient for 'b' should not be null");
        assertArrayEquals(new double[]{5, 6, 7, 8}, a.grad.array(), 1e-6);
        assertArrayEquals(new double[]{1, 2, 3, 4}, b.grad.array(), 1e-6);
    }

    @Test
    void testDivScalar() {
        Tensor a = new Tensor(new double[][]{{10, 20}, {30, 40}}, true);
        Tensor result = a.div(2.0);
        assertArrayEquals(new double[]{5, 10, 15, 20}, result.array(), 1e-6);

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{0.5, 0.5, 0.5, 0.5}, a.grad.array(), 1e-6);
    }

    @Test
    void testDivTensorElementWise() {
        Tensor a = new Tensor(new double[][]{{10, 20}, {21, 32}}, true);
        Tensor b = new Tensor(new double[][]{{2, 5}, {3, 4}}, true);
        Tensor result = a.div(b); // Element-wise division
        assertArrayEquals(new double[]{5, 4, 7, 8}, result.array(), 1e-6);

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertNotNull(b.grad, "Gradient for 'b' should not be null");
        assertArrayEquals(new double[]{1.0/2.0, 1.0/5.0, 1.0/3.0, 1.0/4.0}, a.grad.array(), 1e-6);
        assertArrayEquals(new double[]{-10.0/(2*2), -20.0/(5*5), -21.0/(3*3), -32.0/(4*4)}, b.grad.array(), 1e-6);
    }

    @Test
    void testAddBroadcast() {
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // 2x3
        Tensor b = new Tensor(new double[]{10, 20, 30}, true); // 1x3 (row vector)
        Tensor result = a.add(b);
        assertArrayEquals(new double[]{11, 22, 33, 14, 25, 36}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertNotNull(b.grad, "Gradient for 'b' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.grad.array(), 1e-6);
        // Corrected expected gradient for b after broadcasted add:
        // Each element of b contributes to one column of 'a'.
        // So, grad for b[0] is sum of grads for result[0,0] and result[1,0] etc.
        // Since root grad is 1.0 for all elements of result, b.grad should be {2,2,2}
        assertArrayEquals(new double[]{2, 2, 2}, b.grad.array(), 1e-6);
    }

    @Test
    void testMatmul() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}}, true); // 2x2
        Tensor b = new Tensor(new double[][]{{5, 6, 7}, {8, 9, 10}}, true); // 2x3
        Tensor result = a.matmul(b); // Expected: 2x3

        assertEquals(2, result.rows());
        assertEquals(3, result.cols());
        double[] expectedData = {
            1*5 + 2*8, 1*6 + 2*9, 1*7 + 2*10, // row 1
            3*5 + 4*8, 3*6 + 4*9, 3*7 + 4*10  // row 2
            // 5+16=21, 6+18=24, 7+20=27
            // 15+32=47, 18+36=54, 21+40=61
        };
        assertArrayEquals(new double[]{21, 24, 27, 47, 54, 61}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        // grad_A = grad_res * B^T
        // grad_B = A^T * grad_res
        // grad_res is all 1s, same shape as result (2x3)
        // B^T is 3x2: {{5,8},{6,9},{7,10}}
        // A^T is 2x2: {{1,3},{2,4}}

        // grad_A (2x2) = [[1,1,1],[1,1,1]] (2x3) * [[5,8],[6,9],[7,10]] (3x2)
        // = [ [1*5+1*6+1*7, 1*8+1*9+1*10],
        //     [1*5+1*6+1*7, 1*8+1*9+1*10] ]
        // = [ [18, 27], [18, 27] ]
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{18, 27, 18, 27}, a.grad.array(), 1e-6);

        // grad_B (2x3) = [[1,3],[2,4]] (2x2) * [[1,1,1],[1,1,1]] (2x3)
        // = [ [1*1+3*1, 1*1+3*1, 1*1+3*1],
        //     [2*1+4*1, 2*1+4*1, 2*1+4*1] ]
        // = [ [4, 4, 4], [6, 6, 6] ]
        assertNotNull(b.grad, "Gradient for 'b' should not be null");
        assertArrayEquals(new double[]{4, 4, 4, 6, 6, 6}, b.grad.array(), 1e-6);
    }

    @Test
    void testTranspose() {
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // 2x3
        Tensor result = a.transpose(); // Expected: 3x2

        assertEquals(3, result.rows());
        assertEquals(2, result.cols());
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, result.array(), 1e-6); // column-major order for result.array()
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        // grad_A = (grad_res)^T
        // grad_res is 3x2, all 1s. So (grad_res)^T is 2x3, all 1s.
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.grad.array(), 1e-6);
    }

    @Test
    void testRelu() {
        Tensor a = new Tensor(new double[][]{{1, -2, 0}, {-4, 5, -0.5}}, true);
        Tensor result = a.relu();
        assertArrayEquals(new double[]{1, 0, 0, 0, 5, 0}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        // grad_A_i = grad_res_i if a_i >= 0 else 0.
        // Since derivative of relu(x) for x=0 is 1 in current impl, grad for input 0 should be 1.
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.grad.array(), 1e-6);
    }

    @Test
    void testSigmoid() {
        Tensor a = new Tensor(new double[]{0.0, 1.0, -1.0}, true); // 1x3
        // sigmoid(0) = 0.5
        // sigmoid(1) = 1 / (1 + exp(-1)) approx 0.7310585
        // sigmoid(-1) = 1 / (1 + exp(1)) approx 0.2689414
        double sig1 = 1.0 / (1.0 + Math.exp(-1.0));
        double sig_neg1 = 1.0 / (1.0 + Math.exp(1.0));

        Tensor result = a.sigmoid();
        assertEquals(1, result.rows());
        assertEquals(3, result.cols());
        assertArrayEquals(new double[]{0.5, sig1, sig_neg1}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0); // root_grad = 1.0
        // grad_A_i = grad_res_i * sigmoid(a_i) * (1 - sigmoid(a_i))
        // grad_A_i = 1.0 * result_i * (1 - result_i)
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{
            0.5 * (1-0.5),
            sig1 * (1-sig1),
            sig_neg1 * (1-sig_neg1)
        }, a.grad.array(), 1e-6);
    }

    @Test
    void testTanh() {
        Tensor a = new Tensor(new double[]{0.0, 1.0, -1.0}, true); // 1x3
        // tanh(0) = 0
        // tanh(1) approx 0.7615941
        // tanh(-1) approx -0.7615941
        double tanh1 = Math.tanh(1.0);
        double tanh_neg1 = Math.tanh(-1.0);

        Tensor result = a.tanh();
        assertEquals(1, result.rows());
        assertEquals(3, result.cols());
        assertArrayEquals(new double[]{0.0, tanh1, tanh_neg1}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0); // root_grad = 1.0
        // grad_A_i = grad_res_i * (1 - tanh(a_i)^2)
        // grad_A_i = 1.0 * (1 - result_i^2)
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{
            1.0 - (0.0*0.0),
            1.0 - (tanh1*tanh1),
            1.0 - (tanh_neg1*tanh_neg1)
        }, a.grad.array(), 1e-6);
    }

    @Test
    void testReluLeaky() {
        Tensor a = new Tensor(new double[][]{{1, -2, 0}, {-4, 5, -0.5}}, true);
        float alpha = 0.1f;
        Tensor result = a.reluLeaky(alpha);
        assertArrayEquals(new double[]{1, -2*alpha, 0*alpha, -4*alpha, 5, -0.5*alpha}, result.array(), 1e-6); // Corrected 0*alpha for output
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        // grad_A_i = grad_res_i if a_i >= 0 else grad_res_i * alpha
        // If a_i = 0, derivative is 1.
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, alpha, 1, alpha, 1, alpha}, a.grad.array(), 1e-6);
    }

    @Test
    void testElu() {
        Tensor a = new Tensor(new double[]{0.0, 1.0, -1.0, -2.0}, true);
        double alpha = 1.0; // Standard ELU alpha
        // elu(0) = 0
        // elu(1) = 1
        // elu(-1) = 1.0 * (exp(-1) - 1) approx -0.63212
        // elu(-2) = 1.0 * (exp(-2) - 1) approx -0.86466
        double elu_neg1 = alpha * (Math.exp(-1.0) - 1.0);
        double elu_neg2 = alpha * (Math.exp(-2.0) - 1.0);

        Tensor result = a.elu(alpha);
        assertArrayEquals(new double[]{0.0, 1.0, elu_neg1, elu_neg2}, result.array(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0);
        // grad_A_i = grad_res_i if a_i > 0
        // grad_A_i = grad_res_i * (alpha * exp(a_i)) if a_i <= 0
        // The provided op derivative in Tensor.java is: x > 0 ? 1 : alpha * Math.exp(x)
        // So for x=0, derivative is alpha * exp(0) = alpha.
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{
            alpha * Math.exp(0.0), // for input 0.0
            1.0,                   // for input 1.0
            alpha * Math.exp(-1.0),// for input -1.0
            alpha * Math.exp(-2.0) // for input -2.0
        }, a.grad.array(), 1e-6);
    }

    @Test
    void testSumAllElements() {
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // sum = 21
        Tensor result = a.sum();
        assertEquals(1, result.rows());
        assertEquals(1, result.cols());
        assertEquals(21.0, result.scalar(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0); // root_grad = 1.0
        // grad_A_ij = grad_res (which is 1.0)
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.grad.array(), 1e-6);
    }

    @Test
    void testSumCols() { // sum(boolean rowsOrCols) -> sum(true)
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // 2x3
        Tensor result = a.sum(true); // Sum along columns, result 1x3
        assertEquals(1, result.rows());
        assertEquals(3, result.cols());
        assertArrayEquals(new double[]{1+4, 2+5, 3+6}, result.array(), 1e-6); // {5, 7, 9}
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0); // root_grad for each element of result is 1.0
        // Each element in 'a' contributes to one sum in 'result', so grad is 1.
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.grad.array(), 1e-6);
    }

    @Test
    void testSumRows() { // sum(boolean rowsOrCols) -> sum(false)
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // 2x3
        Tensor result = a.sum(false); // Sum along rows, result 2x1
        assertEquals(2, result.rows());
        assertEquals(1, result.cols());
        assertArrayEquals(new double[]{1+2+3, 4+5+6}, result.array(), 1e-6); // {6, 15}
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0); // root_grad for each element of result is 1.0
        // Each element in 'a' contributes to one sum in 'result', so grad is 1.
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.grad.array(), 1e-6);
    }

    @Test
    void testMeanAllElements() {
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}}, true); // sum = 21, count = 6, mean = 3.5
        Tensor result = a.mean();
        assertEquals(1, result.rows());
        assertEquals(1, result.cols());
        assertEquals(3.5, result.scalar(), 1e-6);
        assertTrue(result.hasGrad());

        manualBackward(result, 1.0); // root_grad = 1.0
        // grad_A_ij = grad_res / N = 1.0 / 6
        assertNotNull(a.grad, "Gradient for 'a' should not be null");
        double expected_grad_val = 1.0 / 6.0;
        assertArrayEquals(new double[]{expected_grad_val, expected_grad_val, expected_grad_val, expected_grad_val, expected_grad_val, expected_grad_val}, a.grad.array(), 1e-6);
    }
}
