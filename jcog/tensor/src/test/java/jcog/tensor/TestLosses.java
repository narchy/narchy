package jcog.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestLosses {

    private static final double EPS = 1e-4; // Epsilon for loss and gradient comparisons

    // Helper to assert scalar tensor value
    private void assertScalarTensorEquals(double expected, Tensor actual, String message) {
        assertNotNull(actual, message + ": Tensor is null");
        assertTrue(actual.isScalar() || (actual.rows() == 1 && actual.cols() == 1),
                   message + ": Tensor is not scalar-like. Shape: " + actual.shapeStr());
        assertEquals(expected, actual.scalar(), EPS, message + ": Scalar value mismatch");
    }

    private void assertTensorShape(Tensor t, int rows, int cols, String msg) {
        assertNotNull(t, msg + ": Tensor is null");
        assertEquals(rows, t.rows(), msg + ": Mismatch in rows. Expected " + rows + ", got " + t.rows());
        assertEquals(cols, t.cols(), msg + ": Mismatch in cols. Expected " + cols + ", got " + t.cols());
    }

    private void assertArrayApproximatelyEquals(double[] expected, double[] actual, String messagePrefix) {
        assertEquals(expected.length, actual.length, messagePrefix + ": Array length mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], EPS, messagePrefix + ": Data mismatch at index " + i);
        }
    }

    // --- Tests for Losses.crossEntropyLoss ---

    @Test
    void testCrossEntropyLossBasic() {
        // Raw logits (pre-softmax)
        Tensor logits = Tensor.matrix(new double[][]{
            {1.0, 2.0, 1.5},  // Item 0
            {3.0, 1.0, 1.0}   // Item 1
        });
        Tensor targets = Tensor.matrix(new double[][]{{1}, {0}}); // Target class 1 for item 0, class 0 for item 1

        Tensor loss = Losses.crossEntropyLoss(logits, targets);

        // Calculate expected loss:
        // Item 0: logits [1.0, 2.0, 1.5] -> target class 1 (value 2.0)
        //   exps = [e^1.0, e^2.0, e^1.5] = [2.71828, 7.38906, 4.48169]
        //   sum_exps = 2.71828 + 7.38906 + 4.48169 = 14.58903
        //   probs = [0.18633, 0.50651, 0.30716]
        //   loss_item0 = -log(probs[1]) = -log(0.50651) = 0.68020
        double loss_item0 = -Math.log(Math.exp(2.0) / (Math.exp(1.0) + Math.exp(2.0) + Math.exp(1.5)));

        // Item 1: logits [3.0, 1.0, 1.0] -> target class 0 (value 3.0)
        //   exps = [e^3.0, e^1.0, e^1.0] = [20.08554, 2.71828, 2.71828]
        //   sum_exps = 20.08554 + 2.71828 + 2.71828 = 25.5221
        //   probs = [0.78700, 0.10650, 0.10650]
        //   loss_item1 = -log(probs[0]) = -log(0.78700) = 0.23963
        double loss_item1 = -Math.log(Math.exp(3.0) / (Math.exp(3.0) + Math.exp(1.0) + Math.exp(1.0)));

        double expectedMeanLoss = (loss_item0 + loss_item1) / 2.0;

        assertScalarTensorEquals(expectedMeanLoss, loss, "testCrossEntropyLossBasic");
    }

    @Test
    void testCrossEntropyLossPerfectPrediction() {
        Tensor logits = Tensor.matrix(new double[][]{{-10.0, 10.0}}); // Strongly predicts class 1
        Tensor targets = Tensor.matrix(new double[][]{{1}});
        Tensor loss = Losses.crossEntropyLoss(logits, targets);
        // Expected loss for -log(e^10 / (e^-10 + e^10)) = -log(1 / (e^-20 + 1)) approx -log(1) = 0
        assertScalarTensorEquals(0.0, loss, "testCrossEntropyLossPerfectPrediction");
    }

    @Test
    void testCrossEntropyLossWorstPrediction() {
        Tensor logits = Tensor.matrix(new double[][]{{10.0, -10.0}}); // Strongly predicts class 0, target is 1
        Tensor targets = Tensor.matrix(new double[][]{{1}});
        Tensor loss = Losses.crossEntropyLoss(logits, targets);
        // Expected: -log(e^-10 / (e^10 + e^-10)) = -log(e^-20 / (1 + e^-20)) approx -log(e^-20) = 20
        assertScalarTensorEquals(20.0, loss, "testCrossEntropyLossWorstPrediction");
    }

    @Test
    void testCrossEntropyLossGradient() {
        Tensor logits = Tensor.matrix(new double[][]{
            {1.0, 2.0, 1.5},
            {3.0, 1.0, 1.0}
        }).grad(true);
        Tensor targets = Tensor.matrix(new double[][]{{1}, {0}});

        Tensor loss = Losses.crossEntropyLoss(logits, targets);
        loss.minimize();

        assertNotNull(logits.grad, "Logits gradient should not be null");
        assertTensorShape(logits.grad, 2, 3, "Logits gradient shape");

        // Expected gradients: (softmax_output - one_hot_target) / N
        // N = 2 (batch size)
        // Item 0: logits [1.0, 2.0, 1.5], target_idx 1
        //   probs0 = [0.18633, 0.50651, 0.30716] (from basic test)
        //   one_hot0 = [0, 1, 0]
        //   grad0_row = [probs0[0]-0, probs0[1]-1, probs0[2]-0] / 2
        //             = [0.18633/2, (0.50651-1)/2, 0.30716/2]
        //             = [0.093165, -0.246745, 0.15358]
        double[] expectedGradRow0 = {0.093165, -0.246745, 0.15358};

        // Item 1: logits [3.0, 1.0, 1.0], target_idx 0
        //   probs1 = [0.78700, 0.10650, 0.10650] (from basic test)
        //   one_hot1 = [1, 0, 0]
        //   grad1_row = [probs1[0]-1, probs1[1]-0, probs1[2]-0] / 2
        //             = [(0.78700-1)/2, 0.10650/2, 0.10650/2]
        //             = [-0.1065, 0.05325, 0.05325]
        double[] expectedGradRow1 = {-0.1065, 0.05325, 0.05325};

        assertArrayApproximatelyEquals(expectedGradRow0, logits.grad.slice(0,1,0,3).array(), "Gradient row 0");
        assertArrayApproximatelyEquals(expectedGradRow1, logits.grad.slice(1,2,0,3).array(), "Gradient row 1");
    }

    // --- Tests for Losses.maskedCrossEntropyLoss ---

    @Test
    void testMaskedCrossEntropyLossBasic() {
        Tensor logits = Tensor.matrix(new double[][]{
            {1.0, 2.0, 1.5},  // Item 0, target 1, loss = 0.68020
            {3.0, 1.0, 1.0},  // Item 1, target 0, loss = 0.23963 (masked out)
            {1.0, 1.0, 2.5}   // Item 2, target 2
        });
        // Item 2: logits [1.0, 1.0, 2.5] -> target class 2 (value 2.5)
        //   exps = [e^1.0, e^1.0, e^2.5] = [2.71828, 2.71828, 12.18249]
        //   sum_exps = 2.71828 + 2.71828 + 12.18249 = 17.61905
        //   probs = [0.15428, 0.15428, 0.69144]
        //   loss_item2 = -log(probs[2]) = -log(0.69144) = 0.36900
        double loss_item0 = 0.68020;
        double loss_item2 = 0.36900;

        Tensor targets = Tensor.matrix(new double[][]{{1}, {0}, {2}});
        Tensor lossMask = Tensor.matrix(new double[][]{{1}, {0}, {1}}); // Mask out item 1

        Tensor loss = Losses.maskedCrossEntropyLoss(logits, targets, lossMask);

        double expectedMeanLoss = (loss_item0 + loss_item2) / 2.0; // Averaged over 2 active items
        assertScalarTensorEquals(expectedMeanLoss, loss, "testMaskedCrossEntropyLossBasic");
    }

    @Test
    void testMaskedCrossEntropyLossAllMasked() {
        Tensor logits = Tensor.matrix(new double[][]{{1.0, 2.0, 1.5}});
        Tensor targets = Tensor.matrix(new double[][]{{1}});
        Tensor lossMask = Tensor.matrix(new double[][]{{0}}); // Mask all items

        Tensor loss = Losses.maskedCrossEntropyLoss(logits, targets, lossMask);
        assertScalarTensorEquals(0.0, loss, "testMaskedCrossEntropyLossAllMasked");
    }

    @Test
    void testMaskedCrossEntropyLossNoMasking() {
        Tensor logits = Tensor.matrix(new double[][]{
            {1.0, 2.0, 1.5},
            {3.0, 1.0, 1.0}
        });
        Tensor targets = Tensor.matrix(new double[][]{{1}, {0}});
        Tensor lossMask = Tensor.matrix(new double[][]{{1}, {1}}); // No masking

        Tensor lossNoMask = Losses.maskedCrossEntropyLoss(logits, targets, lossMask);
        Tensor lossPlain = Losses.crossEntropyLoss(logits, targets); // Calculate with non-masked version

        assertScalarTensorEquals(lossPlain.scalar(), lossNoMask, "testMaskedCrossEntropyLossNoMasking");
    }

    @Test
    void testMaskedCrossEntropyLossGradient() {
        Tensor logits = Tensor.matrix(new double[][]{
            {1.0, 2.0, 1.5},  // Item 0, target 1. Grads: [0.093165, -0.246745, 0.15358] (scaled by 1/num_active)
            {3.0, 1.0, 1.0},  // Item 1, target 0. (Masked out)
            {1.0, 1.0, 2.5}   // Item 2, target 2. Grads: [p0/N_act, p1/N_act, (p2-1)/N_act]
        }).grad(true);
        Tensor targets = Tensor.matrix(new double[][]{{1}, {0}, {2}});
        Tensor lossMask = Tensor.matrix(new double[][]{{1}, {0}, {1}}); // Mask out item 1. num_active = 2.

        Tensor loss = Losses.maskedCrossEntropyLoss(logits, targets, lossMask);
        loss.minimize();

        assertNotNull(logits.grad, "Logits gradient should not be null");
        assertTensorShape(logits.grad, 3, 3, "Logits gradient shape");

        // Expected gradients for row 0 (same as crossEntropyLoss but scaled by 1/num_active=1/2)
        double[] expectedGradRow0 = {0.18633/2.0, (0.50651-1)/2.0, 0.30716/2.0};
        assertArrayApproximatelyEquals(expectedGradRow0, logits.grad.slice(0,1,0,3).array(), "Gradient row 0 (masked)");

        // Expected gradients for row 1 (masked item) should be all zeros
        double[] expectedGradRow1_masked = {0.0, 0.0, 0.0};
        assertArrayApproximatelyEquals(expectedGradRow1_masked, logits.grad.slice(1,2,0,3).array(), "Gradient row 1 (masked)");

        // Expected gradients for row 2 (Item 2: target 2)
        // probs2 = [0.15428, 0.15428, 0.69144]
        // one_hot2 = [0,0,1]
        // grad2_row = [probs2[0]-0, probs2[1]-0, probs2[2]-1] / num_active
        //           = [0.15428/2, 0.15428/2, (0.69144-1)/2]
        //           = [0.07714, 0.07714, -0.15428]
        double[] expectedGradRow2 = {0.07714, 0.07714, -0.15428};
        assertArrayApproximatelyEquals(expectedGradRow2, logits.grad.slice(2,3,0,3).array(), "Gradient row 2 (masked)");
    }
}
