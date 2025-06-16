package jcog.tensor;

import org.ejml.simple.SimpleMatrix;

public class Losses {

    /**
     * Computes the cross-entropy loss for single label targets.
     *
     * @param logits  Input tensor of shape [num_items, num_classes] containing raw scores (pre-softmax).
     *                num_items could be sequence_length for token classification or 1 for sequence classification.
     * @param targets Input tensor of shape [num_items, 1] or [num_items] containing integer class labels.
     * @return Scalar Tensor representing the mean cross-entropy loss.
     */
    public static Tensor crossEntropyLoss(Tensor logits, Tensor targets) {
        int numItems = logits.rows();
        int numClasses = logits.cols();

        if (numItems == 0) {
            return Tensor.scalar(0.0).grad(logits.hasGrad());
        }

        if (targets.volume() != numItems) {
            throw new IllegalArgumentException("Number of items in logits (" + numItems +
                                               ") must match number of items in targets (" + targets.volume() + ").");
        }
        if (targets.rows() != numItems && targets.cols() != 1 && numItems > 0) {
             throw new IllegalArgumentException("Targets tensor shape must be [num_items, 1] or [num_items]. Got: " + targets.shapeStr());
        }


        // 1. Apply LogSoftmax to logits
        // Using softmax().log() for now. Consider numerical stability: add epsilon if softmax can be exact zero.
        // Tensor.softmax() is expected to be numerically stable (subtracts max before exp).
        Tensor probabilities = logits.softmax(); // Numerically stable softmax, shape [num_items, num_classes]

        // Add a small epsilon to prevent log(0) if softmax can output exact zeros.
        // However, a well-behaved softmax should not output exact zero unless input was -inf.
        // Let's assume softmax output > 0 for simplicity here.
        // If issues arise, use: probabilities = probabilities.add(1e-9); // Add epsilon
        Tensor logProbabilities = probabilities.log(); // Shape [num_items, num_classes]

        // 2. Create one-hot encoded targets
        Tensor oneHotTargets = Tensor.zeros(numItems, numClasses);
        double[] oneHotData = oneHotTargets.array();
        double[] targetsData = targets.array();

        for (int i = 0; i < numItems; i++) {
            int targetClassIdx = (int) targetsData[i * targets.cols()]; // Assumes targets is [N,1] or [N] effectively
            if (targetClassIdx < 0 || targetClassIdx >= numClasses) {
                throw new IllegalArgumentException("Target class index " + targetClassIdx +
                                                   " at item " + i + " is out of bounds for numClasses " + numClasses);
            }
            oneHotData[i * numClasses + targetClassIdx] = 1.0;
        }
        // Make oneHotTargets require grad if logProbabilities does, for chain rule continuity, though it's not strictly necessary
        // as its values are fixed. The mul op will handle it.
        // oneHotTargets.grad(logProbabilities.hasGrad()); // Not strictly needed as values are 0 or 1

        // 3. Select the log probabilities of the true classes
        // Element-wise multiply logProbabilities with oneHotTargets
        // Then sum along the class dimension to get the log probability for the true class for each item
        Tensor selectedLogProbs = logProbabilities.mul(oneHotTargets).sum(1); // Sum along dim 1 (classes), result shape [num_items, 1]

        // 4. Compute Negative Log Likelihood losses
        Tensor nllLosses = selectedLogProbs.neg(); // Shape [num_items, 1]

        // 5. Compute the mean loss
        // Tensor.mean() computes mean of all elements. If nllLosses is [N,1], this is fine.
        Tensor meanLoss = nllLosses.mean();

        return meanLoss;
    }

    /**
     * Computes the masked cross-entropy loss for single label targets.
     *
     * @param logits    Input tensor of shape [sequence_length, num_classes].
     * @param targets   Input tensor of shape [sequence_length, 1] or [sequence_length] for class labels.
     * @param lossMask  Boolean tensor of shape [sequence_length, 1] or [sequence_length].
     *                  1.0 where loss should be computed, 0.0 where it should be ignored.
     * @return Scalar Tensor representing the mean cross-entropy loss over the masked elements.
     */
    public static Tensor maskedCrossEntropyLoss(Tensor logits, Tensor targets, Tensor lossMask) {
        int sequenceLength = logits.rows();
        int numClasses = logits.cols();

        if (sequenceLength == 0) {
            return Tensor.scalar(0.0).grad(logits.hasGrad());
        }

        if (targets.volume() != sequenceLength || lossMask.volume() != sequenceLength) {
            throw new IllegalArgumentException("Logits, targets, and lossMask must have the same number of items (sequence_length). " +
                                               "logits: " + sequenceLength + ", targets: " + targets.volume() + ", lossMask: " + lossMask.volume());
        }

        // 1. Calculate LogSoftmax
        Tensor probabilities = logits.softmax();
        Tensor logProbabilities = probabilities.log(); // Shape [sequence_length, num_classes]

        // 2. Create one-hot encoded targets
        Tensor oneHotTargets = Tensor.zeros(sequenceLength, numClasses);
        double[] oneHotData = oneHotTargets.array();
        double[] targetsData = targets.array();

        for (int i = 0; i < sequenceLength; i++) {
            int targetClassIdx = (int) targetsData[i * targets.cols()]; // Assumes targets is [N,1] or [N]
             if (targetClassIdx < 0 || targetClassIdx >= numClasses) {
                throw new IllegalArgumentException("Target class index " + targetClassIdx +
                                                   " at item " + i + " is out of bounds for numClasses " + numClasses);
            }
            oneHotData[i * numClasses + targetClassIdx] = 1.0;
        }
        // oneHotTargets.grad(logProbabilities.hasGrad()); // Not strictly needed

        // 3. Select log probabilities of true classes
        Tensor selectedLogProbs = logProbabilities.mul(oneHotTargets).sum(1); // Shape [sequence_length, 1]

        // 4. Compute Negative Log Likelihood per token
        Tensor perTokenNllLosses = selectedLogProbs.neg(); // Shape [sequence_length, 1]

        // 5. Apply the mask
        // Ensure lossMask is [sequence_length, 1] if perTokenNllLosses is [sequence_length, 1] for broadcasting with mul.
        // If lossMask is [sequence_length], it needs reshaping or careful element-wise mul.
        // Tensor.mul should handle broadcasting if shapes are compatible (e.g. [N,1] * [N] might not work, but [N,1] * [N,1] does)
        // Let's ensure mask is [N,1]
        Tensor reshapedLossMask = lossMask.rows() == sequenceLength && lossMask.cols() == 1 ?
                                  lossMask :
                                  new Tensor(new SimpleMatrix(sequenceLength, 1, true, lossMask.array()), lossMask.hasGrad());


        Tensor maskedLosses = perTokenNllLosses.mul(reshapedLossMask); // Element-wise multiplication

        // 6. Compute sum of masked losses
        Tensor sumMaskedLosses = maskedLosses.sum(); // Sum of all elements in maskedLosses

        // 7. Compute number of active elements
        Tensor numActiveElements = reshapedLossMask.sum();

        double numActiveScalar = numActiveElements.scalar();

        if (numActiveScalar == 0) {
            // Return a zero tensor that requires grad if logits did, to allow gradient flow if this loss is part of a larger computation
            return Tensor.scalar(0.0).grad(logits.hasGrad());
        }

        // 8. Compute mean loss
        // sumMaskedLosses is a scalar Tensor, numActiveScalar is a double.
        // To maintain Tensor operations for backprop, divide Tensor by scalar.
        Tensor meanLoss = sumMaskedLosses.div(numActiveScalar);

        return meanLoss;
    }
}
