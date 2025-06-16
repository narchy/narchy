package jcog.tensor;

import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.NotNull;

public class TokenEmbedding {

    public final Tensor weight;
    public final int vocabSize;
    public final int embeddingDim;

    public TokenEmbedding(int vocabSize, int embeddingDim, boolean requiresGrad) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        // Initialize weight tensor
        this.weight = Tensor.randGaussian(vocabSize, embeddingDim, 0.02);
        if (requiresGrad) {
            this.weight.grad(true).parameter();
        }
    }

    public Tensor forward(@NotNull Tensor tokenIds) {
        // Validate tokenIds shape: must be 1D (row or column vector) or empty
        if (tokenIds.rows() != 1 && tokenIds.cols() != 1 && tokenIds.volume() > 0) {
            throw new IllegalArgumentException("tokenIds must be a 1D tensor (either a column or a row vector) or empty. Shape found: " + tokenIds.shapeStr());
        }

        int numTokens = tokenIds.volume();

        if (numTokens == 0) {
            return Tensor.zeros(0, this.embeddingDim).grad(this.weight.hasGrad());
        }

        // Data for the output tensor
        SimpleMatrix outputMatrixData = new SimpleMatrix(numTokens, this.embeddingDim);
        double[] outputArray = Tensor.array(outputMatrixData);

        // Store token IDs in a primitive array for stable access in the backward pass
        final int[] idArray = new int[numTokens];
        double[] tokenIdsData = tokenIds.array(); // Get underlying data array of tokenIds

        for (int i = 0; i < numTokens; i++) {
            int tokenId = (int) tokenIdsData[i];
            idArray[i] = tokenId; // Store for backward pass

            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IllegalArgumentException("Token ID " + tokenId + " at index " + i + " is out of bounds for vocab size " + vocabSize + ".");
            }

            // Source position in weight.array(): tokenId * embeddingDim
            // Destination position in outputArray: i * embeddingDim
            System.arraycopy(this.weight.array(), tokenId * this.embeddingDim, outputArray, i * this.embeddingDim, this.embeddingDim);
        }

        Tensor result = new Tensor(outputMatrixData, this.weight.hasGrad());

        if (result.hasGrad()) {
            // Set up the operation for gradient propagation
            result.op = new TensorOp(this.weight) { // The weight Tensor is the parent
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    // grad: gradient of the loss with respect to the output of this forward method (shape [numTokens, embeddingDim])
                    // gradOut[0]: gradient accumulator for this.weight (shape [vocabSize, embeddingDim])

                    if (gradOut[0] != null) { // Ensure gradient accumulation is needed for the weight
                        SimpleMatrix weightGradMatrix = gradOut[0];
                        double[] weightGradArray = Tensor.array(weightGradMatrix); // Underlying data array for weight's gradient
                        double[] incomingGradArray = Tensor.array(grad); // Underlying data array for incoming gradient

                        for (int i = 0; i < numTokens; i++) {
                            int currentTokenId = idArray[i]; // Get the original token ID

                            // Offset in incomingGradArray for the i-th token's gradient
                            int sourceOffsetIncomingGrad = i * embeddingDim;
                            // Offset in weightGradArray for the currentTokenId's row
                            int targetOffsetWeightGrad = currentTokenId * embeddingDim;

                            // Accumulate the gradient
                            for (int j = 0; j < embeddingDim; j++) {
                                weightGradArray[targetOffsetWeightGrad + j] += incomingGradArray[sourceOffsetIncomingGrad + j];
                            }
                        }
                    }
                }
            };
        }
        return result;
    }
}
