package jcog.tensor;

import java.util.ArrayList;
import java.util.List;

public class FactorizedTokenEmbedding {

    private final TokenEmbedding_E embeddingLayerE;
    private final Models.Linear projectionLayerH;
    private final int vocabSize;
    private final int embeddingDimE;
    private final int hiddenDimH;

    /**
     * Helper class for the first stage of embedding: vocab -> embeddingDimE.
     */
    private static class TokenEmbedding_E {
        private final Tensor weights; // Shape [vocabSize, embeddingDimE]
        private final int vocabSize;
        private final int embeddingDimE;

        /**
         * Constructor for TokenEmbedding_E.
         * @param vocabSize Size of the vocabulary.
         * @param embeddingDimE Dimension of the embedding space (E).
         * @param requiresGrad If true, weights will track gradients and be parameters.
         */
        public TokenEmbedding_E(int vocabSize, int embeddingDimE, boolean requiresGrad) {
            this.vocabSize = vocabSize;
            this.embeddingDimE = embeddingDimE;
            // Initialize weights, e.g., with Xavier/Glorot initialization for embeddings
            // Scale is often sqrt(1.0 / embeddingDimE) or similar.
            // Small random values are generally good.
            double initScale = Math.sqrt(1.0 / embeddingDimE);
            this.weights = Tensor.randGaussian(vocabSize, embeddingDimE, initScale);
            if (requiresGrad) {
                this.weights.grad(true).parameter();
            }
        }

        /**
         * Performs embedding lookup.
         * @param inputIds Tensor of token IDs, shape [sequenceLength, 1] or [sequenceLength].
         * @return Tensor of embeddings, shape [sequenceLength, embeddingDimE].
         */
        public Tensor forward(Tensor inputIds) {
            int sequenceLength = inputIds.rows();
            if (inputIds.cols() != 1 && sequenceLength == 1 && inputIds.cols() > 1) {
                // If inputIds is [1, sequenceLength], treat cols as sequence if rows is 1
                sequenceLength = inputIds.cols();
            } else if (inputIds.cols() != 1) {
                // Or if it's a true 1D tensor represented as [N] (cols might be N, rows 1)
                 if (inputIds.rows() == 1 && inputIds.isVector()) sequenceLength = inputIds.volume();
                 else if (inputIds.cols() != 1)
                    throw new IllegalArgumentException("inputIds must be a column vector [sequenceLength, 1] or a flat vector.");
            }


            List<Tensor> embeddedRows = new ArrayList<>(sequenceLength);
            for (int i = 0; i < sequenceLength; i++) {
                // Assuming inputIds.get(i) or inputIds.get(i,0) gives the token ID scalar
                // and Tensor.get(row, col) for scalar, or need to handle 1D tensor access.
                // Let's assume inputIds.getAsDouble(row, col) or similar for scalar access.
                int tokenId;
                if (inputIds.cols() == 1) {
                    tokenId = (int) inputIds.getAsDouble(i, 0);
                } else { // Assumed to be a flat vector [1, sequenceLength] or [sequenceLength]
                    tokenId = (int) inputIds.getAsDouble(0, i);
                }

                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IllegalArgumentException("Token ID " + tokenId + " at index " + i + " is out of vocab bounds [0, " + (vocabSize - 1) + "]");
                }
                // Slice the row corresponding to the token ID from the weights matrix.
                // slice(rowStart, rowEnd, colStart, colEnd) -> rowEnd and colEnd are exclusive
                Tensor row = this.weights.slice(tokenId, tokenId + 1, 0, this.embeddingDimE);
                embeddedRows.add(row);
            }

            if (embeddedRows.isEmpty()) {
                return Tensor.zeros(0, this.embeddingDimE); // Or handle as error
            }

            // Concatenate the list of row tensors.
            // If each row is [1, embeddingDimE], concatenating them should result in [sequenceLength, embeddingDimE].
            // This assumes Tensor.concat can combine a list of row vectors into a single 2D tensor.
            return Tensor.concatRows(embeddedRows.toArray(new Tensor[0]));
        }
    }

    /**
     * Constructor for FactorizedTokenEmbedding.
     * @param vocabSize      Size of the vocabulary.
     * @param embeddingDimE  Dimension of the intermediate embedding layer (E).
     * @param hiddenDimH     Dimension of the final output hidden layer (H, e.g., dModel).
     * @param requiresGrad   If true, weights of embedding and projection layers will track gradients.
     */
    public FactorizedTokenEmbedding(int vocabSize, int embeddingDimE, int hiddenDimH, boolean requiresGrad) {
        this.vocabSize = vocabSize;
        this.embeddingDimE = embeddingDimE;
        this.hiddenDimH = hiddenDimH;

        this.embeddingLayerE = new TokenEmbedding_E(vocabSize, embeddingDimE, requiresGrad);
        // Models.Linear constructor: (inFeatures, outFeatures, activation, bias)
        // Bias is enabled (true), no activation (null) for the projection.
        this.projectionLayerH = new Models.Linear(embeddingDimE, hiddenDimH, null, true);

        // Models.Linear already marks its weights and bias as parameters requiring grad
        // if they are initialized that way (which they are by default).
        // Explicitly setting requiresGrad on projectionLayerH's parameters is redundant
        // if 'requiresGrad' for FactorizedTokenEmbedding implies all sub-components are grad-enabled.
        // If finer control is needed, one might pass 'requiresGrad' to Models.Linear constructor
        // if it supported it, or set it manually like:
        // if (requiresGrad) {
        //     this.projectionLayerH.weight.grad(true).parameter();
        //     if (this.projectionLayerH.bias != null) {
        //         this.projectionLayerH.bias.grad(true).parameter();
        //     }
        // }
        // For now, relying on Models.Linear internal setup.
    }

    /**
     * Forward pass for FactorizedTokenEmbedding.
     * @param inputIds Tensor of token IDs, shape [sequenceLength, 1] or [sequenceLength].
     * @return Tensor of final embeddings, shape [sequenceLength, hiddenDimH].
     */
    public Tensor forward(Tensor inputIds) {
        // Step 1: Get embeddings from vocab to embeddingDimE
        // Input: [sequenceLength, 1] or [sequenceLength] -> Output: [sequenceLength, embeddingDimE]
        Tensor embeddedE = this.embeddingLayerE.forward(inputIds);

        // Step 2: Project from embeddingDimE to hiddenDimH
        // Input: [sequenceLength, embeddingDimE] -> Output: [sequenceLength, hiddenDimH]
        Tensor embeddedH = this.projectionLayerH.apply(embeddedE);

        return embeddedH;
    }

    // Optional: Main method for basic testing
    public static void main(String[] args) {
        int vocab = 100;
        int eDim = 32; // Intermediate embedding
        int hDim = 64; // Final model dimension (dModel)
        boolean grad = true;

        FactorizedTokenEmbedding factorizedEmb = new FactorizedTokenEmbedding(vocab, eDim, hDim, grad);

        System.out.println("Factorized Embedding Initialized.");
        System.out.println("E_layer weights shape: " + factorizedEmb.embeddingLayerE.weights.shapeStr());
        System.out.println("H_layer weights shape: " + factorizedEmb.projectionLayerH.weight.shapeStr());
        if (factorizedEmb.projectionLayerH.bias != null) {
            System.out.println("H_layer bias shape: " + factorizedEmb.projectionLayerH.bias.shapeStr());
        }

        // Create dummy input IDs
        // Test with a sequence of 3 tokens: [10, 20, 0]
        // Assuming Tensor can be created from a 2D array for [seq_len, 1]
        Tensor input_ids_seq = new Tensor(new double[][]{{10}, {20}, {0}}); // Shape [3,1]

        // Or from a flat array if inputIds is [sequenceLength]
        // Tensor input_ids_flat = new Tensor(new double[]{10, 20, 0}, 1, 3, false); // Shape [1,3] (row vector)
        // Tensor input_ids_flat_as_vec = new Tensor(new double[]{10, 20, 0}); // Shape [3] (vector)

        System.out.println("\nInput IDs (sequence):");
        input_ids_seq.print();

        Tensor output_embeddings_seq = factorizedEmb.forward(input_ids_seq);
        System.out.println("\nOutput Embeddings (from sequence input):");
        output_embeddings_seq.print();
        System.out.println("Output shape: " + output_embeddings_seq.shapeStr()); // Expected: [3, 64]

        // Test with a single token input
        Tensor input_id_single = new Tensor(new double[][]{{5}}); // Shape [1,1]
        System.out.println("\nInput ID (single):");
        input_id_single.print();
        Tensor output_embedding_single = factorizedEmb.forward(input_id_single);
        System.out.println("\nOutput Embedding (single):");
        output_embedding_single.print();
        System.out.println("Output shape: " + output_embedding_single.shapeStr()); // Expected: [1, 64]

        // Test with flat input (row vector representing a sequence)
        // This part depends on how TokenEmbedding_E.forward handles inputIds.cols() != 1
        Tensor input_ids_flat_row_vec = new Tensor(new double[]{15, 25, 5}, 1, 3, false);
        System.out.println("\nInput IDs (flat row vector):");
        input_ids_flat_row_vec.print();
        Tensor output_embeddings_flat = factorizedEmb.forward(input_ids_flat_row_vec);
        System.out.println("\nOutput Embeddings (from flat row vector input):");
        output_embeddings_flat.print();
        System.out.println("Output shape: " + output_embeddings_flat.shapeStr()); // Expected: [3, 64]


        System.out.println("\nTesting TokenEmbedding_E directly with flat input:");
        TokenEmbedding_E directE = new TokenEmbedding_E(vocab, eDim, grad);
        Tensor direct_flat_out = directE.forward(input_ids_flat_row_vec);
        System.out.println("Direct E output shape: " + direct_flat_out.shapeStr()); // Expected [3, eDim]
        direct_flat_out.print();
    }
}
