package jcog.tensor.experimental;

import jcog.tensor.Models;
import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

import java.util.function.UnaryOperator;

import static jcog.tensor.Tensor.randHe;

/**
 * Self-Attention-Based Interaction Layer (SABIL)
 * A drop-in replacement for a fully-connected layer with self-attention over input segments,
 * using only 2D tensors.
 */
public class SABIL extends Models.BiasActivation {
    public final Tensor Wq, Wk, Wv, Wo; // Weight matrices for Q, K, V, and output projection
    protected final int m;              // Number of segments
    protected final int k;              // Size of each segment (inputDim / m)
    protected final int d;              // Attention dimension

    /**
     * Constructor for SABIL
     * @param inputDim Input dimension (n)
     * @param outputDim Output dimension (p)
     * @param segments Number of segments (m)
     * @param d Attention dimension
     * @param activation Optional activation function (nullable)
     * @param bias Whether to include bias
     */
    public SABIL(int inputDim, int outputDim, int segments, int d, @Nullable UnaryOperator<Tensor> activation, boolean bias) {
        super(outputDim, activation, bias);
        this.m = segments;
        this.d = d;

        // Ensure inputDim is divisible by m
        if (inputDim % segments != 0) {
            throw new IllegalArgumentException("inputDim must be divisible by segments");
        }
        this.k = inputDim / segments;

        // Initialize weight matrices
        this.Wq = randHe(k, d).grad(true).parameter();        // (k, d)
        this.Wk = randHe(k, d).grad(true).parameter();        // (k, d)
        this.Wv = randHe(k, d).grad(true).parameter();        // (k, d)
        this.Wo = randHe(m * d, outputDim).grad(true).parameter(); // (m * d, outputDim)
    }

    /**
     * Forward pass: Apply self-attention over segmented input using 2D tensors
     * @param x Input tensor of shape (rows, inputDim)
     * @return Output tensor of shape (rows, outputDim)
     */
    @Override
    public Tensor apply(Tensor x) {
        int rows = x.rows();

        // Step 1: Reshape input to (rows * m, k) to process segments as rows
        Tensor X = x.reshape(rows * m, k);

        // Step 2: Compute Q, K, V
        // (rows * m, k) × (k, d) -> (rows * m, d)
        Tensor Q = X.matmul(Wq);
        Tensor K = X.matmul(Wk);
        Tensor V = X.matmul(Wv);

        // Step 3: Compute attention scores
        // (rows * m, d) × (d, rows * m) -> (rows * m, rows * m)
        Tensor scores = Q.matmulTranspose(K);

        // Normalize scores
        Tensor scoresNormalized = scores.div(Math.sqrt(d));

        // Step 4: Apply softmax to get attention weights
        Tensor attentionWeights = scoresNormalized.softmax(); // (rows * m, rows * m)

        // Step 5: Apply attention to values
        // (rows * m, rows * m) × (rows * m, d) -> (rows * m, d)
        Tensor Y = attentionWeights.matmul(V);

        // Step 6: Reshape back to (rows, m * d)
        Tensor yFlat = Y.reshape(rows, m * d);

        // Step 7: Project to output dimension
        // (rows, m * d) × (m * d, outputDim) -> (rows, outputDim)
        Tensor y = yFlat.matmul(Wo);

        // Apply bias and activation (from BiasActivation)
        return super.apply(y);
    }
}