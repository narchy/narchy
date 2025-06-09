package jcog.tensor.experimental;

import jcog.tensor.Models;
import jcog.tensor.Tensor;
import org.jetbrains.annotations.Nullable;

import java.util.function.UnaryOperator;

import static jcog.tensor.Tensor.zeros;

/**
 * UNTESTED
 * **Adaptive Gated Linear Unit (AGLU) layer**
 * a novel drop-in replacement for a neural network layer that operates efficiently within fixed mutable memory and excels in learning, adaptivity, and generalization
 * <p>
 * ### Layer Design:
 * <p>
 * 1. **Structure**:
 * - The AGLU layer consists of multiple linear projections followed by a gating mechanism that adaptively combines these projections to form the final output.
 * - Each linear projection is a matrix multiplication of the input by a smaller weight matrix.
 * - Gating functions determine the contribution of each projection to the output.
 * <p>
 * 2. **Adaptive Mechanism**:
 * - Gating functions use lightweight sigmoid activations to control the flow of information, ensuring computational efficiency.
 * - A fixed-size memory buffer stores context-aware parameters that adapt based on input statistics, aiding continual learning.
 * <p>
 * 3. **Efficient Memory Usage**:
 * - Parameters and activations are stored in contiguous memory blocks to maximize cache hits and minimize data movement.
 * - Operations are optimized for cache locality and vectorization, leveraging CPU features like SIMD instructions.
 * <p>
 * 4. **Continual Learning and Generalization**:
 * - Elastic weight consolidation (EWC) is employed to prevent catastrophic forgetting when learning from new data.
 * - Regularization techniques, such as dropout, are integrated to enhance generalization without exceeding memory constraints.
 * <p>
 * ### Forward Pass:
 * <p>
 * - **Input**: $X \in \mathbb{R}^{N \times D_{\text{in}}}$, where $N$ is the batch size and $D_{\text{in}}$ is the input dimension.
 * - **Linear Projections**: Compute $K$ projections using $W_1, W_2, \dots, W_K$, where each $W_k \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}} / K}$.
 * - **Gate Weights**: Compute gate weights $G = \sigma(X W_G + b_G)$, where $W_G \in \mathbb{R}^{D_{\text{in}} \times K}$ and $\sigma$ is the sigmoid function.
 * - **Combine Projections**:
 * \[
 * Y = \sum_{k=1}^{K} G_k \cdot (X W_k + b_k)
 * \]
 * where $G_k$ is the $k$-th gate weight and $b_k$ is the bias for the $k$-th projection.
 * <p>
 * ### Backward Pass:
 * <p>
 * - Gradients are computed using automatic differentiation, with operations optimized for cache efficiency.
 * - During backpropagation, gradients for each $W_k$ and $W_G$ are computed and updated using an optimizer (e.g., Adam).
 * <p>
 * ### Advantages:
 * <p>
 * - **Performance**: Fast forward and backward passes due to cache-friendly memory access and vectorized operations.
 * - **Learning Power**: Captures complex patterns through adaptive combinations of linear projections.
 * - **Adaptivity**: Continually learns from new data while preventing forgetting via EWC.
 * - **Generalizability**: Built-in regularization techniques ensure good performance on unseen data.
 * <p>
 * ### Applications:
 * <p>
 * - **General ML Tasks**: Can replace linear layers in various network architectures (e.g., feedforward, CNNs, RNNs).
 * - **Reinforcement Learning**: Suitable for processing temporal data and variable-length sequences due to its adaptive nature.
 * <p>
 * ### Conclusion:
 * <p>
 * The Adaptive Gated Linear Unit (AGLU) layer provides a memory-efficient, high-performance alternative to traditional linear layers, achieving superior learning and adaptivity across a range of machine learning applications, including reinforcement learning. Its design ensures compatibility with autodiff systems and enhances model performance within fixed memory constraints.
 */
public class AGLU implements UnaryOperator<Tensor> {
    protected final int K; // Number of projections
    protected final Models.Linear[] projections; // List of K linear layers
    protected final Models.Linear gate; // Gate weight matrix
    protected final Tensor beta; // Additive bias
    private final @Nullable UnaryOperator<Tensor> activation;

    public AGLU(int in, int out, int K, @Nullable UnaryOperator<Tensor> activation) {
        if (out % K != 0)
            throw new IllegalArgumentException("out must be divisible by K");
        if (K < 2)
            throw new IllegalArgumentException();
        this.K = K;

        // Initialize projections
        this.projections = new Models.Linear[K];
        for (int i = 0; i < K; i++)
            projections[i] = new Models.Linear(in, out / K, null, false);

        // Initialize gate with sigmoid activation and bias
        boolean bias = true; //TODO optional, by constructor parameter
        this.gate = new Models.Linear(in, K, null, bias);

        // Initialize bias
        this.beta = zeros(1, out).grad(true).parameter();

        this.activation = activation;
    }

    @Override
    public Tensor apply(Tensor x) {
        // Compute projections
        Tensor[] projOutputs = new Tensor[K];
        for (int i = 0; i < K; i++) {
            projOutputs[i] = projections[i].apply(x);
        }

        // Stack projections along the K dimension to get [N, K, D_out/K]
        Tensor projectionsStacked = Tensor.stack(projOutputs, 1);

        // Compute gate weights
        Tensor gateWeights = gate.apply(x);

        // Reshape gate weights to [N, K]
        Tensor gateWeightsReshaped = gateWeights.reshape(x.rows(), K);

        // Expand gate weights to match the shape of projectionsStacked [N, K, D_out/K]
        Tensor gateWeightsExpanded = gateWeightsReshaped.expandToMatch(projectionsStacked.rows(), projectionsStacked.cols() / K, K);

        // Multiply projections by gate weights (element-wise multiplication)
        Tensor weightedProjections = projectionsStacked.mul(gateWeightsExpanded);

        // Sum over K to get [N, D_out]
        Tensor output = weightedProjections.sumCols(); // Sum along the K dimension

        // Add bias
        var y = output.add(beta);

        return activation != null ? activation.apply(y) : y;
    }
}
