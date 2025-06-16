package jcog.tensor.experimental;

import jcog.Util;
import jcog.decide.Decide;
import jcog.decide.DecideSoftmax;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.tensor.Models;
import jcog.tensor.Tensor;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

import static jcog.tensor.Tensor.*;

public enum ModelsExperimental { ;

    public static class Transformer implements UnaryOperator<Tensor> {
        private final int dimModel;
        private final int numHeads;
        private final int dimFF;
        private final List<TransformerLayer> layers;
        private final Models.LayerNorm finalLayerNorm;
        private final double dropoutRate;

        public Transformer(int dimModel, int dimFF, int numHeads, int numLayers, float dropoutRate) {
            this.dimModel = dimModel;
            this.numHeads = numHeads;
            this.dimFF = dimFF;
            this.dropoutRate = dropoutRate;

            this.layers = new ArrayList<>(numLayers);
            for (var i = 0; i < numLayers; i++)
                this.layers.add(new TransformerLayer(dimModel, dimFF, numHeads, dropoutRate));

            this.finalLayerNorm = new Models.LayerNorm(dimModel, 1e-5);
        }

        @Override
        public Tensor apply(Tensor x) {
            for (var layer : layers)
                x = layer.apply(x);
            return finalLayerNorm.apply(x);
        }

        private class TransformerLayer implements UnaryOperator<Tensor> {
            private final UnaryOperator<Tensor> attention;
            private final Models.Linear feedForward1, feedForward2;
            private final Models.LayerNorm norm1, norm2;
            private final Models.Dropout dropout;

            TransformerLayer(int dimModel, int dimFF, int numHeads, float dropoutRate) {
                this(dimModel, dimFF, numHeads, dropoutRate, true);
            }

            TransformerLayer(int dimModel, int dimFF, int numHeads, float dropoutRate, boolean transformerOrULT) {
                this.attention =
                        transformerOrULT ?
                                new MultiHeadAttention(dimModel, numHeads, dropoutRate)
                                :
                                new Models.ULT(dimModel, dimModel, dimModel, numHeads, Tensor::gelu, true);
                this.feedForward1 = new Models.Linear(dimModel, dimFF, Tensor::gelu, true);
                this.feedForward2 = new Models.Linear(dimFF, dimModel, null, true);
                this.norm1 = new Models.LayerNorm(dimModel, 1e-5);
                this.norm2 = new Models.LayerNorm(dimModel, 1e-5);
                this.dropout = new Models.Dropout(dropoutRate);
            }

            @Override
            public Tensor apply(Tensor x) {
                var attended = attention.apply(x);
                x = x.add(dropout.apply(attended));
                x = norm1.apply(x);

                var ff = feedForward2.apply(feedForward1.apply(x));
                x = x.add(dropout.apply(ff));
                x = norm2.apply(x);
                return x;
            }
        }

        private class MultiHeadAttention implements UnaryOperator<Tensor> {
            private final Models.Linear qkv;
            private final Models.Linear output;
            private final Models.Dropout dropout;
            private final int numHeads;
            private final int headDim;

            MultiHeadAttention(int dimModel, int numHeads, float dropoutRate) {
                this.numHeads = numHeads;
                this.headDim = dimModel / numHeads;

                this.qkv = new Models.Linear(dimModel, dimModel * 3, null, true);
                this.output = new Models.Linear(dimModel, dimModel, null, true);
                this.dropout = new Models.Dropout(dropoutRate);

                // Initialize weights
                var initScale = Math.sqrt(2.0 / (5 * dimModel));
                this.qkv.weight.setData(randGaussian(dimModel, dimModel * 3, initScale).data);
                this.output.weight.setData(randGaussian(dimModel, dimModel, initScale).data);
            }

            @Override
            public Tensor apply(Tensor x) {
                var batchSize = x.rows();
                var seqLen = x.cols() / dimModel;

                System.out.println("Input shape: " + x.rows() + "x" + x.cols());
                System.out.println("dimModel: " + dimModel);
                System.out.println("numHeads: " + numHeads);
                System.out.println("headDim: " + headDim);

                var qkv = this.qkv.apply(x);
                System.out.println("QKV shape: " + qkv.rows() + "x" + qkv.cols());

                var q = qkv.slice(0, dimModel);
                var k = qkv.slice(dimModel, 2 * dimModel);
                var v = qkv.slice(2 * dimModel, 3 * dimModel);

                System.out.println("Q shape: " + q.rows() + "x" + q.cols());
                System.out.println("K shape: " + k.rows() + "x" + k.cols());
                System.out.println("V shape: " + v.rows() + "x" + v.cols());

                q = splitHeads(q);
                k = splitHeads(k);
                v = splitHeads(v);

                System.out.println("Q after split: " + q.rows() + "x" + q.cols());
                System.out.println("K after split: " + k.rows() + "x" + k.cols());
                System.out.println("V after split: " + v.rows() + "x" + v.cols());

                var scores = q.matmul(k.transpose(-2, -1));
                scores = scores.div(Math.sqrt(headDim));

                var attentionWeights = scores.softmax();
                attentionWeights = dropout.apply(attentionWeights);

                var contextLayer = attentionWeights.matmul(v);
                contextLayer = mergeHeads(contextLayer);

                return output.apply(contextLayer);
            }

            private Tensor splitHeads(Tensor x) {
                var batchSize = x.rows();
                var seqLen = x.cols() / dimModel;
                var reshaped = x.reshape(batchSize, seqLen, numHeads, headDim);
                System.out.println("Split reshape: " + reshaped.rows() + "x" + reshaped.cols());
                var transposed = reshaped.transpose(1, 2);
                System.out.println("Split transpose: " + transposed.rows() + "x" + transposed.cols());
                return transposed;
            }

            private Tensor mergeHeads(Tensor x) {
                var batchSize = x.rows();
                var seqLen = x.cols() / (numHeads * headDim);
                return x.transpose(1, 2).reshape(batchSize, seqLen, dimModel);
            }
        }
    }

    /**
     * Linformer: Efficient linear attention for long sequences
     * Use for: Long document processing, time series with long histories
     * Advantage: O(n) complexity vs O(n^2) for standard attention
     */
    public static class Linformer extends Models.Linear {
        private final Tensor E, F;
        private final int headDim, numHeads, seqLen, k;

        public Linformer(int inputDim, int outputDim, int headDim, int numHeads, int seqLen, int k) {
            super(inputDim, outputDim * 3, null, true);
            this.headDim = headDim;
            this.numHeads = numHeads;
            this.seqLen = seqLen;
            this.k = k;
            var initScale = 1.0 / Math.sqrt(k);
            this.E = randGaussian(k, seqLen, initScale).grad(true).parameter();
            this.F = randGaussian(k, seqLen, initScale).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            var batchSize = x.rows() / seqLen;
            var qkv = super.apply(x).reshape(batchSize, seqLen, 3 * numHeads * headDim);
            var q = qkv.slice(0, numHeads * headDim).reshape(batchSize * numHeads, seqLen, headDim);
            var k = qkv.slice(numHeads * headDim, 2 * numHeads * headDim).reshape(batchSize * numHeads, seqLen, headDim);
            var v = qkv.slice(2 * numHeads * headDim, 3 * numHeads * headDim).reshape(batchSize * numHeads, seqLen, headDim);
            return q.matmul(E.matmul(k.transpose()).transpose()).div(Math.sqrt(headDim)).softmax()
                    .matmul(F.matmul(v.transpose()).transpose())
                    .reshape(batchSize, seqLen, numHeads * headDim);
        }
    }

    /**
     * LSSL: Efficient sequence modeling, alternative to RNNs and Transformers
     * Use for: Time series forecasting, sequence generation, signal processing
     * Advantage: Models long-range dependencies efficiently, parallelizable
     */
    public static class LSSL extends Models.Linear {
        private final Tensor A, B, C, D;
        private final int seqLen, stateSize;

        public LSSL(int inputDim, int outputDim, int stateSize, int seqLen) {
            super(inputDim, outputDim, null, true);
            this.seqLen = seqLen;
            this.stateSize = stateSize;
            this.A = randGaussian(stateSize, stateSize, 0.01).mul(0.99).grad(true).parameter();
            this.B = randGaussian(stateSize, inputDim, 1.0 / Math.sqrt(inputDim)).grad(true).parameter();
            this.C = randGaussian(outputDim, stateSize, 1.0 / Math.sqrt(stateSize)).grad(true).parameter();
            this.D = randGaussian(outputDim, inputDim, 1.0 / Math.sqrt(inputDim)).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            var batchSize = x.rows() / seqLen;
            var h = zeros(batchSize, stateSize);
            var outputs = new ArrayList<Tensor>(seqLen);
            for (var t = 0; t < seqLen; t++) {
                var xt = x.slice(t * batchSize, (t + 1) * batchSize);
                h = h.matmul(A).add(xt.matmul(B.transpose()));
                outputs.add(h.matmul(C.transpose()).add(xt.matmul(D.transpose())));
            }
            return concat(outputs.toArray(EMPTY_ARRAY));
        }
    }

    /**
     * DQN: Value-based reinforcement learning for discrete action spaces
     * Use for: Game playing, robotic control, resource management
     * Advantage: Learns complex strategies, off-policy learning
     */
    public static class DQN {

        final Decide decide = new DecideSoftmax(0.5f, new XoRoShiRo128PlusRandom());

        private final Models.Layers qNetwork, targetNetwork;
        private final Optimizer optimizer;
        private final int actionDim;
        private final ReplayBuffer replayBuffer;

        public DQN(int stateDim, int actionDim, int hiddenDim, int bufferSize, Optimizer o) {
            this.actionDim = actionDim;
            this.qNetwork = new Models.Layers(Tensor::relu, null, stateDim, hiddenDim, hiddenDim, actionDim);
            this.targetNetwork = new Models.Layers(Tensor::relu, null, stateDim, hiddenDim, hiddenDim, actionDim);
            this.optimizer = o;
            this.replayBuffer = new ReplayBuffer(bufferSize);
            updateTargetNetwork(1.0);
        }

        public int selectAction(Tensor state) {
            Tensor t = qNetwork.apply(state);
            return decide.applyAsInt(t.array());
        }

        public void update(int batchSize, double gamma) {
            if (replayBuffer.size() < batchSize) return;
            var batch = replayBuffer.sample(batchSize);
            var states = concat(batch.stream().map(e -> e.state).toArray(Tensor[]::new));
            var actions = batch.stream().mapToInt(e -> e.action).toArray();
            var rewards = row(batch.stream().mapToDouble(e -> e.reward).toArray());
            var nextStates = concat(batch.stream().map(e -> e.nextState).toArray(Tensor[]::new));
            var dones = row(batch.stream().mapToDouble(e -> e.done ? 0.0 : 1.0).toArray());

            var qValues = qNetwork.apply(states);
            var nextQValues = targetNetwork.apply(nextStates).detach();
            var targetQ = rewards.add(dones.mul(gamma).mul(max(nextQValues, scalar(1))));
            var actionQValues = qValues.gather(1, actions);

            actionQValues.sub(targetQ).pow(2).mean().minimize(optimizer, null);
        }

        public void addExperience(Tensor state, int action, double reward, Tensor nextState, boolean done) {
            replayBuffer.add(new Experience(state, action, reward, nextState, done));
        }

        public void updateTargetNetwork(double tau) {
            for (var i = 0; i < qNetwork.layer.size(); i++) {
                var qParams = parameters(qNetwork.layer.get(i)).toList();
                var targetParams = parameters(targetNetwork.layer.get(i)).toList();
                var p = qParams.size();
                for (var j = 0; j < p; j++)
                    targetParams.get(j).setSoft(qParams.get(j), tau);
            }
        }

        private record Experience(Tensor state, int action, double reward, Tensor nextState, boolean done) {
        }

        private static class ReplayBuffer {
            private final int capacity;
            private final List<Experience> buffer = new ArrayList<>();

            ReplayBuffer(int capacity) {
                this.capacity = capacity;
            }

            void add(Experience experience) {
                if (buffer.size() >= capacity) buffer.remove(0);
                buffer.add(experience);
            }

            List<Experience> sample(int batchSize) {
                Collections.shuffle(buffer);
                return buffer.subList(0, Math.min(batchSize, buffer.size()));
            }

            int size() {
                return buffer.size();
            }
        }
    }

    /**
     * UNTESTED
     * var dt = new DecisionTransformer(
     * stateSize, actionSize, returnSize, maxEpisodeLength,
     * embeddingSize, numLayers, numHeads, dropoutRate
     * );
     * <p>
     * dt.train(
     * batchIndex -> DecisionTransformer.prepareBatch(getTrajectories(batchSize)),
     * numBatches,
     * optimizer
     * );
     * <p>
     * // For inference
     * var batch = DecisionTransformer.prepareBatch(List.of(currentTrajectory));
     * var predictedAction = dt.getAction(batch.states(), batch.actions(), batch.returns(), batch.timesteps(), batch.attentionMask());
     */
    public static class DecisionTransformer {
        private final int stateSize;
        private final int actionSize;
        private final int returnSize;
        private final int maxEpisodeLength;
        private final int embeddingSize;
        private final UnaryOperator<Tensor> transformer;
        private final Models.Linear stateEmbedding;
        private final Models.Linear actionEmbedding;
        private final Models.Linear returnEmbedding;
        private final Models.Linear timeEmbedding;
        private final Models.Linear actionProjection;

        public record TrainingBatch(Tensor states, Tensor actions, Tensor returns, Tensor timesteps,
                                    Tensor attentionMask) {
        }

        /**
         * @param transformerOrULT Using ULT in these ways could potentially offer some advantages:
         *                         Simplicity: ULT combines the multi-head attention and feedforward operations into a single layer, which could simplify the architecture.
         *                         Efficiency: Depending on the implementation details, ULT might be more computationally efficient than separate attention and feedforward layers.
         *                         Flexibility: ULT allows for different dimensionalities for input and output, which could be useful in certain scenarios.
         */
        public DecisionTransformer(int stateSize, int actionSize, int returnSize, int maxEpisodeLength,
                                   int embeddingSize, int numLayers, int numHeads, float dropoutRate, boolean transformerOrULT) {
            this.stateSize = stateSize;
            this.actionSize = actionSize;
            this.returnSize = returnSize;
            this.maxEpisodeLength = maxEpisodeLength;
            this.embeddingSize = embeddingSize;

            UnaryOperator<Tensor> activation = Tensor::relu;

            this.stateEmbedding = new Models.Linear(stateSize, embeddingSize, activation, true);
            this.actionEmbedding = new Models.Linear(actionSize, embeddingSize, activation, true);
            this.returnEmbedding = new Models.Linear(returnSize, embeddingSize, activation, true);
            this.timeEmbedding = new Models.Linear(1, embeddingSize, activation, true);
            this.transformer = transformerOrULT ?
                    new ModelsExperimental.Transformer(embeddingSize, embeddingSize * 4, numHeads, numLayers, dropoutRate)
                    :
                    new Models.ULT(embeddingSize, embeddingSize, embeddingSize, numHeads, activation, true);

            this.actionProjection = new Models.Linear(embeddingSize, actionSize, null, true);

            System.out.println("State embedding: " + stateSize + " -> " + embeddingSize);
            System.out.println("Action embedding: " + actionSize + " -> " + embeddingSize);
            System.out.println("Return embedding: " + returnSize + " -> " + embeddingSize);
            System.out.println("Time embedding: 1 -> " + embeddingSize);
        }

        public Tensor forward(Tensor states, Tensor actions, Tensor returns, Tensor timesteps, Tensor attentionMask) {
            System.out.println("States shape: " + states.rows() + "x" + states.cols());
            System.out.println("Actions shape: " + actions.rows() + "x" + actions.cols());
            System.out.println("Returns shape: " + returns.rows() + "x" + returns.cols());
            System.out.println("Timesteps shape: " + timesteps.rows() + "x" + timesteps.cols());

            var batchSize = states.rows();
            var sequenceLength = states.cols() / stateSize;

            // Apply embeddings
            var stateEmb = applyEmbedding(states, stateEmbedding, stateSize, sequenceLength);
            var actionEmb = applyEmbedding(actions, actionEmbedding, actionSize, sequenceLength);
            var returnEmb = applyEmbedding(returns, returnEmbedding, returnSize, sequenceLength);
            var timeEmb = timeEmbedding.apply(timesteps);

            // Interleave embeddings
            var embeddingsList = new ArrayList<Tensor>();
            for (var i = 0; i < sequenceLength; i++) {
                embeddingsList.add(stateEmb.slice(i, i + 1));
                embeddingsList.add(actionEmb.slice(i, i + 1));
                embeddingsList.add(returnEmb.slice(i, i + 1));
            }
            var input = concat(embeddingsList.toArray(EMPTY_ARRAY));

            input = input.add(timeEmb);

            System.out.println("Transformer input shape: " + input.rows() + "x" + input.cols());
            var output = transformer.apply(input);
            System.out.println("Transformer output shape: " + output.rows() + "x" + output.cols());

            // Extract action predictions
            var actionPredsList = new ArrayList<Tensor>();
            for (var i = 0; i < sequenceLength; i++) {
                var actionEmbed = output.slice(i * 3 + 1, i * 3 + 2);
                actionPredsList.add(actionProjection.apply(actionEmbed));
            }
            var actionPreds = concat(actionPredsList.toArray(EMPTY_ARRAY));

            return actionPreds.mul(attentionMask);
        }

        private Tensor applyEmbedding(Tensor input, Models.Linear embedding, int featureSize, int sequenceLength) {
            var embeddedList = new ArrayList<Tensor>();
            for (var i = 0; i < sequenceLength; i++) {
                var slice = input.slice(i * featureSize, (i + 1) * featureSize);
                embeddedList.add(embedding.apply(slice));
            }
            return concat(embeddedList.toArray(EMPTY_ARRAY));
        }

        private Tensor applyTimeEmbedding(Tensor timesteps) {
            var batchSize = timesteps.rows();
            var sequenceLength = timesteps.cols();
            var timeInput = zeros(batchSize * sequenceLength, 1);
            for (var i = 0; i < batchSize * sequenceLength; i++) {
                timeInput.data.set(i, 0, i % sequenceLength);
            }
            var embedded = timeEmbedding.apply(timeInput);

            // Reshape and repeat to match the input shape
            var repeatedEmbedding = zeros(batchSize, sequenceLength * 3 * embeddingSize);
            for (var i = 0; i < batchSize; i++) {
                for (var j = 0; j < sequenceLength; j++) {
                    for (var k = 0; k < 3; k++) {  // Repeat for state, action, and return
                        for (var l = 0; l < embeddingSize; l++) {
                            repeatedEmbedding.data.set(i, (j * 3 + k) * embeddingSize + l,
                                    embedded.data(i * sequenceLength + j, l));
                        }
                    }
                }
            }
            return repeatedEmbedding;
        }

        private Tensor interleaveEmbeddings(Tensor stateEmb, Tensor actionEmb, Tensor returnEmb, int batchSize, int sequenceLength) {
            var totalLength = sequenceLength * 3;
            var interleaved = zeros(batchSize, totalLength * embeddingSize);
            for (var i = 0; i < sequenceLength; i++) {
                for (var j = 0; j < embeddingSize; j++) {
                    interleaved.data.set(0, i * 3 * embeddingSize + j, stateEmb.data(i * embeddingSize + j));
                    interleaved.data.set(0, (i * 3 + 1) * embeddingSize + j, actionEmb.data(i * embeddingSize + j));
                    interleaved.data.set(0, (i * 3 + 2) * embeddingSize + j, returnEmb.data(i * embeddingSize + j));
                }
            }
            return interleaved;
        }


        public Tensor getAction(Tensor states, Tensor actions, Tensor returns, Tensor timesteps, Tensor attentionMask) {
            var predictedActions = forward(states, actions, returns, timesteps, attentionMask);
            return predictedActions.slice(predictedActions.cols() - actionSize, predictedActions.cols());
        }

        public Tensor computeLoss(TrainingBatch batch) {
            var predictedActions = forward(batch.states, batch.actions, batch.returns, batch.timesteps, batch.attentionMask);
            return predictedActions.sub(batch.actions).pow(2).mean();
        }

        public TrainingBatch training(List<List<Tensor>> trajectories) {
            var batchSize = trajectories.size();
            var maxLength = trajectories.stream().mapToInt(List::size).max().orElse(0);

            var states = zeros(batchSize, maxLength * stateSize);
            var actions = zeros(batchSize, maxLength);
            var returns = zeros(batchSize, maxLength);
            var timesteps = zeros(batchSize, maxLength);
            var attentionMask = zeros(batchSize, maxLength);

            for (var i = 0; i < batchSize; i++) {
                var trajectory = trajectories.get(i);
                for (var j = 0; j < trajectory.size(); j++) {
                    var step = trajectory.get(j);
                    for (var k = 0; k < stateSize; k++) {
                        states.data.set(i, j * stateSize + k, step.data(k));
                    }
                    actions.data.set(i, j, step.data(stateSize));
                    returns.data.set(i, j, step.data(stateSize + actionSize));
                    timesteps.data.set(i, j, j);
                    attentionMask.data.set(i, j, 1);
                }
            }

            return new TrainingBatch(states, actions, returns, timesteps, attentionMask);
        }

        public void train(Function<Integer, TrainingBatch> batchProvider, int numBatches, Optimizer optimizer) {
            for (var i = 0; i < numBatches; i++) {
                var batch = batchProvider.apply(i);
                var loss = computeLoss(batch);
                loss.minimize(optimizer, null);

                if (i % 100 == 0) {
                    System.out.printf("Batch %d, Loss: %.4f%n", i, loss.scalar());
                }
            }
        }
    }


    /**
     * Universal Linear Transformer with Hyperspherical Normalization
     * <p>
     * Adapted version of the ULT class, incorporating the key ideas from the Normalized Transformer (nGPT), such as hyperspherical normalization and eigen learning rates:
     * <p>
     * Key Changes:
     * Hyperspherical Normalization: Normalize weights (Wq, Wk, Wv, Wo) along their embedding dimensions to ensure they reside on a hypersphere.
     * Learnable Eigen Learning Rates: Introduce scaling parameters to control the update steps, inspired by the eigen learning rates in nGPT.
     * Adjusted Softmax Scaling: Adjust the scaling factor to align with nGPT's approach.
     * Removal of Redundant Norm Variability: Apply normalization consistently to input and output to stabilize learning across sequences.
     */
    public static class NormalizedULT extends Models.ULT {
        public final Tensor alphaQ, alphaK, alphaV, alphaO;  // Eigen learning rates

        public NormalizedULT(int inputDim, int outputDim, int dModel, int numHeads,
                             @Nullable UnaryOperator<Tensor> activation, boolean bias) {
            super(inputDim, outputDim, dModel, numHeads, activation, bias);

            // Initialize learnable scaling parameters (eigen learning rates)
            this.alphaQ = ones(Wq.cols()).grad(true).parameter();
            this.alphaK = ones(Wk.cols()).grad(true).parameter();
            this.alphaV = ones(Wv.cols()).grad(true).parameter();
            this.alphaO = ones(Wo.cols()).grad(true).parameter();

            // Normalize all weight matrices on initialization
            normalizeWeights();
        }

        /**
         * Override apply to integrate hyperspherical normalization and scaling
         */
        @Override
        public Tensor apply(Tensor x) {
            // Apply linear projections with eigen scaling factors
            var Q = shape(x.matmul(Wq).mul(alphaQ));
            var K = shape(x.matmul(Wk).mul(alphaK));
            var V = shape(x.matmul(Wv).mul(alphaV));

            // Compute normalized attention scores
            var scores = Q.matmulTranspose(K).div(Math.sqrt(dHead));
            var attentionWeights = //logSoftmax ?
                    //scores.logSoftmax().exp() :
                    scores.softmax();

            // Apply attention weights to values
            var output = attentionWeights.matmul(V);

            // Final linear projection with scaling
            var y = unshape(output).matmul(Wo).mul(alphaO);

            // Apply activation and bias if defined
            return biasActivation(y);
        }

        /**
         * Normalize all weight matrices on a hypersphere
         */
        private void normalizeWeights() {
            Wq.setData(Wq.normalizeL2());
            Wk.setData(Wk.normalizeL2());
            Wv.setData(Wv.normalizeL2());
            Wo.setData(Wo.normalizeL2());
        }

    }
//    /** from "Were RNNs All We Needed?" https://arxiv.org/pdf/2410.01201 */
//    public static class MinGRU {
//
//        // Define the parameters of minGRU
//        public final Tensor Wz; // Weight for the update gate
//        public final Tensor Wh; // Weight for the candidate hidden state
//        public final Tensor bz; // Bias for the update gate (optional)
//        public final Tensor bh; // Bias for the candidate hidden state (optional)
//
//        private final int hiddenDim;
//
//        public MinGRU(int inputDim, int hiddenDim) {
//            this.hiddenDim = hiddenDim;
//
//            // Randomly initialize the weights with Gaussian distribution
//            Wz = randGaussian(inputDim, hiddenDim, Math.sqrt(2.0 / hiddenDim)).grad(true).parameter();
//            Wh = randGaussian(inputDim, hiddenDim, Math.sqrt(2.0 / hiddenDim)).grad(true).parameter();
//
//            ht_prev = zeros()batch
//
//            // Initialize the biases to zero (optional)
//            bz = zeros(1, hiddenDim).grad(true).parameter();
//            bh = zeros(1, hiddenDim).grad(true).parameter();
//        }
//
//        /**
//         * Forward pass through the MinGRU for a single time step.
//         *
//         * @param input    Input tensor for the current time step (batch_size, inputDim)
//         * @param ht_prev  Hidden state from the previous time step (batch_size, hiddenDim)
//         * @return Updated hidden state for the current time step
//         */
//        public Tensor apply(Tensor input) {
//            // Compute update gate z_t = sigmoid(Wz * x_t + bz)
//            Tensor zt = input.matmul(Wz).add(bz).sigmoid();
//
//            // Compute candidate hidden state h̃_t = Wh * x_t + bh
//            Tensor h_tilde = input.matmul(Wh).add(bh);
//
//            // Compute the new hidden state h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
//            Tensor ht = zt.neg().add(1).mul(ht_prev).add(zt.mul(h_tilde));
//
//            ht_prev.setData(ht);
//
//            return ht; // Return the new hidden state
//        }
//
//        /**
//         * This function initializes the hidden state to zeros for the first time step.
//         *
//         * @param batchSize Size of the batch.
//         * @return Initialized hidden state (zeros)
//         */
//        public Tensor initHiddenState(int batchSize) {
//            return zeros(batchSize, hiddenDim); // Initial hidden state
//        }
//    }

//    /**
//     * DeepEBM: Flexible unsupervised learning model
//     * Use for: Anomaly detection, image generation, data compression
//     * Advantage: Models complex distributions without explicit density estimation
//     */
//    public static class DeepEBM {
//        private final Layers network;
//        private final Optimizer optimizer;
//        private final double regularizationStrength;
//
//        public DeepEBM(int inputDim, int hiddenDim, double regularizationStrength) {
//            this.network = new Layers(Tensor::swish, null, inputDim, hiddenDim, hiddenDim, 1);
//            this.optimizer = new Optimizer(new Optimizers.Adam(0.001));
//            this.regularizationStrength = regularizationStrength;
//        }
//
//        public Tensor computeEnergy(Tensor x) { return network.apply(x).neg(); }
//
//        public void update(Tensor positiveSamples, Tensor negativeSamples) {
//            var posEnergy = computeEnergy(positiveSamples);
//            var negEnergy = computeEnergy(negativeSamples);
//            posEnergy.mean().sub(negEnergy.mean())
//                    .add(posEnergy.pow(2).mean().add(negEnergy.pow(2).mean()).mul(regularizationStrength))
//                    .minimize(optimizer);
//        }
//
//        public Tensor sampleLangevin(Tensor initial, int steps, double stepSize, double noise, double clipValue) {
//            var sample = initial.detachCopy();
//            for (int i = 0; i < steps; i++) {
//                sample = sample.add(computeEnergy(sample).grad.mul(stepSize))
//                        .add(Tensor.randGaussian(sample.rows(), sample.cols(), noise));
//                if (clipValue > 0) sample = sample.clip(-clipValue, clipValue);
//            }
//            return sample;
//        }
//
//        public Tensor generateSamples(int numSamples, int inputDim, int steps, double stepSize, double noise, double clipValue) {
//            return sampleLangevin(Tensor.randGaussian(numSamples, inputDim, 1.0), steps, stepSize, noise, clipValue);
//        }
//    }
    /**
     * N-bit quantized weights and activations for reduced memory bandwidth.
     */
    public static class QuantizedLinear extends Models.BiasActivation {
        private final Tensor weights;
        private final Tensor scale, maxWeight, minWeight;
        private final int bits;

        public QuantizedLinear(int inFeatures, int outFeatures, UnaryOperator<Tensor> activation,
                               boolean bias, int bits) {
            super(outFeatures, activation, bias);
            this.bits = bits;

            // Initialize trainable weights
            this.weights = randGaussian(inFeatures, outFeatures, Math.sqrt(2.0 / outFeatures))
                    .grad(true).parameter();

            // Initialize quantization parameters as tensors to maintain gradient flow
            this.maxWeight = scalar(weights.maxValue().scalar()).grad(true);
            this.minWeight = scalar(weights.minValue().scalar()).grad(true);
            this.scale = maxWeight.sub(minWeight).div((1 << bits) - 1).grad(true);
        }

        @Override
        public Tensor apply(Tensor x) {
            // Quantize-dequantize maintaining gradient flow
            var xMax = x.maxValue();
            var xScale = xMax.div((1 << bits) - 1);

            // Quantize input and weights with gradient tracking
            var xQ = x.div(xScale);
            var wQ = weights.sub(minWeight).div(scale);

            // Compute with quantized values and dequantize
            return super.apply(xQ.matmul(wQ).mul(scale).mul(xScale));
        }
    }

    /**
     * Toeplitz structured weights for translation-invariant operations. O(m+n) parameters.
     */
    public static class ToeplitzLinear extends Models.BiasActivation {
        private final Tensor coeffs;
        private final int inFeatures, outFeatures;

        public ToeplitzLinear(int inFeatures, int outFeatures, UnaryOperator<Tensor> activation, boolean bias) {
            super(outFeatures, activation, bias);
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            int n = inFeatures + outFeatures - 1;
            this.coeffs = randGaussian(1, n, Math.sqrt(2.0 / n)).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            Tensor output = null;
            int inF1 = inFeatures - 1;

            for (int i = 0; i < outFeatures; i++) {
                Tensor rowSum = null;
                for (int j = 0; j < inFeatures; j++) {
                    var xj = x.slice(j, j + 1);
                    var cij = coeffs.slice(inF1 + i - j, inF1 + i - j + 1);
                    var prod = xj.mul(cij);
                    rowSum = rowSum == null ? prod : rowSum.add(prod);
                }
                output = output == null ? rowSum : output.concat(rowSum);
            }
            return super.apply(output);
        }
    }

    /**
     * Circulant structured weights. Highly parameter efficient for cyclic patterns.
     */
    public static class CirculantLinear extends Models.BiasActivation {
        public final Tensor kernel;

        public CirculantLinear(int features, UnaryOperator<Tensor> activation, boolean bias) {
            super(features, activation, bias);
            this.kernel = randGaussian(1, features, Math.sqrt(2.0 / features)).grad(true).parameter();
        }

        @Override
        public Tensor apply(Tensor x) {
            int cols = x.cols();
            Tensor output = null;

            for (int i = 0; i < cols; i++) {
                Tensor sum = null;
                for (int j = 0; j < cols; j++) {
                    var ij = (i - j + cols) % cols;
                    var prod = x.slice(j, j + 1).mul(kernel.slice(ij, ij + 1));
                    sum = sum == null ? prod : sum.add(prod);
                }
                output = output == null ? sum : output.concat(sum);
            }
            return super.apply(output);
        }
    }

    public static class NTM implements UnaryOperator<Tensor> {
        private final int inputSize, outputSize, controllerSize, memorySize, memoryVectorSize, numHeads;
        private final Models.Linear controller, interfaceWeights, outputWeights;
        private Tensor memory, readWeights, writeWeights, readVectors;

        public NTM(int inputSize, int outputSize, int controllerSize, int memorySize, int memoryVectorSize, int numHeads) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.controllerSize = controllerSize;
            this.memorySize = memorySize;
            this.memoryVectorSize = memoryVectorSize;
            this.numHeads = numHeads;

            var interfaceSize = numHeads * (memoryVectorSize + 1 + 3) + 3 * memoryVectorSize + 1;

            this.controller = new Models.Linear(inputSize + numHeads * memoryVectorSize, controllerSize, Tensor::tanh, true);
            this.interfaceWeights = new Models.Linear(controllerSize, interfaceSize, null, true);
            this.outputWeights = new Models.Linear(controllerSize + numHeads * memoryVectorSize, outputSize, null, true);

            reset(memorySize, memoryVectorSize, numHeads);
        }

        public void reset(int memorySize, int memoryVectorSize, int numHeads) {
            this.memory = zeros(memorySize, memoryVectorSize);
            this.readWeights = zeros(numHeads, memorySize);
            this.writeWeights = zeros(1, memorySize);
            this.readVectors = zeros(numHeads, memoryVectorSize);
        }

        @Override
        public Tensor apply(Tensor input) {
            var flattenedReadVectors = this.readVectors.reshape(1, numHeads * memoryVectorSize);
            var controllerInput = input.concat(flattenedReadVectors);
            var controllerOutput = this.controller.apply(controllerInput);
            updateMemory(this.interfaceWeights.apply(controllerOutput));
            updateReadVectors();

            var output = controllerOutput.concat(flattenedReadVectors);
            return this.outputWeights.apply(output);
        }

        private void updateMemory(Tensor intrfce) {
            var offset = 0;
            var writeKey = intrfce.slice(offset, offset += memoryVectorSize);
            var writeStrength = intrfce.slice(offset, offset += 1);
            var eraseVector = intrfce.slice(offset, offset += memoryVectorSize);
            var addVector = intrfce.slice(offset, offset += memoryVectorSize);

            writeWeights = contentBasedAddressing(writeKey, writeStrength);

            var retainVector = ones(1, memoryVectorSize).sub(eraseVector.mul(writeWeights.transpose()));
            memory = memory.mul(retainVector).add(writeWeights.transpose().mul(addVector));

            for (var i = 0; i < numHeads; i++) {
                var readKey = intrfce.slice(offset, offset += memoryVectorSize);
                var readStrength = intrfce.slice(offset, offset += 1);
                var shift = intrfce.slice(offset, offset += 3);
                var shiftedWeights = shiftWeighting(contentBasedAddressing(readKey, readStrength), shift);
                readWeights.data.set(i, 0, shiftedWeights.data.get(0, 0));
            }
        }

        private void updateReadVectors() {
            for (var i = 0; i < numHeads; i++) {
                var readWeight = readWeights.slice(i);
                readVectors.data.set(i, 0, readWeight.matmul(memory).data.get(0, 0));
            }
        }

        private Tensor contentBasedAddressing(Tensor key, Tensor strength) {
            var similarityScores = memory.mul(key).sum().div(key.pow(2).sum().sqrt());
            return similarityScores.mul(strength).softmax();
        }

        private Tensor shiftWeighting(Tensor weights, Tensor shift) {
            var rotatedWeights = zeros(1, memorySize);
            for (var i = 0; i < memorySize; i++) {
                for (var j = 0; j < 3; j++) {
                    var index = (i - 1 + j + memorySize) % memorySize;
                    rotatedWeights.data.set(0, i, rotatedWeights.data.get(0, i) + weights.data.get(0, index) * shift.data.get(0, j));
                }
            }
            return rotatedWeights;
        }
    }

    public static class PowerNorm extends Models.LayerNorm {
        public final Tensor alpha;
        private static final double minAlpha = 0, maxAlpha = 1;

        public PowerNorm(int dim, double rate) {
            this(dim, rate, 0.5f);
        }

        public PowerNorm(int dim, double rate, double alphaInitial) {
            super(dim, rate);
            this.alpha = scalar(alphaInitial).grad(true).parameter();
        }

        @Override
        protected Tensor power() {
            return alpha.clip(minAlpha, maxAlpha);
            //.clipSigmoid(minAlpha, maxAlpha);
        }
    }

    public static class FastSlowNetwork extends Models.Linear {
        public final Tensor fastWeight;

        private final int inputSize, hiddenSize;

        private final UnaryOperator<Tensor> activation;

        private final double learningRateFast;

        private final double fastDeltaClip =
                Double.POSITIVE_INFINITY; //DISABLED
        //1;

        private final double fastWeightClip =
                1;
        //Double.POSITIVE_INFINITY; //DISABLED
        //1024;

        double weightDecay = 0.999; // Adjust this value as needed

        private static final float defaultLearningRate =
                //1e-6f;
                1e-5f;

        private final Tensor lastHidden;


        public FastSlowNetwork(int inputSize, int hiddenSize, UnaryOperator<Tensor> activation, boolean hasBias) {
            this(inputSize, hiddenSize, activation, hasBias, defaultLearningRate);
        }

        public FastSlowNetwork(int inputSize, int hiddenSize, UnaryOperator<Tensor> activation, boolean hasBias, double learningRateFast) {
            super(inputSize, hiddenSize, null, hasBias);
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.activation = activation;
            this.learningRateFast = learningRateFast;

            // Initialize fast weights [hiddenSize x hiddenSize]
            this.fastWeight = zeros(hiddenSize, hiddenSize);
            //if (!fasterWeights)
            //    fastWeights.grad(true)
            //.parameter() //if parameter, the optimizer may affect it (normalize, grads etc)

            // Initialize last hidden state [1 x hiddenSize]
            this.lastHidden = zeros(1, hiddenSize);
        }

        @Override
        public Tensor apply(Tensor x) {

            // Ensure input is of shape [batchSize x inputSize]
            if (x.cols() != inputSize)
                throw new IllegalArgumentException("Input tensor must have " + inputSize + " columns, but has " + x.cols());

            var batchSize = x.rows();

            // Compute slow pathway [batchSize x hiddenSize]
            var slowOutput = super.apply(x);

            // Compute fast pathway [batchSize x hiddenSize]
            var fastOutput = applyFast(batchSize);

            // Combine slow and fast pathways [batchSize x hiddenSize]
            var combined = slowOutput.add(fastOutput);

            // Apply activation function [batchSize x hiddenSize]
            var output = activation != null ? activation.apply(combined) : combined;

            // Update fast weights [hiddenSize x hiddenSize]
            updateFast(batchSize == 1 ?
                    output :
                    output.slice(batchSize - 1, batchSize) // For batch processing, update based on the last item in the batch
            );

            return output;
        }

        private Tensor applyFast(int batchSize) {
            synchronized (fastWeight) {
                if (batchSize == 1) {
                    return lastHidden.matmul(fastWeight);
                } else {
                    // For batch processing, we need to repeat lastHidden for each item in the batch
                    var fastOutput = zeros(batchSize, hiddenSize);
                    for (var i = 0; i < batchSize; i++)
                        fastOutput.data.set(i, 0, lastHidden.matmul(fastWeight).data(0));
                    return fastOutput;
                }
            }
        }

        private void updateFast(Tensor output) {
            output = output.detachCopy();
            synchronized (fastWeight) {
                var fastDelta = output.sub(lastHidden);

                if (fastDeltaClip != Double.POSITIVE_INFINITY)
                    fastDelta = fastDelta.clip(-fastDeltaClip, +fastDeltaClip);

                // Apply weight decay
                var fastWeightArray = fastWeight.array();

                if (weightDecay != 1)
                    Util.mul(fastWeightArray, weightDecay);

                //System.out.println(n4(deltaFast.data.normF()) + "\t" + n4(fastWeights.data.normF()));

                Tensor t = lastHidden.transposeMatmul(fastDelta);
                Util.addTo(fastWeightArray,
                        t.array(),//TODO optimize with CommonOps_DDRM
                        learningRateFast);
                //fastWeights.addTo(lastHidden.transpose().matmul(deltaFast));

                if (fastWeightClip != Double.POSITIVE_INFINITY)
                    Util.clampSafe(fastWeightArray, -fastWeightClip, +fastWeightClip);

//                if (Util.sumAbs(fastWeightArray)!=0)
//                    System.out.println(n2(fastWeightArray));

                // Store last output of the batch as last hidden state
                lastHidden.setData(output.data);
            }
        }

        // Method to reset the fast weights and last hidden state
        public void reset() {
            synchronized (this) {
                fastWeight.zero();
                lastHidden.zero();
            }
        }
    }

    /**
     * PredictiveCodingLinearUnit implements a predictive coding–inspired linear layer
     * with Feedback Alignment. It updates weights based on local prediction errors without
     * relying on global backpropagation, making it efficient for CPU execution.
     */
    public static class PredictiveCodingLinearUnit implements UnaryOperator<Tensor> {
        // Weight matrices
        public final Tensor W1; // Input to Hidden
        public final Tensor V;  // Hidden to Output
        public final Tensor B;  // Fixed Feedback Alignment matrix

        // Bias vectors
        private final Tensor b1; // Hidden layer bias
        private final Tensor c;  // Output layer bias

        // Learning rates
        private final double lrWeightsW1;
        private final double lrWeightsV;
        private final double lrBiasesB1;
        private final double lrBiasesC;

        // Activation function (e.g., ReLU)
        private final UnaryOperator<Tensor> activation;

        // Last hidden state for potential recurrent connections
        private final Tensor lastHidden;

        // Random number generator for initializing weights
        private final RandomGenerator rand = new XoRoShiRo128PlusRandom();

        public PredictiveCodingLinearUnit(int inputSize, int hiddenSize, int outputSize,
                                          UnaryOperator<Tensor> activation, boolean hasBias) {
            this(inputSize, hiddenSize, outputSize, activation, hasBias, 0.0001f, 0.0001f, 0.0001f, 0.0001f);
        }

        /**
         * Constructor to initialize the Predictive Coding Linear Unit.
         *
         * @param inputSize      Size of the input layer.
         * @param hiddenSize     Size of the hidden layer.
         * @param outputSize     Size of the output layer.
         * @param activation     Activation function for the hidden layer.
         * @param hasBias        Whether to include bias terms.
         * @param learningRateW1 Learning rate for W1.
         * @param learningRateV  Learning rate for V.
         * @param learningRateB1 Learning rate for b1.
         * @param learningRateC  Learning rate for c.
         */
        public PredictiveCodingLinearUnit(int inputSize, int hiddenSize, int outputSize,
                                          UnaryOperator<Tensor> activation, boolean hasBias,
                                          double learningRateW1, double learningRateV,
                                          double learningRateB1, double learningRateC) {

            this.W1 = randHe(inputSize, hiddenSize);
            this.V = randHe(hiddenSize, outputSize);
            this.B = randHe(outputSize, hiddenSize); // Fixed feedback matrix
            this.b1 = hasBias ? zeros(1, hiddenSize) : null;
            this.c = hasBias ? zeros(1, outputSize) : null;
            this.activation = activation;

            this.lrWeightsW1 = learningRateW1;
            this.lrWeightsV = learningRateV;
            this.lrBiasesB1 = learningRateB1;
            this.lrBiasesC = learningRateC;

            this.lastHidden = zeros(1, hiddenSize);
        }

        @Override
        public Tensor apply(Tensor x) {
            // Forward pass computation
            Tensor hLinear = x.matmul(W1).add(b1);
            Tensor hPred = activation!=null ? activation.apply(hLinear) : hLinear;
            Tensor yPred = hPred.matmul(V).add(c);

            var y = new Tensor(yPred.data, true);
            y.op = new TensorOp(x) {
                @Override
                public void backward(SimpleMatrix grad, SimpleMatrix[] gradOut) {
                    // Compute output error
                    Tensor outputError = new Tensor(grad, false);

                    // Propagate error to hidden layer using feedback matrix B
                    Tensor hiddenError = outputError.matmul(B);

                    // Update weights and biases based on local errors
                    synchronized (PredictiveCodingLinearUnit.this) {
                        // Update W1
                        Tensor deltaW1 = hiddenError.transpose().matmul(x).mul(lrWeightsW1);
                        W1.addThis(deltaW1.array());

                        // Update b1
                        if (b1 != null) {
                            Tensor deltaB1 = hiddenError./*mean().*/mul(lrBiasesB1);
                            b1.addThis(deltaB1.array());
                        }

                        // Update V
                        Tensor deltaV = outputError.transpose().matmul(hPred).mul(lrWeightsV);
                        V.addThis(deltaV.array());

                        // Update c
                        if (c != null) {
                            Tensor deltaC = outputError./*mean().*/mul(lrBiasesC);
                            c.addThis(deltaC.array());
                        }

                        // Optionally store last hidden state
                        lastHidden.setData(hPred.array());
                    }

                    // Compute gradient with respect to input 'x' for propagation
                    if (gradOut != null && gradOut[0]!=null) {
                        // Convert hiddenError to a SimpleMatrix form for computation
                        var e = hiddenError.array();
                        // Compute gradient wrt input: dL/dx = W1^T * (propagated error)
                        var w1t = W1.transpose().array();

                        double[] go = array(gradOut[0]);
                        for (int i = 0; i < go.length; i++)
                            go[i] = w1t[i] * e[i];
                    }
                }
            };
            return y;
        }

        /**
         * Resets the weights and biases to their initial states.
         */
        public void reset() {
            synchronized (this) {
                W1.zero();
                V.zero();
                B.zero();
                if (b1 != null) b1.zero();
                if (c != null) c.zero();
                lastHidden.zero();
            }
        }
    }
}
