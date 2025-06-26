package jcog.tensor.rl.pg2.util; // New package

import jcog.Util;
import jcog.tensor.Tensor;

import java.util.Arrays;
import java.util.function.UnaryOperator;

public enum AgentUtils { // Using enum for utility class style
    ; // No instances

    public static final double EPSILON = 1e-8; // Epsilon for numerical stability

    /**
     * Normalizes an array of doubles to have zero mean and unit variance.
     * Modifies the array in-place.
     * @param x The array to normalize.
     */
    public static void normalize(double[] x) {
        if (x == null || x.length == 0) {
            return;
        }
        var meanVar = Util.variance(x); // Assumes Util.variance returns [mean, variance]
        double mean = meanVar[0];
        double stddev = Math.sqrt(meanVar[1]);
        // Handle cases where stddev is zero or very close to zero
        if (stddev < EPSILON) {
            // If stddev is effectively zero, array elements are likely all the same.
            // Normalizing would lead to NaNs or Infs.
            // Set all elements to 0 in this case, or leave as is if that's preferred.
            // For advantages, setting to 0 if all are same (so stddev is 0) is common.
            Arrays.fill(x, 0.0);
            return;
        }
        for (var i = 0; i < x.length; i++) {
            x[i] = (x[i] - mean) / stddev; // No need to add EPSILON here if stddev check is done
        }
    }

    /**
     * Performs a soft update of target network parameters from source network parameters.
     * target_params = (1 - tau) * target_params + tau * source_params
     *
     * @param source The source network (UnaryOperator<Tensor> representing the model).
     * @param target The target network (UnaryOperator<Tensor> representing the model).
     * @param tau The interpolation parameter (typically a small value like 0.005).
     */
    public static void softUpdate(UnaryOperator<Tensor> source, UnaryOperator<Tensor> target, float tau) {
        if (source == null || target == null) {
            System.err.println("Warning: softUpdate called with null source or target network.");
            return;
        }
        var sourceParams = Tensor.parameters(source).toList();
        var targetParams = Tensor.parameters(target).toList();

        if (sourceParams.isEmpty() && targetParams.isEmpty()) {
            return;
        }
        if (sourceParams.size() != targetParams.size()) {
            throw new IllegalArgumentException("Source and target networks have a different number of parameters for soft update: " +
                                               sourceParams.size() + " vs " + targetParams.size());
        }

        for (var i = 0; i < sourceParams.size(); i++) {
            var sParam = sourceParams.get(i);
            var tParam = targetParams.get(i);
            tParam.setData(tParam.mul(1.0f - tau).add(sParam.mul(tau)));
        }
    }

    /**
     * Performs a hard update (tau=1.0) of target network parameters from source network parameters.
     * Essentially copies source parameters to target parameters.
     *
     * @param source The source network.
     * @param target The target network.
     */
    public static void hardUpdate(UnaryOperator<Tensor> source, UnaryOperator<Tensor> target) {
        softUpdate(source, target, 1.0f);
    }

    public record GaussianDistribution(Tensor mu, Tensor sigma) {
        public Tensor sample(boolean deterministic) {
            return deterministic ? mu : rsample();
        }

        public Tensor rsample() {
            return mu.add(sigma.mul(Tensor.randGaussian(sigma.rows(), sigma.cols())));
        }

        public Tensor logProb(Tensor action) {
            var variance = sigma.sqr().add((float) EPSILON);
            var diffSq = action.sub(mu).sqr();
            var term1 = diffSq.div(variance);
            var term2 = variance.log().add(Math.log(2 * Math.PI));
            var logProbPerDim = term1.add(term2).mul(-0.5f);

            if (logProbPerDim.cols() > 1 && logProbPerDim.rows() > 0) { // Ensure not scalar and has columns to sum
                return logProbPerDim.sum(1);
            } else {
                return logProbPerDim;
            }
        }

        public Tensor entropy() {
            var logSigma = sigma.add((float) EPSILON).log();
            var entropyPerDim = logSigma.add(0.5 * (1.0 + Math.log(2 * Math.PI)));

            if (entropyPerDim.cols() > 1 && entropyPerDim.rows() > 0) { // Ensure not scalar and has columns to sum
                return entropyPerDim.sum(1);
            } else {
                return entropyPerDim;
            }
        }
    }
}
