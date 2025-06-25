package jcog.tensor.rl.pg3.networks;

import jcog.tensor.Tensor;
import jcog.tensor.Mod;
import jcog.tensor.rl.pg3.configs.NetworkConfig;
import jcog.tensor.rl.pg3.util.AgentUtils;
import org.jetbrains.annotations.NotNull;

import java.util.Objects;

/**
 * A stochastic policy network for Soft Actor-Critic (SAC) that outputs parameters for a Gaussian
 * distribution, samples an action, and then squashes it into a bounded range (typically [-1, 1])
 * using the tanh function.
 *
 * <p>The log probability of an action must be corrected due to the tanh transformation.
 * If `a_raw ~ N(mu, sigma)` and `a_squashed = tanh(a_raw)`, then:
 * `log p(a_squashed) = log p(a_raw) - sum(log(1 - tanh(a_raw)^2 + epsilon))`
 * where the sum is over action dimensions.
 * </p>
 *
 * <p>This network internally uses a {@link GaussianPolicyNet} to get the initial mu and sigma,
 * then applies the squashing and log-prob correction.
 * </p>
 */
public class SquashedGaussianPolicyNet extends Mod {

    private final GaussianPolicyNet gaussianPolicyNet;
    private static final float EPSILON = 1e-6f; // Small constant for numerical stability in log_prob correction

    public SquashedGaussianPolicyNet(@NotNull NetworkConfig config, int stateDim, int actionDim,
                                     float sigmaMin, float sigmaMax) {
        Objects.requireNonNull(config, "NetworkConfig must not be null");
        // SAC policy typically doesn't use an output activation in the base layers,
        // as mu and log_sigma are direct outputs. Tanh is applied after sampling.
        // We might need a specific NetworkConfig setup for this.
        // For now, assume GaussianPolicyNet handles its own output structure correctly.
        this.gaussianPolicyNet = new GaussianPolicyNet(config, stateDim, actionDim, sigmaMin, sigmaMax);
    }

    /**
     * Performs a forward pass to get the squashed action and its log probability.
     *
     * @param state         The input state tensor.
     * @param deterministic If true, returns the mean of the distribution (squashed) instead of sampling.
     * @return An {@link ActionSampleWithLogProb} containing the squashed action and its log probability.
     */
    public ActionSampleWithLogProb sampleAndLogProb(Tensor state, boolean deterministic) {
        AgentUtils.GaussianDistribution baseDistribution = this.gaussianPolicyNet.getDistribution(state);
        Tensor rawAction;

        if (deterministic) {
            rawAction = baseDistribution.mu();
        } else {
            rawAction = baseDistribution.sample();
        }

        Tensor squashedAction = rawAction.tanh();

        // Calculate log_prob correction term: sum_i log(1 - tanh(raw_action_i)^2 + epsilon)
        // log(1 - a^2) = log((1-a)(1+a)). For tanh, this is log(sech^2(x)) = 2 * log(sech(x))
        // Or, more directly, 1 - tanh(x)^2 = sech(x)^2.
        // So, log(1 - tanh(x)^2) = log(sech(x)^2) = 2 * log(sech(x)).
        // However, the common SAC derivation uses:
        // log_pi(a|s) = log_N(mu,sigma)(tanh_inv(a)) - sum(log(1 - a_i^2)) where a is squashed action
        // Here, we have raw_action and squashed_action.
        // log_prob(squashed_action) = log_prob_gaussian(raw_action) - sum(log(1 - squashed_action^2 + EPS))
        // The sum is over each action dimension.
        Tensor logProbCorrection = Tensor.scalar(1.0f)
            .sub(squashedAction.sqr())
            .add(EPSILON) // for numerical stability
            .log()
            .sum(1); // Sum across action dimensions (axis 1)

        Tensor logProbGaussian = baseDistribution.logProb(rawAction); // log prob of the raw, unsquashed action
        Tensor finalLogProb = logProbGaussian.sub(logProbCorrection);

        return new ActionSampleWithLogProb(squashedAction, finalLogProb, baseDistribution.entropy());
    }

    /**
     * Simpler forward pass that only returns the squashed action.
     * Use {@link #sampleAndLogProb(Tensor, boolean)} if log probability is also needed.
     *
     * @param state         The input state tensor.
     * @param deterministic If true, returns the mean of the distribution (squashed).
     * @return The squashed action tensor.
     */
    @Override
    public Tensor forward(Tensor state) {
        // Defaulting to non-deterministic sampling for a simple forward pass.
        // For SAC, usually need log_prob too, so sampleAndLogProb is preferred.
        return sampleAndLogProb(state, false).action();
    }

    /**
     * Gets the mean action (mu) from the underlying Gaussian policy, then squashes it.
     * This is typically used for deterministic evaluation.
     * @param state The input state.
     * @return Squashed mean action.
     */
    public Tensor meanAction(Tensor state) {
        AgentUtils.GaussianDistribution baseDistribution = this.gaussianPolicyNet.getDistribution(state);
        return baseDistribution.mu().tanh();
    }


    /**
     * Sets the training mode for the underlying Gaussian policy network.
     * @param training True for training mode, false for evaluation.
     */
    @Override
    public void train(boolean training) {
        this.gaussianPolicyNet.train(training);
        super.train(training);
    }

    /**
     * Returns the underlying {@link GaussianPolicyNet} instance.
     * @return The internal Gaussian policy network.
     */
    public GaussianPolicyNet getGaussianPolicyNet() {
        return gaussianPolicyNet;
    }

    @Override
    public String toString() {
        return "SquashedGaussianPolicyNet(" + gaussianPolicyNet.toString() + ")";
    }

    /**
     * Container for an action sample, its log probability, and entropy of the original distribution.
     */
    public record ActionSampleWithLogProb(Tensor action, Tensor logProb, Tensor entropy) {
    }
}
