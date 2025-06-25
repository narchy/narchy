package jcog.tensor.rl.pg3.configs;

/**
 * Configuration for the SAC (Soft Actor-Critic) agent.
 * <p>
 * This class encapsulates all hyperparameters and structural settings for the SAC agent,
 * its actor (policy), critics (Q-networks), and entropy temperature.
 */
public record SACAgentConfig(
    NetworkConfig policyNetworkConfig,   // For the stochastic policy (actor)
    NetworkConfig qNetworkConfig,        // For the Q-networks (critics)
    OptimizerConfig policyOptimizerConfig,
    OptimizerConfig qOptimizerConfig,
    OptimizerConfig alphaOptimizerConfig,  // For automatic temperature tuning
    MemoryConfig memoryConfig,
    HyperparamConfig hyperparams,        // Common ones like gamma
    SACHyperparams sacHyperparams        // SAC specific ones like tau, alpha, target update interval
) {

    /**
     * SAC-specific hyperparameters.
     *
     * @param tau                       Rate for soft target updates (polyak averaging) for Q-networks.
     * @param initialAlpha              Initial value for the temperature parameter (entropy coefficient).
     *                                  If null and automaticAlphaTuning is true, a heuristic might be used.
     * @param automaticAlphaTuning      Whether to automatically tune the temperature parameter 'alpha'.
     * @param targetEntropy             The target entropy for automatic alpha tuning. If null, typically defaults
     *                                  to -action_dimension.
     * @param targetUpdateInterval      Frequency of updates for the target Q-networks (in terms of critic updates).
     * @param useFixedAlpha             If true, alpha is fixed to initialAlpha and not tuned. Overrides automaticAlphaTuning.
     */
    public record SACHyperparams(
        Float tau,
        Float initialAlpha,
        Boolean automaticAlphaTuning,
        Float targetEntropy, // Can be null, agent will use default based on actionDim
        Integer targetUpdateInterval,
        Boolean useFixedAlpha
    ) {
        public SACHyperparams {
            if (tau == null) tau = 0.005f;
            if (initialAlpha == null) initialAlpha = 0.2f; // Common starting point
            if (automaticAlphaTuning == null) automaticAlphaTuning = true; // SAC often uses tuning
            // targetEntropy is often set dynamically based on action dimension, so null is acceptable.
            if (targetUpdateInterval == null) targetUpdateInterval = 1; // How often to update target Q networks
            if (useFixedAlpha == null) useFixedAlpha = false;

            if (useFixedAlpha && automaticAlphaTuning) {
                //System.err.println("SACConfig: useFixedAlpha is true, so automaticAlphaTuning will be ignored.");
                //automaticAlphaTuning = false; // Ensure consistency if fixed alpha is chosen
            }
        }
    }

    public SACAgentConfig {
        if (policyNetworkConfig == null) throw new IllegalArgumentException("Policy network config cannot be null");
        if (qNetworkConfig == null) throw new IllegalArgumentException("Q-network config cannot be null");

        if (policyOptimizerConfig == null) policyOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null, null);
        if (qOptimizerConfig == null) qOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null, null);
        if (alphaOptimizerConfig == null) alphaOptimizerConfig = new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null, null); // For alpha tuning

        if (memoryConfig == null) {
            MemoryConfig.ReplayBufferConfig replayDefaults = new MemoryConfig.ReplayBufferConfig(1_000_000, 256); // Common batch size for SAC
            memoryConfig = new MemoryConfig(null, replayDefaults, null); // Episode length null for off-policy
        }
        if (hyperparams == null) hyperparams = new HyperparamConfig(
            0.99f, null, null, false, false, 0.0f, 1 // Default gamma
        );
        if (sacHyperparams == null) sacHyperparams = new SACHyperparams(null, null, null, null, null, null);
    }

    // Convenience constructor with some defaults
    public SACAgentConfig(int stateDim, int actionDim) {
        this(
            new NetworkConfig(3e-4f, 256, 256), // policy
            new NetworkConfig(3e-4f, 256, 256), // q-networks
            new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null, null), // policy opt
            new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null, null), // q opt
            new OptimizerConfig(OptimizerConfig.OptimizerType.ADAM, 3e-4f, null, null), // alpha opt
            new MemoryConfig(null, new MemoryConfig.ReplayBufferConfig(1_000_000, 256), null),
            new HyperparamConfig(0.99f, null, null, false, false, 0.0f, 1),
            new SACHyperparams(0.005f, 0.2f, true, (float) -actionDim, 1, false) // targetEntropy often -actionDim
        );
    }
}
