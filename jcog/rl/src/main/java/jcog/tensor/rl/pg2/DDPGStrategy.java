package jcog.tensor.rl.pg2;

import jcog.Util;
import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Memory;
import jcog.tensor.rl.pg.util.ReplayBuffer2;
import jcog.tensor.rl.pg2.PGBuilder.DeterministicPolicy;
import jcog.tensor.rl.pg2.PGBuilder.QNetwork;
import jcog.tensor.rl.pg3.configs.NetworkConfig;

import java.util.Objects;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * A DDPG (Deep Deterministic Policy Gradient) strategy implementation within the `pg2` system.
 *
 * @deprecated This class is part of an older API. The `pg3` package is the focus for new development.
 *             For DDPG or similar algorithms, consider building custom solutions using `pg3` components
 *             (e.g., {@link jcog.tensor.rl.pg3.networks.GaussianPolicyNet} adapted for deterministic actions,
 *             {@link jcog.tensor.rl.pg3.networks.ValueNet} potentially adapted for Q-values, and custom update logic).
 */
@Deprecated
public class DDPGStrategy extends OffPolicyStrategy {
    public final PGBuilder.HyperparamConfig h;
    public final PGBuilder.ActionConfig a; // Added to store the action config
    public final DeterministicPolicy policy;
    public final DeterministicPolicy targetPolicy;
    public final QNetwork critic;
    public final QNetwork targetCritic;
    public final Noise noise;
    public final Tensor.Optimizer policyOpt;
    public final Tensor.Optimizer criticOpt;
    public final int outputs; // Storing for clarity, as it's used for noise creation

    // Internal state for updates, not part of the public final configuration fields
    private final Tensor.GradQueue vCtx = new Tensor.GradQueue();
    private final Tensor.GradQueue pCtx = new Tensor.GradQueue();

    public DDPGStrategy(PGBuilder.HyperparamConfig h, PGBuilder.ActionConfig a, PGBuilder.MemoryConfig m,
                        Memory memory, // Changed to Memory interface
                        DeterministicPolicy policy, QNetwork critic,
                        DeterministicPolicy targetPolicy, QNetwork targetCritic,
                        Tensor.Optimizer policyOpt, Tensor.Optimizer criticOpt,
                        int outputs) {
        super(m, memory);
        this.h = Objects.requireNonNull(h);
        this.a = Objects.requireNonNull(a); // Store action config
        this.policy = Objects.requireNonNull(policy);
        this.critic = Objects.requireNonNull(critic);
        this.targetPolicy = Objects.requireNonNull(targetPolicy);
        this.targetCritic = Objects.requireNonNull(targetCritic);
        this.policyOpt = Objects.requireNonNull(policyOpt);
        this.criticOpt = Objects.requireNonNull(criticOpt);
        this.outputs = outputs;
        this.noise = Noise.create(a.noise(), outputs); // Noise uses the stored action config
    }

    @Override
    public boolean isOffPolicy() {
        return true;
    }

    @Override
    public PGBuilder.MemoryConfig getMemoryConfig() {
        return m;
    }

    @Override
    public UnaryOperator<Tensor> getPolicy() {
        return policy;
    }

    @Override
    public void update(long totalSteps) {
        setTrainingMode(true, policy, critic);
        updateSteps++;
        var batch = toBatch(memory.sample(m.replayBuffer().batchSize()));
        Tensor y;
        try (var ignored = Tensor.noGrad()) {
            var nextActions = targetPolicy.apply(batch.nextStates());
            var targetQ = targetCritic.apply(batch.nextStates(), nextActions);

            var nonTerminal = batch.dones().neg().add(1.0f);
            y = batch.rewards().add(targetQ.mul(h.gamma()).mul(nonTerminal));

        }
        critic.apply(batch.states(), batch.actions()).loss(y, Tensor.Loss.MeanSquared).minimize(vCtx);
        vCtx.optimize(criticOpt);

        if (updateSteps % h.policyUpdateFreq() == 0) {
            critic.apply(batch.states(), policy.apply(batch.states())).mean().neg().minimize(pCtx);
            pCtx.optimize(policyOpt);
            RLUtils.softUpdate(policy, targetPolicy, h.tau());
            RLUtils.softUpdate(critic, targetCritic, h.tau());
        }
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        setTrainingMode(false, policy);
        try (var ignored = Tensor.noGrad()) {
            var action = policy.apply(state).array();
            if (!deterministic) {
                noise.apply(action, model.rng);
                for (var i = 0; i < action.length; i++) action[i] = Util.clampSafePolar(action[i]);
            }
            //System.out.println(Str.n2(action));
            return action;
        }
    }

    private interface Noise {
        static Noise create(PGBuilder.ActionConfig.NoiseConfig config, int actionDim) {
            return switch (config.type()) {
                case OU -> new OUNoise(actionDim, config.stddev());
                case GAUSSIAN -> (action, rng) -> {
                    for (var i = 0; i < action.length; i++) action[i] += rng.nextGaussian(0, config.stddev());
                };
                case NONE -> (action, rng) -> {};
            };
        }

        void apply(double[] action, RandomGenerator rng);
    }

    private static class OUNoise implements Noise {
        private final double[] state;
        private final double mu = 0, theta = 0.15, sigma;

        OUNoise(int size, double sigma) {
            this.state = new double[size];
            this.sigma = sigma;
        }

        @Override
        public void apply(double[] action, RandomGenerator rng) {
            for (var i = 0; i < state.length; i++) {
                var dx = theta * (mu - state[i]) + sigma * rng.nextGaussian();
                state[i] += dx;
                action[i] += state[i];
            }
        }
    }

    public static DDPGStrategy ddpgStrategy(int i, int o, PGBuilder.ActionConfig actionConfig, NetworkConfig policyNetConfig, NetworkConfig valueNetConfig, PGBuilder.MemoryConfig memoryConfig, PGBuilder.HyperparamConfig hyperparams) {
        var ddpgActionConfig = actionConfig.withDistribution(PGBuilder.ActionConfig.Distribution.DETERMINISTIC);
        var ddpgPolicyNetConfig = policyNetConfig.withOutputActivation(
            Tensor.UNITIZE_POLAR
            //Tensor.TANH
        );

        var ddpgPolicy = new DeterministicPolicy(ddpgPolicyNetConfig, i, o);
        var ddpgTargetPolicy = new DeterministicPolicy(ddpgPolicyNetConfig, i, o);
        var ddpgPolicyOpt = ddpgPolicyNetConfig.optimizer().build();

        var ddpgCritic = new QNetwork(valueNetConfig, i, o); // DDPG uses a Q-network as critic
        var ddpgTargetCritic = new QNetwork(valueNetConfig, i, o);
        var ddpgCriticOpt = valueNetConfig.optimizer().build();

        RLUtils.hardUpdate(ddpgPolicy, ddpgTargetPolicy);
        RLUtils.hardUpdate(ddpgCritic, ddpgTargetCritic);

        var ddpgMemory = new ReplayBuffer2(memoryConfig.replayBuffer().capacity());
        return new DDPGStrategy(hyperparams, ddpgActionConfig, memoryConfig, ddpgMemory, ddpgPolicy, ddpgCritic,
                ddpgTargetPolicy, ddpgTargetCritic, ddpgPolicyOpt, ddpgCriticOpt, o);
    }

}
