package jcog.tensor.rl.pg2;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg.util.Memory;

import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * A PPO (Proximal Policy Optimization) strategy implementation within the `pg2` system.
 *
 * @deprecated This class is part of an older API. Prefer using {@link jcog.tensor.rl.pg3.PPOAgent}
 *             from the `pg3` package for new development, which offers a more standardized and
 *             configurable PPO implementation.
 */
@Deprecated
public class PPOStrategy extends AbstractStrategy {
    public final PGBuilder.HyperparamConfig h;
    public final PGBuilder.ActionConfig a;
    public final PGBuilder.MemoryConfig m;
    public final PGBuilder.GaussianPolicy policy;
    public final Memory memory; // Changed to Memory interface
    public final PGBuilder.ValueNetwork value;
    public final Tensor.Optimizer policyOpt;
    public final Tensor.Optimizer valueOpt;

    public PPOStrategy(PGBuilder.HyperparamConfig h, PGBuilder.ActionConfig a, PGBuilder.MemoryConfig m,
                       Memory memory, // Changed to Memory interface
                       PGBuilder.GaussianPolicy policy, PGBuilder.ValueNetwork value,
                       Tensor.Optimizer policyOpt, Tensor.Optimizer valueOpt) {
        this.h = Objects.requireNonNull(h);
        this.a = Objects.requireNonNull(a);
        this.m = Objects.requireNonNull(m);
        this.memory = Objects.requireNonNull(memory);
        this.policy = Objects.requireNonNull(policy);
        this.value = Objects.requireNonNull(value);
        this.policyOpt = Objects.requireNonNull(policyOpt);
        this.valueOpt = Objects.requireNonNull(valueOpt);
    }

    @Override
    public Memory getMemory() {
        return memory; // Already returns Memory type
    }

    @Override
    public UnaryOperator<Tensor> getPolicy() {
        return policy;
    }

    @Override
    public PGBuilder.MemoryConfig getMemoryConfig() {
        return m;
    }

    @Override
    public void record(Experience2 e) {
        memory.add(e);
        // --- FIX: The original logic `(bufferFull || e.done())` caused double updates ---
        // in tests where an episode ended at the exact moment the buffer became full.
        // Tying the update to the episode's end OR the buffer filling, and then immediately clearing,
        // resolves the test failures.
        boolean bufferFull = memory.size() >= m.episodeLength();
        if (bufferFull || e.done()) {
            if (memory.size() > 0) {
                update(0);
            }
            memory.clear();
        }
    }

    @Override
    public void update(long totalSteps) {
        setTrainingMode(true, policy, value);
        var episode = memory.getAll();
        if (episode.isEmpty()) {
            setTrainingMode(false, policy, value);
            return;
        }
        updateSteps++;
        var states = Tensor.concatRows(episode.stream().map(Experience2::state).toList());
        var actions = Tensor.concatRows(episode.stream().map(e -> Tensor.row(e.action())).toList());
        var oldLogProbs = Tensor.concatRows(episode.stream().map(Experience2::oldLogProb).filter(Objects::nonNull).toList());
        if (oldLogProbs.rows() != episode.size()) {
            System.err.println("Warning: Mismatch in oldLogProbs size. Recalculating as a fallback.");
            try (var ignored = Tensor.noGrad()) {
                oldLogProbs = policy.getDistribution(states, a.sigmaMin(), a.sigmaMax()).logProb(actions).detach();
            }
        }

        Tensor advantages, returns;
        try (var ignored = Tensor.noGrad()) {
            var values = value.apply(states);
            var advRet = computeGAE(episode, values);
            advantages = advRet[0];
            returns = advRet[1];
        }

        for (var i = 0; i < h.epochs(); i++) {
            var pCtx = new Tensor.GradQueue();
            var vCtx = new Tensor.GradQueue();

            var newDist = policy.getDistribution(states, a.sigmaMin(), a.sigmaMax());
            var newLogProbs = newDist.logProb(actions);
            var ratio = newLogProbs.sub(oldLogProbs).exp();
            var clippedRatio = ratio.clip(1.0f - h.ppoClip(), 1.0f + h.ppoClip());
            var policyLoss = Tensor.min(ratio.mul(advantages), clippedRatio.mul(advantages)).mean().neg();
            var entropy = newDist.entropy().mean();
            policyLoss.sub(entropy.mul(h.entropyBonus())).minimize(pCtx);
            pCtx.optimize(policyOpt);

            value.apply(states).loss(returns, Tensor.Loss.MeanSquared).minimize(vCtx);
            vCtx.optimize(valueOpt);
        }
        setTrainingMode(false, policy, value);
    }

    private Tensor[] computeGAE(List<Experience2> episode, Tensor values) {
        var n = episode.size();
        double[] advantages = new double[n], returns = new double[n];
        double lastGaeLambda = 0;

        var lastExperience = episode.get(n - 1);
        double nextValue = 0.0;
        if (!lastExperience.done()) {
            try (var ignored = Tensor.noGrad()) {
                nextValue = value.apply(lastExperience.nextState()).scalar();
            }
        }

        for (var t = n - 1; t >= 0; t--) {
            var exp = episode.get(t);
            var nonTerminal = exp.done() ? 0.0 : 1.0;
            var delta = exp.reward() + h.gamma() * nextValue * nonTerminal - values.data(t);
            advantages[t] = lastGaeLambda = delta + h.gamma() * h.lambda() * lastGaeLambda * nonTerminal;
            returns[t] = advantages[t] + values.data(t);
            nextValue = values.data(t);
        }

        if (h.normalizeAdvantages()) RLUtils.normalize(advantages);
        if (h.normalizeReturns()) RLUtils.normalize(returns);
        return new Tensor[]{Tensor.row(advantages).transpose(), Tensor.row(returns).transpose()};
    }

    @Override
    public double[] selectAction(Tensor state, boolean deterministic) {
        setTrainingMode(false, policy, value);
        try (var ignored = Tensor.noGrad()) {
            var dist = policy.getDistribution(state, a.sigmaMin(), a.sigmaMax());
            return dist.sample(deterministic).clipUnitPolar().array();
        }
    }
}
