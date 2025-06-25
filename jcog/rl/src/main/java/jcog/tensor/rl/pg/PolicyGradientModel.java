package jcog.tensor.rl.pg;

import jcog.tensor.Tensor;
import jcog.tensor.rl.pg.util.Experience2;
import jcog.tensor.rl.pg2.AbstrPG;
import jcog.tensor.rl.pg2.PGStrategy;
import jcog.tensor.rl.pg2.PPOStrategy;

import java.util.Objects;
import java.util.function.Consumer;

public class PolicyGradientModel extends AbstrPG {
    public final PGStrategy strategy; // Made public final
    public final boolean isOffPolicy; // Made public final

    // Internal state variables, remain private and mutable
    private Tensor lastState;
    private double[] lastAction;
    private long totalSteps;
    private Tensor lastLogProb;

    // Constructor now accepts actionFilter to pass to AbstrPG
    public PolicyGradientModel(int inputs, int outputs,
                               PGStrategy strategy,
                               Consumer<double[]> actionFilter) {
        super(inputs, outputs, actionFilter); // Pass actionFilter to super constructor
        this.strategy = Objects.requireNonNull(strategy, "strategy cannot be null");
        this.isOffPolicy = strategy.isOffPolicy();
        this.strategy.initialize(this); // Initialize strategy after all final fields are set
    }

    // Overloaded constructor for default actionFilter
    public PolicyGradientModel(int inputs, int outputs, PGStrategy strategy) {
        this(inputs, outputs, strategy, a -> {
        }); // Default no-op action filter
    }

    @Override
    public double[] act(double[] input, double reward, boolean done) {
        totalSteps++;
        var currentState = Tensor.row(input);
        if (lastState != null) {
            strategy.record(new Experience2(lastState, lastAction, reward, currentState, done, lastLogProb));
        }
        var currentAction = action(currentState, !strategy.isTrainingMode() && isOffPolicy);
        if (done) {
            lastState = null;
            lastAction = null;
            lastLogProb = null;
        } else {
            lastState = currentState;
            lastAction = currentAction;
        }
        return currentAction;
    }

    @Override
    protected double[] _action(Tensor state, boolean deterministic) {
        // For PPO, we need to capture the log probability of the action taken.
        if (strategy instanceof PPOStrategy ppo) {
            var dist = ppo.policy.getDistribution(state, ppo.a.sigmaMin(), ppo.a.sigmaMax());
            var actionTensor = dist.sample(deterministic).clipUnitPolar();
            this.lastLogProb = dist.logProb(actionTensor).detach();
            return actionTensor.array();
        } else {
            this.lastLogProb = null;
            return strategy.selectAction(state, deterministic);
        }
    }

    @Override
    protected void reviseAction(double[] actionPrev) {
        if (this.lastAction != null) System.arraycopy(actionPrev, 0, this.lastAction, 0, actionPrev.length);
    }
}
