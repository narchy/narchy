package jcog.tensor.rl.pg;

import jcog.Fuzzy;
import jcog.Util;
import jcog.agent.Agent;
import jcog.math.normalize.FloatNormalizer;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.signal.FloatRange;
import jcog.tensor.Tensor;
import org.eclipse.collections.api.block.function.primitive.FloatToFloatFunction;
import org.jetbrains.annotations.Nullable;

import java.util.function.Consumer;

public abstract class AbstrPG {
    public final int inputs; // Changed to public final
    public final int outputs; // Changed to public final
    protected final RandomBits rng = new RandomBits(new XoRoShiRo128PlusRandom()); // Keep protected final for internal use
    public final Consumer<double[]> actionFilter; // Changed to public final

    protected AbstrPG(int inputs, int outputs, Consumer<double[]> actionFilter) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.actionFilter = Objects.requireNonNull(actionFilter, "actionFilter cannot be null");
    }

    // Overloaded constructor for default actionFilter
    protected AbstrPG(int inputs, int outputs) {
        this(inputs, outputs, a -> {}); // Default no-op action filter
    }


    public abstract double[] act(double[] input, double reward, boolean done);

    protected abstract double[] _action(Tensor state, boolean deterministic);

    protected void reviseAction(double[] actionPrev) {
    }

    protected final double[] action(Tensor currentState, boolean deterministic) {
        var a = _action(currentState, deterministic);
        actionFilter.accept(a);
        return a;
    }

    public PGAgent agent() {
        return new PGAgent(this);
    }

    public static class PGAgent extends Agent {
        public final AbstrPG pg;
        public final FloatRange actionRevise = FloatRange.unit(1.0f);
        private final FloatToFloatFunction rewardNorm = new FloatNormalizer(2, 1000);
        private final double[] lastAction;
        public boolean rewardNormalize, inputPolarize, rewardPolarize;

        public PGAgent(AbstrPG pg) {
            super(pg.inputs, pg.outputs);
            this.pg = pg;
            this.lastAction = new double[pg.outputs];
        }

        @Override
        public void apply(@Nullable double[] inputPrev, double[] actionPrev, float reward, double[] input, double[] actionNext) {
            Util.replaceNaNwithRandom(input, pg.rng);
            double r = Double.isNaN(reward) ? pg.rng.nextFloat() : reward;
            if (inputPolarize) Fuzzy.polarize(input);
            if (pg.rng.nextBoolean(actionRevise.asFloat())) pg.reviseAction(this.lastAction);
            if (rewardNormalize) r = rewardNorm.valueOf((float) r);
            if (rewardPolarize) r = Fuzzy.polarize(r);
            var a = pg.act(input, r, false);
            Fuzzy.unpolarize(a);
            System.arraycopy(a, 0, actionNext, 0, actionNext.length);
            System.arraycopy(a, 0, this.lastAction, 0, a.length);
        }
    }
}
