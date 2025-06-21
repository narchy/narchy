package jcog.tensor.rl.dqn;

import jcog.agent.Policy;
import jcog.signal.FloatRange;
import jcog.tensor.Predictor;
import org.jetbrains.annotations.Nullable;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Supplier;

public abstract class PredictorPolicy implements Policy {
    public final Predictor p;

    /** https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html */
    protected final Predictor pTarget;

    /** p -> pAct */
    public final FloatRange transferRate;

    /** "alpha" learning rate */
    public final FloatRange learn = new FloatRange(
        //1E-5f
        //5E-5f
        1E-4f
        //1.0E-3f
        //0.01f
        //0.05f
        //1.0E-4f
        //0.005f
    , 0, 1);

    /** warning: LERP mode may never reach its target */
    private boolean updateHardOrSoft = true;

    protected PredictorPolicy(Supplier<Predictor> p, @Nullable Supplier<Predictor> pTarget) {
        this.p = p.get();
        this.pTarget = pTarget != null ? p.get() : this.p;
        transferRate = p != pTarget ?
            new FloatRange(0.05f, 0, 1) : null;
    }

    @Override
    public void clear(Random rng) {
        p.clear(rng);
        if (p != pTarget)
            pTarget.copyLerp(this.p, 1);
    }

    public void update() {
        if (pTarget != p) {
            if (updateHardOrSoft) {
                if (ThreadLocalRandom.current().nextFloat() < transferRate.floatValue())
                    pTarget.copyLerp(p, 1);
            } else
                pTarget.copyLerp(p, transferRate.floatValue());
        }
    }

    public final double[] predict(double[] input) {
        return p.get(input);
    }
}