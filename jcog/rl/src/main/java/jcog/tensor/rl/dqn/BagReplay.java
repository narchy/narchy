package jcog.tensor.rl.dqn;

import jcog.Util;
import jcog.math.normalize.FloatNormalizer;
import jcog.pri.PLink;
import jcog.pri.Prioritized;
import jcog.pri.bag.impl.PriReferenceArrayBag;
import jcog.pri.op.PriMerge;
import jcog.tensor.rl.dqn.replay.Experience;
import jcog.tensor.rl.dqn.replay.Replay;
import org.jetbrains.annotations.Nullable;

import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * prioritized replay using a Bag<></>
 */
public class BagReplay extends Replay {

    private final PriReferenceArrayBag<Experience, PLink<Experience>> memory;
    final FloatNormalizer errNormalizer =
            new FloatNormalizer(0, 1024)
                .range(0,1);

    public BagReplay(int cap, int replaysPerIteration) {
        this(cap, 1, replaysPerIteration);
    }

    public BagReplay(int cap, float rememberProb, int replaysPerIteration) {
        super(1, rememberProb, replaysPerIteration);
        this.memory = new PriReferenceArrayBag<>(PriMerge.replace, cap);
    }

    @Override
    @Nullable
    protected Experience sample(Random rng) {
        return memory.sample(rng).get();
    }

    @Override
    public int capacity() {
        return memory.capacity();
    }

    @Override
    public int size() {
        return memory.size();
    }

    protected float importance(double[] q) {
        float qMean = (float) (Util.sumAbs(q) / q.length);
        if (!Float.isFinite(qMean)) return 1;
        float errNorm = errNormalizer.valueOf(qMean);
        //System.out.println(errNormalizer + " " + n4(qMean) + " -> " + n2(errNorm));
        return Prioritized.EPSILON + errNorm;
    }

    @Override
    protected void add(Experience m, double[] qNext) {
        put(m, qNext);
    }

    @Override
    protected void rerun(Experience m, float pri, PolicyAgent agent) {
        double[] dq = agent.run(m, null, pri);

        PLink<Experience> exist = memory.get(m);

        /* priBefore > nextPri indicates learning */ //float priBefore = exist.pri();

        exist.pri(importance(dq));
        memory.commit(null);
    }

    private void put(Experience m, double[] qNext) {
        memory.put(new PLink<>(m, importance(qNext)));
        memory.commit(/*null*/);
    }

    @Override
    protected void pop(RandomGenerator rng) {
        /* nothing needs done as bag handles capacity */
    }
}