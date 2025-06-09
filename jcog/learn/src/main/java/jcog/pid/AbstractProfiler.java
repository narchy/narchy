package jcog.pid;

import jcog.random.RandomBits;

import static java.lang.System.nanoTime;

public abstract class AbstractProfiler {

    protected final RandomBits rng;
    public float profileProb = 1/64f;

    public AbstractProfiler(RandomBits rng) {
        this.rng = rng;
    }

    public final void run() {
        if (!profile())
            _run();
    }

    private boolean profile() {
        if (!rng._nextBooleanFast8(profileProb))
            return false;

        long s = nanoTime();

        _run();

        profiled(nanoTime() - s);
        return true;
    }

    protected abstract void _run();

    protected abstract void profiled(long timeNS);

}
