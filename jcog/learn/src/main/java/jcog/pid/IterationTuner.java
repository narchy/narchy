package jcog.pid;

import jcog.Util;
import jcog.math.FloatMeanWindow;
import jcog.random.RandomBits;

import static jcog.Str.n2;
import static jcog.Str.timeStr;

/** decides # of iterations for a target total time, by periodically profiling a procedure */
abstract public class IterationTuner extends AbstractProfiler {

    private boolean trace = false;

    protected int itersMax = 32 * 1024;

    float momentumIncrease =
            //0.9f,
            0.93f,
        momentumDecrease =
            0.88f;
    //0.8f; /* medium */
    //0.25f; /* fast */
    //0 /* instant */;

    private static final int history = 16;


    private final FloatMeanWindow iterTimes = new FloatMeanWindow(history).minSize(1);

    protected int iters = 1;

    protected IterationTuner(RandomBits rng) {
        super(rng);
    }


    @Override
    protected void _run() {
        next(iters);
    }

    @Override
    protected void profiled(long timeNS) {
        double _timePerIter = timeNS / ((double) iters);
        double timePerIter = iterTimes.acceptAndGetMean((float)_timePerIter);
        //if (iterTimes.size() < iterTimes.capacity()) return; //else iterTimes.clear();
        _update(timePerIter);
    }

    private void _update(double timePerIter) {
        long target = targetPeriodNS();

        double actual = timePerIter * iters;

        iters = Util.clampSafe(iterations(iters, actual, target), 1, itersMax);

        if (trace)
            log(actual, target);
    }

    private void log(double time, long targetTime) {
        double accuracy = Util.pctDiff(time, targetTime);
        System.out.println(timeStr(time) + " -> " + timeStr(targetTime) + " " +
                (time >= targetTime ? '+' : '-') + n2(accuracy) + " => " + iters);
    }

    private int iterations(int iters, double time, double targetTime) {
        //return (int) Math.round(pid.out(time, targetTime));

//        BANG BANG control
//        int step = 10;
//        if (time > targetTime) {
//            return iters-step;
//        } else if (time < targetTime) {
//            return iters+step;
//        } else
//            return iters;

        //proportional with split momentum
        double d = targetTime/time;
        float momentum = targetTime > time ? momentumIncrease : momentumDecrease;
        return (int) Math.round(Util.lerpSafe(momentum, iters * d, iters));
    }

    abstract protected void next(int iterations);

    /** in nanoseconds  */
    abstract public long targetPeriodNS();
}
