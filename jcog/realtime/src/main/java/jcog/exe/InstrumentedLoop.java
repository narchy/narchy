package jcog.exe;

import jcog.exe.realtime.AbstractTimer;

import java.util.SortedMap;

import static java.lang.System.nanoTime;

/** loop which collects timing measurements */
public abstract class InstrumentedLoop extends Loop {

    protected InstrumentedLoop() {
        super();
    }

    protected InstrumentedLoop(AbstractTimer timer) {
        super(-1, timer);
    }

    //protected static final int windowLength = 4;
    //public final FloatAveragedWindow dutyTime = new FloatAveragedWindow(windowLength, 0.5f, false);
    //public final FloatAveragedWindow cycleTime =
            //new FloatAveragedWindow(windowLength, 1f/windowLength /* == non-exponential mean? */, false);

//    /** the current cycle time delta (nanoseconds) */
//    private long cycleTimeNS = 0;

    /** actual measured (not ideal) current cycle time delta (seconds) */
    public /* volatile */ float cycleTimeS = 0;
    public /* volatile */ float dutyTimeS;

    protected long last;
    protected long beforeIteration;
    private long lagNS;

    @Override
    protected void beforeNext() {
        beforeIteration = nanoTime();
    }

    @Override
    protected void afterNext() {
        long afterIteration = nanoTime();
        long lastIteration = this.last;
        this.last = afterIteration;

        long cycleTimeNS = afterIteration - lastIteration;
        lagNS = cycleTimeNS - periodNS();

        //this.cycleTime.commit((float) cycleTimeS);

        cycleTimeS = (float) (cycleTimeNS / 1.0E9);

        long dutyTimeNS = afterIteration - beforeIteration;
        this.dutyTimeS = (float) (dutyTimeNS / 1.0E9);
        //this.dutyTime.commit((float) dutyTimeS);
    }


    public void stats(String prefix, SortedMap<String, Object> x) {
//        long periodNS = periodNS();
        x.put(prefix + " lag (ms)", lagNS());

        var cycleTimeS = this.cycleTimeS; //cycleTime.asDouble();
        x.put(prefix + " cycle time mean (s)", cycleTimeS);

        //x.put(prefix + " cycle time vary", cycleTime.getVariance());

        var dutyTimeS = this.dutyTimeS; //dutyTime.asDouble()
        x.put(prefix + " duty time mean (s)", dutyTimeS);

        //x.put(prefix + " duty time vary", dutyTime.getVariance());
    }

    public long lagNS() {
        return lagNS;
    }

    @Override
    protected void starting() {
        last = nanoTime();
        super.starting();
    }

}
