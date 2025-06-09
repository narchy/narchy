package jcog.exe.realtime;

import java.util.concurrent.Delayed;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public abstract class FixedRateTimedFuture extends AbstractTimedFuture<Void> {

    int offset;
    /**
     * adjustable while running
     */
    protected final/* volatile */ AtomicLong periodNS = new AtomicLong();

    protected FixedRateTimedFuture() {
        super();
    }

    FixedRateTimedFuture(int rounds, long periodNS) {
        super(rounds);
        setPeriodNS(periodNS);
    }

    @Override
    public void execute(AbstractTimer t) {
        super.execute(t);
        t.schedule(this);
    }

    @Override
    public final boolean isPeriodic() {
        return true;
    }

    public final void setPeriodMS(long periodMS) {
        setPeriodNS(periodMS * 1_000_000);
    }

    public final void setPeriodNS(long periodNS) {
        this.periodNS.set(periodNS);
    }

    public final boolean setPeriodNSChanged(long periodNS) {
        return this.periodNS.getAndSet(periodNS)!=periodNS;
    }

    public final int offset(long resolution) {
        return offset;
    }

    public final long periodNS() {
        return periodNS.getOpaque();
    }


    /** period in seconds */
    public final double periodSec() {
        return periodNS()/1.0E9;
    }

    @Override
    public final Void get() {
        return null;
    }

    @Override
    public final Void get(long timeout, TimeUnit unit) {
        return null;
    }

    @Override
    public final int compareTo(Delayed o) {
        throw new UnsupportedOperationException();
    }

}