package jcog.exe.realtime;

import jcog.Log;
import jcog.Util;
import org.slf4j.Logger;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * thread-based timer
 */
public final class ThreadTimer implements AbstractTimer, Runnable {

    private final ScheduledExecutorService executor =
        Executors.newSingleThreadScheduledExecutor();
//        Executors.newSingleThreadScheduledExecutor(r -> {
//                var t = new Thread(r);
//                //t.setPriority(threadPriority);
//                return t;
//            });
        //Executors.newScheduledThreadPool(1);

    private volatile ScheduledFuture<?> scheduledFuture;
    private volatile FixedRateTimedFuture exe;

    private final AtomicBoolean running = new AtomicBoolean(false);

    @Override
    public <D> TimedFuture<D> schedule(TimedFuture<D> r) {
        synchronized(this) {
            stop();

            var f = (FixedRateTimedFuture) r;
            f.rounds = 1;
            var periodNS = f.periodNS.longValue();
            this.exe = f;
            setPeriodNS(periodNS);
            scheduledFuture = executor.scheduleAtFixedRate(this, 0, periodNS, TimeUnit.NANOSECONDS);
        }
        return r;
    }

    @Override
    public final void run() {
        if (Util.enterAlone(running)) {
            try {
                exe.run();
            } catch (Exception e) {
                logger.error("run", e);
                exe.cancel(true);
            } finally {
                if (exe.isCancelled() || exe.isDone())
                    stop();
                Util.exitAlone(running);
            }
        }
    }


    @Override
    public void execute(Runnable r) {
        throw new UnsupportedOperationException();
    }

    public void stop() {
        synchronized(this) {
            if (scheduledFuture != null) {
                scheduledFuture.cancel(false);
                scheduledFuture = null;
            }
        }
    }

    public final void setPeriodNS(long periodNS) {
        if (exe.setPeriodNSChanged(periodNS))
            schedule(exe); //reschedule
    }

    public final void setPeriodMS(int ms) {
        setPeriodNS(ms * 1_000_000L);
    }

    private static final Logger logger = Log.log(ThreadTimer.class);

}