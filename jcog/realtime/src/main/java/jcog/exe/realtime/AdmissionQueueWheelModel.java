package jcog.exe.realtime;

import jcog.data.list.Lst;
import jcog.util.ArrayUtil;
import org.jctools.queues.MessagePassingQueue;
import org.jctools.queues.MpmcArrayQueue;

import java.util.Arrays;
import java.util.Collection;

import static jcog.exe.realtime.TimedFuture.CANCELLED;
import static jcog.exe.realtime.TimedFuture.READY;

/**
 * uses central concurrent admission queue which is drained each cycle.
 * the wheel queues are (hopefully fast) ArrayDeque's safely accessed from one thread only
 */
public class AdmissionQueueWheelModel extends WheelModel {

    private final MessagePassingQueue<TimedFuture> incoming;

    private final Lst<TimedFuture>[] wheel;

    public AdmissionQueueWheelModel(int wheels, long resolution, int BUCKET_CAPACITY) {
        super(wheels, resolution);

        this.wheel = new Lst[wheels];
        for (var i = 0; i < wheels; i++)
            wheel[i] = new Lst<>(0, new TimedFuture[BUCKET_CAPACITY]);

        var ADMISSION_CAPACITY = (wheels * BUCKET_CAPACITY)/4;
        incoming = new MpmcArrayQueue<>(ADMISSION_CAPACITY);
    }


    @Override public boolean accept(TimedFuture<?> r, HashedWheelTimer t) {
        return incoming.offer(r);
    }

    @Override public boolean reschedule(int wheel, TimedFuture r) {
        return this.wheel[wheel].add(r);
    }

    /**
     * HACK TODO note this method isnt fair because it implicitly prioritizes 'tenured' items that were inserted and remained.
     * instead it should behave like ConcurrentQueueWheelModel's impl
     */
    @Override public int run(int c) {
        incoming.drain(this);

        var Q = wheel[c];

        var n = Q.size();
        if (n == 0) return 0;

        var q = Q.array();
        var removed = 0;
        for (var i = 0; i < n; i++) {
            if (!run(q, i))
                removed++;
        }
        if (removed > 0)
            clean(n, Q);
        return n;
    }

    private static void clean(int n, Lst<TimedFuture> q) {
        q.removeNulls();
        q.setSize(n - ArrayUtil.sortNullsToEnd(q.array(), 0, n));
    }

    private boolean run(TimedFuture[] q, int i) {
        var r = q[i];
        if (switch (r.queueState()) {
            case CANCELLED -> true;
            case READY -> { r.execute(timer); yield true;  }
            default -> false;
        }) {
            q[i] = null;
            return false;
        }
        return true;
    }

    @Override
    public boolean isEmpty() {
        return incoming.isEmpty() && size()==0;
    }

    @Override
    public int size() {
        return Arrays.stream(wheel).mapToInt(Collection::size).sum();
    }


}