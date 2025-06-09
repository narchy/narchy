package jcog.exe.realtime;

import jcog.Util;
import org.jctools.queues.MessagePassingQueue;
import org.jctools.queues.MpmcArrayQueue;

import java.util.Arrays;
import java.util.function.ToIntFunction;

import static jcog.exe.realtime.TimedFuture.*;

/** where each wheel is simply its own concurrent queue
 * TODO this should not involve both poll and offer if the item remains in the wheel
 * */
public  class ConcurrentQueueWheelModel extends WheelModel {

    /** the wheels (array of queues) */
    private final MessagePassingQueue<TimedFuture>[] q;

    public ConcurrentQueueWheelModel(int wheels, int queueCapacity, long resolution) {
        super(wheels, resolution);
        assert(wheels > 1);
        q = new MessagePassingQueue[wheels];
        for (int i = 0; i < wheels; i++)
            q[i] = new MpmcArrayQueue<>(queueCapacity);
    }

    @Override
    public int run(int c) {
        var q = this.q[c];
        int n = q.size();
        //TODO if n=2 and the previous or next queue is empty try moving one of the items there. this will distribute items across wheels so each has an ideal 0 or 1 size
        switch (n) {
            case 0 -> { return 0; }
            case 1 -> {
                //special optimized case: the only element can be peek'd without poll/offer in case it remains pending
                var r = q.peek();
                switch (r.queueState()) {
                    case CANCELLED -> q.poll();
                    case READY -> {
                        q.poll();
                        r.execute(timer);
                    }
                    case PENDING -> { } //<--- ideally most common path
                }
            }
            default -> {
                for (int i = 0; i < n; i++) {
                    TimedFuture timedFuture = q.poll();
                    switch (timedFuture.queueState()) {
                        case CANCELLED -> { }
                        case READY -> timedFuture.execute(timer);
                        case PENDING -> q.offer(timedFuture); //re-insert
                    }
                }
            }
        }
        return n;
    }

    @Override
    public boolean accept(TimedFuture<?> r, HashedWheelTimer t) {
        int o = r.offset(resolution);
		if (reschedule(idx(t.cursorActive() + o), r)) {
			return true;
		} else {
			t.reject(r);
			return false;
		}
    }

    @Override
    public boolean reschedule(int wheel, TimedFuture r) {

        int remain = q.length - 1;
        do {
            if (q[wheel].offer(r))
                return true;
            if (++wheel == q.length) wheel = 0;
        } while (--remain > 0);

        return false;
    }

    @Override
    public int size() {
        return Util.sum((ToIntFunction<MessagePassingQueue>) MessagePassingQueue::size, q);
    }

    @Override
    public boolean isEmpty() {
        return Arrays.stream(q).allMatch(MessagePassingQueue::isEmpty);
    }


}

//package jcog.exe.realtime;
//
//import jcog.TODO;
//import jcog.Util;
//
//import java.util.Arrays;
//import java.util.Collection;
//import java.util.Queue;
//import java.util.function.Supplier;
//import java.util.function.ToIntFunction;
//
//import static jcog.exe.realtime.TimedFuture.*;
//
///** TODO this should not involve both poll and offer if the item remains in the wheel */
//public class QueueWheelModel extends WheelModel {
//
//	/**
//	 * the wheels (array of queues)
//	 */
//	final Queue<TimedFuture>[] q;
//
//	public QueueWheelModel(int wheels, long resolution, Supplier<Queue<TimedFuture>> queueBuilder) {
//		super(wheels, resolution);
//		assert (wheels > 1);
//		q = new Queue[wheels];
//		for (int i = 0; i < wheels; i++)
//			q[i] = queueBuilder.get();
//	}
//
//	@Override
//	public int run(int c) {
//
//
//		//TODO if n=2 and the previous or next queue is empty try moving one of the items there. this will distribute items across wheels so each has an ideal 0 or 1 size
//
//		int n = 0, limit = Integer.MAX_VALUE;
//		Queue<TimedFuture> q = this.q[c];
//
//		TimedFuture r;
//		while ((r = q.poll()) != null) {
//			n++;
//			switch (r.queueState()) {
//				case CANCELLED:
//					/* nop */
//					break;
//				case READY:
//					r.execute(timer);
//					break;
//				case PENDING:
//				    if (limit == Integer.MAX_VALUE)
//                        limit = n + q.size(); //defer calculating queue size until the first reinsert otherwise it will keep polling what is offered in this loop
//					//re-insert
//					if (!q.offer(r)) {
//						//OVERFLOW
//						if (!reschedule((c+1)%wheels, r))
//							throw new TODO(); //TODO try all other queues in sequence
//
//					}
//					break;
//			}
//			if (n >= limit)
//			    break; //exit before polling PENDING tasks recycled to the back of the queue
//		}
//
//		return n;
//	}
//
//	@Override
//	public boolean accept(TimedFuture<?> r, HashedWheelTimer t) {
//		return t.reschedule(r); //immediately
//	}
//
//	@Override
//	public boolean reschedule(int wheel, TimedFuture r) {
//
//		Queue<TimedFuture>[] q = this.q;
//		int remain = q.length - 1;
//		do {
//			if (q[wheel].offer(r))
//				return true;
//			if (++wheel == q.length) wheel = 0;
//		} while (--remain > 0);
//
//		return false;
//	}
//
//	@Override
//	public int size() {
//		return Util.sum((ToIntFunction<Queue>) Queue::size, q);
//	}
//
//	@Override
//	public boolean isEmpty() {
//		return Arrays.stream(q).allMatch(Collection::isEmpty);
//	}
//
//
//}
