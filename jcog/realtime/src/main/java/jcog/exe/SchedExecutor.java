package jcog.exe;

import jcog.data.list.Lst;
import org.jctools.queues.MessagePassingQueue;
import org.jctools.queues.MpscArrayQueue;

import java.util.PriorityQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public class SchedExecutor implements AutoCloseable, Executor {

    public ExecutorService exe;

    private final MpscArrayQueue<Task> preScheduled = new MpscArrayQueue<>(QUEUE_CAPACITY);
    private final PriorityQueue<Task> scheduled = new PriorityQueue<>(QUEUE_CAPACITY /* estimate capacity */);
    private final MessagePassingQueue.Consumer<Task> preScheduler = this::preSchedule;
    private static final int QUEUE_CAPACITY = 2 * 1024;

    public SchedExecutor(ExecutorService exe) {
        this.exe = exe;
    }

    @Override
    public void execute(Runnable r) {
        exe.execute(r);
        //runAt(r, Long.MIN_VALUE);
    }

    public void runAt(Runnable task, long next) {
        add(new Task(task, next));
    }

    @Override
    public void close() {
        shutdown();
        try {
            if (!exe.awaitTermination(60, TimeUnit.SECONDS)) shutdownNow();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            shutdownNow();
        }
    }

    public void shutdown() {
        exe.shutdown();
    }

    public void shutdownNow() {
        clear();
        shutdown();
    }

    final Lst<Task> ready = new Lst<>(32);

    public final void run(long now) {
        synchronized (scheduled) {
            pre();
            sched(now);
            exe();
        }
    }

    /** drain scheduled tasks ready to be executed */
    private void pre() {
        preScheduled.drain(preScheduler);
        //Task u; while ((u = preScheduled.poll())!=null) scheduled.add(u);
    }

    /** skim the latest tasks into ready list */
    private void sched(long now) {
        Task t;
        while (((t = scheduled.peek()) != null) && (t.next <= now)) {
            ready.add(scheduled.poll());
        }
    }

    /** execute and clear the ready list */
    private void exe() {
        for (int i = 0, size = ready.size(); i < size; i++)
            exe.execute(ready.getAndNull(i).task);
        ready.clearFast();
    }

    public void clear() {
        synchronized (scheduled) {
            preScheduled.clear();
            scheduled.clear();
        }
    }

    public final void add(Task t) {
        if (!preScheduled.offer(t))
            executeJammed(t);
    }

    private void executeJammed(Task t) {
        throw new RuntimeException("jammed: " + t);
    }

    private void preSchedule(Task x) {
        if (!scheduled.offer(x))
            throw new RuntimeException("scheduled priority queue overflow");
    }

    private final static class Task implements Comparable<Task> {
        final Runnable task;
        final long next;

        Task(Runnable task, long next) {
            this.task = task;
            this.next = next;
        }

        @Override
        public int compareTo(SchedExecutor.Task o) {
            return Long.compare(next, o.next);
        }

    }

}
