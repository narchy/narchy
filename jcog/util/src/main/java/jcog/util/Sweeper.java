//package jcog.util;
//
//import com.google.common.collect.Sets;
//import jcog.pri.PLink;
//
//import java.lang.ref.PhantomReference;
//import java.lang.ref.ReferenceQueue;
//import java.lang.ref.SoftReference;
//import java.lang.ref.WeakReference;
//import java.util.ArrayList;
//import java.util.Collection;
//import java.util.List;
//import java.util.Set;
//import java.util.concurrent.*;
//import java.util.concurrent.atomic.AtomicBoolean;
//import java.util.concurrent.atomic.AtomicLong;
//import java.util.function.Consumer;
//import java.util.function.Supplier;
//
///**
// * https:
// * This class is responsible for monitoring and performing clean-up on garbage collected objects.
// */
//public class Sweeper {
//
//    private final ReferenceQueue<Object> queue = new ReferenceQueue<>();
//    private final Executor executor;
//
//    /**
//     * strong ref holding the handlers
//     */
//    private final Set<Runnable> handlers = Sets.newConcurrentHashSet();
//
//    private volatile boolean shutdown;
//
//    /**
//     * Constructs a default instance that uses daemon threads at maximum priority for the polling and cleanup
//     * tasks.
//     */
//    public Sweeper() {
//        this(createThreadFactory());
//    }
//
//    /**
//     * Constructs an instance that uses the given ThreadFactory to create threads for polling and cleanup tasks.
//     * The ThreadFactory is used to construct an {@link ExecutorService}. Currently, this is a {@link ThreadPoolExecutor},
//     * but that may change in a later implementation.
//     */
//    public Sweeper(final ThreadFactory forExecutor) {
//        this(
//                forExecutor == null ?
//                        npe(ExecutorService.class, "Cannot accept a null ThreadFactory as an argument") :
//                        createExecutor(forExecutor)
//        );
//    }
//
//    /**
//     * Constructs an instance that uses the given {@link Executor} to execute polling and cleanup tasks,
//     * including a background sweeping thread.
//     * Note that the executor is assumed to be able to execute blocking tasks. If (for instance)
//     * the executor executes all the tasks directly, the calling thread will hang.
//     */
//    public Sweeper(final Executor executor) {
//        this(executor, true);
//    }
//
//    /**
//     * Constructs an instance that uses the given {@link Executor} to execute polling and cleanup tasks.
//     * If the second argument is {@code false}, the background sweeping thread will not be launched, and
//     * the user will need to use {@link #sweep()} or {@link #queueingSweep()} to manually trigger polling
//     * of the queue.
//     */
//    public Sweeper(final Executor executor, final boolean backgroundSweeping) {
//        if (executor == null) npe("Cannot accept a null Executor as an argument");
//        this.executor = executor;
//        if (backgroundSweeping) this.executor.execute(new Runnable() {
//            @Override
//            public void run() {
//                Runnable ref = null;
//                try {
//                    for (ref = (Runnable) queue.remove(); ref != null; ref = (RunnableReference) queue.poll()) {
//
//
//                        handlers.remove(ref);
//                        executor.execute(ref);
//                    }
//
//
//                    if (isShutdown()) {
//                    } else {
//
//                        executor.execute(this);
//                    }
//                } catch (InterruptedException e) {
//
//                }
//            }
//        });
//    }
//
//    private static ThreadFactory createThreadFactory() {
//        final AtomicLong ctr = new AtomicLong(Long.MAX_VALUE);
//        return r -> {
//            final Thread t = new Thread(r);
//            t.setName("Sweeper Cleanup Thread #" + Long.toHexString(ctr.getAndDecrement()));
//            t.setPriority(Thread.MIN_PRIORITY);
//            t.setDaemon(true);
//            return t;
//        };
//    }
//
//    private static Executor createExecutor(ThreadFactory threadFactory) {
//        final int coreThreads = 2;
//        final ThreadPoolExecutor executor = new ThreadPoolExecutor(
//                coreThreads, Math.max(coreThreads, Runtime.getRuntime().availableProcessors()),
//                1, TimeUnit.MINUTES,
//                new LinkedBlockingQueue(),
//                threadFactory
//        );
//        executor.prestartAllCoreThreads();
//        return executor;
//    }
//
//    private static void npe(String message) {
//        npe(Object.class, message);
//    }
//
//    private static <T> T npe(Class<T> returnType, String message) {
//        throw new NullPointerException(message);
//    }
//
//
//    /**
//     * Registers a shutdown hook to execute the clean-up on anything that didn't get GC'ed.
//     * <b>Note that cleanup will be executed for any keys that are still alive.</b>
//     * See also caveats from {@link Runtime#addShutdownHook(Thread)}.
//     */
//    public void registerShutdownHook() {
//        Thread t = new Thread(() -> {
//
//            for (Runnable r : shutdown()) {
//                r.run();
//            }
//        });
//        t.setPriority(Thread.MIN_PRIORITY);
//        Runtime.getRuntime().addShutdownHook(t);
//    }
//
//    public <T> RunnableWeakReference<T> onWeakGC(final T key, final SweepAction<T> behavior) {
//        ensureAlive(key);
//        RunnableWeakReference<T> w = new RunnableWeakReference<>(key, behavior);
//        handlers.addAt(w);
//        return w;
//    }
//
//    /**
//     * Register an action to be performed when the key is eligible to be garbage collected.
//     * More precisely, it is performed at some point after the key becomes weakly reachable.
//     */
//    public <T> RunnableWeakReference<T> onWeakGC(final T key, final Runnable behavior) {
//        ensureAlive(key);
//        RunnableWeakReference<T> w = new RunnableWeakReference<>(key, behavior);
//        handlers.addAt(w);
//        return w;
//    }
//
//    /**
//     * Register an action to be performed when the key is eligible to be garbage collected.
//     * More precisely, it is performed at some point after the key becomes weakly reachable.
//     */
//    public <K, V> ConsumingWeakReference<V> onWeakGC(K key, final V val, final Consumer<K> keyConsumer) {
//        ensureAlive(val);
//        ConsumingWeakReference<V> w = new ConsumingWeakReference<>(key, val, keyConsumer);
//        handlers.addAt(w);
//        return w;
//    }
//
//    private final <V> void ensureAlive(Object val) {
//        if (shutdown) throw new RejectedExecutionException("Shutdown. Will not process: " + val);
//    }
//
//    /**
//     * Register an action to be performed after the key is garbage collected.
//     */
//    public RunnablePhantomReference onGC(final Object key, final Runnable behavior) {
//        ensureAlive(key);
//        RunnablePhantomReference reference = new RunnablePhantomReference(key, behavior);
//        handlers.addAt(reference);
//        return reference;
//    }
//
//    /**
//     * Prevents the Sweeper from processing new objects, and returns a
//     * collection of {@link Runnable} objects that have not been cleaned up. Note:
//     * the keys may not have been grabage collected for the returned Runnables.
//     */
//    public Iterable<Runnable> shutdown() {
//        shutdown = true;
//        List<Runnable> toProcess = new ArrayList<>(handlers);
//        if (executor instanceof ExecutorService) {
//            ((ExecutorService) executor).shutdown();
//        }
//        return toProcess;
//    }
//
//    /**
//     * Performs a sweep and cleanup in the current thread.
//     *
//     * @return Whether work was found during this poll.
//     */
//    public boolean sweep() {
//        boolean workFound = false;
//        Runnable action = null;
//        while ((action = (Runnable) queue.poll()) != null) {
//            handlers.remove(action);
//            action.run();
//            action = null;
//            workFound = true;
//            Thread.yield();
//        }
//        return workFound;
//    }
//
//    /**
//     * Performs a sweep in the current thread, but cleanup work is done via the {@link Executor}.
//     *
//     * @return Whether work was found during this poll.
//     */
//    public boolean queueingSweep() {
//        boolean workFound = false;
//        RunnableReference action = null;
//        while ((action = (RunnableReference) queue.poll()) != null) {
//            workFound = true;
//
//            handlers.remove(action);
//            executor.execute(action);
//            action = null;
//            Thread.yield();
//        }
//        return workFound;
//    }
//
//    /**
//     * Enqueue a {@link #queueingSweep()} to be performed by the {@link Executor}.
//     * There is no feedback fromt his sweep, so it is fire-and-forget.
//     */
//    public void runSweep() {
//        executor.execute(new DoSweep());
//    }
//
//    /**
//     * Enqueue a {@link #queueingSweep()} to be performed by the {@link Executor}.
//     * <p>
//     * <b>Implementation Note:</b><br />
//     * There is a nontrivial effficiency hit if you used {@link #Sweeper(Executor}}
//     * and did <i>not</i> pass in a {@link ExecutorService}. We end up having to
//     * allocate a wrapper for each call to this method to make pretend it's a
//     * service. See {@link #runSweep()} for a better option in that case.
//     *
//     * @return A {@link Future} denoting whether work was found during this sweep.
//     */
//    public Future<Boolean> enqueueSweep() {
//        return executor instanceof ExecutorService ? enqueueSweepInService((ExecutorService) executor) : enqueueSweepInExecutor();
//    }
//
//    private Future<Boolean> enqueueSweepInService(final ExecutorService service) {
//        return service.submit((Callable<Boolean>) new DoSweep());
//    }
//
//    private Future<Boolean> enqueueSweepInExecutor() {
//        return new ExecutorCompletionService(
//                executor, new ArrayBlockingQueue<Future<Boolean>>(1, false)
//        ).submit(new DoSweep());
//    }
//
//    /**
//     * Whether this sweeper has been shut down.
//     */
//    public boolean isShutdown() {
//        return shutdown;
//    }
//
//    private interface RunnableReference extends Runnable {
//        Runnable consumeAction();
//    }
//
//    /**
//     * An action performed when a sweep occurs and the target is known.
//     */
//    public static abstract class SweepAction<T> implements Runnable {
//
//        private T target;
//
//        /**
//         * Retrives the object which triggered the cleanup action.
//         */
//        public T getTarget() {
//            return target;
//        }
//
//        /**
//         * Sets the object which triggered the cleanup action.
//         */
//        public void setTarget(T target) {
//            this.target = target;
//        }
//
//        /**
//         * Performs the action to clean up. When this class is called by {@link Sweeper},
//         * {@link #setTarget(Object)} is guaranteed to have been called with a non-{@code null}
//         * value.
//         */
//
//
//    }
//
//    private final class RunnablePhantomReference extends PhantomReference<Object> implements RunnableReference {
//        private volatile Runnable action;
//
//        public RunnablePhantomReference(Object toRun, Runnable action) {
//            super(toRun, queue);
//            this.action = action;
//        }
//
//        @Override
//        public synchronized Runnable consumeAction() {
//            if (action == null) return null;
//            Runnable toReturn = action;
//            action = null;
//            return toReturn;
//        }
//
//        @Override
//        public void run() {
//            Runnable action = consumeAction();
//            if (action != null) action.run();
//        }
//    }
//
//    private final class RunnableWeakReference<T> extends WeakReference<T> implements RunnableReference {
//        private volatile Runnable action;
//
//        public RunnableWeakReference(T toRun, Runnable action) {
//            super(toRun, queue);
//            this.action = action;
//        }
//
//        @Override
//        public /*synchronized*/ Runnable consumeAction() {
//            if (action == null) return null;
//            Runnable toReturn = action;
//            action = null;
//            if (toReturn instanceof SweepAction) {
//                ((SweepAction<T>) toReturn).setTarget(this.get());
//            }
//            return toReturn;
//        }
//
//        @Override
//        public void run() {
//            Runnable action = consumeAction();
//            if (action != null) action.run();
//        }
//    }
//
//    private final class ConsumingSoftReference<T> extends SoftReference<T> implements Runnable, Supplier<T> {
//        private Object holder;
//        private volatile Consumer action;
//
//        public ConsumingSoftReference(Object holder, T value, Consumer action) {
//            super(value, queue);
//            this.holder = holder;
//            this.action = action;
//        }
//
//        @Override
//        public void run() {
//            Object h = holder;
//            Consumer a = action;
//            this.holder = action = null;
//            a.accept(h);
//        }
//    }
//
//    private final class ConsumingWeakReference<T> extends WeakReference<T> implements Runnable, Supplier<T> {
//        private Object holder;
//        private volatile Consumer action;
//
//        public ConsumingWeakReference(Object holder, T value, Consumer action) {
//            super(value, queue);
//            this.holder = holder;
//            this.action = action;
//        }
//
//        @Override
//        public void run() {
//            Object h = holder;
//            Consumer a = action;
//            this.holder = action = null;
//            a.accept(h);
//        }
//    }
//
//    private final class RemoveFromBag implements Runnable {
//        private final RunnableReference toRemove;
//
//        public RemoveFromBag(RunnableReference toRemove) {
//            if (toRemove == null) npe("Cannot remove a null RunnableReference");
//            this.toRemove = toRemove;
//        }
//
//        @Override
//        public void run() {
//            handlers.remove(toRemove);
//        }
//    }
//
//    private class DoSweep implements Callable<Boolean>, Runnable {
//        @Override
//        public Boolean call() {
//            return queueingSweep();
//        }
//
//        @Override
//        public void run() {
//            queueingSweep();
//        }
//    }
//
//    public static void main(String[] args) {
//        final Sweeper sweeper = new Sweeper();
//        sweeper.registerShutdownHook();
//
//
//        final AtomicBoolean stop = new AtomicBoolean(false);
//        Consumer<Object> handler = (v) -> {
//            System.out.println("onGC action run! " + v);
//            stop.setAt(true);
//        };
//
//        Object key = new Object();
//        Object val = new PLink("xyz", 0.5f);
//        ConsumingWeakReference<Object> x = sweeper.onWeakGC(val, key, handler);
//
//
//        for (int i = 0; !stop.get(); i++) {
//            key = new ArrayList<Integer>(i);
//            for (int k = 0; k < i; k++) {
//                ((Collection) key).addAt(k);
//            }
//            Thread.yield();
//        }
//        System.out.println(((Collection) key).size());
//        System.out.println(x + " " + x.get());
//
//        sweeper.shutdown();
//    }
//
//}