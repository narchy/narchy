package jcog.exe.realtime;

import jcog.Log;
import jcog.Str;
import jcog.Util;
import org.slf4j.Logger;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

import static java.util.concurrent.TimeUnit.NANOSECONDS;

/**
 * Hash Wheel Timer, as per the paper:
 * <p>
 * Hashed and hierarchical timing wheels:
 * http:
 * <p>
 * More comprehensive slides, explaining the paper can be found here:
 * http:
 * <p>
 * Hash Wheel timer is an approximated timer that allows performant execution of
 * larger amount of tasks with better performance compared to traditional scheduling.
 *
 * @author Oleksandr Petrov
 */
public class HashedWheelTimer implements AbstractTimer, ScheduledExecutorService, Runnable {

	private static final Logger logger = Log.log(HashedWheelTimer.class);

	private static final int THREAD_PRI =
		Thread.NORM_PRIORITY;
		//Thread.MAX_PRIORITY;

	/**
	 * how many epochs to delay while empty before the thread attempts to end (going into a re-activatable sleep mode)
	 */
	private static final int HIBERNATE_EPOCHS =
			1024;

	private static final int SHUTDOWN = Integer.MIN_VALUE;

	/** in nanoseconds */
	private final long resolution;

	private final int wheels;
    private final WheelModel model;
	private final Executor executor;

	private final AtomicInteger cursor = new AtomicInteger(-1);
	private volatile Thread loop;

	/**
	 * Create a new {@code HashedWheelTimer} using the given timer resolution and wheelSize. All times will
	 * rounded up to the closest multiple of this resolution.
	 * <p>
	 * for sparse timeouts. Sane default is 512.
	 *
	 * @param exec Executor instance to submit tasks to
	 */
	public HashedWheelTimer(WheelModel m, Executor exec) {
		model = m;

		this.resolution = m.resolution;

		this.executor = exec;

		this.wheels = m.wheels;
	}

	private static Callable<?> constantlyNull(Runnable r) {
		return () -> {
			r.run();
			return null;
		};
	}

	public void join() {
		Thread t = loop;
		if (t!=null) {
			try {
				t.join();
			} catch (InterruptedException e) {
                Thread.currentThread().interrupt(); // Restore the interrupted state.
                logger.error("Interrupted while joining timer thread: " + e.getMessage());
			}
		}
	}

	@Override
	public void run() {
		restart();
		loop();
		stop();
	}

	private void loop() {
		int empties = 0;
		long deadline = System.nanoTime();
		do {
			int cNext;
			int c = cursor();
			while ((cursor.compareAndExchangeRelease(c, cNext = (c + 1) % wheels)==c)) {

				if (model.run(c) == 0)
					empties++; //increment empties
				else
					empties = 0; //reset empties

				deadline = await(deadline + resolution);
				c = cNext;
			}
		} while (cursor() >= 0 && (!model.isEmpty() || empties < wheels * HIBERNATE_EPOCHS));
	}

	private void restart() {
		logger.info("{} restart", this);

		model.restart(this);
	}

	private void stop() {
		logger.info("{} {}", this, cursor() == SHUTDOWN ? "off" : "sleep");
		loop = null;
	}

	private long await(long deadline) {
		long now = System.nanoTime();
		long sleepNS = deadline - now;
		//System.out.println(sleepNS);
		if (sleepNS < 0) {
			//LATE:
			//return now;

			//advance to next deadline. corrects jitter
			return now - Math.min(-sleepNS, resolution);

//			long lag = -sleepNS;
//			if (lag <= resolution) {
				//minor lag: don't sleep, keep next deadline on schedule
//				return deadline;
//			} else {
//				//major lag; maintain tempo by skipping to next beat
//				sleepNS = resolution - (lag % resolution);
//				deadline = now + sleepNS;
//			}
		}

		Util.sleepNS(sleepNS);

		return deadline;
	}

	@Override
	public TimedFuture<?> submit(Runnable runnable) {
		return schedule(new Soon.Run(runnable));
	}

	@Override public final <D> TimedFuture<D> schedule(TimedFuture<D> r) {

		if (r.isCancelled())
			return null;

		if (r instanceof FixedRateTimedFuture ff)
			_schedule(ff);

		if (!model.accept(r, this))
			return null;

		assertRunning();
		return r;
	}

	private void _schedule(FixedRateTimedFuture fr) {
		long resolution = this.resolution;
		double epoch = resolution * wheels;
		long periodNS = fr.periodNS.get();
		int rounds = Math.min((int) (periodNS / epoch), Integer.MAX_VALUE);
		fr.rounds = rounds;
		fr.offset = Math.max(1, (int)Math.round(((periodNS - rounds * epoch) / resolution)));
	}

	protected static <X> void reject(Future<X> r) {
		r.cancel(false);
		logger.error("reject {}", r);
	}

	/**
	 * equivalent to model.idx() since its wheels is equal
	 */
	public final int idx(int cursor) {
		return cursor % wheels;
	}

	int cursor() {
		return cursor.getAcquire();
	}

	final int cursorActive() {
		int c = cursor();
		if (c != -1)
			return c;
		else {
			assertRunning();
			return cursor();
		}
	}

	@Override
	public ScheduledFuture<?> schedule(Runnable runnable,
									   long delay,
									   TimeUnit timeUnit) {
		return scheduleOneShot(NANOSECONDS.convert(delay, timeUnit), constantlyNull(runnable));
	}

	@Override
	public <V> ScheduledFuture<V> schedule(Callable<V> callable, long delay, TimeUnit timeUnit) {
		return scheduleOneShot(NANOSECONDS.convert(delay, timeUnit), callable);
	}

	@Override
	public FixedRateTimedFuture scheduleAtFixedRate(Runnable runnable, long delay, long period, TimeUnit unit) {
		return scheduleFixedRate(NANOSECONDS.convert(period, unit), NANOSECONDS.convert(delay, unit),
			runnable);
	}

	@Override
	public FixedDelayTimedFuture<?> scheduleWithFixedDelay(Runnable runnable, long initialDelay, long delay, TimeUnit unit) {
		return scheduleFixedDelay(NANOSECONDS.convert(delay, unit),
			NANOSECONDS.convert(initialDelay, unit),
			constantlyNull(runnable));
	}

	@Override
	public String toString() {
		return String.format("HashedWheelTimer { Buffer Size: %d, Resolution: %s }",
			wheels,
			Str.timeStr(resolution));
	}

	/**
	 * Executor Delegate, invokes immediately bypassing the timer's ordered scheduling.
	 * Use submit for invokeLater-like behavior
	 */
	@Override
	public final void execute(Runnable r) {
		executor.execute(r);
	}

	@Override
	public void shutdown() {
		_shutdown();
		if (executor instanceof ExecutorService es)
			es.shutdown();
	}

	@Override
	public List<Runnable> shutdownNow() {
		_shutdown();
		return executor instanceof ExecutorService es ? es.shutdownNow() : Collections.EMPTY_LIST;
	}


	private void _shutdown() {
		cursor.setRelease(Integer.MIN_VALUE);
	}

	@Override
	public boolean isShutdown() {
		return cursor() >= 0 &&
			(!(executor instanceof ExecutorService es) || es.isShutdown());
	}

	@Override
	public boolean isTerminated() {
		return cursor() >= 0 &&
			(!(executor instanceof ExecutorService es) || es.isTerminated());
	}

	@Override
	public boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
		return
			(!(executor instanceof ExecutorService es) || es.awaitTermination(timeout, unit));
	}

	@Override
	public <T> Future<T> submit(Callable<T> task) {
		return e().submit(task);
	}

	private ExecutorService e() {
		return (ExecutorService) this.executor;
	}

	@Override
	public <T> Future<T> submit(Runnable task, T result) {
		return e().submit(task, result);
	}

	@Override
	public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks) throws InterruptedException {
		return e().invokeAll(tasks);
	}

	@Override
	public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> tasks, long timeout,
										 TimeUnit unit) throws InterruptedException {
		return e().invokeAll(tasks, timeout, unit);
	}

	@Override
	public <T> T invokeAny(Collection<? extends Callable<T>> tasks) throws InterruptedException, ExecutionException {
		return e().invokeAny(tasks);
	}

	@Override
	public <T> T invokeAny(Collection<? extends Callable<T>> tasks, long timeout,
						   TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
		return e().invokeAny(tasks, timeout, unit);
	}


	/**
	 * INTERNALS
	 */
	private <V> ScheduledFuture<V> scheduleOneShot(long delayNS, Callable<V> callable) {

		if (delayNS <= resolution / 2) {
			return scheduleOneShotImmediately(callable);
		}
//		} else if (delayNS < resolution) {
//			//round-up
//			delayNS = resolution;
//		}

		long cycleLen = wheels * resolution;
		int rounds = (int) (((double) delayNS) / cycleLen);
		int firstFireOffset = Util.longToInt(delayNS - rounds * cycleLen);

		return schedule(new OneTimedFuture(Math.max(0, firstFireOffset), rounds, callable));
	}

	private <V> ImmediateFuture<V> scheduleOneShotImmediately(Callable<V> callable) {
		ImmediateFuture<V> f = new ImmediateFuture<>(callable);
		executor.execute(f);
		return f;
	}


	public FixedRateTimedFuture scheduleFixedRate(long recurringTimeout,
												  long firstDelay,
												  Runnable callable) {
		return scheduleFixedRate(recurringTimeout, firstDelay, new MyFixedRateRunnable(recurringTimeout, callable));
	}

	private FixedRateTimedFuture scheduleFixedRate(long recurringTimeout, long firstDelay, FixedRateTimedFuture r) {
		assert (recurringTimeout >= resolution) : "Cannot schedule tasks for amount of time less than timer precision.";

		if (firstDelay > 0) {
			scheduleOneShot(firstDelay, () -> {
				schedule(r);
				return null;
			});
		} else {
			schedule(r);
		}

		return r;
	}

	private <V> FixedDelayTimedFuture<V> scheduleFixedDelay(long recurringTimeout,
															long firstDelay,
															Callable<V> callable) {
		assert (recurringTimeout >= resolution) : "Cannot schedule tasks for amount of time less than timer precision.";


		FixedDelayTimedFuture<V> r = new FixedDelayTimedFuture<>(
			callable,
			recurringTimeout, resolution, wheels,
			this::schedule);

		if (firstDelay > resolution) {
			scheduleOneShot(firstDelay, () -> {
				schedule(r);
				return null;
			});
		} else {
			schedule(r);
		}

		return r;
	}


	void assertRunning() {
		if (cursor.compareAndExchangeRelease(-1, 0)==-1)
			start();
	}

	private synchronized void start() {
//		if (this.loop != null) {

//			//HACK time grap between cursor==-1 and loop==null (final thread stop signal)
//			Util.sleepMS(10);

			if (this.loop != null)
				throw new RuntimeException("loop exists");
//		}

		Thread t = this.loop = new Thread(this, HashedWheelTimer.class.getSimpleName() + '_' + hashCode());
        boolean daemon = false;
        t.setDaemon(daemon);
		t.setPriority(THREAD_PRI);
		t.start();
	}

	public int size() {
		return model.size();
	}

//	public enum WaitStrategy {
//
//		/**
//		 * Yielding wait strategy.
//		 * <p>
//		 * Spins in the loop, until the deadline is reached. Releases the flow control
//		 * by means of Thread.yield() call. This strategy is less precise than BusySpin
//		 * one, but is more scheduler-friendly.
//		 */
//		YieldingWait {
//			@Override
//			public void waitUntil(long deadline) {
////				Thread t = null;
//				do {
//					Thread.yield();
////					if ((t == null ? (t = Thread.currentThread()) : t).isInterrupted()) {
////						//throw new InterruptedException();
////						throw new RuntimeException();
////					}
//				} while (deadline >= System.nanoTime());
//			}
//		},
//
//		/**
//		 * BusySpin wait strategy.
//		 * <p>
//		 * Spins in the loop until the deadline is reached. In a multi-core environment,
//		 * will occupy an entire core. Is more precise than Sleep wait strategy, but
//		 * consumes more resources.
//		 */
//		BusySpinWait {
//			@Override
//			public void waitUntil(long deadline) {
////				Thread t = null;
//				do {
//					Thread.onSpinWait();
////					if ((t == null ? (t = Thread.currentThread()) : t).isInterrupted()) {
////						//throw new InterruptedException();
////						throw new RuntimeException("interrupted");
////					}
//				} while (deadline >= System.nanoTime());
//			}
//		},
//
//		/**
//		 * Sleep wait strategy.
//		 * <p>
//		 * Will release the flow control, giving other threads a possibility of execution
//		 * on the same processor. Uses less resources than BusySpin wait, but is less
//		 * precise.
//		 */
//		SleepWait {
//			@Override
//			public void waitUntil(long deadline) {
//				Util.sleepNS(deadline - System.nanoTime());
//			}
//
//		};
//
//		/**
//		 * Wait until the given deadline, deadlineNanoseconds
//		 *
//		 * @param deadlineNanoseconds deadline to wait for, in milliseconds
//		 */
//		public abstract void waitUntil(long deadlineNanoseconds);
//
//		void waitNanos(long nanos) {
//			Util.sleepNS(nanos);
//		}
//
//	}

	private static final class ImmediateFuture<V> implements ScheduledFuture<V>, Runnable {
		private Callable<V> callable;
		private Object result = this;

		ImmediateFuture(Callable<V> callable) {
			this.callable = callable;
		}

		@Override
		public long getDelay(TimeUnit timeUnit) {
			return 0;
		}

		@Override
		public int compareTo(Delayed delayed) {
            throw new UnsupportedOperationException();
		}

		@Override
		public boolean cancel(boolean mayInterruptIfRunning) {
			Callable<V> c = callable;
			if (c != null) {
				result = null;
				callable = null;
				return true;
			}
			return false;
		}

		@Override
		public boolean isCancelled() {
			return result == this && callable == null;
		}

		@Override
		public boolean isDone() {
			return result != this;
		}

		@Override
		public V get() {
			Object r = this.result;
			return r == this ? null : (V) r;
		}

		@Override
		public V get(long l, TimeUnit timeUnit) {
			return AbstractTimedCallable.poll(this, l, timeUnit);
		}

		@Override
		public void run() {
			try {
				result = callable.call();
			} catch (Exception e) {
				result = e;
			}
		}
	}

	private static final class MyFixedRateRunnable extends FixedRateTimedFuture {
		private final Runnable callable;

		MyFixedRateRunnable(long recurringTimeout, Runnable callable) {
			super(1, recurringTimeout);
			this.callable = callable;
		}

		@Override public void run() {
			callable.run();
		}
	}
}