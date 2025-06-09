package jcog.util;

import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/** usage: call reset().. then call your runnables/consumers with run(...)
 * TODO this is probably the same as CountedCompleter
 * */
public class CountDownThenRun {

    private final AtomicInteger count = new AtomicInteger();
    private /*volatile*/ Runnable onFinish;

    public void reset(int count, Runnable onFinish) {
//        if (count < 1)
//            throw new RuntimeException("count > 0");
        this.count.set(count);
//        if (!compareAndSet(0, count))
//            System.exit(0); //HACK //throw new RuntimeException(this + " not ready");
        this.onFinish = onFinish;
    }

    private void countDown() {
        if (count.decrementAndGet() == 0) {
            var f = onFinish;
            onFinish = null;
            f.run();
        }
    }

    public <V> void run(Consumer<V>[] cc, V x, Runnable finish, Executor exe) {
        reset(cc.length, finish);
        for (var c: cc)
            exe.execute(new MyConsumerRunnable<>(c, x));
    }

    private final class MyConsumerRunnable<V> implements Runnable {
        private final Consumer<V> c;
        private final V x;

        MyConsumerRunnable(Consumer<V> c, V x) {
            this.c = c;
            this.x = x;
        }

        @Override
        public String toString() {
            return c.toString();
        }

        @Override
        public void run() {
            try {
                c.accept(x);
            } finally {
                countDown();
            }
        }
    }
}