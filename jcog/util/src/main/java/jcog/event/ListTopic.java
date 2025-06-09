package jcog.event;

import jcog.data.list.FastCoWList;
import jcog.util.CountDownThenRun;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

/**
 * arraylist implementation, thread safe.  creates an array copy on each update
 * for fastest possible iteration during emitted events.
 */
public class ListTopic<V> extends FastCoWList<Consumer<V>> implements Topic<V> {

    private final CountDownThenRun busy = new CountDownThenRun();

    public ListTopic() {
        this(8);
    }

    public ListTopic(int capacity) {
        super(capacity, Consumer[]::new);
    }

    @Override
    public void accept(V x) {
        //forEachWith(Consumer::accept, x);
        var cc = this.array();
        for (var c: cc)
            c.accept(x);
    }

    @Override
    public void emitAsync(V x, Executor exe) {
        var cc = this.array();
        for (Consumer c: cc)
            exe.execute(() -> c.accept(x));
        //forEachWith((c,X)->exe.execute(() -> c.accept(X)), x);
    }

    @Override
    public void emitAsyncAndWait(V x, Executor executorService) throws InterruptedException {
        var cc = this.array();
        if (cc != null) {
            var n = cc.length;
            if (n != 0) {
                var l = new CountDownLatch(n);

                for (var c : cc) {
                    executorService.execute(() -> {
                        try {
                            c.accept(x);
                        } finally {
                            l.countDown();
                        }

                    });
                }
                l.await();
            }
        }
    }

    @Override
    public void emitAsync(V x, Executor exe, Runnable onFinish) {
        var cc = this.array();
        switch (cc.length) {
            case 0:
                return;
//            case 1: {
//                exe.execute(() -> {
//                    try {
//                        cc[0].accept(x);
//                    } finally {
//                        onFinish.run();
//                    }
//                });
//                break;
//            }
            default: {
                busy.run(cc, x, onFinish, exe);
                break;
            }
        }
    }

    @Override
    public void start(Consumer<V> o) {
        //assert (o != null);
        add(o);
    }

    @Override
    public void stop(Consumer<V> o) {
        //assert (o != null);
        remove(o);
    }


}