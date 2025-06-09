package nars.exe;

import jcog.Log;
import jcog.data.list.Lst;
import jcog.pri.PLink;
import nars.NAR;
import nars.NARPart;
import nars.derive.Deriver;
import nars.derive.impl.TaskBagDeriver;
import nars.derive.reaction.ReactionModel;
import nars.focus.Focus;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.random.RandomGenerator;

public class DeriverWorkers0 extends NARPart {

    private final ExecutorService exe;
    volatile boolean running;

    private final ReactionModel rules;
    private final int threads;
    private List<DeriverWorker> workers;

    public DeriverWorkers0(ReactionModel rules, int threads) {
        this.rules = rules;
        this.threads = threads;
        this.workers = new Lst<>(threads);

        this.exe = Executors.newFixedThreadPool(threads);
    }

    @Override
    protected void starting(NAR n) {
        n.exe.exe = ForkJoinPool.commonPool();

        running = true;
        for (int i = 0; i < threads; i++) {
            var w = new DeriverWorker(deriver(n));
            workers.add(w);
            exe.execute(w);
        }
    }

    @Override
    protected void stopping(NAR nar) {
        running = false;
        workers.clear();
    }

    protected Deriver deriver(NAR n) {
        return new TaskBagDeriver(rules, n);
    }

    protected @Nullable PLink<Focus> focus(RandomGenerator rng) {
        return nar.focus.sample(rng);
    }

    private class DeriverWorker implements Runnable {
        private final Deriver d;

        public DeriverWorker(Deriver d) {
            this.d = d;
        }

        @Override
        public void run() {
            while (running) {
                try {
                    var f = focus(d.rng);
                    if (f != null)
                        d.next(f.id);
                    else
                        Thread.yield();
                } catch (Throwable t) {
                    logger.error("run",t);
                }
            }
        }

        private static final Logger logger = Log.log(DeriverWorker.class);
    }
}
