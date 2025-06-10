package nars.deriver.impl;

import jcog.TODO;
import jcog.data.map.MRUMap;
import jcog.data.set.ArrayHashSet;
import jcog.signal.IntRange;
import nars.Deriver;
import nars.NAR;
import nars.Premise;
import nars.action.resolve.TaskResolve;
import nars.deriver.TaskBagAttentionSampler;
import nars.entity.NALTask;
import nars.action.transform.TemporalComposer;
import nars.deriver.reaction.ReactionModel;
import nars.premise.NALPremise;
import nars.premise.SubPremise;
import nars.term.Termed;
import org.jetbrains.annotations.Nullable;

import java.util.Map;

public class SerialDeriver extends Deriver {

    private final TaskBagAttentionSampler taskSampler = new TaskBagAttentionSampler(this);

    /** breadth */
    public final IntRange iter = new IntRange(60, 1, 4096);

    public final IntRange batch = new IntRange(1, 1, 32);

    private final Map<Termed, ArrayHashSet<Premise>> queues;

//    private static final boolean DEQUEUE_ORDERED = false;
//    private static final boolean removeOnlyIfTaskPremise = true;

    private static final boolean BIN_FROM = false;

    /** TODO refine */
    static final int queueCapacity =
        BIN_FROM ? 512 : 24;
        //depth.intValue() * 8;


    static final boolean clearQueuesAfterIteration = false;

    @Nullable transient private Premise current;
    @Nullable transient private ArrayHashSet<Premise> queue;

    public SerialDeriver(ReactionModel model, NAR nar) {
        super(model, nar);
        //this.taskSampler = new TaskBagAttentionSampler(this); //already assigned

        /* TODO tune */
        int cacheCapacity = (int) Math.pow(iter.intValue(), 2);

        queues = new MRUMap<>(cacheCapacity);
    }

    @Override protected void next() {
        int iter = this.iter.intValue(), subIter = this.batch.intValue();
        for (int i = 0; i < iter; i++) {

            Premise seed = seed();

            if (seed == null) break;

            this.current = seed;

            Termed k = key(seed);
            this.queue = queues.get(k);
            boolean newQueue = false;
            if (queue == null) {
                queue = new ArrayHashSet<>();
                newQueue = true;
            }

            //if (queue.size() > 6) System.out.println(k + " " + queue.size());

            trySeed(seed);

            runQueue(subIter);

            updateQueue(k, newQueue);

            this.current = null;
            this.queue = null;
        }

        if (clearQueuesAfterIteration)
            queues.clear();

    }

    private @Nullable Premise seed() {
        //throw new TODO("follow seed pattern in TaskBagDeriver");
        var tasks = taskSampler.seed(this.focus, 1, this.rng);
        if (tasks != null && !tasks.isEmpty()) {
            NALTask t = tasks.get(0);
            //TODO verify this is the correct way to create a NALPremise.SeedTaskPremise
            //tasks.remove(0); //TODO is this necessary? TaskBagAttentionSampler.seed does not remove
            return new NALPremise.SeedTaskPremise(t);
        }
        return null;
    }

    private void trySeed(Premise seed) {
        int qs = queue.size();
        if (qs == 0 || rng.nextBoolean(
            /* TODO tune */
            //(float)Math.pow(1 - qs/(1f + queueCapacity), 7))
            (1 - qs/(1f + queueCapacity))/20
        ))
            run(seed);
    }

    private void updateQueue(Termed k, boolean newQueue) {
        boolean emptyAfter = queue.isEmpty();
        if (newQueue && !emptyAfter)
            queues.put(k, queue);
        else if (!newQueue && emptyAfter)
            queues.remove(k);
    }

    /** key(premise) - determines premise binning */
    private Termed key(Premise/*TaskLink*/ seed) {
        return BIN_FROM ?
            seed.from() //bin by 'from'
            :
            seed; //bin by entire tasklink
    }

    protected void runQueue(int depth) {
        var q = queue;
        for (int j = 0; j < depth; j++) {
//            if (queue.size() > 2)
//                System.out.println(current + " "+ queue.size());

            Premise n = premise(q);
            if (n == null) return;
            
            run(n);
        }
    }

    private Premise premise(ArrayHashSet<Premise> q) {
        int i = rng.nextInt(q.size());
        Premise p = q.get(i);
        if (!keep(p))
            q.remove(i);
        return p;
//        return DEQUEUE_ORDERED ?
//                q.removeFirst() :
//                q.remove(rng);
    }

    private static boolean keep(Premise p) {
        return
            isTaskResolve(p) ||
                    (
                    !(p instanceof NALPremise) &&
                    p.self() &&
                    (!(p instanceof SubPremise s) || !(s.reaction() instanceof TemporalComposer))
                    )
        ;
    }

    @Override
    protected void acceptPremise(Premise p) {
        var q = queue;

        //replace, since existing tasklink may be old, remove existing instance and replace
        //HACK
        if (isTaskResolve(p))
            q.remove(p);

        if (q.add(p)) {
            if (q.size() > queueCapacity)
                q.removeRandom(rng);
        }
    }

    private static boolean isTaskResolve(Premise p) {
        return p.parent instanceof Premise &&
               p.reaction() instanceof TaskResolve;
    }

//    protected void runQueue(int subIter) {
//        runQueueRandom(subIter);
//        //runQueueLinear(subIter);
//        //runQueueRandomFrontLine(subIter);
//
//        queue.clear();
//    }
//
//    private Premise popQueue() {
//        return queue.removeRandom(rng);
//        //return ((ArrayHashSet<Premise>)queue).remove(rng);
//    }
//
////    private void runQueueRandomFrontLine(int depth) {
////
////        int qs0 = queue.size();
////
////        //frontline
////        var l = new Lst<Premise>(Math.min(qs0, depth));
////        for (int j = 0; j < depth; j++) {
////            Premise n = popQueue();
////            if (n == null)
////                break;
////            l.add(n);
////        }
////        for (var n : l)
////            run(n);
////
////        depth -= l.size();
////        l.delete();
////
////        if (depth > 0) {
////            //remainder
////            runQueueRandom(depth);
////        }
////
////    }
//
//    private void runQueueRandom(int depth) {
//        for (int j = 0; j < depth; j++) {
//            Premise n = popQueue();
//            if (n == null)
//                return;
//            run(n);
//        }
//    }

    public static class CachedSerialDeriver extends SerialDeriver {

        long hit, miss;

        final MRUMap<Premise, Premise> cache;

        public CachedSerialDeriver(ReactionModel m, NAR n, int capacity) {
            super(m, n);
            cache = new MRUMap<>(capacity);
        }

        @Override
        protected void acceptPremise(Premise p) {
            super.acceptPremise(cached(p));
        }

        private Premise cached(Premise p) {
            var q = cache.putIfAbsent(p,p);
            if (q == null) {
                miss++;
                return p;
            } else {
                hit++;
                ((TaskLink)q).pri(((TaskLink)p).priElseZero()); //replace pri
                return q;
            }
        }
    }
}