package jcog.pri.bag;

import jcog.Is;
import jcog.TODO;
import org.jetbrains.annotations.Nullable;

import java.util.Iterator;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;

import static jcog.pri.bag.Sampler.SampleReaction.*;

@Is("Reservoir_sampling")
@FunctionalInterface public interface Sampler<X> {


    /* sample the bag, optionally removing each visited element as decided by the visitor's
     * returned value */
    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    void sample(RandomGenerator rng, Function<? super X, SampleReaction> each);


    /**
     * lowest-common denominator basic sample one-instance-at-a-time;
     * not necessarily most efficient for a given impl.
     * gets the next value without removing changing it or removing it from any index.  however
     * the bag is cycled so that subsequent elements are different.
     */
    default @Nullable X sample(RandomGenerator rng) {
        Object[] result = new Object[1];
        sample(rng, ((Predicate<? super X>) (x) -> {
            result[0] = x;
            return false;
        }));
        return (X) result[0];
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    default Sampler<X> sample(RandomGenerator rng, Predicate<? super X> each) {
        sample(rng, (Function<X, SampleReaction>) (x -> each.test(x) ? Next : Stop));
        return this;
    }

    default Sampler<X> sample(RandomGenerator rng, int max, Consumer<? super X> each) {
        sampleOrPop(rng, false, max, each);
        return this;
    }

    default Sampler<X> sampleOrPop(RandomGenerator rng, boolean pop, int max, Consumer<? super X> each) {
        if (max > 0) {
            sample(rng, new Function<>() {

                int count = max;

                @Override
                public SampleReaction apply(X x) {
                    each.accept(x);
                    boolean y = --count > 0;
                    return pop ?
                            (y ? Remove : RemoveAndStop)
                            :
                            (y ? Next : Stop);
                }
            });
        }
        return this;
    }


    /**
     * implementations can provide a custom sampler meant for evaluating the items uniquely.
     * in other words, it means to search mostly iteratively through a container,
     * though psuedorandomness can factor into how this is done - so long as the
     * sequence of returnd items contains minimal amounts of duplicates.
     * <p>
     * the iteration will end early if the container has been exhaustively iterated, if this is possible to know.
     */
    default Iterator<X> sampleUnique(RandomGenerator rng) {
        throw new TODO();
    }

    default @Nullable X sampleUniqueFirst(Predicate<X> filter, RandomGenerator rng) {
        Iterator<? extends X> i = sampleUnique(rng);
        while (i.hasNext()) {
            X x = i.next();
            if (filter.test(x))
                return x;
        }

        return null;
    }

    /**
     * action returned from bag sampling visitor indicating what to do with the current
     * item
     */
    enum SampleReaction {
        Next(false, false),
        Remove(true, false),
        Stop(false, true),
        RemoveAndStop(true, true);

        public final boolean remove;
        public final boolean stop;

        SampleReaction(boolean remove, boolean stop) {
            this.remove = remove;
            this.stop = stop;
        }

        public static SampleReaction the(boolean remove, boolean stop) {
            if (remove)
                return stop ? RemoveAndStop : Remove;
            else
                return stop ? Stop : Next;
        }
    }




//    class RoundRobinSampler<X> extends FasterList<X> implements Sampler<X> {
//
//        @Override
//        public void sample(Random rng, Function<? super X, SampleReaction> each) {
//            int limit;
//            X[] ii;
//            restart: while ((limit = Math.min(size, (ii=items).length)) > 0) {
//
//                if (limit == 0)
//                    return;
//                int next = rng.nextInt(limit);
//                int missesBeforeRestart = 4, misses = 0;
//                X n;
//                do {
//                    do {
//                        next++;
//                        if (next == limit)
//                            next = 0; //loop
//                        n = ii[next];
//                        if (n == null) {
//                            if (misses++ >= missesBeforeRestart) {
//                                break restart;
//                            }
//                        } else {
//                            misses = 0;
//                        }
//                    } while (n == null);
//
//                } while (!each.apply(n).stop);
//                return;
//            }
//        }
//    }

}