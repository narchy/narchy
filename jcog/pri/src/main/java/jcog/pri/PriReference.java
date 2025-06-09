package jcog.pri;

import jcog.pri.distribution.DistributionApproximator;
import org.eclipse.collections.api.block.function.primitive.FloatFunction;

import java.util.Iterator;
import java.util.function.Supplier;

/**
 * prioritized reference
 */
public interface PriReference<X> extends Supplier<X>, Prioritizable {

    FloatFunction<Prioritized> PRI = Prioritized::priElseZero;

    static float[] histogram(Iterable<? extends Prioritized> pp, float[] x) {
        return histogram(pp, x, PRI);
    }

    static <P extends Prioritized> float[] histogram(Iterable<? extends P> pp, float[] x, FloatFunction<P> pri) {
        var bins = x.length;
        var total = 0;

        for (var y : pp) {
            if (y == null) continue;
            var p = pri.floatValueOf(y);
            DistributionApproximator.bin(p, 1, x, bins);
            total++;
        }

        if (total > 0) {
            for (var i = 0; i < bins; i++)
                x[i] /= total;
        }
        return x;
    }

    static boolean equals(PriReference x, Object y) {
        return (x == y) ||
               (y instanceof PriReference Y && x.get().equals(Y.get()));
    }

    static <X> Iterator<X> get(Iterator<PriReference<X>> iterator) {
        return new RefGetIterator<>(iterator);
    }

    /** effectively: Iterators.transform(iterator, Supplier::get), with less overhead */
    final class RefGetIterator<X> implements Iterator<X> {
        private final Iterator<? extends PriReference<X>> iter;

        public RefGetIterator(Iterator<? extends PriReference<X>> i) {
            this.iter = i;
        }

        @Override public boolean hasNext() {
            return iter.hasNext();
        }

        @Override public X next() {
            return iter.next().get();
        }

        @Override public void remove() {
            iter.remove();
        }
    }

//    /**
//     * double[histogramID][bin]
//     */
//    static <X, Y> double[][] histogram(Iterable<PriReference<Y>> pp,  BiConsumer<PriReference<Y>, double[][]> each,  double[][] d) {
//
//        for (var y : pp)
//            each.accept(y, d);
//
//        for (var e : d) {
//            var total = Util.sum(e);
//            if (total > 0) {
//                for (int i = 0, eLength = e.length; i < eLength; i++)
//                    e[i] /= total;
//            }
//        }
//
//        return d;
//    }

}