package jcog.data.set;


import com.google.common.base.Joiner;
import jcog.TODO;
import jcog.Util;
import jcog.data.array.IntComparator;
import jcog.data.bit.MetalBitSet;
import jcog.data.list.Lst;
import jcog.sort.QuickSort;
import jcog.util.ArrayUtil;
import org.eclipse.collections.api.block.predicate.primitive.LongObjectPredicate;
import org.eclipse.collections.api.block.procedure.primitive.IntIntProcedure;
import org.eclipse.collections.api.block.procedure.primitive.LongObjectProcedure;
import org.eclipse.collections.api.iterator.LongIterator;
import org.eclipse.collections.api.iterator.MutableLongIterator;
import org.eclipse.collections.impl.iterator.ImmutableEmptyLongIterator;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.function.BiPredicate;
import java.util.function.IntPredicate;
import java.util.function.Predicate;


/**
 * a set of (long,object) pairs as 2 array lists
 * TODO sort and binary search lookup
 */
public class LongObjectArraySet<X> extends Lst<X> implements IntIntProcedure  {

    public long[] when = ArrayUtil.EMPTY_LONG_ARRAY;

    public LongObjectArraySet() {
        this(0);
    }

    public LongObjectArraySet(int initialSize, X[] array) {
        super(initialSize, array);
    }

    public LongObjectArraySet(int initialCapacity) {
        super(initialCapacity);
    }

    /** public void swap(int a, int b) { */
    @Override public final void value(int a, int b) {
        if (a != b) {
            super.swap(a,b);
            ArrayUtil.swapLong(when, a, b);
        }
    }

    @Override
    public final void swap(int a, int b) {
        value(a, b);
    }

    public LongObjectArraySet<X> sortThis(IntComparator cmp) {
        int size = this.size;
        if (size > 1) {
            //TODO SmoothSort ?
            QuickSort.quickSort(0, size, cmp, this);
        }
        return this;
    }

    @Override
    public final LongObjectArraySet<X> sortThis() {
        return size > 1 ? sortThis(this::compareTimeValue) : this;
    }

    /** but doesnt sort entries within the same time */
    public LongObjectArraySet<X> sortThisByTime() {
        return sortThis((a,b)-> Long.compare(when[a],when[b]));
    }

    /** complete sort but sorts by item natural order first */
    public LongObjectArraySet<X> sortThisByValue() {
        return sortThis(this::compareValueTime);
    }

    private int compareTimeValue(int ia, int ib) {
        if (ia == ib) return 0;
        int ab = compareTime(ia, ib);
        if (ab != 0)
            return ab;

        X[] ii = this.items;
        X a = ii[ia], b = ii[ib];
        return ((Comparable)a).compareTo(b); //TODO non-Comparable compare by obj identity
    }

    public int compareTime(int ia, int ib) {
        long[] ww = this.when;
        return Long.compare(ww[ia], ww[ib]);
    }

    private int compareValueTime(int a, int b) {
        if (a == b) return 0;
        int ab = compareValue(a, b);
        return ab != 0 ? ab : compareTime(a, b);
    }

    public int compareValue(int a, int b) {
        X[] i = this.items;
        X A = i[a], B = i[b];
        int ab = ((Comparable)A).compareTo(B); //TODO non-Comparable compare by obj identity
        return ab;
    }

    @Override
    public void trimToSize() {
        super.trimToSize();
        int s = this.size;
        if (when.length > s)
            when = s > 0 ? Arrays.copyOf(when, s) : ArrayUtil.EMPTY_LONG_ARRAY;
    }

    public boolean contains(long w, X what) {
        return contains(w, what, 0, size());
    }
    public final boolean contains(long w, X what, int startIndex, int finalIndexExc) {
        long[] longs = this.when;
        X[] ii = this.items;
        for (int i = startIndex; i < finalIndexExc; i++) {
            if (longs[i] == w && ii[i].equals(what))
                return true;
        }
        return false;
    }

    /** assumes its been sorted */
    public int _indexOf(long w, X what, int startIndex, int finalIndexExc, BiPredicate<X, X> equal, int dtTolerance, IntPredicate hit) {
        long[] when = this.when; X[] items = this.items;
        boolean fwd = finalIndexExc >= startIndex;
        int s = size();
        for (int i = startIndex; ; i+= fwd ? 1 : -1) {
            if ((fwd && i >= s) || (!fwd && i < 0))
                break; //past the target
            if (!hit.test(i)) {
                long ll = when[i];
                if (Util.equals(ll, w, dtTolerance) && equal.test(items[i], what))
                    return i;
                if ((fwd && ll > w) || (!fwd && ll < w))
                    break; //past the target
            }
        }
        return -1;
    }

    public String toItemString() { return super.toString(); }

    @Override
    public String toString() {
        //HACK this could be better
        int[] i = {0};;
        return Joiner.on(',').join(this.stream().map(n -> when[i[0]++] + ":" + n).iterator());
    }

    /**
     * List semantics are changed in this overridden method. this is just an alias for the addAt(long, X) method
     */
    @Override
    public final void add(int when, X t) {
        throw new UnsupportedOperationException();
    }


    public boolean add(long w, X t) {
        return add(w, t, false);
    }


    protected boolean add(long w, X t, boolean valueIfExists) {
        //var existing = getContained()//TODO

        int n = size();
        long[] when = this.when; X[] items = this.items;
        for (int i = 0; i < n; i++) {
            if (when[i] == w && items[i].equals(t))
                return valueIfExists; //existing found
        }

        addDirect(w, t);
        return true;
    }

    /** add without testing for existing */
    public final void addDirect(long w, X t) {
        int s = addAndGetSize(t);

        //match long[] to the Object[] capacity
        long[] when = this.when;
        if (when.length < s)
            this.when = when = Arrays.copyOf(when, items.length);

        when[s - 1] = w;
    }

    @Override
    public final boolean add(X x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public final X remove(int index) {
        throw new UnsupportedOperationException("use removeThe(index)");
    }

    public boolean remove(long at, X t) {
        return removeIf((when, what) -> at == when && what.equals(t));
    }

    /**
     * removes the ith tuple
     */
    public final X removeThe(int i) {
        removeWhen(i);
        return super.remove(i);
    }

    private void removeWhen(int i) {
        int s = size;
        //if (i < s - 1)
            System.arraycopy(when, i + 1, when, i, s - i - 1);
    }

    public boolean removeIf(long at, LongObjectPredicate<X> iff) {
        int s = size();
        long[] when = this.when; X[] items = this.items;
        MetalBitSet m = null;
        for (int i = 0; i < s; i++) {
            long w = when[i];
            if (w == at && iff.accept(w, items[i]))
                m = bitset(i, s, m);
        }
        return removeAll(m, s);
    }

    @Override
    public boolean removeBelow(int index) {
        Arrays.fill(items, 0, index, null);
        return removeNulls();
    }
    @Override
    public boolean removeAbove(int index) {
        Arrays.fill(items, index, size(), null);
        return removeNulls();
    }

    public void forEachEvent(LongObjectProcedure<X> each) {
        int n = size;
        long[] when = this.when; X[] items = this.items;
        for (int i = 0; i < n; i++)
            each.value(when[i], items[i]);
    }
    public boolean AND(LongObjectPredicate<X> each) {
        int n = size;
        long[] when = this.when; X[] items = this.items;
        for (int i = 0; i < n; i++) {
            if (!each.accept(when[i], items[i]))
                return false;
        }
        return true;
    }
    public boolean OR(LongObjectPredicate<X> each) {
        int n = size;
        long[] when = this.when; X[] items = this.items;
        for (int i = 0; i < n; i++)
            if (each.accept(when[i], items[i]))
                return true;
        return false;
    }

    @Override
    public final void delete() {
        when = null;
        super.delete();
    }

    @Override
    public boolean removeInstances(@Nullable X x) {
        //return removeIf(i -> i == term);
        int s = size; if (s < 1) return false;
        MetalBitSet m = null;
        X[] items = this.items;
        for (int i = 0; i < s; i++)
            if (items[i] == x)
                m = bitset(i, s, m);
        return removeAll(m, s);
    }

    @Override
    public final boolean removeIf(Predicate<? super X> p) {
        //return removeIf((when,what)->predicate.test(what));
        int s = size; if (s < 1) return false;
        MetalBitSet m = null;
        X[] items = this.items;
        for (int i = 0; i < s; i++)
            if (p.test(items[i]))
                m = bitset(i, s, m);
        return removeAll(m, s);
    }

    private static MetalBitSet bitset(int i, int s, MetalBitSet m) {
        ((m == null) ? m = MetalBitSet.bits(s) : m).set(i);
        return m;
    }

    public boolean removeIf(LongObjectPredicate<X> iff) {
        int s = size();
        long[] when = this.when; X[] items = this.items;
        switch (s) {
            case 0:
                return false;
            case 1:
                if (iff.accept(when[0], items[0])) {
                    clear();
                    return true;
                } else
                    return false;
            default:
                return removeIfN(iff, s, when, items);
        }

    }

    private boolean removeIfN(LongObjectPredicate<X> iff, int s, long[] when, X[] items) {
        MetalBitSet m = null;
        for (int i = 0; i < s; i++)
            if (iff.accept(when[i], items[i]))
                m = bitset(i, s, m);
        return removeAll(m, s);
    }


    @Override
    public void removeFast(int i) {
        removeWhen(i);
        super.removeFast(i);
    }

    private boolean removeAll(@Nullable MetalBitSet m, int s) {
        if (m == null) return false;
        int toRemove = m.cardinality();
        if (toRemove == 0)
            return false;
        else if (toRemove == 1) {
            removeFast(m.first(true));
        } else {
            int next = -1, removed = 0;
            while (toRemove-- > 0) {
                next = m.next(true, next + 1, s);
                removeFast(next - removed++);
            }
        }
        return true;
    }

    @Override
    public final boolean removeNulls() {
        return removeInstances(null);
    }

    @Override
    public X removeLast() {
        return removeThe(size-1);
    }

    @Override
    public boolean removeFirstInstance(X x) {
        throw new TODO();
    }

    @Override
    public boolean removeOnce(X x) {
        throw new TODO();
    }

    @Override
    public void removeLastFast() {
        removeFast(size-1);
    }

    public void reverse() {
        int s = size;
        if (s > 1) {
            reverseThis();
            ArrayUtil.reverse(when, 0, s);
        }
    }
//    public LongObjectPair<X> removeEvent(int i) {
//        long w = when[i];
//
//        removeWhen(i, size());
//
//        X x = super.remove(i);
//
//        return PrimitiveTuples.pair(w, x);
//    }


    public boolean removeAll(X X) {
        return switch (size()) {
            case 0 -> false;
            case 1 -> {
                if (getFirst().equals(X)) {
                    removeFast(0);
                    yield true;
                } else
                    yield false;
            }
            default -> removeIf(X::equals);
        };
    }

    public LongIterator longIterator() {
        int s = size;
        //case 1:  //TODO return LongIterator
        return switch (s) {
            case 0 -> ImmutableEmptyLongIterator.INSTANCE;
            default -> new InternalLongIterator(when, s);
        };
    }

    public final long when(int i) {
        return when[i];
    }

    public final boolean removeAll(MetalBitSet toRemove) {
        return removeAll(toRemove, size());
    }

    public boolean symmetricDifference(LongObjectArraySet<X> y) {
//        boolean assumeSorted = false; //TODO

        int xn = size(), yn = y.size();
        MetalBitSet xr = null, yr = null;
        main: for (int xi = 0; xi < xn; xi++) {
            long xw = when[xi];
            X X = items[xi];

            for (int yi = 0; yi < yn; yi++) {
                if (yr!=null && yr.test(yi)) continue; //already removed
                long yw = y.when[yi];
//                if (assumeSorted && yw > xw) break;
                if (xw == yw) {
                    if (X.equals(y.items[yi])) {
                        if (xr == null) {
                            xr = MetalBitSet.bits(xn); yr = MetalBitSet.bits(yn);
                        }
                        xr.set(xi);
                        yr.set(yi);
                        continue main;
                    }
                }
            }
        }
        if (xr!=null) {
            removeAll(xr, xn);
            y.removeAll(yr, yn);
            return true;
        }

        return false;
    }

    private static final class InternalLongIterator implements MutableLongIterator {

        private final long[] data;
        private final int size;
        /**
         * Index of element to be returned by subsequent call to next.
         */
        private int index;

        InternalLongIterator(long[] data, int size) {
            this.data = data;
            this.size = size;
        }

        @Override
        public boolean hasNext() {
            return this.index < size;
        }

        @Override
        public long next() {
            if (!this.hasNext())
                throw new NoSuchElementException();

            return data[this.index++];
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }
}