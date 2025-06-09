package jcog.data.bit;

import jcog.TODO;
import jcog.data.array.IntComparator;
import jcog.data.iterator.AbstractIntIterator;
import org.eclipse.collections.api.block.procedure.primitive.IntProcedure;
import org.eclipse.collections.api.iterator.IntIterator;

import java.util.Random;
import java.util.function.IntPredicate;

/**
 * Bare Metal Fixed-Size BitSets
 * <p>
 * for serious performance. implementations will not check index bounds
 * nor grow in capacity
 *
 * TODO methods from: http://hg.openjdk.java.net/jdk/jdk/file/2cc1ae79b303/src/java.xml/share/classes/com/sun/org/apache/xalan/internal/xsltc/dom/BitArray.java
 */
public abstract sealed class MetalBitSet implements IntPredicate permits AtomicIntBitSet, AtomicMetalBitSet, IntArrayBitSet, IntBitSet, LongArrayBitSet {

    public static MetalBitSet bits(boolean[] arr) {
        var b = bits(arr.length);
        for (int i = 0, arrLength = arr.length; i < arrLength; i++) {
            if (arr[i])
                b.set(i);
        }
        return b;
    }

    public abstract MetalBitSet clone();

    public abstract void set(int i, boolean v);

    public MetalBitSet set(int i) {
        set(i, true);
        return this;
    }

    public MetalBitSet clear(int i) {
        set(i, false);
        return this;
    }

    public final MetalBitSet set(int... ii) {
        assert(ii.length>0);
        for (var i : ii)
            set(i);
        return this;
    }

    public abstract void clear();

    public abstract int cardinality();

    public boolean isEmpty() {
        return cardinality() == 0;
    }

    public void setRange(boolean v, int start, int end) {
        for (var i = start; i < end; i++)
            if (v) set(i); else clear(i);
    }

    /** use caution if capacity exceeds what you expect */
    public int first(boolean what) {
        return next(what, 0, capacity());
    }

    public int next(boolean what, int from, int to) {
        return next(what, from, to, +1);
    }

    /**
     * finds the next bit matching 'what' between from (inclusive) and to (exclusive), or -1 if nothing found
     * TODO use Integer bit methods
     */
    public int next(boolean what, int from, int to, int inc) {
        if (inc == 0) throw new UnsupportedOperationException();
        for (var i = from; i != to; i+=inc)
            if (test(i) == what)
                return i;
        return -1;
    }

    public void forEach(IntProcedure each) {
        int next = -1, n;
        var cap = capacity();
        while ((n = next(true, next + 1, cap)) >= 0) {
            each.accept(n);
            next = n;
        }
    }

    /** modifies this by OR-ing the values with another bitset of equivalent (or smaller?) capacity */
    public void orThis(MetalBitSet other) {
        throw new TODO();
    }

    /** modifies this by AND-ing the values with another bitset of equivalent (or smaller?) capacity */
    public void andThis(MetalBitSet other) {
        throw new TODO();
    }

    public abstract int capacity();

    public final IntComparator indexComparator() {
        return (a,b) -> Boolean.compare(test(a), test(b));
    }

    public final IntComparator indexComparatorReverse() {
        return (a,b) -> Boolean.compare(test(b), test(a));
    }

    public void swap(int a, int b) {
        if (a!=b) {
            boolean A = test(a), B = test(b);
            if (A!=B) {
                set(a, B);
                set(b, A);
            }
        }
    }

    /** modifies this instance by inverting all the bit values
     *  warning this may modify bits beyond the expected range, causing unexpected cardinality changes
     *  returns this instance
     * */
    public abstract MetalBitSet negateThis();

    public void negateBit(int b) {
        set(b, !test(b));
    }

    public MetalBitSet negateAll(int lowestBits) {
        for (var b = 0; b < lowestBits; b++)
            negateBit(b);
        return this;
    }


    /** TODO optimized impl in subclasses */
    public MetalBitSet and(MetalBitSet other) {
        var c = cardinality();
        assert(c == other.cardinality());
        if (this == other) return this;

        //TODO if (this.equals(other)

        var m = bits(c);
        for (var i = 0; i < c; i++)
            m.set(i, test(i) & other.test(i));

        //TODO if (m.equals(this)) else if m.equals(...)
        return m;
    }

    @Override
    public boolean equals(Object obj) {
        throw new TODO();
    }

    public int random(Random rng) {
        var c = cardinality();
        if (c <= 1) {
            return first(true);
        } else {
            var which = rng.nextInt(c);
            var at = -1;
            for ( ; which >= 0; which--)
                at = next(true, at+1, Integer.MAX_VALUE);
            return at;
        }
    }

    public boolean getAndSet(int i, boolean b) {
        var was = test(i);
        if (b) set(i); else clear(i);
        return was;
    }

    /** TODO optimized impl in subclasses */
    public int count(boolean b) {
        var n = cardinality();
        var s = 0;
        for (var i = 0; i < n; i++) {
            if (test(i)==b) s++;
        }
        return s;
    }


    static final IntIterator EmptyIntIterator = new IntIterator() {
        @Override public int next() {
            throw new UnsupportedOperationException();
        }
        @Override public boolean hasNext() {
            return false;
        }
    };

    /** interator */
    public IntIterator iterator(int from, int to) {
        var start = MetalBitSet.this.next(true, from, to);
        if (start < 0 || start >= to)
            return EmptyIntIterator;

        //TODO if (start < 0) return new EmptyIntIterator();
        return new MetalBitSetIntIterator(start, to);
    }


    public static MetalBitSet full() {
        return bits(32);
    }

    public static MetalBitSet bits(int size) {
		return size <= 32 ?
            new IntBitSet() :
            new IntArrayBitSet(size);
            //new LongArrayBitSet(size);
    }

    /**
     * @param from starting index, inclusive
     * @param to ending index, exclusive
     * @return cardinality in the range
     */
    public int cardinality(int from, int to) {
        var count = 0;
        for (var i = from; i < to; i++) {
            if (test(i)) count++;
        }
        return count;
    }

    public void toArray(int fromBit, int toBit, short[] tgt) {
        var hh = iterator(fromBit, toBit);
        var j = 0;
        while (hh.hasNext()) {
            tgt[j++] = (short) hh.next();
        }
        //assert(j==c);
    }

    public int setNext(int cap) {
        var y = next(false, 0, cap);
        if (y >= 0)
            set(y);
        return y;
    }

    abstract public void setAll(boolean b);

    private final class MetalBitSetIntIterator extends AbstractIntIterator {

        MetalBitSetIntIterator(int start, int to) {
            super(start, to);
        }

        @Override protected int next(int next) {
            return MetalBitSet.this.next(true, next + 1, to);
        }
    }
}