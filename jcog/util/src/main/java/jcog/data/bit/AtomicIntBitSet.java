package jcog.data.bit;

import jcog.TODO;
import jcog.data.list.FastAtomicIntArray;

/** from: https://stackoverflow.com/a/12425007 */
public non-sealed class AtomicIntBitSet extends MetalBitSet {

    //TODO volatile int capacity; etc..

    public static final AtomicIntBitSet EMPTY = new AtomicIntBitSet(0);

    private final FastAtomicIntArray array;

    public AtomicIntBitSet(int length) {
        int intLength = (length + 31)/32; // unsigned / 32
        array = new FastAtomicIntArray(intLength);
    }

    public void set(int n, boolean value) {
        int mask = 1 << n;
        if (!value) mask = ~mask;
        int i = n / 32;
        int p, next;
        do {
            p = array.getAcquire(i);
            next = value ? (p | mask) : (p & mask);
        } while (p != next && array.compareAndExchangeRelease(i, p, next)!=p);
    }

    /** clears all */
    public void clear() {
        array.fill(0);
    }

    @Override
    public int cardinality() {
        return 0;
    }

    @Override
    public int capacity() {
        return array.length();
    }

    @Override
    public MetalBitSet negateThis() {
        throw new TODO();
    }

    public boolean test(int n) {
        return (array.getAcquire(n / 32) & (1 << n)) != 0;
    }

    @Override
    public int next(boolean what, int from, int to) {
        return !what ?
            nextClearBit(from, to) :
            super.next(what, from, to); //TODO adapt from java.util.BitSet
    }

    /** from java.util.BitSet */
    private int nextClearBit(int from, int cap) {
        int u = from / 32;
        int mask = 0xffffffff << from;

        FastAtomicIntArray a = this.array;
        cap = Math.min(a.length()*32, cap); //HACK in case array changed in the call

        long word = ~a.getAcquire(u) & mask;

        while (true) {
            if (word != 0)
                return (u * 32) + Long.numberOfTrailingZeros(word);
            if (++u >= cap)
                return -1;
            word = ~a.getAcquire(u);
        }
    }

    public int size() {
        int n = array.length(), c = 0;
        for (int i= 0; i < n; i++)
            c += Integer.bitCount( array.getAcquire(i) );
        return c;
    }

    @Override
    public MetalBitSet clone() {
        throw new TODO();
    }

    @Override
    public void setAll(boolean b) {
        throw new TODO();
    }
}