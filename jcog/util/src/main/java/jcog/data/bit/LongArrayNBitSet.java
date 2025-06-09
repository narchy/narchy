package jcog.data.bit;

import org.roaringbitmap.IntIterator;
import org.roaringbitmap.PeekableIntIterator;

public class LongArrayNBitSet extends LongArrayBitSet {

    private final int capacity;

    public LongArrayNBitSet(int bits) {
        super(bits);
        this.capacity = bits;
    }

    @Override
    public int capacity() {
        return capacity;
    }

    public IntIterator iterator() {
        return new MyIntIterator();
    }

    public final int first() { return next(true, 0, capacity); }
    public final int last() { return next(true, capacity - 1, 0, -1); }

    public void set(LongArrayNBitSet values) {
        System.arraycopy(values.data, 0, data, 0, values.data.length);
    }

    private final class MyIntIterator implements IntIterator {
        int next = -1;

        @Override
        public boolean hasNext() {
            int n = LongArrayNBitSet.this.next(true, next + 1, capacity+1);
            if (n < 0)
                return false;

            this.next = n;
            return true;
        }

        @Override
        public int next() {
            return next;
        }

        @Override
        public PeekableIntIterator clone() {
            throw new UnsupportedOperationException();
        }
    }

}
