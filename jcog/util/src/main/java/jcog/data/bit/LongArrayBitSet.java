package jcog.data.bit;

import com.google.common.base.Joiner;
import jcog.TODO;
import jcog.Util;
import jcog.util.ArrayUtil;

import java.util.Arrays;

/**
 * TODO implement better bulk setAt(start,end,v) impl
 */
public non-sealed class LongArrayBitSet extends MetalBitSet {
    long[] data;

    public LongArrayBitSet(long[] data) {
        assert data.length > 0;
        this.data = data;
    }

    public LongArrayBitSet(int bits) {
        resize(bits);
    }

    public final int next(final boolean what, int from, int to) {
        long[] data = this.data;
        if (from >= to) return -1;

        int wordIndex = from >>> 6;
        if (wordIndex >= data.length) return -1;

        // Pre-calculate all constants used in loop
        long xmask = what ? 0L : -1L;
        long initialMask = -1L << (from & 63);
        int maxWordIndex = Math.min(data.length, (to + 63) >>> 6);

        // Single XOR operation at loop start with mask
        while (true) {
            long word = (data[wordIndex] ^ xmask) & initialMask;

            if (word != 0) {
                int bit = Long.numberOfTrailingZeros(word);
                int result = (wordIndex << 6) | bit;
                return result >= to ? -1 : result;
            }

            wordIndex++;
            if (wordIndex >= maxWordIndex) {
                return -1;
            }

            initialMask = -1L; // All bits set for subsequent iterations
        }
    }

//    int nextOld(boolean what, int from, int to) {
//        long[] data = this.data;
//        for (int i = from; i < to; i++) {
//            if (i % 64 == 0) {
//                //determine skip ahead by highest index
//                long d = data[i / 64];
//                int skip = Long.numberOfTrailingZeros(what ? d : ~d);
//                if (skip != 0) {
//                    i += skip;
//                    if (skip == 64) {
//                        i--;
//                        continue;
//                    }
//                }
//                return i;
//            }
//
//            if (test(i) == what)
//                return i;
//        }
//        return -1;
//    }

    public int capacity() {
        return data.length * 64;
    }

    @Override
    public MetalBitSet clone() {
        return new LongArrayBitSet(data.clone());
    }

    @Override
    public MetalBitSet negateThis() {
        throw new TODO();
    }

    private void resize(long bits) {
        long[] prev = data;

        if (bits == 0)
            data = ArrayUtil.EMPTY_LONG_ARRAY;
        else {
            data = new long
                    [Math.max(1, (int) Math.ceil(((double) bits) / Long.SIZE))];
                    //[Math.max(1, Util.longToInt(bits / Long.SIZE))];

            if (prev != null)
                System.arraycopy(prev, 0, data, 0, Math.min(data.length, prev.length));
        }
    }

    public void clear() {
        Arrays.fill(data, 0);
    }


    /**
     * number of bits set to true
     */
    public int cardinality() {
        int sum = 0;
        for (long l : data)
            sum += Long.bitCount(l);
        return sum;
    }

    @Override
    public boolean isEmpty() {
        for (long l : data)
            if (l != 0)
                return false;
        return true;
    }

    @Override
    public final void set(int i, boolean v) {
        long m = (1L << i);
        long[] d = this.data;
        int I = i >>> 6;
        if (v)  d[I] |=  m;
        else    d[I] &= ~m;
    }

    /**
     * sets 2 contiguous bits; warning should be aligned so that they are on the same word
     */
    void _set2(int i, boolean a, boolean b) {
        long[] d = this.data;
        int I = i >>> 6;
        long dI = d[I];
        long m = (1L << i);
        long x = a ? (dI | m) : (dI & ~m);
        m <<= 1;
        d[I] = b ? (x | m) : (x & ~m);
    }

    public boolean getAndSet(int i, boolean next) {
        long[] d = this.data;

        int I = i >>> 6;

        long di = d[I];

        long j = (1L << i);
        boolean prev = (di & j) != 0;

        if (prev != next)
            d[I] = next ? (di | j) : (di & ~j);

        return prev;
    }

    @Override
    public void setAll(boolean b) {
        Arrays.fill(data, b ? ~0L : 0L);
    }

    /**
     * Returns true if the bit is set in the specified index.
     */
    @Override
    public boolean test(int i) {
        return (data[i >>> 6] & (1L << i)) != 0;
    }

    public long bitSize() {
        return (long) data.length * Long.SIZE;
    }

    /**
     * Combines the two BitArrays using bitwise OR.
     */
    public void putAll(LongArrayBitSet array) {
        long[] d = this.data;
        int n = d.length;
        assert n == array.data.length :
                "BitArrays must be of equal length (" + n + "!= " + array.data.length + ')';
        for (int i = 0; i < n; i++)
            d[i] |= array.data[i];
    }

    @Override
    public String toString() {
        return Joiner.on(" ").join(Util.arrayOf(i -> Long.toHexString(data[data.length - 1 - i]), new String[data.length]));
    }
}