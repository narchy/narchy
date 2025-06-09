package jcog.data.bit;

import com.google.common.base.Joiner;
import jcog.TODO;
import jcog.Util;
import jcog.util.ArrayUtil;

import java.util.Arrays;

/**
 * Optimized implementation of MetalBitSet using an int[] to store bits.
 */
public non-sealed class IntArrayBitSet extends MetalBitSet {
    int[] data;

    public IntArrayBitSet(int bits) {
        resize(bits);
    }

    public IntArrayBitSet(int[] data) {
        assert data.length > 0;
        this.data = data;
    }
    public final int next(final boolean what, int from, int to) {
        int[] data = this.data;
        if (from >= to) return -1;

        int wordIndex = from >>> 5;
        if (wordIndex >= data.length) return -1;

        // Pre-calculate xmask and initial mask
        int xmask = what ? 0 : -1;
        int mask = -1 << (from & 31);
        int maxWordIndex = Math.min(data.length, (to + 31) >>> 5);

        // Single XOR operation at loop start with mask
        while (true) {
            int word = (data[wordIndex] ^ xmask) & mask;
            if (word != 0) {
                int result = (wordIndex << 5) | Integer.numberOfTrailingZeros(word);
                return result >= to ? -1 : result;
            } else if (++wordIndex >= maxWordIndex)
                return -1;

            mask = -1; // All bits set for subsequent iterations
        }
    }
//    @Override
//    public final int next(final boolean what, int from, int to) {
//        int[] data = this.data;
//        int wordIndex = from >>> 5;
//        if (wordIndex >= data.length) return -1;
//        int bitIndex = from & 31;
//        var xmask = what ? 0 : -1;
//        int word = data[wordIndex] ^ xmask;
//        word &= (-1 << bitIndex);
//
//        while (true) {
//            if (word != 0)
//                return trailing(to, wordIndex, word);
//
//            if (++wordIndex == data.length || (wordIndex << 5) >= to)
//                return -1;
//
//            word = data[wordIndex] ^ xmask;
//        }
//    }

    private static int trailing(int to, int wordIndex, int word) {
        int result = (wordIndex << 5) | Integer.numberOfTrailingZeros(word);
        return result < to ? result : -1;
    }

    public int capacity() {
        return data.length * 32;
    }

    @Override
    public MetalBitSet clone() {
        return new IntArrayBitSet(data.clone());
    }

    @Override
    public MetalBitSet negateThis() {
        throw new TODO();
    }

    public void resize(int bits) {
        int[] prev = data;

        if (bits == 0)
            data = ArrayUtil.EMPTY_INT_ARRAY;
        else {
            data = new int[Math.max(1, (int) Math.ceil(((double) bits) / Integer.SIZE))];

            if (prev != null)
                System.arraycopy(prev, 0, data, 0, Math.min(data.length, prev.length));
        }
    }

    public final void clear() {
        Arrays.fill(data, 0);
    }

    /**
     * number of bits set to true
     */
    public final int cardinality() {
        int sum = 0;
        for (int i : data)
            sum += Integer.bitCount(i);
        return sum;
    }

    @Override
    public final boolean isEmpty() {
        for (int i : data)
            if (i != 0)
                return false;
        return true;
    }

    @Override
    public final void set(int i, boolean v) {
        int m = (1 << i);
        int I = i >>> 5;
        int[] d = this.data;
        if (v)  d[I] |=  m;
        else    d[I] &= ~m;
    }

    @Override
    public final MetalBitSet set(int i) {
        this.data[i >>> 5] |= (1 << i);
        return this;
    }
    @Override
    public final MetalBitSet clear(int i) {
        this.data[i >>> 5] &= ~(1 << i);
        return this;
    }

    /**
     * Returns true if the bit is set in the specified index.
     */
    @Override public final boolean test(int i) {
        return (data[i >>> 5] & (1 << i)) != 0;
    }

    /**
     * sets 2 contiguous bits; warning should be aligned so that they are on the same word
     */
    final void _set2(int i, boolean a, boolean b) {
        int[] d = this.data;
        int I = i >>> 5;
        int dI = d[I];
        int m = (1 << i);
        int x = a ? (dI | m) : (dI & ~m);
        m <<= 1;
        d[I] = b ? (x | m) : (x & ~m);
    }

    public final boolean getAndSet(int i, boolean next) {
        int[] d = this.data;
        int I = i >>> 5;
        int di = d[I];
        int j = (1 << i);
        boolean prev = (di & j) != 0;

        if (prev != next)
            d[I] = next ? (di | j) : (di & ~j);

        return prev;
    }

    @Override
    public final void setAll(boolean b) {
        Arrays.fill(data, b ? ~0 : 0);
    }


    /**
     * Combines the two BitArrays using bitwise OR.
     */
    public void putAll(IntArrayBitSet array) {
        int[] d = this.data;
        int n = d.length;
        assert n == array.data.length :
                "BitArrays must be of equal length (" + n + "!= " + array.data.length + ')';
        for (int i = 0; i < n; i++)
            d[i] |= array.data[i];
    }

    @Override
    public String toString() {
        return Joiner.on(" ").join(Util.arrayOf(i -> Integer.toBinaryString(data[data.length - 1 - i]), new String[data.length]));
    }
}