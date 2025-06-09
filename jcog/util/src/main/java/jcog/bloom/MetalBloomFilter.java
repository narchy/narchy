package jcog.bloom;

import jcog.bloom.hash.BytesHasher;
import jcog.bloom.hash.Hasher;

import java.util.function.Function;

public class MetalBloomFilter<E> {
    protected final Hasher<? super E> hasher;
    protected final byte[] cells;
    protected final int cap, hashes;

    public MetalBloomFilter(Function<E, byte[]> hashProvider, int cap, int hashes) {
        this(new BytesHasher(hashProvider), cap, hashes);
    }

    public MetalBloomFilter(Hasher<? super E> hasher, int cap, int hashes) {
        this.hasher = hasher;
        this.cells = new byte[cap];
        this.cap = cap;
        this.hashes = hashes;
    }

    public void add(E element) {
        add(hash(element));
    }

    /** possibly contains */
    public boolean contains(E element) {
        return contains(hash(element));
    }


    private void increment(int idx) {
        if (cells[idx] < Byte.MAX_VALUE)
            cells[idx]++;
    }

    public void add(int[] indices) {
        for (var i : indices)
            increment(i);
    }

    /** possibly contains */
    public boolean contains(int[] indices) {
        for (int i = 0; i < hashes; i++) {
            if (cells[indices[i]] > 0)
                return true;
        }
        return false;
    }

    public int[] hash(E element) {
        var h = this.hashes;
        int[] hashes = new int[h];
        int h1 = hasher.hash1(element);
        int h2 = hasher.hash2(element);
        for (int i = 0; i < h; i++)
            hashes[i] = Math.abs(((h1 + i * h2) % cap));
        return hashes;
    }

}
