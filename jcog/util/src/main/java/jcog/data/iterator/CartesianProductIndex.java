package jcog.data.iterator;

import java.util.Iterator;

/**
 * indexed cartesian product iterator
 * TODO optional shuffle, weighted shuffle
 * from: https://stackoverflow.com/a/10946027 */
public class CartesianProductIndex implements Iterable<int[]>, Iterator<int[]> {

    private final int[] lens, index;
    private boolean hasNext = true;
    private transient int[] result;

    public CartesianProductIndex(int[] lengths) {
        lens = lengths;
        index = new int[lengths.length];
    }

    public boolean hasNext() {
        return hasNext;
    }

    public int[] next() {
        int[] indices = index;
        int n = indices.length;

        int[] result = this.result;
        if (result == null) result = this.result = new int[n];

        System.arraycopy(indices, 0, result, 0, n);

        for (int i = n - 1; i >= 0; i--) {
            if (indices[i] == lens[i] - 1) {
                indices[i] = 0;
                if (i == 0)
                    hasNext = false; //end
            } else {
                indices[i]++;
                break;
            }
        }

        return result;
    }

    public Iterator<int[]> iterator() {
        return this;
    }

    public void remove() {
        throw new UnsupportedOperationException();
    }

}