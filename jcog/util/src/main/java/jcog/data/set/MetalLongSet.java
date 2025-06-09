package jcog.data.set;


import jcog.Util;
import jcog.data.array.LongArrays;

import java.util.Arrays;

public class MetalLongSet  {

    private long[] data;
    private int size;
    private static final float GROWTH_FACTOR =
        Util.PHIf;
        //3/2f;

    public MetalLongSet(int initialCapacity) {
        data = new long[initialCapacity];
        size = 0;
    }

    public boolean add(long x) {
        var s = size;
        if (s == 0 || x > data[s - 1]) {
            append(x);
            return true;
        } else {
            return insert(x);
        }
    }

    private boolean insert(long x) {
        int i = search(x);
        return i < 0 && insert(x, -i - 1);
    }

    private int search(long x) {
        return LongArrays.binarySearch(data, 0, size, x);
    }

    private boolean insert(long x, int index) {
        growMaybe();
        var d = data;
        System.arraycopy(d, index, d, index + 1, (size++) - index);
        d[index] = x;
        return true;
    }

    private void append(long x) {
        growMaybe();
        data[size++] = x;
    }

    private void growMaybe() {
        var d = data;
        var cap = d.length;
        if (size == cap)
            data = Arrays.copyOf(d, (int) (cap * GROWTH_FACTOR));
    }

    public int size() { return size; }

    public boolean contains(long value) {
        return search(value) >= 0;
    }

    /** may return 'data' array as-is */
    public long[] toSortedArray() {
        trim();
        return data;
    }

    public void trim() {
        if (data.length!=size)
            data = Arrays.copyOf(data, size);
    }

    public void addAll(long[] xx) {
        for (var x : xx)
            add(x);
    }

    public boolean removeValue(long v) {
        int index = search(v);
        if (index >= 0) {
            // Shift elements to the left to overwrite the removed element
            System.arraycopy(data, index + 1, data, index, size - index - 1);
            size--;
            return true;
        }
        return false;
    }
}