package jcog.data.set;

import jcog.Util;
import jcog.WTF;
import jcog.data.iterator.ArrayIterator;
import jcog.util.ArrayUtil;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.stream.Stream;

/**
 * adapted from: https://github.com/andrey-stekov/SortedArrayMap/blob/master/src/main/java/org/andrey/collections/SortedArrayMap.java
 * TODO test
 */
public class SortedArraySet<X> implements Iterable<X> {

    private X[] array;
    public int size = 0;
    private final Comparator/*<? super X>*/ cmp;

    private static final SortedArraySet EMPTY = new SortedArraySet(null, ArrayUtil.EMPTY_OBJECT_ARRAY);

    public SortedArraySet(Comparator/*<? super X>*/ comparator, X[] array) {
        this.array = array;
        this.cmp = comparator;
    }

    @Override public Iterator<X> iterator() {
        return ArrayIterator.iterateN(array, size);
    }

    public Stream<X> stream() {
        return ArrayIterator.stream(array, size);
    }

    public final X[] array() {
        return array;
    }

    @Override
    public SortedArraySet<X> clone() {
        var s = new SortedArraySet<>(cmp, array.clone());
        s.size = size;
        return s;
    }

    @Override
    public String toString() {
        //from: AbstractCollection.toString()
        var it = iterator();
        if (! it.hasNext())
            return "[]";

        var sb = new StringBuilder();
        sb.append('[');
        for (;;) {
            var e = it.next();
            sb.append(e == this ? "(this Collection)" : e);
            if (!it.hasNext())
                return sb.append(']').toString();
            sb.append(',').append(' ');
        }
    }

    public int binarySearchNearest(Object key) {
        var i = binarySearch(key);
        return Util.clampSafe(i < 0 ? -i - 1 : i, 0, size-1);
    }

    public int binarySearch(Object key) {
        var size = this.size;
        if (size == 0)
            return -1;

        int low = 0, high = size - 1;
        var a = array;
        while (low <= high) {
            var mid = (low + high) / 2;
            var m = a[mid];
            if (m == null)
                break;
            var c = cmp.compare(m, key);
            if (c < 0) {
                low = mid + 1;
            } else if (c > 0) {
                high = mid - 1;
            } else
                return mid; // key found
        }
        return -(low + 1);  // key not found.
    }

    public X first() {
        return size > 0 ? array[0] : null;
    }

    public X last() {
        return size > 0 ? array[size - 1] : null;
    }

    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    public boolean contains(Object key) {
        return binarySearch(key) >= 0;
    }

    public boolean containsExhaustive(Object value) {
        return ArrayUtil.indexOf(array, value::equals/*Objects.equals(z, value)*/, 0, size)!=-1;
    }

    public void resize(int req) {
        _resize(req);
    }

    private void _resize(int req) {
        if (req > array.length)
            array = Arrays.copyOf(array,
                    req
                    //Math.max(req, 1 + (int)(size * growthRate))
            );
    }

    public X put(X x) {
        var index = binarySearch(x);
        if (index >= 0) {
            var x0 = array[index];
            array[index] = x;
            return x0;
        } else {
            var target = -index - 1;
            putInsert(x, target);
            return null;
        }
    }

    protected void putInsert(X x, int target) {
        target = Math.min(size, target); //HACK
        _resize(size + 1);

        if (target != size)
            System.arraycopy(array, target, array, target + 1, size - target);

        array[target] = x;
        size++;
    }


    /** internal capacity */
    protected int _capacity() {
        return array.length;
    }


    public X remove(Object x) {
        var index = binarySearch(x);
        return index < 0 ? null : remove(index);
    }

    public X get(int i) {
        return array[i];
    }

    @Nullable public X remove(int index) {
        var a = array;
        var old = a[index];
        if (index == size - 1) {
            a[index] = null;
        } else {
            if (size - index - 1 <= 0)
                throw new WTF();
            System.arraycopy(a, index + 1, a, index, size - index - 1);
        }
        size--;
        return old;
    }

    public void clear() {
        Arrays.fill(array, null);  //this.array = new Entry[DEFAULT_CAPACITY];
        this.size = 0;
    }

}
