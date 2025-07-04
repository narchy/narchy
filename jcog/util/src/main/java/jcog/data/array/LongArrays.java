

/* Generic definitions */




/* Assertions (useful to generate conditional code) */
/* Current type and class (and size, if applicable) */
/* Value methods */
/* Interfaces (keys) */
/* Interfaces (values) */
/* Abstract implementations (keys) */
/* Abstract implementations (values) */
/* Static containers (keys) */
/* Static containers (values) */
/* Implementations */
/* Synchronized wrappers */
/* Unmodifiable wrappers */
/* Other wrappers */
/* Methods (keys) */
/* Methods (values) */
/* Methods (keys/values) */
/* Methods that have special names depending on keys (but the special names depend on values) */
/* Equality */
/* Object/Reference-only definitions (keys) */
/* Primitive-type-only definitions (keys) */
/* Object/Reference-only definitions (values) */
/*		 
 * Copyright (C) 2002-2010 Sebastiano Vigna 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 *
 *
 *
 * For the sorting and binary search code:
 *
 * Copyright (C) 1999 CERN - European Organization for Nuclear Research.
 *
 *   Permission to use, copy, modify, distribute and sell this software and
 *   its documentation for any purpose is hereby granted without fee,
 *   provided that the above copyright notice appear in all copies and that
 *   both that copyright notice and this permission notice appear in
 *   supporting documentation. CERN makes no representations about the
 *   suitability of this software for any purpose. It is provided "as is"
 *   without expressed or implied warranty. 
 */
package jcog.data.array;


/**
 * A class providing static methods and objects that do useful things with type-specific arrays.
 * <p>
 * <p>In particular, the <code>ensureCapacity()</code>, <code>grow()</code>,
 * <code>trim()</code> and <code>setLength()</code> methods allow to handle
 * arrays much like array lists. This can be very useful when efficiency (or
 * syntactic simplicity) reasons make array lists unsuitable.
 * <p>
 * <P>Note that {@link it.unimi.dsi.fastutil.io.BinIO} and {@link it.unimi.dsi.fastutil.io.TextIO}
 * contain several methods make it possible to load and save arrays of primitive types as sequences
 * of elements in {@link java.io.DataInput} format (i.e., not as objects) or as sequences of lines of text.
 *
 * @see java.util.Arrays
 */
public enum LongArrays {
    ;


    /**
     * The inverse of the golden ratio times 2<sup>16</sup>.
     */
    public static final long ONEOVERPHI = 106039;

    /**
     * A static, final, empty array.
     */
    public static final long[] EMPTY_ARRAY = {};

    /**
     * Ensures that an array can contain the given number of entries.
     * <p>
     * <P>If you cannot foresee whether this array will need again to be
     * enlarged, you should probably use <code>grow()</code> instead.
     *
     * @param array  an array.
     * @param length the new minimum length for this array.
     * @return <code>array</code>, if it contains <code>length</code> entries or more; otherwise,
     * an array with <code>length</code> entries whose first <code>array.length</code>
     * entries are the same as those of <code>array</code>.
     */
    public static long[] ensureCapacity(long[] array, int length) {
        if (length > array.length) {
            long[] t =
                    new long[length];
            System.arraycopy(array, 0, t, 0, array.length);
            return t;
        }
        return array;
    }

    /**
     * Ensures that an array can contain the given number of entries, preserving just a part of the array.
     *
     * @param array    an array.
     * @param length   the new minimum length for this array.
     * @param preserve the number of elements of the array that must be preserved in case a new allocation is necessary.
     * @return <code>array</code>, if it can contain <code>length</code> entries or more; otherwise,
     * an array with <code>length</code> entries whose first <code>preserve</code>
     * entries are the same as those of <code>array</code>.
     */
    public static long[] ensureCapacity(long[] array, int length, int preserve) {
        if (length > array.length) {
            long[] t =
                    new long[length];
            System.arraycopy(array, 0, t, 0, preserve);
            return t;
        }
        return array;
    }

    /**
     * Grows the given array to the maximum between the given length and
     * the current length divided by the golden ratio, provided that the given
     * length is larger than the current length.
     * <p>
     * <P> Dividing by the golden ratio (&phi;) approximately increases the array
     * length by 1.618. If you want complete control on the array growth, you
     * should probably use <code>ensureCapacity()</code> instead.
     *
     * @param array  an array.
     * @param length the new minimum length for this array.
     * @return <code>array</code>, if it can contain <code>length</code>
     * entries; otherwise, an array with
     * max(<code>length</code>,<code>array.length</code>/&phi;) entries whose first
     * <code>array.length</code> entries are the same as those of <code>array</code>.
     */
    public static long[] grow(long[] array, int length) {
        if (length > array.length) {
            int newLength = (int) Math.min(Math.max((ONEOVERPHI * array.length) >>> 16, length), Integer.MAX_VALUE);
            long[] t =
                    new long[newLength];
            System.arraycopy(array, 0, t, 0, array.length);
            return t;
        }
        return array;
    }

    /**
     * Grows the given array to the maximum between the given length and
     * the current length divided by the golden ratio, provided that the given
     * length is larger than the current length, preserving just a part of the array.
     * <p>
     * <P> Dividing by the golden ratio (&phi;) approximately increases the array
     * length by 1.618. If you want complete control on the array growth, you
     * should probably use <code>ensureCapacity()</code> instead.
     *
     * @param array    an array.
     * @param length   the new minimum length for this array.
     * @param preserve the number of elements of the array that must be preserved in case a new allocation is necessary.
     * @return <code>array</code>, if it can contain <code>length</code>
     * entries; otherwise, an array with
     * max(<code>length</code>,<code>array.length</code>/&phi;) entries whose first
     * <code>preserve</code> entries are the same as those of <code>array</code>.
     */
    public static long[] grow(long[] array, int length, int preserve) {
        if (length > array.length) {
            int newLength = (int) Math.min(Math.max((ONEOVERPHI * array.length) >>> 16, length), Integer.MAX_VALUE);
            long[] t =
                    new long[newLength];
            System.arraycopy(array, 0, t, 0, preserve);
            return t;
        }
        return array;
    }

    /**
     * Trims the given array to the given length.
     *
     * @param array  an array.
     * @param length the new maximum length for the array.
     * @return <code>array</code>, if it contains <code>length</code>
     * entries or less; otherwise, an array with
     * <code>length</code> entries whose entries are the same as
     * the first <code>length</code> entries of <code>array</code>.
     */
    public static long[] trim(long[] array, int length) {
        if (length >= array.length) return array;
        long[] t =
                length == 0 ? EMPTY_ARRAY : new long[length];
        System.arraycopy(array, 0, t, 0, length);
        return t;
    }

    /**
     * Sets the length of the given array.
     *
     * @param array  an array.
     * @param length the new length for the array.
     * @return <code>array</code>, if it contains exactly <code>length</code>
     * entries; otherwise, if it contains <em>more</em> than
     * <code>length</code> entries, an array with <code>length</code> entries
     * whose entries are the same as the first <code>length</code> entries of
     * <code>array</code>; otherwise, an array with <code>length</code> entries
     * whose first <code>array.length</code> entries are the same as those of
     * <code>array</code>.
     */
    public static long[] setLength(long[] array, int length) {
        if (length == array.length) return array;
        if (length < array.length) return trim(array, length);
        return ensureCapacity(array, length);
    }

//    /**
//     * Returns a copy of a portion of an array.
//     *
//     * @param array  an array.
//     * @param offset the first element to copy.
//     * @param length the number of elements to copy.
//     * @return a new array containing <code>length</code> elements of <code>array</code> starting at <code>offset</code>.
//     */
//    public static long[] copy(long[] array, int offset, int length) {
//        ensureOffsetLength(array, offset, length);
//        long[] a =
//                length == 0 ? EMPTY_ARRAY : new long[length];
//        System.arraycopy(array, offset, a, 0, length);
//        return a;
//    }

    /**
     * Returns a copy of an array.
     *
     * @param array an array.
     * @return a copy of <code>array</code>.
     */
    public static long[] copy(long[] array) {
        return array.clone();
    }

    /**
     * Fills the given array with the given value.
     * <p>
     * <P>This method uses a backward loop. It is significantly faster than the corresponding
     * method in {@link java.util.Arrays}.
     *
     * @param array an array.
     * @param value the new value for all elements of the array.
     */
    public static void fill(long[] array, long value) {
        int i = array.length;
        while (i-- != 0) array[i] = value;
    }

//    /**
//     * Fills a portion of the given array with the given value.
//     * <p>
//     * <P>If possible (i.e., <code>from</code> is 0) this method uses a
//     * backward loop. In this case, it is significantly faster than the
//     * corresponding method in {@link java.util.Arrays}.
//     *
//     * @param array an array.
//     * @param from  the starting index of the portion to fill.
//     * @param to    the end index of the portion to fill.
//     * @param value the new value for all elements of the specified portion of the array.
//     */
//    public static void fill(long[] array, int from, int to, long value) {
//        ensureFromTo(array, from, to);
//        if (from == 0) while (to-- != 0) array[to] = value;
//        else for (int i = from; i < to; i++) array[i] = value;
//    }

    /**
     * Returns true if the two arrays are elementwise equal.
     * <p>
     * <P>This method uses a backward loop. It is significantly faster than the corresponding
     * method in {@link java.util.Arrays}.
     *
     * @param a1 an array.
     * @param a2 another array.
     * @return true if the two arrays are of the same length, and their elements are equal.
     */
    public static boolean equals(long[] a1, long[] a2) {
        int i = a1.length;
        if (i != a2.length) return false;
        while (i-- != 0) if (a1[i] != a2[i]) return false;
        return true;
    }

//    /**
//     * Ensures that a range given by its first (inclusive) and last (exclusive) elements fits an array.
//     * <p>
//     * <P>This method may be used whenever an array range check is needed.
//     *
//     * @param a    an array.
//     * @param from a start index (inclusive).
//     * @param to   an end index (inclusive).
//     * @throws IllegalArgumentException       if <code>from</code> is greater than <code>to</code>.
//     * @throws ArrayIndexOutOfBoundsException if <code>from</code> or <code>to</code> are greater than the array length or negative.
//     */
//    public static void ensureFromTo(long[] a, int from, int to) {
//        ArrayUtil.ensureFromTo(a.length, from, to);
//    }




//    /**
//     * Ensures that a range given by an offset and a length fits an array.
//     * <p>
//     * <P>This method may be used whenever an array range check is needed.
//     *
//     * @param a      an array.
//     * @param offset a start index.
//     * @param length a length (the number of elements in the range).
//     * @throws IllegalArgumentException       if <code>length</code> is negative.
//     * @throws ArrayIndexOutOfBoundsException if <code>offset</code> is negative or <code>offset</code>+<code>length</code> is greater than the array length.
//     */
//    public static void ensureOffsetLength(long[] a, int offset, int length) {
//        ArrayUtil.ensureOffsetLength(a.length, offset, length);
//    }

    private static final int SMALL = 7;
    private static final int MEDIUM = 40;

    private static void swap(long[] x, int a, int b) {
        long t = x[a];
        x[a] = x[b];
        x[b] = t;
    }

    private static void vecSwap(long[] x, int a, int b, int n) {
        for (int i = 0; i < n; i++, a++, b++) swap(x, a, b);
    }

    private static int med3(long[] x, int a, int b, int c, LongComparator comp) {
        int ab = comp.compare(x[a], x[b]);
        int ac = comp.compare(x[a], x[c]);
        int bc = comp.compare(x[b], x[c]);
        return (ab < 0 ?
                (bc < 0 ? b : ac < 0 ? c : a) :
                (bc > 0 ? b : ac > 0 ? c : a));
    }

    /**
     * Sorts the specified range of elements according to the order induced by the specified
     * comparator using quicksort.
     * <p>
     * <p>The sorting algorithm is a tuned quicksort adapted from Jon L. Bentley and M. Douglas
     * McIlroy, &ldquo;Engineering a Sort Function&rdquo;, <i>Software: Practice and Experience</i>, 23(11), pages
     * 1249&minus;1265, 1993.
     *
     * @param x    the array to be sorted.
     * @param from the index of the first element (inclusive) to be sorted.
     * @param to   the index of the last element (exclusive) to be sorted.
     * @param comp the comparator to determine the sorting order.
     */
    public static void quickSort(long[] x, int from, int to, LongComparator comp) {
        int len = to - from;
        
        if (len < SMALL) {
            for (int i = from; i < to; i++)
                for (int j = i; j > from && comp.compare(x[j - 1], x[j]) > 0; j--) swap(x, j, j - 1);
            return;
        }
        
        int m = from + len / 2; 
        if (len > SMALL) {
            int l = from;
            int n = to - 1;
            if (len > MEDIUM) { 
                int s = len / 8;
                l = med3(x, l, l + s, l + 2 * s, comp);
                m = med3(x, m - s, m, m + s, comp);
                n = med3(x, n - 2 * s, n - s, n, comp);
            }
            m = med3(x, l, m, n, comp); 
        }
        long v = x[m];

        int c = to - 1;
        int a = from;
        int b = a, d = c;
        while (true) {
            int comparison;
            while (b <= c && (comparison = comp.compare(x[b], v)) <= 0) {
                if (comparison == 0) swap(x, a++, b);
                b++;
            }
            while (c >= b && (comparison = comp.compare(x[c], v)) >= 0) {
                if (comparison == 0) swap(x, c, d--);
                c--;
            }
            if (b > c) break;
            swap(x, b++, c--);
        }

        {
            int s = Math.min(a - from, b - a);
            vecSwap(x, from, b - s, s);
        }
        {
            int n = to;
            int s = Math.min(d - c, n - d - 1);
            vecSwap(x, b, n - s, s);

            if ((s = b - a) > 1) quickSort(x, from, from + s, comp);
            if ((s = d - c) > 1) quickSort(x, n - s, n, comp);
        }
    }

    /**
     * Sorts an array according to the order induced by the specified
     * comparator using quicksort.
     * <p>
     * <p>The sorting algorithm is a tuned quicksort adapted from Jon L. Bentley and M. Douglas
     * McIlroy, &ldquo;Engineering a Sort Function&rdquo;, <i>Software: Practice and Experience</i>, 23(11), pages
     * 1249&minus;1265, 1993.
     *
     * @param x    the array to be sorted.
     * @param comp the comparator to determine the sorting order.
     */
    public static void quickSort(long[] x, LongComparator comp) {
        quickSort(x, 0, x.length, comp);
    }

    @SuppressWarnings("unchecked")
    private static int med3(long[] x, int a, int b, int c) {
        int ab = (Long.compare(x[a], x[b]));
        int ac = (Long.compare(x[a], x[c]));
        int bc = (Long.compare(x[b], x[c]));
        return (ab < 0 ?
                (bc < 0 ? b : ac < 0 ? c : a) :
                (bc > 0 ? b : ac > 0 ? c : a));
    }

    /**
     * Sorts the specified range of elements according to the natural ascending order using quicksort.
     * <p>
     * <p>The sorting algorithm is a tuned quicksort adapted from Jon L. Bentley and M. Douglas
     * McIlroy, &ldquo;Engineering a Sort Function&rdquo;, <i>Software: Practice and Experience</i>, 23(11), pages
     * 1249&minus;1265, 1993.
     *
     * @param x    the array to be sorted.
     * @param from the index of the first element (inclusive) to be sorted.
     * @param to   the index of the last element (exclusive) to be sorted.
     */
    @SuppressWarnings("unchecked")
    public static void quickSort(long[] x, int from, int to) {
        int len = to - from;
        
        if (len < SMALL) {
            for (int i = from; i < to; i++)
                for (int j = i; j > from && ((x[j]) < (x[j - 1])); j--) swap(x, j, j - 1);
            return;
        }
        
        int m = from + len / 2; 
        if (len > SMALL) {
            int l = from;
            int n = to - 1;
            if (len > MEDIUM) { 
                int s = len / 8;
                l = med3(x, l, l + s, l + 2 * s);
                m = med3(x, m - s, m, m + s);
                n = med3(x, n - 2 * s, n - s, n);
            }
            m = med3(x, l, m, n); 
        }
        long v = x[m];

        int c = to - 1;
        int a = from;
        int b = a, d = c;
        while (true) {
            int comparison;
            while (b <= c && (comparison = (Long.compare(x[b], v))) <= 0) {
                if (comparison == 0) swap(x, a++, b);
                b++;
            }
            while (c >= b && (comparison = (Long.compare(x[c], v))) >= 0) {
                if (comparison == 0) swap(x, c, d--);
                c--;
            }
            if (b > c) break;
            swap(x, b++, c--);
        }

        int s = Math.min(a - from, b - a);
        vecSwap(x, from, b - s, s);
        int n = to;
        s = Math.min(d - c, n - d - 1);
        vecSwap(x, b, n - s, s);
        
        if ((s = b - a) > 1) quickSort(x, from, from + s);
        if ((s = d - c) > 1) quickSort(x, n - s, n);
    }

    /**
     * Sorts an array according to the natural ascending order using quicksort.
     * <p>
     * <p>The sorting algorithm is a tuned quicksort adapted from Jon L. Bentley and M. Douglas
     * McIlroy, &ldquo;Engineering a Sort Function&rdquo;, <i>Software: Practice and Experience</i>, 23(11), pages
     * 1249&minus;1265, 1993.
     *
     * @param x the array to be sorted.
     */
    public static void quickSort(long[] x) {
        quickSort(x, 0, x.length);
    }

    /**
     * Sorts the specified range of elements according to the natural ascending order using mergesort, using a given support array.
     * <p>
     * <p>This sort is guaranteed to be <i>stable</i>: equal elements will not be reordered as a result
     * of the sort. Moreover, no support arrays will be allocated.
     *
     * @param a    the array to be sorted.
     * @param from the index of the first element (inclusive) to be sorted.
     * @param to   the index of the last element (exclusive) to be sorted.
     * @param supp a support array, at least as large as <code>a</code>.
     */
    @SuppressWarnings("unchecked")
    public static void mergeSort(long[] a, int from, int to, long[] supp) {
        int len = to - from;
        
        if (len < SMALL) {
            for (int i = from; i < to; i++)
                for (int j = i; j > from && ((a[j]) < (a[j - 1])); j--)
                    swap(a, j, j - 1);
            return;
        }
        
        int mid = (from + to) >>> 1;
        mergeSort(supp, from, mid, a);
        mergeSort(supp, mid, to, a);
        
        
        if (((supp[mid - 1]) <= (supp[mid]))) {
            System.arraycopy(supp, from, a, from, len);
            return;
        }
        
        for (int i = from, p = from, q = mid; i < to; i++) {
            a[i] = supp[q >= to || p < mid && supp[p] <= supp[q] ? p++ : q++];
        }
    }

    /**
     * Sorts the specified range of elements according to the natural ascending order using mergesort.
     * <p>
     * <p>This sort is guaranteed to be <i>stable</i>: equal elements will not be reordered as a result
     * of the sort. An array as large as <code>a</code> will be allocated by this method.
     *
     * @param a    the array to be sorted.
     * @param from the index of the first element (inclusive) to be sorted.
     * @param to   the index of the last element (exclusive) to be sorted.
     */
    public static void mergeSort(long[] a, int from, int to) {
        mergeSort(a, from, to, a.clone());
    }

    /**
     * Sorts an array according to the natural ascending order using mergesort.
     * <p>
     * <p>This sort is guaranteed to be <i>stable</i>: equal elements will not be reordered as a result
     * of the sort. An array as large as <code>a</code> will be allocated by this method.
     *
     * @param a the array to be sorted.
     */
    public static void mergeSort(long[] a) {
        mergeSort(a, 0, a.length);
    }

    /**
     * Sorts the specified range of elements according to the order induced by the specified
     * comparator using mergesort, using a given support array.
     * <p>
     * <p>This sort is guaranteed to be <i>stable</i>: equal elements will not be reordered as a result
     * of the sort. Moreover, no support arrays will be allocated.
     *
     * @param a    the array to be sorted.
     * @param from the index of the first element (inclusive) to be sorted.
     * @param to   the index of the last element (exclusive) to be sorted.
     * @param comp the comparator to determine the sorting order.
     * @param supp a support array, at least as large as <code>a</code>.
     */
    @SuppressWarnings("unchecked")
    public static void mergeSort(long[] a, int from, int to, LongComparator comp, long[] supp) {
        int len = to - from;
        
        if (len < SMALL) {
            for (int i = from; i < to; i++)
                for (int j = i; j > from && comp.compare(a[j - 1], a[j]) > 0; j--)
                    swap(a, j, j - 1);
            return;
        }
        
        int mid = (from + to) >>> 1;
        mergeSort(supp, from, mid, comp, a);
        mergeSort(supp, mid, to, comp, a);
        
        
        if (comp.compare(supp[mid - 1], supp[mid]) <= 0) {
            System.arraycopy(supp, from, a, from, len);
            return;
        }
        
        for (int i = from, p = from, q = mid; i < to; i++) {
            a[i] = supp[q >= to || p < mid && comp.compare(supp[p], supp[q]) <= 0 ? p++ : q++];
        }
    }

    /**
     * Sorts the specified range of elements according to the order induced by the specified
     * comparator using mergesort.
     * <p>
     * <p>This sort is guaranteed to be <i>stable</i>: equal elements will not be reordered as a result
     * of the sort. An array as large as <code>a</code> will be allocated by this method.
     *
     * @param a    the array to be sorted.
     * @param from the index of the first element (inclusive) to be sorted.
     * @param to   the index of the last element (exclusive) to be sorted.
     * @param comp the comparator to determine the sorting order.
     */
    public static void mergeSort(long[] a, int from, int to, LongComparator comp) {
        mergeSort(a, from, to, comp, a.clone());
    }

    /**
     * Sorts an array according to the order induced by the specified
     * comparator using mergesort.
     * <p>
     * <p>This sort is guaranteed to be <i>stable</i>: equal elements will not be reordered as a result
     * of the sort.  An array as large as <code>a</code> will be allocated by this method.
     *
     * @param a    the array to be sorted.
     * @param comp the comparator to determine the sorting order.
     */
    public static void mergeSort(long[] a, LongComparator comp) {
        mergeSort(a, 0, a.length, comp);
    }

    /**
     * Searches a range of the specified array for the specified value using
     * the binary search algorithm. The range must be sorted prior to making this call.
     * If it is not sorted, the results are undefined. If the range contains multiple elements with
     * the specified value, there is no guarantee which one will be found.
     *
     * @param a    the array to be searched.
     * @param from the index of the first element (inclusive) to be searched.
     * @param to   the index of the last element (exclusive) to be searched.
     * @param key  the value to be searched for.
     * @return index of the search key, if it is contained in the array;
     * otherwise, <tt>(-(<i>insertion point</i>) - 1)</tt>.  The <i>insertion
     * point</i> is defined as the the point at which the value would
     * be inserted into the array: the index of the first
     * element greater than the key, or the length of the array, if all
     * elements in the array are less than the specified key.  Note
     * that this guarantees that the return value will be &gt;= 0 if
     * and only if the key is found.
     * @see java.util.Arrays
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static int binarySearch(long[] a, int from, int to, long key) {
        if (to==from)
            return -1;
        while (from <= to) {
            int mid = (from + to) >>> 1;
            long midVal = a[mid];
            if (midVal < key) from = mid + 1;
            else if (midVal > key) to = mid - 1;
            else return mid;
        }
        return -(from + 1);
    }

    /**
     * Searches an array for the specified value using
     * the binary search algorithm. The range must be sorted prior to making this call.
     * If it is not sorted, the results are undefined. If the range contains multiple elements with
     * the specified value, there is no guarantee which one will be found.
     *
     * @param a   the array to be searched.
     * @param key the value to be searched for.
     * @return index of the search key, if it is contained in the array;
     * otherwise, <tt>(-(<i>insertion point</i>) - 1)</tt>.  The <i>insertion
     * point</i> is defined as the the point at which the value would
     * be inserted into the array: the index of the first
     * element greater than the key, or the length of the array, if all
     * elements in the array are less than the specified key.  Note
     * that this guarantees that the return value will be &gt;= 0 if
     * and only if the key is found.
     * @see java.util.Arrays
     */
    public static int binarySearch(long[] a, long key) {
        return binarySearch(a, 0, a.length, key);
    }

    /**
     * Searches a range of the specified array for the specified value using
     * the binary search algorithm and a specified comparator. The range must be sorted following the comparator prior to making this call.
     * If it is not sorted, the results are undefined. If the range contains multiple elements with
     * the specified value, there is no guarantee which one will be found.
     *
     * @param a    the array to be searched.
     * @param from the index of the first element (inclusive) to be searched.
     * @param to   the index of the last element (exclusive) to be searched.
     * @param key  the value to be searched for.
     * @param c    a comparator.
     * @return index of the search key, if it is contained in the array;
     * otherwise, <tt>(-(<i>insertion point</i>) - 1)</tt>.  The <i>insertion
     * point</i> is defined as the the point at which the value would
     * be inserted into the array: the index of the first
     * element greater than the key, or the length of the array, if all
     * elements in the array are less than the specified key.  Note
     * that this guarantees that the return value will be &gt;= 0 if
     * and only if the key is found.
     * @see java.util.Arrays
     */
    public static int binarySearch(long[] a, int from, int to, long key, LongComparator c) {
        while (from <= to) {
            int mid = (from + to) >>> 1;
            long midVal = a[mid];
            int cmp = c.compare(midVal, key);
            if (cmp < 0) from = mid + 1;
            else if (cmp > 0) to = mid - 1;
            else return mid; 
        }
        return -(from + 1);
    }

    /**
     * Searches an array for the specified value using
     * the binary search algorithm and a specified comparator. The range must be sorted following the comparator prior to making this call.
     * If it is not sorted, the results are undefined. If the range contains multiple elements with
     * the specified value, there is no guarantee which one will be found.
     *
     * @param a   the array to be searched.
     * @param key the value to be searched for.
     * @param c   a comparator.
     * @return index of the search key, if it is contained in the array;
     * otherwise, <tt>(-(<i>insertion point</i>) - 1)</tt>.  The <i>insertion
     * point</i> is defined as the the point at which the value would
     * be inserted into the array: the index of the first
     * element greater than the key, or the length of the array, if all
     * elements in the array are less than the specified key.  Note
     * that this guarantees that the return value will be &gt;= 0 if
     * and only if the key is found.
     * @see java.util.Arrays
     */
    public static int binarySearch(long[] a, long key, LongComparator c) {
        return binarySearch(a, 0, a.length, key, c);
    }

























}