package jcog.sort;

import jcog.data.array.IntComparator;
import jcog.util.ArrayUtil;
import org.eclipse.collections.api.block.function.primitive.FloatFunction;
import org.eclipse.collections.api.block.function.primitive.IntToDoubleFunction;
import org.eclipse.collections.api.block.function.primitive.IntToFloatFunction;
import org.eclipse.collections.api.block.procedure.primitive.IntIntProcedure;
import org.eclipse.collections.api.tuple.primitive.IntIntPair;

import java.util.ArrayDeque;
import java.util.Comparator;

import static org.eclipse.collections.impl.tuple.primitive.PrimitiveTuples.pair;

/**
 * Optimized QuickSort implementation aligned with academic literature and industry best practices.
 * https://algs4.cs.princeton.edu/23quicksort/
 * https://en.wikipedia.org/wiki/Sorting_algorithm#Comparison_sorts
 */
public enum QuickSort {
    ;

    /**
     * threshold below which reverts to bubblesort
     * TODO tune
     * <p>
     * bubble sort mean = n * n
     * quick sort mean  = 2 * n * log(n)
     */
    public static final int SMALL = 3;
    /**
     * Threshold below which the sort switches to insertion sort.
     * Empirically tuned for better performance.
     */
    private static final int INSERTION_SORT_THRESHOLD = 16;
    /**
     * threshold for median-of-3
     * TODO tune
     */
    private static final int MEDIUM = SMALL * 10;

    /**
     * Sorts the specified range of elements using the specified swapper and comparator using an optimized quicksort.
     *
     * @param from    the index of the first element (inclusive) to be sorted.
     * @param to      the index of the last element (exclusive) to be sorted.
     * @param cmp     the comparator to determine the order of the elements.
     * @param swapper an object that knows how to swap the elements at any two positions.
     */
    public static void quickSort(int from, int to, IntComparator cmp, IntIntProcedure swapper) {
        var stack = new ArrayDeque<IntIntPair>(2);
        stack.push(pair(from, to));

        while (!stack.isEmpty()) {
            var range = stack.pop();
            int left = range.getOne();
            int right = range.getTwo();
            int len = right - left;

            if (len <= 1)
                continue;

            if (len <= INSERTION_SORT_THRESHOLD) {
                insertionSort(left, right, cmp, swapper);
                continue;
            }

            // Pivot selection: median-of-three
            int pivotIndex = medianOfThree(left, left + len / 2, right - 1, cmp, swapper);
            swapper.value(pivotIndex, right - 1); // Move pivot to end

            // Partitioning
            int partition = partition(left, right - 1, cmp, swapper);

            // Push subarrays to stack, process smaller first to ensure O(log n) stack depth
            if (partition - left < right - (partition + 1)) {
                stack.push(pair(partition + 1, right));
                stack.push(pair(left, partition));
            } else {
                stack.push(pair(left, partition));
                stack.push(pair(partition + 1, right));
            }
        }
    }

    private static int partition(int left, int right, IntComparator cmp, IntIntProcedure swapper) {
        int pivot = left;
        int i = left;
        int j = right - 1;

        while (i <= j) {
            while (i <= j && cmp.compare(i, pivot) < 0) {
                i++;
            }
            while (i <= j && cmp.compare(j, pivot) > 0) {
                j--;
            }
            if (i <= j) {
                swapper.value(i, j);
                if (i == pivot) {
                    pivot = j;
                } else if (j == pivot) {
                    pivot = i;
                }
                i++;
                j--;
            }
        }
        return i;
    }

    private static int medianOfThree(int a, int b, int c, IntComparator cmp, IntIntProcedure swapper) {
        if (cmp.compare(a, b) > 0)
            swapper.value(a, b);
        if (cmp.compare(a, c) > 0)
            swapper.value(a, c);
        if (cmp.compare(b, c) > 0)
            swapper.value(b, c);
        return b;
    }

    private static void insertionSort(int from, int to, IntComparator cmp, IntIntProcedure swapper) {
        for (int i = from + 1; i < to; i++) {
            int j = i;
            while (j > from && cmp.compare(j - 1, j) > 0) {
                swapper.value(j - 1, j);
                j--;
            }
        }
    }

//    /**
//     * Sorts the specified range of elements using the specified swapper and according to the order induced by the specified
//     * comparator using quicksort.
//     * <p>
//     * <p>The sorting algorithm is a tuned quicksort adapted from Jon L. Bentley and M. Douglas
//     * McIlroy, &ldquo;Engineering a Sort Function&rdquo;, <i>Software: Practice and Experience</i>, 23(11), pages
//     * 1249&minus;1265, 1993.
//     *
//     * @param from    the index of the first element (inclusive) to be sorted.
//     * @param to      the index of the last element (exclusive) to be sorted.
//     * @param cmp     the comparator to determine the order of the generic data.
//     * @param swapper an object that knows how to swap the elements at any two positions.
//     */
//    public static void quickSort(int from, int to, IntComparator cmp, IntIntProcedure swapper) {
//        int len;
//        while ((len = to - from) > 1) {
//
//            if (len <= SMALL) {
//                bubbleSort(from, to, cmp, swapper);
//                return;
//            }
//
//            int m = mid(from, to, cmp, len);
//
//            int a = from;
//            int b = a;
//            int c = to - 1;
//            int d = c;
//            while (true) {
//                int comparison;
//                while (b <= c && ((comparison = cmp(b, m, cmp)) <= 0)) {
//                    if (comparison == 0) {
//                        if (a == m) m = b;
//                        else if (b == m) m = a;
//                        swapper.value(a++, b);
//                    }
//                    b++;
//                }
//                while (c >= b && ((comparison = cmp(c, m, cmp)) >= 0)) {
//                    if (comparison == 0) {
//                        if (c == m) m = d;
//                        else if (d == m) m = c;
//                        swapper.value(c, d--);
//                    }
//                    c--;
//                }
//                if (b > c) break;
//                if (b == m) m = d;
//                swapper.value(b++, c--);
//            }
//
//
//            vecSwapUntil(swapper, from, b, min(a - from, b - a));
//
//            vecSwapUntil(swapper, b, to, min(d - c, to - d - 1));
//
//            {
//                int s = b - a;
//                if (s > 1)
//                    //TODO push
//                    quickSort(from, from + s, cmp, swapper); //TODO non-recursive
//                //TODO pop
//            }
//
//            {
//                int s = d - c;
//                if (s <= 1)
//                    break; //done //TODO pop , else done
//
//                from = to - s;
//            }
//
//        }
//    }
//
//    private static int cmp(int x, int y, IntComparator cmp) {
//        return x == y ? 0 : cmp.compare(x, y);
//    }
//
//    private static int mid(int from, int to, IntComparator cmp, int len) {
//        int m = from + len / 2;
//        if (len > SMALL) {
//            int l = from;
//            int n = to - 1;
//            if (len > MEDIUM) {
//                int s = len / 8;
//                l = med3(l, l + s, l + 2 * s, cmp);
//                m = med3(m - s, m, m + s, cmp);
//                n = med3(n - 2 * s, n - s, n, cmp);
//            }
//            m = med3(l, m, n, cmp);
//        }
//        return m;
//    }
//
//    private static void bubbleSort(int from, int to, IntComparator cmp, IntIntProcedure swapper) {
//        for (int i = from; i < to; i++)
//            for (int j = i; j > from && cmp.compare(j - 1, j) > 0; j--)
//                swapper.value(j - 1, j);
//    }
//
//    /**
//     * Returns the index of the median of the three indexed chars.
//     */
//    private static int med3(int a, int b, int c, IntComparator cmp) {
//        int ab = cmp.compare(a, b);
//        int ac = cmp.compare(a, c);
//        int bc = cmp.compare(b, c);
//        return (ab < 0 ?
//                (bc < 0 ? b : ac < 0 ? c : a) :
//                (bc > 0 ? b : ac > 0 ? c : a));
//    }
//
//    private static void vecSwapUntil(IntIntProcedure swapper, int from, int to, int s) {
//        int t = to - s;
//        for (; s > 0; s--, from++, t++)
//            swapper.value(from, t);
//    }
//
    /**
     * sorts descending
     */
    public static void sort(int[] a, IntToFloatFunction v) {
        sort(a, 0, a.length, v);
    }

    /**
     * sorts descending
     */
    public static void sort(byte[] a, IntToFloatFunction v) {
        sort(a, 0, a.length, v);
    }

    private static void sort(int[] x, int left, int right, IntToFloatFunction v) {
        quickSort(left, right, (a, b) -> a == b ? 0 :
                        Float.compare(v.valueOf(a), v.valueOf(b)),
                ArrayUtil.swapper(x));
    }

    private static void sort(byte[] x, int left, int right, IntToFloatFunction v) {
        quickSort(left, right, (a, b) -> a == b ? 0 :
                        Float.compare(v.valueOf(a), v.valueOf(b)),
                (a, b) -> ArrayUtil.swapByte(x, a, b));
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public static <X> void sort(X[] x, int left, int right, IntToDoubleFunction v) {
        quickSort(left, right, (a, b) -> a == b ? 0 :
                        Double.compare(v.valueOf(a), v.valueOf(b)),
                ArrayUtil.swapper(x));
    }

    public static <X> void sort(X[] x, int left, int right, Comparator<X> v) {
        quickSort(left, right, (a, b) -> a == b ? 0 :
                        v.compare(x[a], x[b]),
                ArrayUtil.swapper(x));
    }

    /**
     * sorts descending
     */
    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public static <X> void sort(X[] x, int left, int right, FloatFunction<X> v) {
        quickSort(left, right, (a, b) -> a == b ? 0 :
                        Float.compare(v.floatValueOf(x[a]), v.floatValueOf(x[b])),
                ArrayUtil.swapper(x));
    }

    /**
     * modifies order of input array
     */
    public static <X> X[] sort(X[] x, FloatFunction<X> v) {
        sort(x, 0, x.length, v);
        return x;
    }

//    /**
//     * Interface for swapping elements. Extends IntIntProcedure for compatibility.
//     */
//    @FunctionalInterface
//    public interface IntSwapper extends IntIntProcedure {
//        void swap(int a, int b);
//
//        @Override
//        default void value(int a, int b) {
//            swap(a, b);
//        }
//    }
}