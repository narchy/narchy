package com.github.fge.grappa.stack;

import com.google.common.annotations.VisibleForTesting;
import jcog.data.iterator.ArrayIterator;
import jcog.util.ArrayUtil;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Iterator;

/**
 * A {@link ValueStack} implementation using arrays
 *
 * <p>This is the default implementation currently used.</p>
 *
 * @param <X> type parameter of the stack's element
 */
public final class ArrayValueStack<X> implements ValueStack<X> {

    @VisibleForTesting
    static final int SIZE_INCREASE = 8;
    @VisibleForTesting
    private static final String SWAP_BADARG = "argument to swap(...) must be >= 2";

    private int arraySize;
    private X[] array = (X[]) ArrayUtil.EMPTY_OBJECT_ARRAY;

    private void doPush(int down, X value) {
        ensureCapacity();
        System.arraycopy(array, down, array, down + 1, arraySize - down);
        array[down] = value;
        arraySize++;
    }

    private X doPop(int down) {
        X ret = array[down];
        arraySize--;
        System.arraycopy(array, down + 1, array, down, arraySize - down);
        array[arraySize] = null;
        shrinkIfNecessary();
        return ret;
    }

    private X doPeek(int down) {
        return array[down];
    }

    private void doPoke(int down, X value) {
        array[down] = value;
    }

    private void doDup() {
        ensureCapacity();
        System.arraycopy(array, 0, array, 1, arraySize);
        arraySize++;
    }

    private void doSwap(int n) {

        int swapIndex = n / 2; // this also works for odd numbers

        for (int index = 0; index < swapIndex; index++) {
            X tmp = array[index];
            array[index] = array[n - index - 1];
            array[n - index - 1] = tmp;
        }
    }

    @Override
    public int size() {
        return arraySize;
    }

    @Override
    public void clear() {
        arraySize = 0;
        array = (X[]) ArrayUtil.EMPTY_OBJECT_ARRAY;
    }

    @Override
    public Object takeSnapshot() {
        int s = this.arraySize;
        if (s == 0) return null;
        X[] copy = array.length > 0  ? Arrays.copyOf(array, s) : (X[]) ArrayUtil.EMPTY_OBJECT_ARRAY;
        return new ArrayWithSize<>(copy, s);
    }

    @Override
    public void restoreSnapshot(@Nullable Object snapshot) {
        if (snapshot == null) {
            arraySize = 0;
            array = (X[])ArrayUtil.EMPTY_OBJECT_ARRAY;
        } else {
            ArrayWithSize<X> s = (ArrayWithSize<X>) snapshot;
            array = s.array;
            arraySize = s.arraySize;
        }
    }

    @Override
    public Iterator<X> iterator() {
        return ArrayIterator.iterateN(array, arraySize);
    }

    private void ensureCapacity() {
        if (arraySize == array.length)
            array = Arrays.copyOf(array, arraySize + SIZE_INCREASE);
    }

    private void shrinkIfNecessary() {
    }

    @Override
    public final boolean isEmpty()
    {
        return size() == 0;
    }

    @Override
    public final void push(X value)
    {
        push(0, value);
    }

    @Override
    public final void push(int down, X value)
    {
        /*
         * It is legal to append at the end! We must therefore check that the
         * index - 1 is strictly less than size, not the index itself
         */
//        if (down < 0)
//            throw new IllegalArgumentException(NEGATIVE_INDEX);
//        Objects.requireNonNull(value);
        doPush(down, value);
    }

    //
        @Override
        public final X pop()
        {
            return pop(0);
        }

    //
        @Override
        public final X pop(int down)
        {
    //        if (down < 0)
    //            throw new IllegalArgumentException(NEGATIVE_INDEX);
            return doPop(down);
        }

    @Override
    public final <T extends X> T popAs(Class<T> type)
    {
        return type.cast(pop(0));
    }

    @Override
    public final <T extends X> T popAs(Class<T> type, int down)
    {
        return type.cast(pop(down));
    }

    @Override
    public final X peek()
    {
        return peek(0);
    }

    @Override
    public final X peek(int down)
    {
//        if (down < 0)
//            throw new IllegalArgumentException(NEGATIVE_INDEX);
        return doPeek(down);
    }

    @Override
    public final <T extends X> T peekAs(Class<T> type)
    {
        return type.cast(peek(0));
    }

    @Override
    public final <T extends X> T peekAs(Class<T> type, int down)
    {
        return type.cast(peek(down));
    }

    @Override
    public final void poke(X value)
    {
        poke(0, value);
    }

    @Override
    public final void poke(int down, X value)
    {
//        if (down < 0)
//            throw new IllegalArgumentException(NEGATIVE_INDEX);
//        Objects.requireNonNull(value);
        doPoke(down, value);
    }

    @Override
    public final void swap(int n)
    {
        if (n < 2)
            throw new IllegalArgumentException(SWAP_BADARG);
        /*
         * As for .push(n, value), we need to check for n - 1 here
         */
        doSwap(n);
    }

    @Override
    public final void swap()
    {
        swap(2);
    }

    /**
     * Duplicates the top value. Equivalent to push(peek()).
     */
    @Override
    public final void dup() {
        doDup();
    }
//    private void shrinkIfNecessary()
//    {
//        final int length = array.length;
//        final int lengthSizeDiff = length - arraySize;
//        if (lengthSizeDiff >= SIZE_INCREASE)
//            array = Arrays.copyOf(array, length - SIZE_INCREASE);
//    }

    private static final class ArrayWithSize<T>
    {
        private final T[] array;
        private final int arraySize;

        private ArrayWithSize(T[] array, int size)
        {
            this.array = array;
            arraySize = size;
        }
    }

//    private static final class ArrayIterator<T>
//        implements Iterator<T>
//    {
//        private final T[] array;
//        private final int arraySize;
//
//        private int index = 0;
//
//        private ArrayIterator(T[] array, int size)
//        {
//            this.array = array;
//            arraySize = size;
//        }
//
//        @Override
//        public boolean hasNext()
//        {
//            return index < arraySize;
//        }
//
//        @Override
//        public T next()
//        {
//            if (!hasNext())
//                throw new NoSuchElementException();
//            return array[index++];
//        }
//
//        @Override
//        public void remove()
//        {
//            throw new UnsupportedOperationException();
//        }
//    }
}