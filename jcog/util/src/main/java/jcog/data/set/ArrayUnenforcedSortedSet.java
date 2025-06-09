package jcog.data.set;

import jcog.TODO;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.stream.Stream;

import static com.google.common.collect.Iterators.singletonIterator;

/**
 * use with caution
 */
public abstract class ArrayUnenforcedSortedSet<X> extends ArrayUnenforcedSet<X> implements SortedSet<X> {

    public static final SortedSet empty = new ArrayUnenforcedSortedSet<>() {

        @Override
        public Stream<Object> stream() {
            return Stream.empty();
        }

//        @Override
//        public Object first() {
//            throw new UnsupportedOperationException();
//        }
//
//        @Override
//        public Object last() {
//            throw new UnsupportedOperationException();
//        }
    };

    @SafeVarargs
    private ArrayUnenforcedSortedSet(X... xx) {
        super(xx);
    }
    /**
     * assumes u is already sorted and deduplicated
     */
    public static <X> SortedSet<X> the(X[] u) {
        return switch (u.length) {
            case 0 -> empty;
            case 1 -> the(u[0]);
            case 2 -> new Two(u[0], u[1]);
            default -> new ArrayArrayUnenforcedSortedSet<>(u);
        };
    }

    public static <X> SortedSet<X> the(X x) {
        return new One(x);
    }

    public static <X extends Comparable> SortedSet<X> the(X x, X y) {
        int c = x.compareTo(y);
        return switch (c) {
            case 0 -> new One(x);
            case 1 -> new Two(y, x);
            default -> new Two(x, y);
        };
    }

    @Override
    public boolean add(X x) {
        throw new TODO();
    }

    @Override
    public @Nullable Comparator<? super X> comparator() {
        return null;
    }

    @Override
    public SortedSet<X> subSet(X x, X e1) {
        throw new TODO();
    }

    @Override
    public SortedSet<X> headSet(X x) {
        throw new TODO();
    }

    @Override
    public SortedSet<X> tailSet(X x) {
        throw new TODO();
    }

    //    private static class One<X> extends AbstractSet<X> implements SortedSet<X> {
//
//        private One(X x) {
//            super(x);
//        }
//
//        @Override
//        public int size() {
//            return 1;
//        }
//
//        @Override
//        public Stream<X> stream() {
//            return Stream.of(this.first());
//        }
//
//    }
    public static class One<X> extends AbstractSet<X> implements SortedSet<X> {

        public final X element;

        One(X x) {
            element = x;
        }

        public Iterator<X> iterator() {
            return singletonIterator(element);
        }

        @Override
        public boolean add(X x) {
            return false;
        }

        @Override
        public boolean addAll(Collection<? extends X> c) {
            return false;
        }

        public int size() {
            return 1;
        }

        public boolean contains(Object o) {
            return element.equals(o);
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(element);
        }

        @Override
        public Stream<X> stream() {
            return Stream.of(element);
        }

        @Override
        public @Nullable Comparator<? super X> comparator() {
            throw new UnsupportedOperationException();
        }

        @Override
        public SortedSet<X> subSet(X fromElement, X toElement) {
            throw new UnsupportedOperationException();
        }

        @Override
        public SortedSet<X> headSet(X toElement) {
            throw new UnsupportedOperationException();
        }

        @Override
        public SortedSet<X> tailSet(X fromElement) {
            throw new UnsupportedOperationException();
        }

        @Override
        public X first() {
            return element;
        }

        @Override
        public X last() {
            return element;
        }
    }

//    private static class One<X> extends ArrayUnenforcedSortedSet<X> {
//
//        private One(X x) {
//            super(x);
//        }
//
//        @Override
//        public int size() {
//            return 1;
//        }
//
//        @Override
//        public Stream<X> stream() {
//            return Stream.of(this.first());
//        }
//
//    }

    private static class Two<X> extends ArrayUnenforcedSortedSet<X> {

        private Two(X x, X y) {
            super(x, y);
        }

        @Override
        public int size() {
            return 2;
        }

        @Override
        public Stream<X> stream() {
            return Stream.of(first(), last());
        }

    }


    @Deprecated
    private static final class ArrayArrayUnenforcedSortedSet<X> extends ArrayUnenforcedSortedSet<X> {

        ArrayArrayUnenforcedSortedSet(X[] u) {
            super(u);
        }

    }
}