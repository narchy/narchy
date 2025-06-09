package jcog.data.iterator;

import com.google.common.collect.Iterators;
import jcog.util.SingletonIterator;

import java.util.Iterator;

import static java.util.Collections.EMPTY_SET;
import static jcog.Util.emptyIterable;
import static jcog.Util.emptyIterator;

/**
 * modified from eclipse collections "CompositeIterator"
 * TODO 3-ary tests?
 */
public final class Concaterator<E> implements Iterator<E> {

	private final Iterator meta;
	private Iterator<E> inner;

	public static Iterator concat(Iterator... ii) {
        return switch (ii.length) {
            case 0 -> emptyIterator;
            case 1 -> ii[0];
            case 2 -> concat2(ii);
            default -> new Concaterator<>((Object[]) ii);
        };
	}

	public static <E> Iterator<E> concat(Object... ii) {
        return switch (ii.length) {
            case 0 -> emptyIterator;
            case 1 -> iterator(ii[0]);
            //case 2 -> concat2(ii); //guava's isn't high-efficiency
            default -> new Concaterator<>(ii);
        };
    }

	private static <E> Iterator<E> concat2(Object[] ii) {
		Object a = ii[0], b = ii[1];
		if (empty(a)) return iterator(b);
		else if (empty(b)) return iterator(a);
		else return Iterators.concat(iterator(a), iterator(b));
	}

	private static boolean empty(Object a) {
		return a == emptyIterator || a == emptyIterable || a == EMPTY_SET;
	}

	private static Iterator concat2(Iterator[] ii) {
		Iterator a = ii[0], b = ii[1];
		if (a == emptyIterator) return b;
		else if (b == emptyIterator) return a;
		else return Iterators.concat(a, b);
	}

	public static <E> Iterator<E> iterator(Object x) {
		if (x instanceof Iterator xx)
			return xx;
		else if (x instanceof Iterable xxx)
			return xxx.iterator();
		else
			return new SingletonIterator<>((E)x);
	}

	private Concaterator(Object... ii) {
		meta = ArrayIterator.iterate(ii);
		inner = null;
	}

	@Override
	public boolean hasNext() {
		while (true) {

			if (inner!=null && this.inner.hasNext())
				return true;

			if (!meta.hasNext()) {
				inner = null;
				return false;
			}

			this.inner = iterator(meta.next());
		}
	}

	@Override
	public E next() {
		return this.inner.next();
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	private static final class ConcatenatedIterator2<T> implements Iterator<T> {
		private Iterator<T> a, b;

		ConcatenatedIterator2(Iterator<T> a, Iterator<T> b) {
			this.a = a;
			this.b = b;
		}

		@Override
		public boolean hasNext() {
			if (a!=null) {
                if (a.hasNext()) return true; else a = null;
			}
			if (a == null && b!=null) {
				if (b.hasNext()) return true; else b = null;
			}
			return false;
		}

		@Override
		public T next() {
            return (a != null ? a : b).next();
		}
	}
}
