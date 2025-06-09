package jcog.data.set;

import com.google.common.base.Joiner;
import jcog.data.iterator.Concaterator;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.Consumer;

/**
 * TODO make a caching subclass which
 * generates an arraylist copy during the first iteration for fast
 * subsequent iterations. this copy can be linked via SoftReference<>
 */
public class UnenforcedConcatSet<X> extends AbstractSet<X> {

    final Collection<X> a, b;
    final int size;

    public UnenforcedConcatSet(Collection<X> a, Collection<X> b) {
        this.a = a;
        this.b = b;
        this.size = a.size() + b.size();
    }

    @Override
    public void forEach(Consumer<? super X> action) {
        a.forEach(action);
        b.forEach(action);
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean contains(Object o) {
        return a.contains(o) || b.contains(o);
    }

    @Override
    public boolean isEmpty() {
        return size==0;
    }

    @Override
    public Iterator<X> iterator() {
        return Concaterator.concat(a, b);
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean equals(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int hashCode() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return Joiner.on(',').join(iterator());
    }


    /** if a or b are null, they are considered empty sets */
    public static <X> Set<X> concat(@Nullable Set<X> a, @Nullable Set<X> b) {
        boolean aEmpty = a == null || a.isEmpty();
        boolean bEmpty = b == null || b.isEmpty();
        if (bEmpty && aEmpty) return Collections.EMPTY_SET;
        else if (aEmpty) return b;
        else if (bEmpty) return a;
        else
            return new UnenforcedConcatSet<>(a, b);
    }

}
