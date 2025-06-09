package jcog.data.set;

import jcog.util.LambdaStampedLock;

import java.util.Comparator;

public class LockedSortedArraySet<X> extends SortedArraySet<X> {

    public final LambdaStampedLock lock = new LambdaStampedLock();

    public LockedSortedArraySet(Comparator<? super X> comparator, X[] array) {
        super(comparator, array);
    }

    @Override
    public X put(X x) {
        long l = lock.writeLock();
        try {
            return _put(x);
        } finally {
            lock.unlock(l);
        }
    }

    public X _put(X x) {
        return super.put(x);
    }

    @Override
    public X remove(Object x) {
        long l = lock.writeLock();
        try {
            return _remove(x);
        } finally {
            lock.unlock(l);
        }
    }

    @Override
    public X remove(int index) {
        return _remove(index);
    }

    public X _remove(int index) {
        return super.remove(index);
    }

    public X _remove(Object x) {
        return super.remove(x);
    }

    @Override
    public void resize(int cap) {
        lock.write(()-> super.resize(cap));
    }

    @Override
    public void clear() {
        long l = lock.writeLock();
        try {
            super.clear();
        } finally {
            lock.unlock(l);
        }
    }
}
