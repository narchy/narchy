package jcog.sort;

import jcog.util.LambdaStampedLock;
import org.eclipse.collections.api.block.function.primitive.FloatFunction;

import java.util.function.Consumer;

public abstract class LockedFloatSortedArray<X> extends FloatSortedArray<X> implements FloatFunction<X> {

    protected final LambdaStampedLock lock = new LambdaStampedLock();


    /**
     * direct insert; not ordinarily invoked from external
     */
    protected void insert(X incoming) {
        int r = add(incoming, this);
        assert (r != -1);
    }


    @Override
    public final void forEach(Consumer<? super X> each) {
        lock.read(() -> {
            int s = size;
            X[] xx = items;
            for (int i = 0; i < s; i++)
                each.accept(xx[i]);
        });
    }

    @Override
    public final void clear() {

        long l = lock.writeLock();
        try {
            if (size > 0) {
                //forEach(ScalarValue::delete);
                super.delete();
            }
        } finally {
            lock.unlock(l);
        }

    }

    public X remove(X x) {
        //TODO use optimistic read here
        long l = lock.writeLock();
        try {

            int index = indexOf(x, this);

            if (index != -1) {
                X[] items = this.items;
                X xx = items[index];

                if (items[index] != xx) //moved while waiting for lock, retry:
                    index = indexOf(x, this);

                if (index != -1) {
                    boolean wasRemoved = removeFast(xx, index);
                    assert (wasRemoved);
                    return xx;
                }
            }
        } finally {
            lock.unlock(l);
        }

        return null;
    }
}
