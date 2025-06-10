package nars.table.eternal;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.sort.LockedFloatSortedArray;
import jcog.util.ArrayUtil;
import nars.*;
import nars.action.memory.Remember;
import nars.task.SerialTask;
import nars.truth.Stamp;
import nars.truth.proj.IntegralTruthProjection;
import nars.utils.Profiler;
import nars.truth.proj.MutableTruthProjection;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;

import static nars.BeliefTable.eternalOriginality;


/**
 * Created by me on 5/7/16.
 */
public class EternalTable extends LockedFloatSortedArray<NALTask> implements BeliefTable {

    private final Map<NALTask, NALTask> taskMap = new HashMap<>();

    /** initializes with zero capacity. */
    public EternalTable() {
        super();
        items = NALTask.EmptyNALTaskArray; //HACK
    }

    @Override
    public final void forEachTask(long minT, long maxT, Consumer<? super NALTask> x) {
        forEachTask(x);
    }

    @Override
    public final void forEachTask(Consumer<? super NALTask> x) {
        forEach(x);
    }

    @Override
    public final int taskCount() {
        return size();
    }

    @Override
    public final Stream<? extends NALTask> taskStream() {
        return stream();
    }

    @Override
    public void match(Answer a) {
        if (size == 0)
            return; //empty
        long r = lock.readLock();
        try {
            //lock.read(() -> whileEach(a));
            whileEach(a);
        } finally {
            lock.unlockRead(r);
        }
    }

    @Override
    public void taskCapacity(int c) {
        assert (c >= 0);

        Lst<Task> trash = null;

        long l = lock.writeLock();
        try {
            if (this.capacity() == c)
                return;

            int excess = size - c;
            if (excess > 0)
                trash = compress(c, excess, trash);

            capacity(c);
        } finally {
            lock.unlock(l);
        }

        if (trash != null)
            trash.forEach(Task::delete);
    }

    private Lst<Task> compress(int c, int excess, Lst<Task> trash) {
        for (int e = 0; e < excess; e++) {
            if (!compressFast(null))
                break; //stuck
        }

        excess = size - c;
        if (excess > 0) {
            //evict
            trash = new Lst<>(excess);
            do {
                trash.addFast(removeLast());
            } while (--excess > 0);
        }
        return trash;
    }

    @Override
    public NALTask[] taskArray() {

        //long l = lock.readLock();
//        try {
        int s = this.size;
        if (s == 0)
            return NALTask.EmptyNALTaskArray;
        else {
            NALTask[] list = this.items;
            return Arrays.copyOf(list, Math.min(s, list.length));
        }
//        } finally {
//            lock.unlock(l);
//        }

    }

    /**
     * for ranking purposes.  returns negative for descending order
     */
    @Override
    public final float floatValueOf(NALTask w) {
        return (float) -eternalOriginality(w);
    }

    /** weakest to weakest, first valid is taken */
    private boolean compressFast(@Nullable Remember r) {
        Profiler.startTime("EternalTable.compressFast");
        try {
            int s = size();

            NAR nar = r != null ? r.nar() : null;
            EternalReviser ar = new EternalReviser(nar != null ? nar.timeRes() : 1);

            for (int a = s - 1; a >= 1; a--) {
                NALTask A = get(a);
                ar.set(A);
                for (int b = a - 1; b >= 0; b--) {
                    NALTask B = get(b);
                    NALTask AB = ar.apply(B);
                    if (AB != null) {
                        //assert(a > b); //remove the higher index, A, first and B remains at same index
                        removeFast(a);
                        removeFast(b);
                        if (!contains(AB)) {
                            insert(AB);
//                        if (r!=null)
//                            r.focus.activate(new Remember(AB, r.concept));
                        }
                        return true;
                    }
                }
            }
            return false;
        } finally {
            Profiler.recordTime("EternalTable.compressFast");
        }
    }

    @Override
    public boolean insert(NALTask x) {
        long l = lock.writeLock();
        try {
            if (super.insert(x)) {
                taskMap.put(x, x);
                return true;
            }
            return false;
        } finally {
            lock.unlockWrite(l);
        }
    }

    @Override
    protected NALTask removeLast() {
        long l = lock.writeLock();
        try {
            NALTask removed = super.removeLast();
            if (removed != null) {
                taskMap.remove(removed);
            }
            return removed;
        } finally {
            lock.unlockWrite(l);
        }
    }


    @Override
    protected boolean removeFast(int index) {
        long l = lock.writeLock();
        try {
            NALTask item = get(index); // Get item before removing from array
            if (super.removeFast(index)) {
                if (item != null) {
                    taskMap.remove(item);
                }
                return true;
            }
            return false;
        } finally {
            lock.unlockWrite(l);
        }
    }


    @Override
    public void clear() {
        long l = lock.writeLock();
        try {
            super.clear();
            taskMap.clear();
        } finally {
            lock.unlockWrite(l);
        }
    }

    @Override
    public void delete() {
        long l = lock.writeLock();
        try {
            super.delete();
            taskMap.clear();
        } finally {
            lock.unlockWrite(l);
        }
    }

    private boolean old_compressFast(@Nullable Remember r) {
        int s = size();

        NAR nar = r != null ? r.nar() : null;
        EternalReviser ar = new EternalReviser(nar != null ? nar.timeRes() : 1);

        for (int a = s - 1; a >= 1; a--) {
            NALTask A = get(a);
            ar.set(A);
            for (int b = a - 1; b >= 0; b--) {
                NALTask B = get(b);
                NALTask AB = ar.apply(B);
                if (AB != null) {
                    //assert(a > b); //remove the higher index, A, first and B remains at same index
                    removeFast(a); // Already handles taskMap removal
                    removeFast(b); // Already handles taskMap removal
                    if (!contains(AB)) { // contains might need to check taskMap too, or be consistent with super
                        insert(AB); // Already handles taskMap addition
//                        if (r!=null)
//                            r.focus.activate(new Remember(AB, r.concept));
                    }
                    return true;
                }
            }
        }
        return false;
    }

    public final Truth truth() {
        NALTask s = first();
        return s != null ? s.truth() : null;
    }

    @Override
    public boolean remove(NALTask x, boolean delete) {

        if (cantContain(x))
            return false;

        // lock is handled by super.remove and taskMap removal is handled by overridden remove(NALTask)
        NALTask removedTask = null;
        long rl = lock.writeLock(); // remove(x) in superclass needs write lock potentially
        try {
            removedTask = super.remove(x); // Call super.remove to avoid recursive call to this method
            if (removedTask != null) {
                taskMap.remove(removedTask);
            }
        } finally {
            lock.unlockWrite(rl);
        }


        if (removedTask != null) {
            if (delete)
                removedTask.delete();
            return true;
        } else {
            return false;
        }
    }


    @Override
    public final void remember(Remember r) {
        NALTask x = r.input;
        if (cantContain(x))
            return;

        long l = lock.readLock();
        try {
            // First, try to look up the task in taskMap
            NALTask existingTask = taskMap.get(x);
            if (existingTask != null) {
                r.store(existingTask);
                return;
            }

            //If not in map, proceed with original logic (linear scan, compress, insert)
            //The linear scan is now a fallback or for verification if map gets out of sync (though it shouldn't)
            int s = size;
            if (s > 0) {
                NALTask[] ii = items;
                // This indexOf is less critical now but kept for robustness / if map ever fails
                int existingInArray = ArrayUtil.indexOf(ii, x, 0, Math.min(s, ii.length));
                if (existingInArray != -1) {
                    // Should have been found in taskMap. If not, map is inconsistent.
                    // For now, trust the array result and update map.
                    NALTask taskFromArray = ii[existingInArray];
                    if (taskMap.put(taskFromArray, taskFromArray) == null) {
                        // Log inconsistency if needed: task in array but not map
                    }
                    r.store(taskFromArray);
                    return;
                }
            }

            int c = capacity();
            if (s >= c) {
                l = Util.readToWrite(l, lock);

                if (c == 0) {
                    if (s > 0)
                        delete(); // calls super.delete() and taskMap.clear() via override
                    return;
                }

                if (s > 1)
                    old_compressFast(r); // Call the original compressFast logic, now renamed
                int excess = 1 + size() - c;
                if (excess > 0) {
                    //compression failed; evict
                    NALTask weakestPresent = last();
                    if (weakestPresent != null) {
                        if (!stronger(x, weakestPresent)) {
                            //rejected
                            r.unstore(x);
                            if (--excess <= 0)
                                return;
                        } else {
                            evict(excess, r);
                            //continue...
                        }
                    }
                }
            }

            l = Util.readToWrite(l, lock); // Ensure we have a write lock for insertion

            if (capacity() > 0) { //check again in case it got deleted while waiting for lock
                // insert(x) already handles taskMap.put(x,x) and is synchronized
                if (insert(x)) { // if insert succeeded
                    r.store(x);
                } else {
                    // Task might already be there due to a race condition resolved by insert's internal check,
                    // or insert failed for other reasons. If it's already there, try to get it from map.
                    NALTask current = taskMap.get(x);
                    if (current != null) {
                        r.store(current);
                    } else {
                        // Insert failed and not in map, treat as unstore
                        r.unstore(x);
                    }
                }
            } else {
                 r.unstore(x); // No capacity
            }


        } finally {
            if (l != 0)
                lock.unlock(l);
        }

    }

    private void evict(int n, Remember r) {
        for (int i = 0; i < n; i++)
            r.unstore(removeLast());
    }

    private static final class EternalReviser implements Function<NALTask,NALTask> {

        long[] xStamp;
        private NALTask x;

        MutableTruthProjection l;
        final int dtDither;

        /** ditherDT used in intermpolation */
        EternalReviser(int dtDither) {
            this.dtDither = dtDither;
        }

        void set(NALTask x) {
            this.x = x;
            this.xStamp = x.stamp();
        }

        @Override
        public NALTask apply(NALTask y) {
            return Stamp.overlapsAny(xStamp, y.stamp()) ? null : projection(y);
        }

        private NALTask projection(NALTask y) {
            if (l == null) {
                l = new IntegralTruthProjection(2);
                l.add(x);
            }

            //l.removeIf(z -> z != x); //remove all but 'x'
            l.add(y);
            return l.task();
        }

    }

    private static boolean stronger(NALTask a, NALTask b) {
        return eternalOriginality(b) <= eternalOriginality(a);
    }

    private static boolean cantContain(NALTask input) {
        return input instanceof SerialTask || !input.ETERNAL();
    }


}