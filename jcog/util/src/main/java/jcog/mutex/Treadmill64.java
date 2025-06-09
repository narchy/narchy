package jcog.mutex;

import jcog.Util;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLongArray;

/** striping via 64bit (pair of 32bit codes) global exclusion locking via busy spin
 *  on a linear probed atomic array of fixed size */
public final class Treadmill64  {

    private final AtomicBoolean writing = new AtomicBoolean(false);
    private final int size, offset;

    /** extra space for additional usage */
    public Treadmill64(int size, int offset) {
        this.size = size;
        this.offset = offset;
    }

    public int start(long hash, AtomicLongArray buf) {
        if (hash == 0) hash = 1; //reserve 0 value

        int end = size + offset;

        int cycles = 0;
        while (true) {

            /* optimistic pre-scan determined free slot */
            int maybeFree = -1;

            for (int i = offset; i < end; i++) {
                long v = buf.getAcquire(i);
                if (v == hash) {
                    //collision
                    maybeFree = -1; //reset incase it's found an empty
                    //continue below
                    break;
                } else if (v == 0 && maybeFree==-1) {
                    //first empty cell candidate
                    maybeFree = i;
                    //continue to detect hash in subsequent cells
                }
            }

            if (maybeFree != -1) {
                if (Util.enterAlone(writing)) {
                    try {
                        if (available(buf, maybeFree, hash))
                            return maybeFree;
                        for (int j = offset; j < end; j++)
                            if (j != maybeFree && available(buf, j, hash))
                                return j;
                    } finally {
                        Util.exitAlone(writing);
                    }
                }
            }

            Util.pauseSpin(cycles++);
        }
    }

    private static boolean available(AtomicLongArray buf, int i, long hash) {
        return buf.compareAndExchangeRelease(i, 0, hash) == 0;
    }

}



