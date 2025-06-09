package jcog.util;

import java.util.concurrent.locks.StampedLock;
import java.util.function.BooleanSupplier;
import java.util.function.IntSupplier;
import java.util.function.Supplier;

/**
 * https://www.javaspecialists.eu/archive/Issue215.html
 */
public class LambdaStampedLock extends StampedLock {

    public void write(Runnable writeProcedure) {
        long stamp = writeLock();
        try {
            writeProcedure.run();
        } finally {
            unlockWrite(stamp);
        }
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public <T> T write(Supplier<T> writeProcedure) {
        long stamp = writeLock();

        try {
            return writeProcedure.get();
        } finally {
            unlockWrite(stamp);
        }

    }


    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public boolean write(BooleanSupplier writeProcedure) {
        long stamp = writeLock();

        try {
            return writeProcedure.getAsBoolean();
        } finally {
            unlockWrite(stamp);
        }

    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public int write(IntSupplier writeProcedure) {
        long stamp = writeLock();
        try {
            return writeProcedure.getAsInt();
        } finally {
            unlockWrite(stamp);
        }
    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public <T> T read(Supplier<T> readProcedure) {
        long stamp = readLock();

        try {
            return readProcedure.get();
        } finally {
            unlockRead(stamp);
        }

    }

    @SuppressWarnings("LambdaUnfriendlyMethodOverload")
    public int read(IntSupplier readProcedure) {
        int result;
        long stamp = readLock();
        try {
            result = readProcedure.getAsInt();
        } finally {
            unlockRead(stamp);
        }
        return result;
    }
    public boolean read(BooleanSupplier readProcedure) {
        long stamp = readLock();
        boolean result;
        try {
            result = readProcedure.getAsBoolean();
        } finally {
            unlockRead(stamp);
        }
        return result;
    }
    public void read(Runnable readProcedure) {
        long stamp = readLock();
        try {
            readProcedure.run();
        } finally {
            unlockRead(stamp);
        }
    }

    public void readOptimistic(Runnable readProcedure) {
        long stamp = tryOptimisticRead();

        if (stamp != 0) {
            readProcedure.run();
            if (validate(stamp))
                return;
        }

        read(readProcedure);
    }

    public boolean readAndMaybeWrite(BooleanSupplier conditionRead, Runnable actionWrite) {
        long stamp = readLock();
        try {
            if (conditionRead.getAsBoolean()) {
                long writeStamp = tryConvertToWriteLock(stamp);
                if (writeStamp != 0) {
                    stamp = writeStamp;
                } else {
                    unlockRead(stamp);
                    stamp = writeLock();
                }
                actionWrite.run();
                return true;
            }
            return false;
        } finally {
            unlock(stamp);
        }
    }
}