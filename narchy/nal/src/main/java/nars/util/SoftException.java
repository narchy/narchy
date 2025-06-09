package nars.util;

import com.google.common.base.Joiner;
import nars.NAL;
import org.jetbrains.annotations.Nullable;

/**
 * Created by me on 10/30/16.
 */
public class SoftException extends RuntimeException {

    private static final int depth = 7;

    private @Nullable StackWalker.StackFrame[] stack;

    public SoftException() {
        super();
    }

    public SoftException(String message) {
        super(message);
    }

    @Override
    public final Throwable fillInStackTrace() {
        if (NAL.DEBUG) {
            synchronized(this) {
                stack = StackWalker.getInstance().walk(s -> s
                        .skip(5).limit(depth).toArray(StackWalker.StackFrame[]::new)
                );
            }
        }
        return this;
    }

    @Override
    public String toString() {
        var m = getMessage();
        return (m!=null ? m + '\n' : "") + (stack!=null ? Joiner.on("\n").join(stack) : "");
    }

    @Override
    public void printStackTrace() {
        if (stack!=null) {
            for (var stackFrame : stack)
                System.err.println(stackFrame);
        }
        //super.printStackTrace();
    }
}
