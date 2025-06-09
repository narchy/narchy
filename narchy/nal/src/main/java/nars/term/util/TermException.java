package nars.term.util;

import nars.Op;
import nars.Term;
import nars.subterm.Subterms;
import nars.term.Termlike;
import nars.util.SoftException;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;

import static nars.Op.DTERNAL;

/**
 * contains fields for storing target construction parameters
 */
public final class TermException extends SoftException {

    private final byte op;
    private final int dt;
    private final Term[] args;

    public TermException(String reason) {
        this(reason, null, Op.EmptyTermArray);
    }

    public TermException(String reason, Termlike t) {
        this(reason, null, DTERNAL, t);
    }

    public TermException(String reason, Op op, Term... args) {
        this(reason, op, DTERNAL, args);
    }

    public TermException(String reason, Op op, int dt, Termlike args) {
        this(reason, op, dt, args instanceof Subterms s ? s.arrayShared() : new Term[] { (Term) args });
    }

    public TermException(String reason, Op op, int dt, Term... args) {
        super(reason);
        this.op = op!=null ? op.id : -1;
        this.dt = dt;
        this.args = args;
    }

    @Nullable
    private Op op() { return (op == -1 ? null : Op.op(op)); }

    @Override
    public String getMessage() {
        return super.getMessage() + " {" +
                op() +
                ", dt=" + dt +
                ", args=" + Arrays.toString(args) +
                '}';
    }

}