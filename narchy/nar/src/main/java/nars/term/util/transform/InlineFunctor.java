package nars.term.util.transform;

import nars.Term;
import nars.subterm.Subterms;

import java.util.function.BiFunction;

/** marker interface for functors which are allowed to be applied during
 * transformation or target construction processes.
 * these are good for simple functors that are guaranteed to return quickly.
 */
@FunctionalInterface
public interface InlineFunctor<E> extends BiFunction<E /*Evaluation*/, Subterms, Term> {

    default Term applyInline(Subterms args) {
        return apply(null, args);
    }

}