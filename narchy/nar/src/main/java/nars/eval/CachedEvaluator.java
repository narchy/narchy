package nars.eval;

import jcog.memoize.Memoizers;
import nars.term.Compound;
import nars.term.Functor;
import nars.term.atom.Atom;
import org.jetbrains.annotations.Nullable;

import java.util.function.Function;

public non-sealed class CachedEvaluator extends Evaluator {

    private final Function<Compound, EvaluationPhase> memoize;

    public CachedEvaluator(Function<Atom, Functor> funcResolver, int capacity) {
        super(funcResolver);
        memoize = Memoizers.the.memoize(CachedEvaluator.class.getSimpleName(),
                super::compile, capacity);
    }

    @Override
    @Nullable
    EvaluationPhase compile(Compound x) {
        var y = memoize.apply(x);
        // HACK: The caching mechanism (memoize) might use EvaluationPhase.NULL as a sentinel
        // for cached "null" results, as some cache implementations do not support storing null directly.
        // This line translates that sentinel back to a true null, as expected by the compile method's contract.
        return y == EvaluationPhase.NULL ? null : y;
    }
}