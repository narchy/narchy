package nars.term.functor;

import nars.eval.Evaluation;
import nars.term.Functor;
import nars.term.atom.Atom;
import nars.term.util.transform.InlineFunctor;

public abstract class AbstractInlineFunctor extends Functor implements InlineFunctor<Evaluation> {


    protected AbstractInlineFunctor(Atom atom) {
        super(atom);
    }

    protected AbstractInlineFunctor(String atom) {
        super(atom);
    }

}
