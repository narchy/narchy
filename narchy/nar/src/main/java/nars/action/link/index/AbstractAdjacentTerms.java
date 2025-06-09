package nars.action.link.index;

import nars.Term;
import nars.premise.Premise;

import java.util.function.Function;

/** TODO decaying hidden markov (Term.bytes()) implementation */
public abstract class AbstractAdjacentTerms implements AdjacentTerms {

    abstract protected Function<Premise, Term> test(Term from, Term to);

}