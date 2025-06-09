package nars.action.link.index;

import jcog.pri.PriReference;
import nars.Term;
import nars.derive.Deriver;
import org.jetbrains.annotations.Nullable;

import java.util.Iterator;

/**
 * implementations resolve adjacent concepts to a concept in a context by a particular strategy
 */
public interface AdjacentTerms {

	@Nullable Iterator<PriReference<Term>> adjacencies(Term from, Term to, Deriver d);

}