package nars.eval;

import nars.NAL;
import nars.NAR;
import nars.Term;
import nars.concept.TaskConcept;
import nars.table.TaskTable;
import nars.unify.Unify;
import nars.unify.UnifyAny;

import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Stream;

import static nars.Op.IMPL;

/**
 * adapter for reifying NARS beliefs (above a certain confidence threshold) as
 * target-level facts for use during evaluation
 */
public record Facts(NAR nar, float expMin, boolean beliefsOrGoals) implements Function<Term, Stream<Term>> {

    @Override
    public Stream<Term> apply(Term x) {


        //TODO filter or handle temporal terms appropriately
        /* stages
            1. resolve the target itself
            2. check its termlinks, and local neighborhood graph termlinks
            3. exhaustive concept index scan
        */
        int xo = x.opID();
        Unify u = new UnifyAny(nar.random());

        return
                Stream.concat(
                        Stream.of(nar.concept(x)).filter(Objects::nonNull), //Stage 1
                        //TODO Stage 2
                        nar.concepts() //Stage 3
                )
                        .filter(c -> {
                            if (!(c instanceof TaskConcept))
                                return false;

                            Term yt = c.term();
                            int yo = yt.opID();
                            if (beliefsOrGoals && yo == IMPL.id) {
                                Term head = yt.sub(1);
                                return head.opID() == xo && u.clear(NAL.derive.TTL_UNISUBST).unify(head, x);
                            }

                            //TODO prefilter
                            //TODO match implication predicate, store the association in a transition graph
                            return xo == yo && u.clear(NAL.derive.TTL_UNISUBST).unify(x, yt);


                        })
                        .map(c -> {


                            TaskTable table = beliefsOrGoals ? c.beliefs() : c.goals();
                            if (table.isEmpty())
                                return null;

                            boolean t = polarized(table, true), f = polarized(table, false);
                            if (t == f)
                                return null;

                            Term ct = c.term();
                            /*if (!t && f)*/
                            return t ? ct : ct.neg();

                        }).filter(Objects::nonNull)
                ;
    }

    private boolean polarized(TaskTable table, boolean trueOrFalse) {
        return table.taskStream().anyMatch(
    t -> exp( (trueOrFalse ? t.expectation() : 1 - t.expectation())));
    }

    /**
     * whether to accept the given expectation
     */
    private boolean exp(double exp) {
        //TODO handle negative expectation
        return exp >= this.expMin;
    }


}