package nars;

import jcog.util.ArrayUtil;
import nars.action.decompose.*;
import nars.action.link.STMLinker;
import nars.action.link.TermLinking;
import nars.action.link.index.BagAdjacentTerms;
import nars.action.resolve.BeliefResolve;
import nars.action.transform.*;
import nars.derive.reaction.ReactionModel;
import nars.derive.reaction.Reactions;
import nars.focus.time.FocusTiming;
import nars.focus.time.TaskWhen;
import nars.premise.NALPremise;

import static nars.NAL.temporal.*;
import static nars.Op.*;
import static nars.action.resolve.Answerer.AnyTaskResolver;

/**
 * utility class and recipes for building Deriver's
 *
 * half of it is patrick's original meta-NAL, which is whats in the .nal files.  the other half of the 'rules' aka reactions are natively coded.  both share a common predicate system that gets compiled into a tree of folded common preconditions for maximum elimination.
 *
 * these rules are triggered by matching properties against the current Premise in a Deriver.  the premise consists of, at minimum, a Term from() and optionally Term to(), NALTask fromTask(), NALTask toTask().  this allows all varieties of single and double premises.
 *
 * TaskLinks are the elemental Premise.  it derives single and double premises through a dynamic task lookup process that can be biased in different ways, for example, occurrence time (start.. end range).
 *
 * the various premise implementations can cache (memoize) data in them to accelerate if they are called again.
 *
 * the rates of generating the different 'daughter-product' premises in this chain reaction are individually controllable but seems that even if this is flat it can work with the careful choices of rules.
 *
 * in terms of forming new links in the graph search process, compound decomposition is a spiralling inwards.  and termlinking is a spiral outwards.  the priority rates of each of these can be seperately controlled too for different mental dynamics.
 */
public class Derivers extends Reactions {

    private final TaskWhen when =
        new FocusTiming();
        //new MultiFocusTiming();
        //new JitterTiming(when),

    private static final boolean implTwoStep = false;

    public static final boolean beliefResolveAuto = false;

    final BeliefResolve beliefResolver = new BeliefResolve(
            true, true, true, true,
            when,
            AnyTaskResolver
    );

    public Derivers core() {
        return core(true);
    }

    /**
     * standard derivation behaviors
     */
    public Derivers core(boolean varIntro) {

        //add(new TaskResolve(when, AnyTaskResolver));

        //if (!beliefResolveAuto)
        add(beliefResolver);

        decomposers();

        termlinking();

        if (varIntro)
            varIntro();

        add(new Evaluate());

//        addAll(
//                new AnswerQuestionsFromBeliefTable(
//                        new FocusTiming(),
//                        true, true, AnyTaskResolver),
//                new AnswerQuestionsFromConcepts.AnswerQuestionsFromTaskLinks(when)
//                        //.log(true)
//        );

        return this;
    }

    public Derivers temporalInduction() {
        temporalInductionImpl(TEMPORAL_INDUCTION_IMPL_SUBJ_PN, TEMPORAL_INDUCTION_IMPL_BIDI);
        temporalInductionConj(false, TEMPORAL_INDUCTION_DISJ);
        return this;
    }

    /** standard ruleset */
    @Deprecated public static Derivers nal(int minLevel, int maxLevel, String... extraFiles) {
        Derivers r = new Derivers();

        for (int level = minLevel; level <= maxLevel; level++) {
            switch (level) {
                case 1 -> r.structural();
                case 2 -> r.sets();
                case 3,5,6 -> {
                    r.procedural();
                    r.analogy();
                }
            }
        }

        r.files(extraFiles);

        return r;
    }

    /** TODO ensure called only once */
    public Derivers structural() {
        files(
           "inh.nal"
            //,"inh.goal.nal"
            ,"sim.nal"
        );

        return this;
    }

    /** optional */
    public Derivers images() {

        add(new ImageUnfold(true));

        add(new ImageAlign.ImageAlignBidi());

			//new ImageAlign.ImageAlignUni_Root()
//			new ImageAlign.ImageAlignUni_InCompound(true),
//			new ImageAlign.ImageAlignUni_InCompound(false),

        return this;
    }

    /** same */
    public Derivers diff() {
        files(
                "diff.nal"
            , "diff.goal.nal" //TODO needs updated if comparator model is used
        );
        return this;
    }

    public Derivers sets() {
        files("set.compose.nal",
                "set.decompose.nal",
                "set.guess.nal");
        return this;
    }

    public Derivers analogy() {
        files(
          "analogy.anonymous.conj.nal"
              , "analogy.anonymous.impl.nal"
              , "analogy.mutate.nal"
                //,"analogy.goal.nal"
        );
        return this;
    }
    public Derivers procedural() {

        files(
                "cond.decompose.nal",

                /* X!,  C.  |- before(C,X)! */        "cond.decompose.must.nal",
                // /* X!,  C.  |- afterOrDuring(C,X)! */ "cond.decompose.might.nal",

                /* C!,  X.  |- afterOrDuring(C,X)! */ "cond.decompose.would.nal",

                /* C/!, X.  |- before(C,X)!  */       "cond.decompose.wouldve.nal",

                // /* C!,--X0. |- X0! */                 "cond.decompose.should.nal",

                // /* C!,X0 |- X0! */ "cond.decompose.start.nal",

                //"nal3.question.nal",

                "contraposition.nal",
                "conversion.nal",

                "impl.syl.nal",
                "impl.syl.cond.nal",
                "impl.syl.combine.nal",

                "impl.strong.nal",
                "impl.strong.cond.nal",

                "impl.compose.nal"
                //,"impl.compose.specific.nal"

              , "impl.decompose.self.nal"
              //, "impl.decompose.subcond.nal"
              , "impl.decompose.specific.nal"

              , "impl.recompose.nal"

              //, "impl.decompose.or.nal"

              //,"impl.decompose.inner.nal"
              //,"impl.decompose.inner.question.nal",


                //,"quest_induction.nal"
                //"equal.nal"
                //"xor.nal"
                //"impl.disj.nal"
                //"nal6.layer2.nal"
                //"nal6.mutex.nal"
        );
        return this;
    }

    private void varIntro() {
        add(
            new VariableIntroduction()./*anon().*/taskPunc(true,
                NAL.derive.VARIABLE_INTRODUCE_GOALS,
                NAL.derive.VARIABLE_INTRODUCE_QUESTIONS,
                NAL.derive.VARIABLE_INTRODUCE_QUESTS)
        );
    }

    private void termlinking() {
        add(
            new TermLinking(new BagAdjacentTerms())
        );
    }

    public Derivers stm() {
        add(new STMLinker(true, true, true, true));
        return this;
    }

    private void temporalInductionConj(boolean full, boolean disj) {
        if (!full) {
            //auto-negate
            add(new TemporalInduction.ConjInduction(0, 0));

            if (disj)
                add(new TemporalInduction.DisjInduction(0, 0));
        } else {
            //disj = ignored

            addAll(
                  new TemporalInduction.ConjInduction(+1, +1)
                , new TemporalInduction.ConjInduction(-1, +1)
                , new TemporalInduction.ConjInduction(+1, -1)
                , new TemporalInduction.ConjInduction(-1, -1)
            );

//					.iff(  TheTask, TermMatcher.ConjSequence.the, false)
//					.iff(TheBelief, TermMatcher.ConjSequence.the, false)
        }
    }

    private void temporalInductionImpl(boolean implBothSubjs, boolean bidi) {
        int implDir = bidi ? 0 : +1;
        if (implBothSubjs) {
            addAll(
				new TemporalInduction.ImplInduction(+1, implDir),
				new TemporalInduction.ImplInduction(-1, implDir)
            );
        } else {
            add(
                new TemporalInduction.ImplInduction(2 /* AUTO-stochastic */, implDir)
                //new TemporalInduction.ImplTemporalInduction(0 /* AUTO */, implDir)
            );
        }
    }


    private void decomposers() {

        /* may be necessary to be true for certain delta.goal.nal rules */
        boolean deltaDecompose =
                true;
                //false;

        var special = new Op[] { INH, SIM, IMPL, CONJ };

        addAll(
            //default compound decomposition
            new Decompose1().taskIsNot(
                deltaDecompose ? special : ArrayUtil.add(special, DELTA)
            ),

            new DecomposeCond().bidi(AnyTaskResolver, when),

            new DecomposeStatement(INH, SIM)
        );

        if (implTwoStep) {
            addAll(
                //2-step IMPL decompose (progressive - less combinatorial explosion)
                new DecomposeImpl(false).bidi(AnyTaskResolver, when),
                new DecomposeCondSubterm(IMPL, true)//.bidi(AnyTaskResolver, when)
            );
        } else {
            //1-step IMPL decompose (immediate)
            //add(new DecomposeImpl(true).bidi());

            addAll(
                //new DecomposeImpl(true)
                new DecomposeImpl(true).bidi(AnyTaskResolver, when)
                //new DecomposeImpl(false).bidi()
                //new DecomposeImpl(true), new DecomposeImpl(false).reverse()
            );
            //add(DecomposeTerm.either(
        }
    }

    @Override
    public ReactionModel compile(NAR n) {
        var m = super.compile(n);

        if (beliefResolveAuto) {
            m.premisePreProcessor = (x, d) -> {
                if (x instanceof NALPremise.SingleTaskPremise s) {
                    if (s.task.BELIEF_OR_GOAL()) {
                        var y = beliefResolver.resolveBelief(s, d);
                        if (y != null)
                            return y;
                    }
                }
                return x;
            };
        }
        return m;
    }


}