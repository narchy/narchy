package nars.deriver.op;

import jcog.Fuzzy;
import jcog.data.map.UnifriedMap;
import jcog.math.Intervals;
import nars.Deriver;
import nars.NAL;
import nars.Term;
import nars.premise.NALPremise;
import nars.term.Compound;
import nars.term.atom.Atomic;
import nars.truth.MutableTruthInterval;
import org.jetbrains.annotations.Nullable;

import java.util.Map;

import static nars.Op.*;

public enum DerivedOccurrence {
    Task() {

    },

    Belief() {
        @Override
        public boolean taskOrBelief(Deriver d) {
            return false;
        }
    },

    /** task range + seqDur (assuming its CONJ), if it's non-point */
    TaskSeq() {
        @Nullable
        @Override
        protected void when(NALPremise i, byte punc, Deriver d, MutableTruthInterval t) {
            var tt = i.task;
            var ts = tt.start();
            if (ts != ETERNAL) {
                var te = tt.end() + tt.term().seqDur();
                if (ts != te)
                    t.occurr(ts, te);
            }
        }
    },

    /** smudges derivation to the union of the premise tasks.  resulting truth is diluted in proportion to any non-intersecting range  */
    Union() {
        @Override
        protected void when(MutableTruthInterval t, long ts, long te, long bs, long be, byte punc, Deriver d) {
            t.occurr(Math.min(ts, bs), Math.max(te, be));
        }
    },

    Intersection() {
        @Override
        protected void when(MutableTruthInterval t, long ts, long te, long bs, long be, byte punc, Deriver d) {
            var i = Intervals.intersectionRaw(ts, te, bs, be);
            if (i == null)
                t.clear(); //fail
            else
                t.occurr(i[0], i[1]);
        }
    },

    /**
     * dynamically choose either TaskRel and BeliefRel
     */
    EitherRel() {

        @Override
        public boolean taskOrBelief(Deriver d) {
            return d.randomBoolean();
        }

        @Override
        public boolean relative() {
            return true;
        }
    },

    TaskRel() {
        @Override
        public boolean relative() {
            return true;
        }
    },

    BeliefRel() {
        @Override
        public boolean relative() { return true; }

        @Override
        public boolean taskOrBelief(Deriver d) {
            return false;
        }
    },

    Pre() {
        @Override
        protected void when(NALPremise i, byte punc, Deriver d, MutableTruthInterval t) {
            Term taskTerm = i.task().term();
            Term beliefTerm = i.belief().term(); // Assuming NonSeedTaskPremise or similar access

            Compound implicationTerm;
            long knownEventStart, knownEventEnd;

            // Pre: deriving antecedent (A) from consequent (B), with (A ==> B) or (A && dt B)
            // So, the component we know is B (the consequent).
            if (taskTerm.IMPL() && !beliefTerm.IMPL()) { // Task is implication, belief is consequent B
                implicationTerm = (Compound) taskTerm;
                knownEventStart = i.beliefStart();
                knownEventEnd = i.beliefEnd();
            } else if (beliefTerm.IMPL() && !taskTerm.IMPL()) { // Belief is implication, task is consequent B
                implicationTerm = (Compound) beliefTerm;
                knownEventStart = i.taskStart();
                knownEventEnd = i.taskEnd();
            } else {
                // Neither or both are implications - this rule shouldn't apply or premises are malformed.
                t.clear(); // Invalidates the truth interval
                return;
            }

            if (knownEventStart == ETERNAL) {
                t.occurr(ETERNAL, ETERNAL); // If known event is eternal, derived is eternal
                return;
            }

            prePost(false, knownEventStart, knownEventEnd, implicationTerm, t);
        }
    },

    Post() {
        @Override
        protected void when(NALPremise i, byte punc, Deriver d, MutableTruthInterval t) {
            Term taskTerm = i.task().term();
            Term beliefTerm = i.belief().term(); // Assuming NonSeedTaskPremise or similar access

            Compound implicationTerm;
            long knownEventStart, knownEventEnd;

            // Post: deriving consequent (B) from antecedent (A), with (A ==> B) or (A && dt B)
            // So, the component we know is A (the antecedent).
            if (taskTerm.IMPL() && !beliefTerm.IMPL()) { // Task is implication, belief is antecedent A
                implicationTerm = (Compound) taskTerm;
                // This case is tricky for Post: if task is (A==>B) and belief is A,
                // then this is a standard forward derivation.
                // knownEvent is A, which is the belief.
                knownEventStart = i.beliefStart();
                knownEventEnd = i.beliefEnd();

            } else if (beliefTerm.IMPL() && !taskTerm.IMPL()) { // Belief is implication, task is antecedent A
                implicationTerm = (Compound) beliefTerm;
                knownEventStart = i.taskStart();
                knownEventEnd = i.taskEnd();
            } else {
                // Neither or both are implications - this rule shouldn't apply or premises are malformed.
                t.clear(); // Invalidates the truth interval
                return;
            }

            if (knownEventStart == ETERNAL) {
                t.occurr(ETERNAL, ETERNAL); // If known event is eternal, derived is eternal
                return;
            }

            prePost(true, knownEventStart, knownEventEnd, implicationTerm, t);
        }
    },

//    /** custom solver for goal-based conjunction decomposition */
//    CondAfter() {
//        @Override
//        public boolean relative() {
//            return true;
//        }
//
//        @Override
//        public Pair<Term, MutableTruthInterval> solve(NALPremise i, Term x, MutableTruthInterval t, byte punc, Deriver d) {
//            return conjShift(i, x, false, t, d);
//        }
//
//        @Override protected boolean taskOrBelief(Deriver d) {
//            return false; //BELIEF
//        }
//    },

    NearestNow() {
        @Override
        public boolean relative() {
            return true;
        }

        @Override
        public boolean taskOrBelief(Deriver d) {
            return nearestNow(d);
        }

        private static boolean nearestNow(Deriver d) {
            var P = (NALPremise) d.premise;

            var now = d.now();
            var t = P.task.timeBetweenTo(now);//P.taskMid();

            var b = P.belief().timeBetweenTo(now); //, P.beliefMid();

            //return abs(now - t) <= abs(now - b);
            return t <= b;
        }
    },

    Later() {
        @Override
        protected void when(MutableTruthInterval t, long ts, long te, long bs, long be, byte punc, Deriver d) {
            var start = Math.max(ts, bs);
            var range =
                    //te - ts; //task's range
                    Math.min(te - ts, be - bs); //min range
            t.occurr(start, start + range);
        }

        @Override
        public boolean taskOrBelief(Deriver d) {
            return later((NALPremise.DoubleTaskPremise)NALPremise.nalPremise(d.premise));
        }

        @Override
        public boolean relative() {
            return true;
        }

        /** choose the later of the two premise tasks.
         @return true = Task, false = Belief */
        private static boolean later(NALPremise.DoubleTaskPremise p) {
            return p.taskStart() >= p.beliefStart();
        }

    },

    TaskBeliefSpan() {
        @Override
        protected void when(MutableTruthInterval t, long ts, long te, long bs, long be, byte punc, Deriver d) {
            var beliefSpan = d.premise.to().seqDur();
            var taskSpan = te - ts;
            if (beliefSpan > taskSpan)
                t.occurr(Intervals.range(Fuzzy.mean(ts,te), beliefSpan));
            else
                t.occurr(ts, te);
        }
    },
    ;

    /**
     * Calculates the occurrence time for a derived event based on an implication and a known component event.
     *
     * @param fwd              True for Post-derivation (calculating consequent time from antecedent),
     *                         False for Pre-derivation (calculating antecedent time from consequent).
     * @param knownEventStart  Start time of the known component event.
     * @param knownEventEnd    End time of the known component event.
     * @param implicationTerm  The Compound term representing the implication (e.g., (A ==> B) or (A && dt B)).
     * @param out              MutableTruthInterval to store the calculated occurrence time.
     */
    private static void prePost(boolean fwd, long knownEventStart, long knownEventEnd, Compound implicationTerm, MutableTruthInterval out) {

        if (knownEventStart == ETERNAL) {
            out.occurr(ETERNAL, ETERNAL);
            return;
        }

        long dt = implicationTerm.dt();

        if (dt == XTERNAL) {
            // If dt is XTERNAL, the derived time is considered ETERNAL,
            // unless specific XTERNAL semantics dictate otherwise in the future.
            out.occurr(ETERNAL, ETERNAL);
            return;
        }
        if (dt == DTERNAL) {
            dt = 0;
        }

        long derivedStart, derivedEnd;
        if (fwd) { // Post: deriving consequent B from antecedent A. knownEvent is A.
            derivedStart = knownEventStart + dt;
            derivedEnd = knownEventEnd + dt;
        } else {   // Pre: deriving antecedent A from consequent B. knownEvent is B.
            // antecedentDuration is the duration of A, the first component of the implication.
            long antecedentDuration = implicationTerm.seqDurSub(0, false);
            derivedStart = knownEventStart - dt - antecedentDuration;
            derivedEnd = knownEventEnd - dt - antecedentDuration;
        }
        out.occurr(derivedStart, derivedEnd);
    }


    public static final Map<Term, DerivedOccurrence> solvers;

    static {
        var tm = new UnifriedMap<Term, DerivedOccurrence>();
        for (var m : values())
            tm.put(Atomic.atomic(m.name()), m);
        tm.trimToSize();
        solvers = tm;
    }

    public final Atomic term;

    DerivedOccurrence() {
        this.term = Atomic.atom(name());
    }

    /**
     * whether full occurrence solving is involved
     */
    public boolean relative() {
        return false;
    }

    /**
     *
     * @param i the premise which the occurrence is being derived from
     * @param punc the punctuator for the premise (e.g. question mark)
     * @param d the current context deriver
     * @return a new occurrence interval, or `null` if the result is invalid.
     */
    public final MutableTruthInterval when(NALPremise i, byte punc, Deriver d) {
        var t = new MutableTruthInterval();
        when(i, punc, d, t);
        return t.s == TIMELESS ? null : t;
    }

    /** if overriding, handle ETERNAL cases */
    protected @Nullable void when(NALPremise i, byte punc, Deriver d, MutableTruthInterval t) {
        long ts = i.taskStart(), te = i.taskEnd();
        if (i instanceof NALPremise.DoubleTaskPremise dp) {
            var bs = dp.beliefStart();
            if (bs != ETERNAL) {
                var be = dp.beliefEnd();
                if (ts == ETERNAL) {
                    t.occurr(bs, be);
                } else {
                    when(t, ts, te, bs, be, punc, d); //BOTH temporal, solve
                }
                return;
            }
        }

        t.occurr(ts, te);
    }


    /** implementations need to call 'o.occurr()' with the conclusion time */
    protected void when(MutableTruthInterval t, long ts, long te, long bs, long be, byte punc, Deriver d) {
        var taskOrBelief = taskOrBelief(d);
        t.occurr(occurrence(
            taskOrBelief ? new long[] {ts, te} : new long[] {bs, be},
            ts, te, bs, be, taskOrBelief, d));
    }

    private static long[] occurrence(long[] SE, long ts, long te, long bs, long be, boolean taskOrBelief, Deriver d) {
        if (NAL.premise.OCC_TRIM)
            Intervals.shrink(SE, taskOrBelief ? bs : ts, taskOrBelief ? be : te);

        return SE;
    }

    /** whether the task (true) or belief (false) is the dominant temporal target
     * @param d*/
    public boolean taskOrBelief(Deriver d) {
        return true; //default
    }
}