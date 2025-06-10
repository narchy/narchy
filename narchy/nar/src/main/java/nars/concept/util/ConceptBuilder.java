package nars.concept.util;

import nars.*;
import nars.concept.NodeConcept;
import nars.concept.TaskConcept;
import nars.control.Emotion;
import nars.table.BeliefTables;
import nars.table.EmptyBeliefTable;
import nars.table.dynamic.ImageBeliefTable;
import nars.table.question.QuestionTable;
import nars.table.util.DynTables;
import nars.term.Compound;
import nars.time.Time;
import nars.utils.Profiler;
import org.eclipse.collections.api.block.function.primitive.ObjectBooleanToObjectFunction;
import org.jetbrains.annotations.Nullable;

import java.util.function.BiFunction;

/**
 * TODO make this BiFunction<Term,Concept,Concept>
 */
public abstract class ConceptBuilder implements BiFunction<Term, Concept, Concept> {

	private transient Emotion emotion;
	private transient Time time;

	protected abstract QuestionTable questionTable();

	protected abstract BeliefTable eternalTable(Term t);

	public abstract BeliefTable temporalTable(Term t, boolean beliefOrGoal);

	protected abstract Concept nodeConcept(Term t);

	private static boolean beliefOrGoalTable(Term t, boolean beliefOrGoal) {
		if (!(t instanceof Compound))
			return true; //assume its not a variable, etc.

        var o = t.op();
		if (beliefOrGoal ? !o.beliefable : !o.goalable)
			return false;

		return !t.hasAny(beliefOrGoal ? Op.UnBeliefableStructure : Op.UnGoalableStructure);
	}

	public final QuestionTable questionTable(Term t, boolean questionOrQuest) {
		return (questionOrQuest || !t.IMPL()) ?
			questionTable() :
			QuestionTable.Empty;
	}


	private BeliefTable beliefTable(Term t, boolean beliefOrGoal, boolean mutable, @Nullable ObjectBooleanToObjectFunction<Term, BeliefTable> dyn) {
		return beliefOrGoalTable(t, beliefOrGoal) ?
			beliefTable(t, beliefOrGoal, mutable, dyn != null ? dyn.valueOf(t, beliefOrGoal) : null) :
			EmptyBeliefTable.Empty;
	}

	public final BeliefTable beliefTable(Term t, boolean beliefOrGoal) {
		return beliefOrGoalTable(t, beliefOrGoal) ?
				beliefTable(t, beliefOrGoal, true, (BeliefTable)null) :
				EmptyBeliefTable.Empty;
	}

	private BeliefTable beliefTable(Term t, boolean beliefOrGoal, boolean mutable, @Nullable BeliefTable overlay) {
		//HACK ImageBeliefTable proxies for all other sub-tables

        var overlays = overlay!=null ? 1 : 0;
		if (overlays == 1 && (!mutable || overlay instanceof ImageBeliefTable))
			return overlay; //special case

        var b = new BeliefTables(overlays + (mutable ? 2 : 0));

		/* add the mutable tables first, so that dynamic tables can cache into them when computed after*/
		if(mutable) {
			b.add(temporalTable(t, beliefOrGoal));
			b.add(eternalTable(t));
		}

		if (overlays > 0)
			b.add(overlay);

		return b;
	}

	/**
	 * @param mutable whether to include mutable belief tables, or only dynamic
	 */
	@Nullable private Concept taskConcept(Term t, boolean mutable, ObjectBooleanToObjectFunction<Term, BeliefTable> d) {
		return !mutable && d == null ? null : new TaskConcept(t,
				//BELIEF
				beliefTable(t, true, mutable, d),
				//GOAL
				beliefTable(t, false, mutable, d),
				//QUESTION
				mutable ? questionTable(t, true) : QuestionTable.Empty,
				//QUEST
				mutable ? questionTable(t, false) : QuestionTable.Empty
		);
	}


	@Override
	public final Concept apply(Term t, Concept prev) {
		return ((prev != null) && !prev.isDeleted()) ? prev : apply(t);
	}


	public final Concept apply(Term t) {
		return apply(t, true, true);
	}

	/**
	 * constructs a concept but does no capacity allocation (result will have zero capacity, except dynamic abilities)
	 */
	public final Concept apply(Term t, boolean mutable, boolean dynamic) {
        long profilerStartTime = Profiler.startTime();
        Concept c = null;
        boolean newConceptCreated = false;
        try {
            c = concept(t, mutable, dynamic); // Actual creation happens here, timed internally by `concept()`

            if (c == null) {
    //			if (NAL.DEBUG)
    //				throw new TermException("unconceptualizable: mutable=" + mutable + ", dynamic=" + dynamic, t);
            } else {
                newConceptCreated = true;
                emotion.conceptNew.increment();
                start(c); // start() is profiled in DefaultConceptBuilder
            }
            return c;
        } finally {
            Profiler.recordTime("ConceptBuilder.apply.wrapper", profilerStartTime);
            Profiler.incrementCounter("ConceptBuilder.apply.wrapper.calls");
            if (newConceptCreated) {
                Profiler.incrementCounter("ConceptBuilder.apply.newConceptCreated");
            }
        }
	}

	private @Nullable Concept concept(Term t, boolean mutable, boolean dynamic) {
        long profilerStartTime = Profiler.startTime();
        try {
            return NALTask.TASKS(t) ?
                    taskConcept(t, mutable, dynamic ? DynTables.tableDyn(t) : null) :
                    nodeConcept(t);
        } finally {
            Profiler.recordTime("ConceptBuilder.concept.internalCreate", profilerStartTime);
            Profiler.incrementCounter("ConceptBuilder.concept.internalCreate.calls");
            if (NALTask.TASKS(t)) {
                Profiler.incrementCounter("ConceptBuilder.concept.taskConceptPath");
            } else {
                Profiler.incrementCounter("ConceptBuilder.concept.nodeConceptPath");
            }
        }
	}

	/**
	 * called after constructing a new concept, or after a permanent concept has been installed
	 */
	public void start(Concept c) {
		if (c.isDeleted())
			((NodeConcept) c).meta.clear(); //HACK remove any deleted state
		c.creation = time.now();
	}

	public void init(Emotion e, Time time) {
		this.emotion = e;
		this.time = time;
	}
}