package nars.action.answer;

import nars.Op;
import nars.Term;
import nars.action.resolve.TaskResolver;
import nars.derive.Deriver;
import nars.focus.time.TaskWhen;
import nars.task.NALTask;

import static nars.task.NALTask.TASKS;

public class AnswerQuestionsFromBeliefTable extends AnswerQuestions {

	public static final boolean ACCEPT_REFINED_QUESTION = true;

	public AnswerQuestionsFromBeliefTable(TaskWhen timing, boolean question, boolean quest, TaskResolver resolver) {
		super(timing, question, quest, resolver);
		assert(question||quest);

		hasAny(PremiseTask, Op.VAR_QUERY, false);

//		store(false);
	}

	@Override
	protected void run(NALTask Q, Deriver d) {
		Term q = Q.term();
		Term qq = d.nar.eval(q);
		if (q==qq || TASKS(qq, (byte)0, true)) {
			var A = answer(Q, qq, true, ACCEPT_REFINED_QUESTION, () -> timing.whenRelative(Q, d), d);
			if (A != null)
				react(A, d);
		}
	}
}