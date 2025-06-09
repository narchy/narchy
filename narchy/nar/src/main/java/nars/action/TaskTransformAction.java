package nars.action;

import nars.derive.Deriver;
import nars.derive.reaction.TaskReaction;
import nars.task.NALTask;
import org.jetbrains.annotations.Nullable;

public abstract class TaskTransformAction extends TaskReaction {

	protected abstract @Nullable NALTask transform(NALTask x, Deriver d);

	@Override
	protected final void run(NALTask x, Deriver d) {
		NALTask y = transform(x, d);
		if (y!=null) {
			if (copyMeta())
				copyMeta(x, y);
			react(y, d);
		}
	}

	private static void copyMeta(NALTask x, NALTask y) {
		y.copyMeta(x);
		y.setCreation(x.creation());
	}

	protected boolean copyMeta() {
		return false;
	}


}