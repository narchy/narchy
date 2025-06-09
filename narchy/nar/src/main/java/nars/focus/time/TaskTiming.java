package nars.focus.time;

import nars.derive.Deriver;
import nars.task.NALTask;

/** simply matches the task time exactly. */
public class TaskTiming extends PresentTiming {

    @Override
    public long[] whenRelative(NALTask t, Deriver d) {
        return t.startEndArray();
    }

}
