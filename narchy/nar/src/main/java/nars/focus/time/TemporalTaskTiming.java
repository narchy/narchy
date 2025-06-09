package nars.focus.time;

import nars.derive.Deriver;
import nars.task.NALTask;

public class TemporalTaskTiming extends TaskTiming {
    @Override
    public long[] whenRelative(NALTask t, Deriver d) {
        return t.ETERNAL() ? whenAbsolute(d) : super.whenRelative(t, d);
    }
}
