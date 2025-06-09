package nars.input;

import nars.task.NALTask;

/**
 * immediate, no buffering
 */
public class DirectTaskInput extends TaskInput {

    @Override protected void remember(NALTask x) {
        rememberNow(x);
    }


}
