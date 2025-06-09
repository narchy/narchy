package nars.task;

import jcog.util.ArrayUtil;
import nars.NAR;
import nars.NARS;
import nars.Narsese;
import nars.Task;
import nars.unify.UnifyAny;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static nars.$.$$;
import static nars.Op.BELIEF;
import static nars.Op.ETERNAL;
import static org.junit.jupiter.api.Assertions.*;

class AnswerTest {

    @Test
    void testMatchPartialXternal() throws Narsese.NarseseException {
        beliefQuery("((x &&+1 y) ==> z)",
                new String[]{"((x &&+- y) ==> z)", "((x &&+- y) ==> z)", "((x &&+- y) ==>+- z)", "((x && y) ==> z)"});
    }
    @Disabled
    @Test
    void testMatchPartialXternalDifferentVolume() throws Narsese.NarseseException {
        beliefQuery("((x &&+1 (x &&+1 y)) ==> z)",
            new String[]{"((x &&+- y) ==>+- z)"});
    }
    @Test
    void testMatchPartialXternalDifferentVolume_Partial() {
        assertFalse(new UnifyAny().uni($$("((x &&+1 (x &&+1 y)) ==> z)"), $$("((x &&+- y)          ==> z)")));
//        assertTrue($$("((x &&+1 (x &&+1 y)) ==> z)")
//            .unify($$("((x &&+- y)          ==> z)"), new UnifyAny()));
//        beliefQuery("((x &&+1 (x &&+1 y)) ==> z)",
//                new String[]{"((x &&+- y)       ==> z)"});
    }

    static void beliefQuery(String belief, String[] queries) throws Narsese.NarseseException {
        NAR n = NARS.shell();
        n.believe(belief);
        queries = ArrayUtil.add(queries, belief);
        for (String q : queries) {
            @Nullable Task a = n.answer($$(q), BELIEF, ETERNAL);
            assertNotNull(a, ()->q + " did not match " + belief);
            assertEquals($$(belief), a.term());
        }
    }
}