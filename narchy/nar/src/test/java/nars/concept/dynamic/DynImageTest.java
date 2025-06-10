package nars.concept.dynamic;

import nars.*;
import nars.table.dynamic.ImageBeliefTable;
import nars.term.util.Image;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static nars.$.$;
import static nars.$.$$;
import static nars.Op.BELIEF;
import static nars.Op.ETERNAL;
import static org.junit.jupiter.api.Assertions.*;

class DynImageTest extends AbstractDynTaskTest {

    @Test
    void testImageExtIdentityEternal() throws Narsese.NarseseException {
        assertImageIdentity(ETERNAL, "((x,y)-->z)", "(y --> (z,x,/))", "(x --> (z,/,y))");
    }
    @Test
    void testImageExtIdentityEternal_withVar1() throws Narsese.NarseseException {
        assertImageIdentity(ETERNAL, "((#1,y)-->z)", "(y --> (z,#1,/))", "(#1 --> (z,/,y))");
    }

    /** image transforms with multiple variables disabled */
//    @Test
//    void testImageExtIdentityEternal_withVar2() throws Narsese.NarseseException {
    @Test void testMultiVariableImage() {
        //assertImageIdentity(ETERNAL, "((#1,#2)-->z)", "(#1 --> (z,#2,/))", "(#1 --> (z,/,#2))");
        assertTrue(isDynamicBeliefTable("(#1 --> (z,#2,/))"));
        assertTrue(isDynamicBeliefTable("(#1 --> (z,#2,(--,/)))"));
        assertTrue(isDynamicBeliefTable("(#1 --> (z,/,#2))"));
    }

    @Test void testTemporalTermImage1() {
        //assertFalse(isDynamicTable("((x&&y) --> (b,a,/))")); //actually this can be dynamic via intersection decomposition
        assertTrue(isDynamicBeliefTable("(a --> (b,(x&&y),/))"));
    }

    @Disabled
    @Test void testInvalidTemporalTermImage2() {
        assertFalse(isDynamicBeliefTable("(a --> (b,(x==>y),/))"));
        assertFalse(isDynamicBeliefTable("((x==>y) --> (b,a,/))"));
    }

    @Test
    void testImageExtIdentityTemporal() throws Narsese.NarseseException {
        assertImageIdentity(0, "((x,y)-->z)", "(y --> (z,x,/))", "(x --> (z,/,y))");
    }
    @Test
    void testImageIntIdentityEternal() throws Narsese.NarseseException {
        assertImageIdentity(ETERNAL, "(z-->(x,y))", "((z,x,\\)-->y)", "((z,\\,y)-->x)");
    }
    @Test
    void testImageIntIdentityTemporal() throws Narsese.NarseseException {
        assertImageIdentity(0,"(z-->(x,y))", "((z,x,\\)-->y)", "((z,\\,y)-->x)");
    }

    private void assertImageIdentity(long when, String x, String x1, String x2) throws Narsese.NarseseException {
        Term t = $$(x);

        Term i1 = $(x1);
        assertEquals(t, Image.imageNormalize(i1).normalize());
        Term i2 = $(x2);
        assertEquals(t, Image.imageNormalize(i2).normalize());

        n.believe($$(x), when,0.75f, 0.50f);


        assertEquals(
                "%.75;.50%", n.beliefTruth(i1, when).toString()
        );
        String tStr = when == ETERNAL ? "" : " " + when;
        assertEquals(
                i1 + "." + tStr + " %.75;.50%", n.answer(i1, BELIEF, when).toStringWithoutBudget()
        );
        assertEquals(
                i2 + "." + tStr + " %.75;.50%", n.answer(i2, BELIEF, when).toStringWithoutBudget()
        );
    }

    @Test void InternalImageTaskTermRepresentation() {
        NAR n = NARS.tmp(4);
        Term it = $$("(x --> (y,/))");
        n.believe(it);
        Concept i = n.conceptualize(it);
        Term pt = Image.imageNormalize(it);
        Concept p = n.concept(pt);
        assertNotSame(i, p);

        assertEquals(ImageBeliefTable.class, i.beliefs().getClass());

    }
}