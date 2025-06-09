package nars.derive;

import nars.$;
import nars.NARS;
import nars.Narsese;
import nars.test.TestNAR;
import nars.unify.constraint.RelationConstraint;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static nars.Op.BELIEF;
import static nars.Op.ETERNAL;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;


class UnifyConstraintTest {

    private static final int CYCLES = 64;

    @Test
    void testNeqComRecursiveConstraint() throws Narsese.NarseseException {
        RelationConstraint.NotRecursiveSubtermOf c = new RelationConstraint.NotRecursiveSubtermOf($.varQuery(1), $.varQuery(2));
        assertTrue(c.valid($.$("X"), $.$("Y"), null));
        assertFalse(c.valid($.$("X"), $.$("(X && Y)"), null));
        assertFalse(c.valid($.$("X"), $.$("(X|Y)"), null));
        //assertTrue(c.valid($.$("X"), $.$("f(X,Y)"), null));
    }


    @Disabled
    @Test
    void testNeqComRecursiveConstraintAllRules() {

        /*
        SHOULD NOT HAPPEN:
        $.02;.09$ ((o-(i-happy))-->happy). 497⋈527 %.55;.18% {497⋈527: æ0IáËÑþKn;æ0IáËÑþKM;æ0IáËÑþKÄ;æ0IáËÑþKÉ;æ0IáËÑþKÌ} (((%1-->%2),(%1-->%3),neqCom(%2,%3)),((%3-->%2),((Abduction-->Belief),(Weak-->Goal),(Backward-->Permute))))
            $.04;.75$ happy(L). 497⋈512 %.55;.75% {497⋈512: æ0IáËÑþKÄ}
            $.05;.53$ ((L)-->(o-(i-happy))). 527 %.54;.53% {527: æ0IáËÑþKn;æ0IáËÑþKM;æ0IáËÑþKÉ;æ0IáËÑþKÌ} Dynamic
        */
        new TestNAR(NARS.tmp(1))
                .believe("happy(L)", 1f, 0.9f)
                .believe("((L)-->(o-(i-happy)))", 1f, 0.9f)
                .mustNotOutput(CYCLES, "((o-(i-happy))-->happy)", BELIEF, ETERNAL);
    }

    @Disabled @Test
    void testNeqComRecursiveConstraintOneRule() {

        /*
        SHOULD NOT HAPPEN:
        $.02;.09$ ((o-(i-happy))-->happy). 497⋈527 %.55;.18% {497⋈527: æ0IáËÑþKn;æ0IáËÑþKM;æ0IáËÑþKÄ;æ0IáËÑþKÉ;æ0IáËÑþKÌ} (((%1-->%2),(%1-->%3),neqCom(%2,%3)),((%3-->%2),((Abduction-->Belief),(Weak-->Goal),(Backward-->Permute))))
            $.04;.75$ happy(L). 497⋈512 %.55;.75% {497⋈512: æ0IáËÑþKÄ}
            $.05;.53$ ((L)-->(o-(i-happy))). 527 %.54;.53% {527: æ0IáËÑþKn;æ0IáËÑþKM;æ0IáËÑþKÉ;æ0IáËÑþKÌ} Dynamic
        */
        new TestNAR(NARS.tmp(1))
                .believe("happy(L)", 1f, 0.9f)
                .believe("((L)-->(o-(i-happy)))", 1f, 0.9f)
                .mustNotOutput(CYCLES, "((o-(i-happy))-->happy)", BELIEF, ETERNAL);
    }

    @Test
    void testNeqComConstraint() {
        /*
        SHOULD NOT HAPPEN:
        $.05;.07$ ((((L)~(i|(L)))|(L))-->happy). 1866⋈1876 %.10;.16% {1866⋈1876: êbaîCóòmh;êbaîCóòoÁ;êbaîCóòoÃ;êbaîCóòrj;êbaîCóòrm;êbaîCóòrÏ} (((%1-->%2),(%3-->%2),notSet(%3),notSet(%1),neqCom(%3,%1)),(((%1|%3)-->%2),((Intersection-->Belief),(Intersection-->Goal))))
            $.08;.75$ happy(L). 1866⋈1876 %1.0;.75% {1866⋈1876: êbaîCóòrj}
            $.04;.43$ ((((L)~(i|(L)))|(L))-->happy). 1876 %.10;.21% {1876: êbaîCóòmh;êbaîCóòoÁ;êbaîCóòoÃ;êbaîCóòrm;êbaîCóòrÏ} Dynamic
        */
        new TestNAR(NARS.tmp(1))
                .believe("happy(L)", 1f, 0.9f)
                .believe("(((i)|(L))-->happy)", 1f, 0.9f)
                .mustNotOutput(CYCLES, "(((i)|(L))-->happy)", BELIEF, 1f, 1f, 0.81f, 0.81f, ETERNAL);
    }

//    @Test public void testExtractConstantPath() {
//        $constantPath
//    }

}