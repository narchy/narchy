package nars.nal.nal8;

import nars.*;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.Test;

import static jcog.Str.n2;
import static nars.Op.*;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * exhaustive parametric implication deduction/induction belief/goal tests
 */
class ImplicationTest {

    private static final Term x = $.atomic("x");
    private static final Term y = $.atomic("y");
    private static final boolean[] B = {true, false};
    private static final int CYCLES = 20;
    private static final int TERM_VOL_MAX = 5;

    private static void assertContains(String oo, String c) {
        assertTrue(oo.contains(c));
    }

//    static void assertNotContains(String oo, String c) {
//        assertFalse(oo.contains(c), () -> "matched:\n\t" +
//                Joiner.on("\n\t").join(Stream.of(oo.split("\n")).filter(x -> x.contains(c)).collect(toList())));
//    }

    @Test
    void testBelief() {

        StringBuilder o = new StringBuilder();
        for (float condFreq : new float[]{0, 1, 0.5f}) {
            for (boolean sp : B) {
                Term z = sp ? x : y;
                for (boolean xx : B) {
                    for (boolean yy : B) {
                        NAR n = NARS.tmp(6);
                        n.complexMax.set(TERM_VOL_MAX);


                        Term impl = IMPL.the(x.negIf(!xx), y.negIf(!yy));

                        n.believe(impl);
                        n.believe(z, condFreq, n.confDefault(BELIEF));
                        n.run(CYCLES);


                        Term nz = sp ? y : x;


                        @Nullable Truth nzt = n.beliefTruth(nz, ETERNAL);

                        o.append(z).append(". %").append(n2(condFreq)).append("% ").append(impl).append(". ").append(nz).append('=').append(nzt).append('\n');
                    }
                }
            }
        }

        String oo = o.toString();
        System.out.println(oo);

        assertContains(oo, "x. %1.0% (x==>y). y=%1.0;.81%");
        assertContains(oo, "x. %0.0% ((--,x)==>y). y=%1.0;.81%");

        assertContains(oo, "y. %1.0% (x==>y). x=%1.0;.45%");
        assertContains(oo, "y. %1.0% ((--,x)==>y). x=%0.0;.45%");
        assertContains(oo, "y. %0.0% (--,(x==>y)). x=%1.0;.45%");
        assertContains(oo, "y. %0.0% (--,((--,x)==>y)). x=%0.0;.45%");

        //assertNotContains(oo, "y. %1.0% (--,((--,x)==>y)). x=%1.0;.45%");
        //assertNotContains(oo, "y. %0.0% ((--,x)==>y). x=%1.0;.45%");


    }

    @Test
    void testGoal() {

        StringBuilder o = new StringBuilder();
        for (boolean sp : B) {
            Term z = sp ? x : y;
            for (boolean zz : B) {
                for (boolean xx : B) {
                    for (boolean yy : B) {
                        NAR n = NARS.tmp(6);
                        n.complexMax.set(TERM_VOL_MAX);

                        Term cond = z.negIf(!zz);
                        Term impl = IMPL.the(x.negIf(!xx), y.negIf(!yy));

                        n.believe(impl);
                        n.want(cond);
                        n.run(CYCLES);

                        Term nz = sp ? y : x;
                        @Nullable Truth nzt = n.goalTruth(nz, ETERNAL);
                        o.append(cond).append("! ").append(impl).append(". ").append(nz).append('=').append(nzt).append('\n');
                    }
                }
            }
        }

        String oo = o.toString();

        System.out.println(oo);


        assertContains(oo, "y! (x==>y). x=%1.0;.81%");
        assertContains(oo, "y! ((--,x)==>y). x=%0.0;.81%");
        assertContains(oo, "(--,y)! (--,(x==>y)). x=%1.0;.81%");
        assertContains(oo, "(--,y)! (--,((--,x)==>y)). x=%0.0;.81%");


        assertContains(oo, "x! (x==>y). y=%1.0;.45%");
        assertContains(oo, "x! (--,(x==>y)). y=%0.0;.45%");


    }

}