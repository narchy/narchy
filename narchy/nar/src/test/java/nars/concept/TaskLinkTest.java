package nars.concept;

import jcog.pri.op.PriMerge;
import nars.NARS;
import nars.Narsese;
import nars.link.MutableTaskLink;
import org.junit.jupiter.api.Test;

import static jcog.util.PriReturn.Delta;
import static nars.$.$;
import static nars.$.$$;
import static nars.Op.BELIEF;
import static nars.Op.GOAL;
import static org.junit.jupiter.api.Assertions.*;

public class TaskLinkTest {

    @Test void TaskLinkComponent_Init_Deleted() {
        MutableTaskLink t = MutableTaskLink.link($$("x"));
        assertNaN(t.pri());
        //assertEquals(0, t.pri(), 0.001f);
    }

    @Test void TaskLinkComponentPriOverflow() {
        MutableTaskLink t = MutableTaskLink.link($$("x"));
        t.priPunc(BELIEF, 0.9f);
        assertEquals((1*0.9f)/4f, t.pri(), 0.001f);

        float gd = t.mergePunc(GOAL, 0.9f, PriMerge.replace, Delta);
        assertEquals(0.9f/4f, gd, 0.001f);
        assertEquals((2*0.9f)/4f, t.pri(), 0.001f);
        t.mergePunc(GOAL, 0f, PriMerge.replace);

        t.mergePunc(BELIEF, 0.1f, PriMerge.plus);
        assertEquals(0.25f, t.pri(), 0.01f);
        assertEquals(1, t.priPunc(BELIEF), 0.01f);

        //no change:
        t.mergePunc(BELIEF, 0.1f, PriMerge.plus);
        assertEquals(0.25f, t.pri(), 0.01f);
        assertEquals(1, t.priPunc(BELIEF), 0.01f);

    }

    private static void assertNaN(float x) {
        assertTrue(x!=x);
    }

    //    private final NAR n = new NARS().shell();

    @Test void TaskLinkComponentRange() {
        MutableTaskLink t = MutableTaskLink.link($$("x"));
        t.mergePunc(BELIEF, 0.9f, PriMerge.plus);
        t.mergePunc(GOAL, 0.9f, PriMerge.plus);
        assertEquals(0.5f, t.pri(), 0.1f);
        assertEquals("$.45:0.9,0,0.9,0 x", t.toString());
        t.priMul(0.9f);
        assertEquals("$.40:0.81,0,0.81,0 x", t.toString());
    }
    @Test void mergeMaxComponents() {
        MutableTaskLink t = MutableTaskLink.link($$("x"));
        t.mergePunc(BELIEF, 1f, PriMerge.max);
        assertEquals("$.25:1,0,0,0 x", t.toString());
        t.mergePunc(GOAL, 1f, PriMerge.max);
        assertEquals("$.50:1,0,1,0 x", t.toString());
    }

    @Test
    void testConceptualizeNonTaskable_IndepVarUnbalanced() throws Narsese.NarseseException {
        assertNotNull(NARS.tmp(1).conceptualize($("(x --> $1)")));
    }

//    @Test
//    void testConceptFireLinkSelection() throws Narsese.NarseseException {
//        int count = 8;
//
//        n.input("$0.01 a:b.");
//        n.run(1);
//
//        System.out.println("inputs:\n");
//
//        Concept c = n.conceptualize("a:b");
//        for (int n = 0; n < count; n++) {
//            PLink<Term> inserted = new PLink<>($("x" + n + ":a"), ((1 + n) / ((float) count)));
//            System.out.println(inserted);
//            c.termlinks().put(inserted);
//        }
//
//
//        System.out.println();
//
//        HashBag<String> termlinkHits = new HashBag();
//        HashBag<String> taskHits = new HashBag();
//
//        Activate cf = new Activate(c, 1f);
//
//        Term A = $.the("a");
//
//        MatrixDeriver dummy = new MatrixDeriver(Derivers.parse(n,
//        "(A --> B), (A --> C), neqRCom(B,C)      |- (C --> B), (Belief:Abduction, Goal:DesireWeak)"));
//
//        for (int i = 0; i < 100; i++) {
//            final int[] remain = {9};
//            ActivatedLinks linkActivations = new ActivatedLinks();
//            dummy.premiseMatrix(cf, (task, target) -> {
//                Task ptask = task;
//                Term pterm = target.get();
//                System.out.println("tasklink=" + ptask + " termlink=" + pterm);
//                if (pterm instanceof Atom || !A.equals(pterm.sub(0)))
//                    return true;
//                String tls = pterm.toString();
//
//
//                termlinkHits.addOccurrences(/*tasklink.get() + " " +*/ tls, 1);
//                taskHits.addOccurrences(/*tasklink.get() + " " +*/ (ptask + " " + pterm), 1);
//                return --remain[0] > 0;
//            }, 1, 3, n.);
//
//            //TODO analyze linkActivations
//            n.input(linkActivations);
//        }
//
//
//        System.out.println("termlinks pri (after):\n");
//        c.termlinks().print();
//
//        System.out.println("\ntermlink hits:\n");
//        termlinkHits.topOccurrences(termlinkHits.size()).forEach(System.out::println);
//
//
//
//
//
//
//
//
//
//
//
//        ObjectIntPair<String> top = termlinkHits.topOccurrences(1).get(0);
//        ObjectIntPair<String> bottom = termlinkHits.bottomOccurrences(1).get(0);
//        String min = bottom.getOne();
//        assertTrue("(a-->x0)".equals(min) || "(a-->x1)".equals(min));
//        assertEquals("(a-->x" + (count - 1) + ")", top.getOne());
//
//    }
//
//    @Test
//    void testDerivedBudgets() throws Narsese.NarseseException {
//
//
//
//
//
//
//
//        n.input("$0.1$ <a --> b>.");
//        n.input("$0.1$ <b --> a>.");
//        n.run(15);
//
//
//        n.conceptsActive().forEach(System.out::println);
//    }


//    /** when sampling links, weaker beliefs should be selected at a proportional rate.
//     * dont allow return the top belief tasklinks
//     */
//    @Test public void testEnsureFairBeliefSelection() {
//        float confA = 0.9f, confB = 0.5f;
//        Frequency f = sampleLink((n)->{
//            try {
//                n.believe("x", 1f, confA);
//                n.believe("x", 1f, confB);
//            } catch (Narsese.NarseseException e) {
//                e.printStackTrace();
//            }
//        }, 64);
//
//        System.out.println(f);
//
//        //test relative selection frequency
//        assertEquals(2, f.getUniqueCount());
//        Iterator<Comparable<?>> ff = f.valuesIterator();
//        Comparable<?> aa = ff.next();
//        Comparable<?> bb = ff.next();
//        assertTrue((f.getPct(bb) / f.getPct(aa)) > 1.5f);  //some significant difference
//
//
//    }
//
//
//
//    @Test public void testEnsureFairQuestionSelection() {
//        Frequency f = sampleLink((n)->{
//
//                Task a = n.question("x");
//                a.pri(0.25f);
//                Task b = n.question("x");
//                b.pri(0.5f);
//
//        },64);
//
//        System.out.println(f);
//        assertEquals(2, f.getUniqueCount());
//
//        assertEquals(2, f.getUniqueCount());
//        Iterator<Comparable<?>> ff = f.valuesIterator();
//        Comparable<?> aa = ff.next();
//        Comparable<?> bb = ff.next();
//        assertTrue((f.getPct(bb) / f.getPct(aa)) > 1.75f);  //some significant difference
//    }
//
//    private static Frequency sampleLink(Consumer<NAR> setup, int samples) {
//        NAR n = NARS.shell();
//        setup.accept(n);
//        n.run(1);
//
//        Bag<?, TaskLink> links = n.concept("x").tasklinks();
//        assertEquals(1, links.size());
//        @Nullable TaskLink l = links.iterator().next();
//        assertNotNull(l);
//
//        Frequency f = new Frequency();
//        for (int i = 0; i < samples; i++)
//            f.addValue(l.apply(n).toString());
//        return f;
//    }
}