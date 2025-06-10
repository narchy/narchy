package nars.term.util.transform;

import jcog.random.RandomBits;
import jcog.random.XorShift128PlusRandom;
import nars.*;
import nars.term.Compound;
import nars.term.var.Variable;
import nars.unify.Unify;
import nars.unify.mutate.CommutivePermutations;
import nars.unify.mutate.Termutator;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;

import static nars.$.$;
import static nars.$.v;
import static org.junit.jupiter.api.Assertions.assertEquals;

//import nars.term.var.ellipsis.Ellipsis;

/**
 * Created by me on 12/22/15.
 */
class TermutatorTest {

    private static final int TTL = 1024*10;

    private final Unify unifier = new Unify(Op.VAR_PATTERN, new RandomBits(new XorShift128PlusRandom(1)),
            NAL.unify.UNIFICATION_STACK_CAPACITY, TTL) {
        @Override public boolean match() {
            return false;
        }
    };

//    @Test
//    void testChoose1_2() {
//
//        assertTermutatorProducesUniqueResults(
//                new Choose1(e1, p2,
//                        ((Compound)p("a", "b")).toSetSorted()), 2);
//
//    }
//
//    @Test
//    void testChoose1_3() {
//
//        assertTermutatorProducesUniqueResults(
//                new Choose1(e1, p2,
//                        ((Compound)p("a", "b", "c")).toSetSorted()), 3);
//    }
//
//    @Test
//    void testChoose1_4() {
//
//        assertTermutatorProducesUniqueResults(
//                new Choose1(e1, p2,
//                        ((Compound)p("a", "b", "c", "d")).toSetSorted()), 4);
//    }


    private static final Term e0;
    static {
        
        Term ee0;
        try {
            ee0 = $("%A..+");
        } catch (Narsese.NarseseException e) {
            ee0 = null;
            e.printStackTrace();
            System.exit(1);
        }
        e0 = ee0;
    }
//    private static final Ellipsis e1 = Ellipsis.EllipsisPrototype.make((byte) 1,1);


    private static final Variable p2= v(Op.VAR_PATTERN, (byte) 2);
    private static final SortedSet<Term> p2p3 = ((Compound)$.p( p2, v(Op.VAR_PATTERN, (byte) 3) )).toSetSorted();
//
//    @Test
//    void testChoose2_2() {
//
//
//
//        assertTermutatorProducesUniqueResults(
//                new Choose2(e1, p2p3, ((Compound)p("a", "b")).toSetSorted()
//                ), 2);
//    }
//
//    @Test
//    void testChoose2_3() {
//
//        assertTermutatorProducesUniqueResults(
//                new Choose2(e1, p2p3, ((Compound)p("a", "b", "c")).toSetSorted()
//                ), 6);
//    }
//    @Test
//    void testChoose2_4() {
//
//        Set<String> series = IntStream.range(0, 5).mapToObj(i -> assertTermutatorProducesUniqueResults(
//                new Choose2(e1, p2p3, ((Compound) p("a", "b", "c", "d")).toSetSorted()
//                ), 12)).collect(Collectors.toSet());
//
//        assertTrue(series.size() > 1);
//    }



    @Test
    void testComm2() throws Narsese.NarseseException {
        assertTermutatorProducesUniqueResults(
                new CommutivePermutations((Compound)$("{%A,%B}"), (Compound)$("{x,y}")), 2);
    }
    @Test
    void testComm3() throws Narsese.NarseseException {
        assertTermutatorProducesUniqueResults(
                new CommutivePermutations((Compound)$("{%A,%B,%C}"), (Compound)$("{x,y,z}")), 6);
    }
    @Test
    void testComm3Conj() throws Narsese.NarseseException {
        assertTermutatorProducesUniqueResults(
                new CommutivePermutations((Compound)$("(&&,%A,%B,%C)"), (Compound)$("(&&,x,y,z)")), 6);
    }
    @Test
    void testComm4() throws Narsese.NarseseException {
        assertTermutatorProducesUniqueResults(
                new CommutivePermutations((Compound)$("{%A,%B,%C,%D}"), (Compound)$("{w,x,y,z}")), 24);
    }

    private String assertTermutatorProducesUniqueResults(Termutator t, int num) {


        unifier.setTTL(TTL);


        int[] duplicates = {0};
        int[] actual = {0};
        Set<String> s = new LinkedHashSet<>();
        t.apply(new Termutator[]{t, new Termutator($.the(Termutator.class)) {

            @Override
            public boolean apply(Termutator[] chain, int current, Unify u) {
                TreeMap t1 = new TreeMap(u.xy.map);

                if (s.add(t1.toString())) {
                    actual[0]++;
                } else {
                    duplicates[0]++;
                }
                return true;
            }

        }}, 0, unifier);


        String res = s.toString();
//        System.out.println(res);

        assertEquals(num, s.size());
        assertEquals(num, actual[0]);
        assertEquals(0, duplicates[0]);

        return res;
    }

}