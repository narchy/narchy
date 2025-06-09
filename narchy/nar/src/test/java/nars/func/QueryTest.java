//package nars.op;
//
//import nars.$;
//import nars.NAR;
//import nars.NARS;
//import nars.Narsese;
//import nars.eval.FactualEvaluator;
//import nars.term.Term;
//import org.junit.jupiter.api.Test;
//
//import java.util.Set;
//import java.util.TreeSet;
//
//import static nars.$.$$;
//import static org.junit.jupiter.api.Assertions.assertEquals;
//import static org.junit.jupiter.api.Assertions.assertTrue;
//
//public class QueryTest {
//
//    @Test void FactPos() throws Narsese.NarseseException {
//        NAR n = NARS.shell();
//        n.believe("f(x)");
//
//        Set<Term> e = FactualEvaluator.queryAll($$("f(?what)"), n);
//        assertEquals("[f(x)]", e.toString());
//    }
//
//    @Test void FactsPos() throws Narsese.NarseseException {
//        NAR n = NARS.shell();
//        n.believe("f(x)");
//        n.believe("f(y)");
//
//        Set<Term> e = FactualEvaluator.queryAll($$("f(?what)"), n);
//        assertEquals(Set.of($$("f(x)"), $$("f(y)")), e);
//    }
//
//    @Test void FactNeg() throws Narsese.NarseseException {
//        NAR n = NARS.shell();
//        n.believe("--f(x)");
//
//        Set<Term> e = FactualEvaluator.queryAll($$("f(?what)"), n);
//        assertEquals("[(--,f(x))]", e.toString());
//    }
//
//    @Test void FactImpliedByFact() throws Narsese.NarseseException {
//        NAR n = NARS.shell();
//        n.believe("(f(x) ==> g(x))");
//        n.believe("f(x)");
//
//        Set<Term> e = FactualEvaluator.queryAll($$("g(?what)"), n);
//        assertEquals("[f(x)]", e.toString());
//    }
//
//    @Test void FactImpliedByConj() throws Narsese.NarseseException {
//        NAR n = NARS.shell();
//        n.believe("((f(#1) && f(#2)) ==> g(#1,#2))");
//        n.believe("f(x)");
//        n.believe("f(y)");
//
//        Set<Term> e = FactualEvaluator.queryAll($$("g(?1,?2)"), n);
//        assertEquals("[g(x,x),g(x,y),g(y,y),g(y,x)]", e.toString());
//    }
//
//    static class FunctorBacktrackingTest {
//        final NAR n = NARS.shell();
//
//        {
//            try {
//                /**
//                 * from: https://en.wikipedia.org/wiki/Prolog#Execution
//                 *
//                 */
//                n.input(
//                        "mother(trude, sally).",
//                        "father(tom, sally).",
//                        "father(tom, erica).",
//                        "father(mike, tom).",
//
//                        "(father($X, $Y) ==> parent($X, $Y)).",
//                        "(mother($X, $Y) ==> parent($X, $Y)).",
//                        "((parent(#Z, $X) && parent(#Z, $Y)) ==> sibling($X, $Y))."
//
//                        //TODO
//                        //"prolog(\"sibling(X, Y)      :- parent_child(Z, X), parent_child(Z, Y)\").",
//
//
//                );
//            } catch (Narsese.NarseseException e) {
//                e.printStackTrace();
//            }
//        }
//
//        @Test
//        public void test2() {
//
//            Set<Term> q = new TreeSet(FactualEvaluator.queryAll($$("father(?Father, ?Child)"), n));
//            assertEquals("[father(tom,erica), father(tom,sally), father(mike,tom)]", q.toString());
//
//
//            //"[father(mike,tom), father(tom,sally), father(tom,erica)]",
////            assertEquals("{father(tom,sally)=[true], father(tom,erica)=[true], father(mike,tom)=[true]}",
////                    e.nodes.toString());
//        }
//
//        @Test
//        public void test3() throws Narsese.NarseseException {
//
//            //        "[wonder(sibling(sally,erica))]",
//            {
//                Set<Term> q = FactualEvaluator.queryAll($.$("sibling(sally,erica)"), n);
//                assertTrue(q.isEmpty());
//            }
//        }
//
//        @Test
//        public void test3b() throws Narsese.NarseseException {
//
//            {
//                n.believe("mother(trude,erica)"); //becomes true only after this missing information
//
//                Set<Term> q = FactualEvaluator.queryAll($$("sibling(sally,erica)"), n);
//                assertEquals("[sibling(sally,erica)]", q.toString());
//            }
//
//        }
//
//        @Test
//        public void test4() {
//            Set<Term> x = FactualEvaluator.queryAll($$("sibling(tom,erica)"), n);
//            System.out.println(x);
//
//
////        assertEquals(
////                "[wonder(sibling(tom,erica))]", //UNKNOWN, not true or falsedate
//
////                .toString()
////        );+
//
//
//
//        /*
//
//        ?- father_child(Father, Child).
//            [enumerates all possibilities]
//         */
//
//        }
//    }
//}
