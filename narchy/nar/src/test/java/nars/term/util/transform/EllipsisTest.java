//package nars.term.util.transform;
//
//import jcog.random.XorShift128PlusRandom;
//import nars.$;
//import nars.NAL;
//import nars.Narsese;
//import nars.derive.premise.PatternTermBuilder;
//import nars.term.Compound;
//import nars.term.Term;
//import nars.term.var.Variable;
//import nars.term.atom.Atomic;
//import nars.term.atom.Bool;
////import nars.term.var.ellipsis.EllipsisOneOrMore;
////import nars.term.var.ellipsis.EllipsisZeroOrMore;
////import nars.term.var.ellipsis.Fragment;
//import nars.unify.Unify;
//import org.jetbrains.annotations.NotNull;
//import org.jetbrains.annotations.Nullable;
//import org.junit.jupiter.api.Disabled;
//import org.junit.jupiter.api.Test;
//
//import java.util.HashSet;
//import java.util.Random;
//import java.util.Set;
//
//import static nars.$.$;
//import static nars.$.$$$;
//import static nars.Op.VAR_PATTERN;
//import static nars.term.util.TermTest.assertEq;
//import static nars.term.var.ellipsis.Ellipsis.firstEllipsis;
//import static org.junit.jupiter.api.Assertions.*;
//
///**
// * Created by me on 12/12/15.
// */
//public class EllipsisTest {
//
//
//    interface EllipsisTestCase {
//        Compound getPattern();
//
//        Term getResult() throws Narsese.NarseseException;
//
//        @Nullable
//        Term getMatchable(int arity) throws Narsese.NarseseException;
//
//        default Set<Term> test(int arity, int repeats) throws Narsese.NarseseException {
//
//
//            Term y = /*index.patternify*/(getMatchable(arity));
//
//            assertTrue(!(y instanceof Bool));
//            assertNotNull(y);
//            assertTrue(y.isNormalized());
//
//            assertEquals(0, y.vars());
//            assertEquals(0, y.varPattern());
//
//            Term r = /*index.patternify*/( getResult() );
//
//            Term x = PatternTermBuilder.patternify(PatternTermBuilder.rule( getPattern() ));
//
//
//
//
//
//            @Nullable Variable ellipsisTerm = (Variable) firstEllipsis(x.subterms());
//            assertNotNull(ellipsisTerm);
//
//
//            Set<Term> selectedFixed = new HashSet(arity);
//            for (int seed = 0; seed < Math.max(1, repeats) /* enough chances to select all combinations */; seed++) {
//
//
//
//                //System.out.println(seed + ": " + x + " unify " + y + " => " + r);
//
//                Unify f = new Unify(VAR_PATTERN, new XorShift128PlusRandom(1 + seed), NAL.unify.UNIFICATION_STACK_CAPACITY) {
//
//                    @Override
//                    public boolean match() {
//
//
//
//                        Term a = resolveVar(ellipsisTerm);
//                        if (a instanceof Fragment) {
//                            Fragment varArgs = (Fragment) a;
//
//
//
//                            assertEquals(getExpectedUniqueTerms(arity), varArgs.subs());
//
//                            Term u = apply(varArgs);
//                            if (u == null) {
//                                u = varArgs;
//                            }
//
//                            Set<Term> varArgTerms = new HashSet(1);
//                            if (u instanceof Fragment) {
//                                Fragment m = (Fragment) u;
//                                for (Term term : m) {
//                                    varArgTerms.add(term);
//                                }
//                            } else {
//                                varArgTerms.add(u);
//                            }
//
//                            assertEquals(getExpectedUniqueTerms(arity), varArgTerms.size());
//
//                            testFurther(selectedFixed, this, varArgTerms);
//
//                        } else {
//                            assertNotNull(a);
//
//                        }
//
//
//                        Term s = apply(r);
//                        if (s != null) {
//
//                            if (s.varPattern() == 0)
//                                selectedFixed.add(s);
//
//                            assertEquals(0, s.varPattern(), () -> s + " should be all subbed by " + this.xy);
//                        }
//
//                        return true;
//                    }
//                };
//
//                f.setTTL(128);
//                f.unify(x, y);
//
//
//
//
//            }
//
//
//            return selectedFixed;
//
//        }
//
//        int getExpectedUniqueTerms(int arity);
//
//        default void testFurther(Set<Term> selectedFixed, Unify f, Set<Term> varArgTerms) {
//
//        }
//
//        default void test(int arityMin, int arityMax, int repeats) throws Narsese.NarseseException {
//            for (int arity = arityMin; arity <= arityMax; arity++) {
//                test(arity, repeats);
//            }
//        }
//    }
//
//    abstract static class CommutiveEllipsisTest implements EllipsisTestCase {
//        final String prefix;
//        final String suffix;
//        final @NotNull Compound p;
//        final String ellipsisTerm;
//
//        CommutiveEllipsisTest(String ellipsisTerm, String prefix, String suffix) throws Narsese.NarseseException {
//            this.prefix = prefix;
//            this.suffix = suffix;
//            this.ellipsisTerm = ellipsisTerm;
//            p =  getPattern(prefix, suffix);
//        }
//
//
//        static String termSequence(int arity) {
//            StringBuilder sb = new StringBuilder(arity * 3);
//            for (int i = 0; i < arity; i++) {
//                sb.append((char) ('a' + i));
//                if (i < arity - 1)
//                    sb.append(',');
//            }
//            return sb.toString();
//        }
//
//        protected abstract @Nullable Compound getPattern(String prefix, String suffix) throws Narsese.NarseseException;
//
//
//        @Override
//        public @NotNull Compound getPattern() {
//            return p;
//        }
//
//        @Override
//        public @Nullable Term getMatchable(int arity) throws Narsese.NarseseException {
//            String s = termSequence(arity);
//            String p = this.prefix;
//            if ((arity == 0) && (p.endsWith(",")))
//                p = p.substring(0, p.length()-1);
//            return $(p + s + suffix);
//        }
//    }
//
//    public static class CommutiveEllipsisTest1 extends CommutiveEllipsisTest {
//
//        static final Variable fixedTerm = $.varPattern(1);
//
//
//        CommutiveEllipsisTest1(String ellipsisTerm, String[] openClose) throws Narsese.NarseseException {
//            super(ellipsisTerm, openClose[0], openClose[1]);
//        }
//
//        @Override
//        public Set<Term> test(int arity, int repeats) throws Narsese.NarseseException {
//            Set<Term> selectedFixed = super.test(arity, repeats);
//
//            /** should have iterated all */
//            assertEquals(arity, selectedFixed.size(), selectedFixed::toString);
//            return selectedFixed;
//        }
//
//        @Override
//        public int getExpectedUniqueTerms(int arity) {
//            return arity - 1;
//        }
//
//        @Override
//        public void testFurther(Set<Term> selectedFixed, @NotNull Unify f, @NotNull Set<Term> varArgTerms) {
//            assertEquals(2, f.xy.keySet().size());
//            Term fixedTermValue = f.resolveVar(fixedTerm);
//            assertNotNull(fixedTermValue, f::toString);
//            assertTrue(fixedTermValue instanceof Atomic);
//            assertFalse(varArgTerms.contains(fixedTermValue));
//        }
//
//
//        @Override
//        public @NotNull Compound getPattern(String prefix, String suffix) throws Narsese.NarseseException {
//            Compound pattern = (Compound) Narsese.term(prefix + "%1, " + ellipsisTerm + suffix, true).term();
//            return pattern;
//        }
//
//
//        @Override
//        public @NotNull Term getResult() throws Narsese.NarseseException {
//            return Narsese.term("<%1 --> (" + ellipsisTerm + ")>", true).normalize().term();
//        }
//
//    }
//
//    /**
//     * for testing zero-or-more matcher
//     */
//    static class CommutiveEllipsisTest2 extends CommutiveEllipsisTest {
//
//        CommutiveEllipsisTest2(String ellipsisTerm, String[] openClose) throws Narsese.NarseseException {
//            super(ellipsisTerm, openClose[0], openClose[1]);
//        }
//
//        @Override
//        public Set<Term> test(int arity, int repeats) throws Narsese.NarseseException {
//            Set<Term> s = super.test(arity, repeats);
//            Term the = s.isEmpty() ? null : s.iterator().next();
//            assertNotNull(the);
//            assertTrue(!the.toString().substring(1).isEmpty(), () -> the + " is empty");
//            assertTrue(the.toString().substring(1).charAt(0) == 'Z', () -> the + " does not begin with Z");
//            return s;
//        }
//
//        @Override
//        public @Nullable Compound getPattern(String prefix, String suffix) throws Narsese.NarseseException {
//            return $(prefix + ellipsisTerm + suffix);
//        }
//
//
//        @Override
//        public Term getResult() throws Narsese.NarseseException {
//            String s = prefix + "Z, " + ellipsisTerm + suffix;
//            Compound c = $(s);
//            assertNotNull(c, () -> s + " produced null compound");
//            return c;
//        }
//
//        @Override
//        public int getExpectedUniqueTerms(int arity) {
//            return arity;
//        }
//    }
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
//
//
//    @Test
//    void testEllipsisOneOrMore() throws Narsese.NarseseException {
//        String s = "%prefix..+";
//        Variable t = $(s);
//        assertNotNull(t);
//        assertEquals("%prefixU..+", t.toString());
//
//
//        assertEquals(EllipsisOneOrMore.class, t.normalizedVariable((byte) 1).getClass());
//
//
//    }
//
//    @Test
//    void testEllipsisZeroOrMore() throws Narsese.NarseseException {
//        String s = "%prefix..*";
//        Variable t = $(s);
//        assertNotNull(t);
//        assertEquals("%prefixU..*", t.toString());
//
//        Term tn = t.normalizedVariable((byte) 1);
//        assertEquals(EllipsisZeroOrMore.class, tn.getClass());
//        assertEquals("%1..*", tn.toString());
//        assertNotEquals($.varPattern(1), tn);
//    }
//
//
//
//
//
//
//    private static String[] p(String a, String b) {
//        return new String[]{a, b};
//    }
//
////    @Disabled @Test
////    void testVarArg0() throws Narsese.NarseseException {
////
////        String rule = "(%S ==> %M), ((&&,%S,%A..+) ==> %M) |- ((&&,%A..+) ==> %M), (Belief:DecomposeNegativePositivePositive, Order:ForAllSame, SequenceIntervals:FromBelief)";
////
////        Compound _x = $.$('<' + rule + '>');
////        assertTrue(_x instanceof PremiseRuleSource, _x.toString());
////        PremiseRuleSource x = (PremiseRuleSource) _x;
////
//////        x = new PremiseRuleProto(x, NARS.shell());
//////
//////
//////        assertEquals(
//////                "(((%1==>%2),((%1&&%3..+)==>%2)),(((&&,%3..+)==>%2),((DecomposeNegativePositivePositive-->Belief),(ForAllSame-->Order),(FromBelief-->SequenceIntervals))))",
//////                x.toString()
//////        );
////
////    }
//
////    @Test
////    void testEllipsisMatchCommutive1_0a() throws Narsese.NarseseException {
////        testSect("||");
////    }
//    @Test
//    void testEllipsisMatchCommutive1_0b() throws Narsese.NarseseException {
//        testSect("&&");
//    }
//
//    private static void testSect(String o) throws Narsese.NarseseException {
//        new CommutiveEllipsisTest1("%2..+", p('(' + o + ',', ")")).test(2, 2, 4);
//    }
//
//
//    @Test
//    void testEllipsisMatchCommutive1_1() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest1("%2..+", p("{", "}")).test(2, 4, 4);
//    }
//
//    @Test
//    void testEllipsisMatchCommutive1_2() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest1("%2..+", p("[", "]")).test(2, 4, 4);
//    }
//
//    @Test
//    void testEllipsisMatchCommutive1_3() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest1("%2..+", p("(&&,", ")")).test(2, 4, 4);
//    }
//
//    @Test
//    void testEllipsisMatchCommutive1_3with() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest1("%2..+", p("(&&,x,", ")")).test(2, 4, 4);
//    }
//
//
//    @Test
//    void testEllipsisMatchCommutive2one_sete() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest2("%1..+", p("{", "}")).test(1, 5, 0);
//    }
//
//
//    @Test
//    void testEllipsisMatchCommutive2one_seti() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest2("%1..+", p("[", "]")).test(1, 5, 0);
//    }
//
//    @Disabled
//    @Test
//    void testEllipsisMatchCommutive2one_prod() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest2("%1..+", p("(Z,", ")")).test(1, 5, 0);
//    }
//
//    @Disabled
//    @Test
//    void testEllipsisMatchCommutive2empty_prod() throws Narsese.NarseseException {
//        new EllipsisTest.CommutiveEllipsisTest2("%1..*", p("(Z,", ")")).test(0, 2, 0);
//    }
//
//
//    private static Set<String> testCombinations(Compound _X, Compound Y, int expect) {
//        Compound X = (Compound) PatternTermBuilder.rule(_X);
//
//        Set<String> results = new HashSet(0);
//        for (int seed = 0; seed < (expect+1)*(expect+1); seed++) {
//
//
//            Random rng = new XorShift128PlusRandom(seed);
//            Unify f = new Unify(VAR_PATTERN, rng, NAL.unify.UNIFICATION_STACK_CAPACITY, 128) {
//                @Override
//                public boolean match() {
//                    results.add(xy.toString());
//                    return true;
//                }
//            };
//
//            f.unify(X, Y);
//
//            //results.forEach(System.out::println);
//
//            assertEquals(expect, results.size(),
//                    ()->"insufficient permutations for: " + X + " .. " + Y);
//
//            results.clear();
//        }
//        return results;
//    }
//
//    @Test
//    void testEllipsisCombinatorics1() throws Narsese.NarseseException {
//
//        testCombinations(
//                $$$("(&&, %1..+, %2)"),
//                $$$("(&&, x, y, z)"),
//                3);
//    }
//
//    @Test
//    void testMatchAll2() throws Narsese.NarseseException {
//        testCombinations(
//                $$$("((|,%1,%2),(|,%2,%3))"),
//                $$$("((|,bird,swimmer),(|,animal,swimmer))"),
//                1);
//    }
//
//    @Test
//    void testMatchAll3() throws Narsese.NarseseException {
//        testCombinations(
//                $$$("((|,%X,%Z,%A) , (|,%Y,%Z,%A))"),
//                $$$("((|,bird,man, swimmer),(|,man, animal,swimmer))"),
//                2);
//    }
//
//    @Test
//    void testRepeatEllipsisAWithoutEllipsis() throws Narsese.NarseseException {
//        testCombinations(
//                $$$("((|,%X,%Y) ,(|,%Y,%Z))"),
//                $$$("((|,bird,swimmer),(|,animal,swimmer))"),
//                1);
//    }
//
//    @Test
//    void testRepeatEllipsisA() throws Narsese.NarseseException {
//
//        assertEq("(%1||%2..+)", "(|,%X,%A..+)");
//        assertNotEquals($$$("(|,%X,%A..+)"), $$$("(|,%Y,%A..+)"));
//        testCombinations(
//                $$$("((|,%X,%A..+) , (|,%Y,%A..+))"),
//                $$$("((|,x,common),(|,y,common))"),
//                1);
//    }
//
//    @Disabled @Test
//    void testRepeatEllipsisA2() throws Narsese.NarseseException {
//
//        testCombinations(
//                $$$("((%X,%A..+) , (%Y,%A..+))"),
//                $$$("((bird,swimmer),(animal,swimmer))"),
//                1);
//    }
//
//    @Test
//    void testRepeatEllipsisA0() throws Narsese.NarseseException {
//        testCombinations(
//                $$$("((%A, %X) --> (%B, %X))"),
//                $$$("((bird,swimmer)-->(animal,swimmer))"),
//                1);
//    }
//
//    @Test
//    void testRepeatEllipsisB() throws Narsese.NarseseException {
//
//
//        testCombinations(
//                $$$("((|,%X,%A..+) ,(|,%X,%B..+))"),
//                $$$("((|,bird,swimmer),(|,animal,swimmer))"),
//                1);
//    }
//
//    @Test
//    void testIntersection1() throws Narsese.NarseseException {
//        testCombinations(
//                $$$("(%M --> (|,%S,%A..+))"),
//                $$$("(m-->(|,s,a))"),
//                2);
//        testCombinations(
//                $$$("(%M --> (&,%S,%A..+))"),
//                $$$("(m-->(&,s,a))"),
//                2);
//    }
//    @Test
//    void conjEllipsisToConjSeq1() throws Narsese.NarseseException {
//        testCombinations(
//                $$$("(a &&+- %A..+)"),
//                $$$("((a &&+1 b) &&+1 c)"),
//                1);
//    }
//    @Test
//    void conjEllipsisToConjSeq2() {
//        testCombinations(
//                $$$("(%X &&+- %A..+)"),
//                $$$("((a &&+1 b) &&+1 c)"),
//                3);
//    }
//
//
////    @Test
////    void testEllipsisInMinArity() {
////        Atomic a = Atomic.the("a");
////        Ellipsis b = new EllipsisOneOrMore($.varPattern(1));
////
////        for (Op o : Op.values()) {
////            if (o.minSubs <= 1) continue;
////
////            if (o.statement) continue;
////
////
////            assertEquals(a, o.the(DTERNAL, a), o + " with normal target");
////
////            assertEquals(o.statement ? VAR_PATTERN : o,
////                    o.the(DTERNAL, b).op(),
////                    o + " with ellipsis not reduced");
////        }
////    }
//
//
//}