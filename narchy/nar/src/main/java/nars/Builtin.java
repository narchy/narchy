package nars;

import jcog.Str;
import jcog.data.map.UnifriedMap;
import nars.eval.Evaluation;
import nars.func.*;
import nars.subterm.Subterms;
import nars.table.BeliefTable;
import nars.table.TaskTable;
import nars.task.NALTask;
import nars.term.Compound;
import nars.term.Functor;
import nars.term.Termed;
import nars.term.atom.Atom;
import nars.term.atom.Bool;
import nars.term.atom.Int;
import nars.term.functor.AbstractInlineFunctor;
import nars.term.functor.AbstractInlineFunctor1;
import nars.term.functor.AbstractInlineFunctor2;
import nars.term.functor.LambdaFunctor;
import nars.term.obj.QuantityTerm;
import nars.term.util.Image;
import nars.term.util.Terms;
import nars.term.util.conj.Cond;
import nars.term.util.conj.CondDiff;
import nars.term.util.conj.ConjList;
import nars.term.util.transform.InstantFunctor;
import nars.term.util.var.DepIndepVarIntroduction;
import nars.term.var.Variable;
import nars.time.Tense;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeSet;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import static nars.Op.*;
import static nars.term.Functor.f0;
import static nars.term.atom.Bool.*;
import static nars.term.util.Image.imageNormalize;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Built-in set of default Functors and Operators, registered into a NAR on initialization
 * Provides the standard core function library
 * <p>
 * see:
 * https:
 */
public enum Builtin {
    ;

    public static final Functor[] statik = {

            Cmp.cmp,

            MathFunc.add,
            MathFunc.mul,
            MathFunc.div,
            MathFunc.pow,

            MathFunc.mod,
            MathFunc.max,
            MathFunc.min,
//            MathFunc.lte,
//            MathFunc.gte,
//            MathFunc.lt,
//            MathFunc.gt,

            //MathFunc.xor,

            Member.member,

            Replace.replace,


            SetFunc.intersect,
            SetFunc.differ,
            SetFunc.union,


            ListFunc.append,
            ListFunc.reverse,
            ListFunc.sub,
            ListFunc.subs,

            new AbstractInlineFunctor2("noEvent") {
                @Override protected Term apply(Term c, Term e) {
                    return c.equals(e) ||
                            c instanceof Compound C && C.condOf(e) ? Null : c;
                }
            },
            new AbstractInlineFunctor2("noEventPN") {
                @Override protected Term apply(Term c, Term e) {
                    return c.equalsPN(e) ||
                            (c instanceof Compound C) && C.condOf(e, 0) ? Null : c;
                }
            },

            new AbstractInlineFunctor1.AbstractInstantFunctor1("unneg") {
                @Override
                protected Term apply1(Term x) {
                    return x.unneg();
                }
            },

            new AbstractInlineFunctor1.AbstractInstantFunctor1("imageNormalize") {
                @Override
                protected Term apply1(Term x) {
                    return imageNormalize(x);
                }
            },

            Functor.f2Inline("imageInt", Image::imageInt),
            Functor.f2Inline("imageExt", Image::imageExt),


            new AbstractInlineFunctor2("eventOf") {
                @Override
                protected Term apply(Term conj, Term event) {
                    return conj instanceof Compound C && C.condOf(event) ? True : False;
                }
            },

            /** removes one matching event, chosen at random  */
            new AbstractInlineFunctor2(atom("condWithoutAny")) {
                @Override
                @Deprecated protected Term apply(Term conj, Term event) {
                    return CondDiff.diffAny(conj, event, false);
                }
            },

            /** removes one matching event, chosen at random, pos or negated */
            new AbstractInlineFunctor2(atom("condWithoutAnyPN")) {
                @Override
                @Deprecated protected Term apply(Term conj, Term event) {
                    return CondDiff.diffAny(conj, event, true);
                }
            },
            new AbstractInlineFunctor2(atom("condIntersect")) {
                @Override
                @Deprecated protected Term apply(Term x, Term y) {
                    return Cond.intersect(x,y);
                }
            },

            /** removes all matching events */
            new AbstractInlineFunctor2(atom("conjWithoutFirst")) {
                @Override
                protected Term apply(Term conj, Term event) {
                    return CondDiff.diffFirst(conj, event);
                }
            },
            new AbstractInlineFunctor2(atom("conjWithoutFirstPN")) {
                @Override
                protected Term apply(Term conj, Term event) {
                    return CondDiff.diffFirstPN(conj, event);
                }
            },
            new AbstractInlineFunctor2(atom("condWithoutAll")) {
                @Override
                protected Term apply(Term conj, Term event) {
                    return CondDiff.diffAll(conj, event);
                }
            },

            /** removes all matching events, pos or negated */
            new AbstractInlineFunctor2(atom("condWithoutAllPN")) {
                @Override
                protected Term apply(Term conj, Term event) {
                    return CondDiff.diffAllPN(conj, event);
                }
            },

//            new AbstractInlineFunctor1("conjWithoutIndepEvents") {
//                @Override
//                protected Term apply1(Term conj) {
//                    if (!conj.CONJ() || !conj.hasAny(VAR_INDEP)) return conj;//unchanged
//                    ConjList c = new ConjList();
//                    boolean xternal = conj.dt() == XTERNAL;
//                    ((Compound) conj).events((when, what) -> {
//                        if (!what.hasAny(VAR_INDEP))
//                            c.add(when, what);
//                    }, xternal ? TIMELESS : 0, true, xternal);
//                    return c.term();
//                }
//            },

            /** applies the changes in structurally similar terms "from" and "to" to the target target */
            Functor.f3((Atom) $.atomic("replaceDiff"), (target, from, to) -> {
                if (from.equals(to))
                    return Null;

                int n;
                if (from.opID() == to.opID() && (n = from.subs()) == to.subs()) {

                    Map<Term, Term> m = null;
                    for (int i = 0; i < n; i++) {
                        Term f = from.sub(i), t = to.sub(i);
                        if (!f.equals(t)) {
                            if (m == null) m = new UnifriedMap<>(1);
                            m.put(f, t);
                        }
                    }
                    if (m != null) {
                        Term y = target.replace(m);
                        if (y != null && !y.equals(target))
                            return y;
                    }
                }
                return Null;
            }),

            /** similar to C/Java "indexOf" but returns a set of all numeric indices where the 2nd argument occurrs as a subterm of the first
             *  if not present, returns Null
             * */

            Functor.f2("indicesOf", (x, y) -> {


                int s = x.subs();
                if (s > 0) {
                    TreeSet<Term> indices = null;
                    for (int i = 0; i < s; i++) {
                        if (x.sub(i).equals(y)) {
                            if (indices == null) indices = new TreeSet<>();
                            indices.add(Int.i(i));
                        }
                    }
                    return indices == null ? Null : SETe.the(indices);
                }
                return Null;
            }),
            Functor.f2("keyValues", (x, y) -> {


                int s = x.subs();
                if (s > 0) {
                    TreeSet<Term> indices = null;
                    for (int i = 0; i < s; i++) {
                        if (x.sub(i).equals(y)) {
                            if (indices == null) indices = new TreeSet<>();
                            indices.add($.p(y, Int.i(i)));
                        }
                    }
                    if (indices == null)
                        return Null;
                    else {
                        return switch (indices.size()) {
                            case 0 -> Null;
                            case 1 -> indices.first();
                            default -> SETe.the(indices);
                        };
                    }
                }
                return Null;
            }),

            Functor.f2("varMask", (x, y) -> {
                int s = x.subs();
                if (s > 0) {
                    Term[] t = IntStream.range(0, s).mapToObj(i -> x.sub(i).equals(y) ? y : $.varDep("z" + i)).toArray(Term[]::new);
                    return $.p(t);
                }
                return Null;
            }),


            Functor.f1Const("reflect", reflect::reflect),


            Functor.f1Const("toString", x -> $.quote(x.toString())),
            Functor.f1Const("toChars", x -> $.p(x.toString().toCharArray(), $::the)),
            Functor.f1Const("complexity", x -> $.the(x.complexity())),

            flat.flatProduct,

            Functor.f2("similaritree", (a, b) ->
                    a instanceof Variable || b instanceof Variable ? null :
                            $.the(Str.levenshteinDistance(a.toString(), b.toString()))
            ),



            Functor.f2("ifThen", (condition, conseq) -> {
                if (!condition.equals(True)) {
                    if (condition == Null)
                        return Null;
                    return null;
                } else
                    return conseq;
            }),

            Functor.f3("ifThenElse", (condition, ifTrue, ifFalse) -> {
                if (!(condition instanceof Bool))
                    return null;
                if (condition == True)
                    return ifTrue;
                if (condition == False)
                    return ifFalse;

                return Null;
            }),

            Functor.f3("ifOrElse", (condition, conseqTrue, conseqFalse) -> {
                if (condition.hasVars())
                    return null;
                if (condition.equals(True))
                    return conseqTrue;
                if (condition.equals(False))
                    return conseqFalse;

                return Null;
            }),

            Functor.f2("ifNeqRoot", (returned, compareTo) ->
                    returned.equalsRoot(compareTo) ? Null : returned
            ),


            Functor.f2("subterm", (x, index) -> {
                try {
                    if (index.INT())
                        return x.sub($.intValue(index));
                } catch (NumberFormatException ignored) {
                }
                return null;
            }),

            Functor.f1Inline("quote", x -> x),

            Functor.f1Inline("condLatest", (Term x) -> {
                if (!x.SEQ())
                    return x;

                try (var e = ConjList.conds(x, false, false)) {
                    long l = e.whenLatest();
                    e.removeIf((w, z) -> w != l);
                    return e.term();
                }
            }),

            /** warning: this returns Null if unchanged */
            new AbstractInlineFunctor2("withoutPN") {
                @Override
                protected Term apply(Term container, Term _content) {
                    Term content = _content.unneg();
                    Term y = Terms.withoutAll(container, content::equalsPN);
                    return y == null || y.equals(container) ? Null : y;
                }
            },

            /** warning: this returns Null if unchanged */
            new AbstractInlineFunctor1("negateConds") {
                @Override protected Term apply1(Term x) {
                    return Cond.negateConds(x,
                        true
                        //!x.SEQ()
                        //false
                    );
                }
            },

            /** warning: this returns Null if unchanged */
            new AbstractInlineFunctor2("without") {
                @Override
                protected Term apply(Term container, Term content) {
                    Term y = Terms.withoutAll(container, content::equals);
                    return y == null || y.equals(container) ? Null : y;
                }
            },

            /** warning: this returns Null if unchanged */
            new AbstractInlineFunctor2("unsect") {
                @Override
                protected Term apply(Term container, Term content) {
                    Term y = Terms.withoutAll(container, content.CONJ() ? ((Compound) content).contains(false, false) : content::equals);
                    return y == null || y.equals(container) ? Null : y;
                }
            },

            Functor.f("slice", args -> {
                if (args.subs() != 2)
                    return null;

                Term x = args.sub(0);
                if (x.subs() > 0) {
                    int len = x.subs();

                    Term index = args.sub(1);
                    Op o = index.op();
                    if (o == INT) {

                        int i = Int.i(index);
                        return i >= 0 && i < len ? x.sub(i) : False;

                    } else if (o == PROD && index.subs() == 2) {
                        Term start = index.sub(0);
                        if (start.INT()) {
                            Term end = index.sub(1);
                            if (end.INT()) {
                                int si = Int.i(start);
                                if (si >= 0 && si < len) {
                                    int ei = Int.i(end);
                                    if (ei >= 0 && ei <= len) {
                                        if (si == ei)
                                            return EmptyProduct;
                                        if (si < ei) {
                                            return $.p(Arrays.copyOfRange(x.subterms().arrayClone(), si, ei));
                                        }
                                    }
                                }

                                return False;
                            }
                        }

                    }
                }
                return null;
            })
    };

//    private static Term nullIfEq(Term mustNotEqual, Term result) {
//        return mustNotEqual.equals(result) ? Null : result;
//    }

    private static final Map<Term, Functor> statiks;

    static {
        UnifriedMap<Term, Functor> s = new UnifriedMap<>(statik.length);
        for (Functor f : statik) s.put(f.term(), f);
        s.trimToSize();
        statiks = s;
    }

    public static @Nullable Functor functor(Term x) {
        return statiks.get(x);
    }

    private static void registerFunctorsStatic(NAR nar) {
        for (Functor t : statik)
            nar.add(t);
    }

    /**
     * instantiates NAR-local functors
     */
    private static void registerFunctorsDynamic(NAR nar) {
        nar.add(SetFunc.sort());

        /** dynamic target builder - useful for NAR specific contexts like clock etc.. */
        nar.add(new TermDynamic(nar));

        /** applies # dep and $ indep variable introduction if possible. returns the input term otherwise  */
        nar.add(Functor.f1Inline("varIntro", x -> {
            if (!(x instanceof Compound c)) return Null;
            Term result = DepIndepVarIntroduction.the.apply(c, nar.random(), null);
            return result != null ? result : Null;
        }));

/** TODO
         * replay(timeStart, timeEnd, [belief|goal|question] [pos|neg]) [!;]
         * functor context parameter to subsume Operator API as functors that trigger
         * async reactions when evaluated in imperative contexts
         */
        nar.add(new AbstractInlineFunctor2("replay") {

            @Override
            protected Term apply(Term a, Term b) {
                return null;
            }
        });


        nar.add(f1Table("beliefTruth", nar, BELIEF, (c, n) -> {
            long when = n.time();
            return $.quote(((BeliefTable) c).truth(when, when, n));
        }));
//        nar.on(Functor.f1Concept("goalTruth", nar, (c, n) -> $.quote(n.goal(c, n.time()))));

        nar.add(f0("self", nar::self));

        nar.add(Functor.f1("the", what -> {


            if (what instanceof Atom) {
                switch (what.toString()) {
                    case "sys" -> {
                        return $.p(
                                $.quote(nar.emotion.summary()),
                                $.quote(nar.memory.summary()),
                                $.quote(nar.emotion.summary()),
                                $.quote(nar.exe.toString())
                        );
                    }
                }
            }

            Termed x = nar.concept(what);
            if (x == null)
                x = what;

            return $.quote($.p($.quote(x.getClass().toString()), $.quote(x.toString())));
        }));


    }

    private static void registerOperators(NAR nar) {

        nar.add(atom("task"), (x, nn) -> {
            //TODO punctuation parameter: Task.BeliefAtom, Task.GoalAtom etc
            NALTask t = NALTask.taskEternal(x.term().sub(0).sub(0), BELIEF, $.t(1.0f, nn.confDefault(BELIEF)), nn);
            t.withPri(nn.priDefault(BELIEF));
            nn.input(t);
            return null;
        });

        nar.addOp1("assertTrue", (x, nn) -> {
            if (x.hasVars()) //???
                assertSame(True, x);
        });

        //TODO separate into assert and equals functors
        nar.addOp2("assertEquals", (x, y, nn) -> {
            if (!x.equals(y) && !x.hasVars() && !y.hasVars())
                assertEquals(/*msg,*/ x, y);
        });

    }


    public static void init(NAR nar) {
        registerFunctorsStatic(nar);
        registerFunctorsDynamic(nar);
        registerOperators(nar);
    }

    public static LambdaFunctor f1Table(String termAtom, NAR nar, byte punc, BiFunction<TaskTable, NAR, Term> ff) {
        assert punc == BELIEF || punc == GOAL;
        boolean beliefOrGoal = punc == BELIEF;
        return Functor.f1(atom(termAtom), t -> {
            TaskTable c = nar.table(t, beliefOrGoal);
            return c != null ? ff.apply(c, nar) : null;
        });
    }

    private static final class TermDynamic extends AbstractInlineFunctor implements InstantFunctor<Evaluation> {

        private final NAR nar;

        TermDynamic(NAR nar) {
            super(TERM);
            this.nar = nar;
        }

        @Override
        public Term apply(Evaluation e, Subterms s) {
            Term opTerm = s.sub(0);
            Op o = Op.op((Atom) opTerm);
            if (o == null)
                return Null; //TODO throw

            int dt;
            if (s.subs() == 3) {
                if (!o.temporal)
                    throw new UnsupportedOperationException("unrecognized modifier argument: " + s);

                Term dtTerm = s.sub(2);
                if (!(dtTerm instanceof QuantityTerm)) {
                    dtTerm = QuantityTerm.the(dtTerm);
                    if (dtTerm == null)
                        return Null;
                }

                dt = Tense.occToDT(nar.time.toCycles(((QuantityTerm) dtTerm).quant));

            } else
                dt = DTERNAL;

            return o.the(dt, s.sub(1).subterms().arrayShared());
        }
    }



}
/*
Below is a comprehensive table listing both **existing functors** from the `Builtin` class in the `nars` package and **planned functors** suggested earlier. Each functor is accompanied by a simple explanation of its functionality. The table is organized by category for clarity.

---

| **Category**               | **Functor Name**       | **Existing (E) / Planned (P)** | **Functionality**                                                                                   |
|----------------------------|------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------|
| **Mathematical Operations**| `add`                  | E                              | Adds two numbers (e.g., `add(2, 3) = 5`).                                                          |
|                            | `mul`                  | E                              | Multiplies two numbers (e.g., `mul(2, 3) = 6`).                                                    |
|                            | `div`                  | E                              | Divides one number by another (e.g., `div(6, 2) = 3`).                                             |
|                            | `pow`                  | E                              | Raises a number to a power (e.g., `pow(2, 3) = 8`).                                                |
|                            | `mod`                  | E                              | Computes the remainder of division (e.g., `mod(7, 3) = 1`).                                        |
|                            | `max`                  | E                              | Returns the larger of two numbers (e.g., `max(3, 5) = 5`).                                         |
|                            | `min`                  | E                              | Returns the smaller of two numbers (e.g., `min(3, 5) = 3`).                                        |
|                            | `sqrt`                 | P                              | Computes the square root of a number (e.g., `sqrt(25) = 5`).                                       |
|                            | `log`                  | P                              | Computes the natural logarithm (e.g., `log(2.718) ≈ 1`).                                           |
|                            | `exp`                  | P                              | Computes the exponential (e.g., `exp(1) ≈ 2.718`).                                                 |
|                            | `round`                | P                              | Rounds a number to the nearest integer (e.g., `round(3.7) = 4`).                                   |
| **Statistical Operations** | `mean`                 | P                              | Calculates the average of a set (e.g., `mean({1, 2, 3}) = 2`).                                     |
|                            | `median`               | P                              | Finds the middle value of a set (e.g., `median({1, 2, 3}) = 2`).                                   |
|                            | `variance`             | P                              | Computes the variance of a set (e.g., `variance({1, 2, 3}) ≈ 0.667`).                              |
|                            | `correlation`          | P                              | Calculates correlation between two lists (e.g., `correlation({1, 2}, {2, 4}) = 1`).                |
| **String Processing**      | `toString`             | E                              | Converts a term to a quoted string (e.g., `toString(A) = "A"`).                                    |
|                            | `toChars`              | E                              | Converts a term to a list of characters (e.g., `toChars("ab") = ("a", "b")`).                      |
|                            | `similaritree`         | E                              | Computes Levenshtein distance between strings (e.g., `similaritree("cat", "bat") = 1`).            |
|                            | `concat`               | P                              | Concatenates two strings (e.g., `concat("hello", "world") = "helloworld"`).                        |
|                            | `substring`            | P                              | Extracts a substring (e.g., `substring("hello", 1, 4) = "ell"`).                                   |
|                            | `split`                | P                              | Splits a string by a delimiter (e.g., `split("a,b", ",") = ("a", "b")`).                           |
|                            | `regexMatch`           | P                              | Checks if a string matches a regex (e.g., `regexMatch("cat", "c.t") = True`).                      |
| **Temporal Operations**    | `condLatest`           | E                              | Extracts the latest event from a sequence (e.g., `condLatest((A&&B)) = B` if B is latest).         |
|                            | `timeShift`            | P                              | Shifts a term’s time (e.g., `timeShift(A, 5) = A shifted by 5 units`).                             |
|                            | `duration`             | P                              | Computes time difference (e.g., `duration(A@5, B@10) = 5`).                                        |
|                            | `sequenceConcat`       | P                              | Combines sequences (e.g., `sequenceConcat((A, B), (C)) = (A, B, C)`).                              |
|                            | `sequenceContains`     | P                              | Checks if a sequence contains a term (e.g., `sequenceContains((A, B), B) = True`).                 |
| **Logical Operations**     | `ifThen`               | E                              | Returns a term if condition is true (e.g., `ifThen(True, A) = A`).                                 |
|                            | `ifThenElse`           | E                              | Chooses between two terms based on condition (e.g., `ifThenElse(True, A, B) = A`).                 |
|                            | `ifOrElse`             | E                              | Similar to ifThenElse, stricter on vars (e.g., `ifOrElse(False, A, B) = B`).                       |
|                            | `eventOf`              | E                              | Checks if an event is in a conjunction (e.g., `eventOf((A&&B), A) = True`).                        |
|                            | `switch`               | P                              | Selects from cases (e.g., `switch(2, ((1, A), (2, B))) = B`).                                      |
|                            | `forall`               | P                              | Verifies a condition for all (e.g., `forall({1, 2}, >0) = True`).                                  |
|                            | `exists`               | P                              | Checks if a condition holds for any (e.g., `exists({0, 1}, >0) = True`).                           |
|                            | `filter`               | P                              | Filters elements by condition (e.g., `filter({1, 2, 3}, >1) = {2, 3}`).                            |
| **Term Manipulation**      | `unneg`                | E                              | Removes negation (e.g., `unneg(--A) = A`).                                                         |
|                            | `imageNormalize`       | E                              | Normalizes image-like terms (e.g., `imageNormalize(A:[B]) = A`).                                   |
|                            | `imageInt`             | E                              | Creates an internal image (e.g., `imageInt(A, B) = A:[B]`).                                        |
|                            | `imageExt`             | E                              | Creates an external image (e.g., `imageExt(A, B) = A:[B]`).                                        |
|                            | `replace`              | E                              | Replaces subterms based on a pattern (e.g., `replace(A, B, C) = A with B->C`).                     |
|                            | `subterm`              | E                              | Extracts a subterm by index (e.g., `subterm((A, B), 1) = B`).                                      |
|                            | `without`              | E                              | Removes a subterm (e.g., `without((A, B), A) = B`).                                                |
|                            | `withoutPN`            | E                              | Removes a subterm (pos/neg) (e.g., `withoutPN((A, --A), A) = Null`).                               |
|                            | `unsect`               | E                              | Removes conjunction-like terms (e.g., `unsect((A&&B), A) = B`).                                    |
|                            | `slice`                | E                              | Extracts a range of subterms (e.g., `slice((A, B, C), (1, 2)) = (B)`).                             |
|                            | `varMask`              | E                              | Masks subterms with variables (e.g., `varMask((A, B), A) = (A, $z1)`).                             |
|                            | `complexity`           | E                              | Returns term complexity (e.g., `complexity((A&&B)) = 3`).                                          |
|                            | `flatProduct`          | E                              | Flattens a product term (e.g., `flatProduct(((A, B), C)) = (A, B, C)`).                            |
|                            | `subterms`             | P                              | Returns all subterms (e.g., `subterms((A&&B)) = {A, B, (A&&B)}`).                                  |
|                            | `depth`                | P                              | Measures nesting depth (e.g., `depth(((A, B), C)) = 2`).                                           |
|                            | `map`                  | P                              | Applies a function to subterms (e.g., `map(+1, (1, 2)) = (2, 3)`).                                 |
|                            | `fold`                 | P                              | Reduces subterms (e.g., `fold(+, 0, (1, 2, 3)) = 6`).                                              |
| **Set Operations**         | `intersect`            | E                              | Finds common elements (e.g., `intersect({A, B}, {B, C}) = {B}`).                                   |
|                            | `differ`               | E                              | Finds differing elements (e.g., `differ({A, B}, {B, C}) = {A}`).                                   |
|                            | `union`                | E                              | Combines sets (e.g., `union({A}, {B}) = {A, B}`).                                                  |
| **List Operations**        | `append`               | E                              | Adds an element to a list (e.g., `append((A, B), C) = (A, B, C)`).                                 |
|                            | `reverse`              | E                              | Reverses a list (e.g., `reverse((A, B)) = (B, A)`).                                                |
|                            | `sub`                  | E                              | Extracts a sublist (e.g., `sub((A, B, C), 1) = B`).                                                |
|                            | `subs`                 | E                              | Returns number of subterms (e.g., `subs((A, B)) = 2`).                                             |
| **Knowledge Queries**      | `beliefTruth`          | E                              | Gets truth value of a belief (e.g., `beliefTruth(A) = <0.8, 0.9>`).                                |
|                            | `self`                 | E                              | Returns the NAR’s self term (e.g., `self() = SELF`).                                                |
|                            | `the`                  | E                              | Quotes a term or concept info (e.g., `the(A) = "A"`).                                              |
|                            | `conceptExists`        | P                              | Checks if a concept exists (e.g., `conceptExists(A) = True`).                                      |
|                            | `beliefCount`          | P                              | Counts beliefs for a term (e.g., `beliefCount(A) = 3`).                                            |
|                            | `taskHistory`          | P                              | Retrieves task history (e.g., `taskHistory(A, 0, 10) = (T1, T2)`).                                 |
| **Probabilistic**          | `prob`                 | P                              | Extracts frequency from truth (e.g., `prob(<0.8, 0.9>) = 0.8`).                                    |
|                            | `conf`                 | P                              | Extracts confidence from truth (e.g., `conf(<0.8, 0.9>) = 0.9`).                                   |
|                            | `randomTerm`           | P                              | Picks a random subterm (e.g., `randomTerm((A, B)) = B`).                                           |
| **Meta-Functors**          | `quote`                | E                              | Returns input unchanged (e.g., `quote(A) = A`).                                                    |
|                            | `reflect`              | E                              | Reflects a term (e.g., `reflect(A) = A’s structure`).                                              |
|                            | `compose`              | P                              | Combines functors (e.g., `compose(+1, *2)(3) = 8`).                                                |
|                            | `apply`                | P                              | Applies a functor (e.g., `apply(+1, (2)) = 3`).                                                    |
| **Debugging**              | `print`                | P                              | Prints a term and returns it (e.g., `print(A) = A`).                                               |
|                            | `trace`                | P                              | Logs a message with a term (e.g., `trace("step", A) = A`).                                         |
| **Conjunction Handling**   | `noEvent`              | E                              | Returns `Null` if term is an event in conj (e.g., `noEvent((A&&B), A) = Null`).                     |
|                            | `noEventPN`            | E                              | Similar to `noEvent`, considers pos/neg (e.g., `noEventPN((A&&--A), A) = Null`).                    |
|                            | `condWithoutAny`       | E                              | Removes one matching event (e.g., `condWithoutAny((A&&B), A) = B`).                                |
|                            | `condWithoutAnyPN`     | E                              | Removes one event, pos/neg (e.g., `condWithoutAnyPN((A&&--A), A) = --A`).                          |
|                            | `condIntersect`        | E                              | Intersects conjunctions (e.g., `condIntersect((A&&B), (B&&C)) = B`).                               |
|                            | `conjWithoutFirst`     | E                              | Removes first matching event (e.g., `conjWithoutFirst((A&&B), A) = B`).                            |
|                            | `conjWithoutFirstPN`   | E                              | Removes first event, pos/neg (e.g., `conjWithoutFirstPN((A&&--A), A) = --A`).                      |
|                            | `condWithoutAll`       | E                              | Removes all matching events (e.g., `condWithoutAll((A&&A), A) = Null`).                            |
|                            | `condWithoutAllPN`     | E                              | Removes all events, pos/neg (e.g., `condWithoutAllPN((A&&--A), A) = Null`).                        |
|                            | `negateConds`          | E                              | Negates conditions in a conj (e.g., `negateConds(A&&B) = (--A&&--B)`).                             |
| **Dynamic Functors**       | `sort`                 | E                              | Sorts a set or list dynamically (e.g., `sort({3, 1, 2}) = {1, 2, 3}`).                            |
|                            | `varIntro`             | E                              | Introduces variables (e.g., `varIntro((A, B)) = (#1, B)`).                                         |
|                            | `replay`               | E                              | Placeholder for replaying tasks (currently returns `Null`).                                        |

---

### Notes
- **Existing Functors (E)**: These are already implemented in the `Builtin` class as per the provided code.
- **Planned Functors (P)**: These are proposed additions to enhance NARS's functionality.
- **Functionality**: Each explanation is kept concise, focusing on the core purpose and typical behavior.

This table provides a unified view of NARS's current and potential capabilities, aiding developers in understanding the system’s scope and planning future enhancements. Let me know if you’d like further details or assistance with implementing any of these functors!
 */