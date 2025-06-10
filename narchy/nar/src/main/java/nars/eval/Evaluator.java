package nars.eval;

import jcog.data.bit.MetalBitSet;
import jcog.data.map.UnifriedMap;
import jcog.data.set.ArrayHashSet;
import nars.$;
import nars.Op;
import nars.Term;
import nars.subterm.Subterms;
import nars.subterm.TermList;
import nars.term.Compound;
import nars.term.Functor;
import nars.term.Neg;
import nars.term.atom.Atom;
import nars.term.atom.Atomic;
import nars.term.builder.SimpleTermBuilder;
import nars.term.builder.TermBuilder;
import nars.term.util.transform.Replace;
import nars.term.util.transform.TermTransform;
import nars.term.var.Variable;
import org.jetbrains.annotations.Nullable;

import java.util.Set;
import java.util.function.Function;

import static nars.Op.EQ;
import static nars.Op.INH;
import static nars.term.atom.Bool.*;

public sealed class Evaluator permits CachedEvaluator {

    private final Function<Atom, Functor> funcResolver;

    public Evaluator(Function<Atom, Functor> funcResolver) {
        this.funcResolver = funcResolver;
    }

    public Evaluator clone() {
        return new Evaluator(funcResolver);
    }

    @Nullable
    final EvaluationPhase clauses(Compound x) {
        if (!Functor.evalable(x)) return null;

        var y = compile(x);
        return y == EvaluationPhase.NULL ? null : y;
    }

    EvaluationPhase compile(Compound x0) {
        var e = new ArrayHashSet<Compound>(8);

        var y = compile(x0, e);

        e.clear();

        return y;
    }

    private EvaluationPhase compile(Compound x0, ArrayHashSet<Compound> e) {
        var x = clauses(x0, e);
        return switch (x instanceof Atomic ? 0 : e.list.size()) {
            case 0 -> compile0(x0, x);
            case 1 -> compile1(e, x);
            default -> new EvaluationPhase.Recipe(x, e.list);
        };
    }

    private static EvaluationPhase compile0(Compound x0, Term x) {
        return x.equals(x0) ?
            EvaluationPhase.NULL :
            EvaluationPhase.Constant.the(x == True ? x0 : x);
    }

    private static EvaluationPhase compile1(ArrayHashSet<Compound> e, Term x) {
        var y0 = e.list.getFirst();
        return x.equals(y0) ?
                (x instanceof Compound c ?
                    new EvaluationPhase.Itself(c) :
                    EvaluationPhase.Constant.the(x)
                )
                :
                new EvaluationPhase.PartOfItself(x, y0);
    }

    private Term clauses(Compound x, ArrayHashSet<Compound> e) {
        var m = new UnifriedMap<Term, Term>(0);
        return clauseRemap(clauseFind(x, null, e, m), m, e);
    }

    private static Term clauseRemap(Term y, UnifriedMap<Term, Term> remap, ArrayHashSet<Compound> e) {
        while (!remap.isEmpty()) {
            var t = Replace.replace(remap, Evaluator.B);
            remap.clear();
            y = t.apply(y);
            if (!e.isEmpty())
                clauses(e, remap, t);
        }
        return y;
    }

    private static void clauses(ArrayHashSet<Compound> e, UnifriedMap<Term, Term> remap, TermTransform t) {
        var l = e.listIterator();
        while (l.hasNext()) {
            var ll = l.next();
            var z = t.apply(ll);
            if (z.EQ()) {
                var zz = equalConstantReduce(z.subtermsDirect(), remap);
                if (zz!=null)
                    z = zz;
            }

            if (z instanceof Compound Z)
                l.set(Z);
            else
                l.remove();
        }
    }

    private Term clauseFind(final Compound X, Term superTerm, Set<Compound> e, UnifriedMap<Term, Term> remap) {

        if (X instanceof Neg)
            return clauseFindNeg(X, e, remap);


        var xChanged = false;
        var yy = X.subtermsDirect();
        if (Functor.evalable(yy)) {
            var n = yy.subs();

            TermList xy = null;

            for (var i = n - 1; i >= 0; i--) {
                var x = yy.sub(i);
                if (Functor.evalable(x)) {

                    var z = clauseFind((Compound) x, X, e, remap);
                    if (z == Null)
                        return Null;

                    if (z!=x) {
                        xChanged = true;
                        x = z;
                    }
                }

                if (xy == null) {
                    if (!xChanged)
                        continue;

                    xy = clone(yy, n, i);
                }

                xy.addFast(x);
            }

            if (xChanged) {
                xy.reverseThis();
                yy = xy;
            }
        }

        return clauseFindResult(X, superTerm, e, remap, X.op(), yy, xChanged);
    }

    private static TermList clone(Subterms x, int n, int i) {
        var y = new TermList(n);
        for (var k = n - 1; k > i; k--) y.addFast(x.sub(k));
        return y;
    }

    private Term clauseFindResult(Compound x, Term superTerm, java.util.Set<Compound> e, UnifriedMap<Term, Term> remap, Op xOp, Subterms yy, boolean xChanged) {
        boolean evalable;

        Term xf = Functor.func(x);
        switch (xf) {
            case Functor functor -> evalable = true;
            case Atom xfa -> {
                evalable = false;
                var h = funcResolver.apply(xfa);
                if (h != null) {
                    if (!xChanged) {
                        yy = yy.toList();
                        xChanged = true;
                    }
                    ((TermList) yy).setFast(1, h);
                    evalable = true;
                }
            }
            case null, default -> evalable = false;
        }

        Term y = xChanged ? xOp.build(B, x.dt(), yy) : x;
        if (xChanged && evalable && !Functor.isFunc(y)) {
            evalable = false;
        }


        if (y.EQ()) {
            if (!(superTerm instanceof Neg)) {
                var z = equalConstantReduce(y.subtermsDirect(), remap);
                if (z != null)
                    return z;
            }
            evalable = true;
        }

        if (evalable && y instanceof Compound yc && yc.isAny(EQ.bit | INH.bit))
            e.add(yc);

        return y;
    }

    private Term clauseFindNeg(Compound x, java.util.Set<Compound> e, UnifriedMap<Term, Term> remap) {
        var ux = x.unneg();
        var uy = ux instanceof Compound uxc ? clauseFind(uxc, x, e, remap) : ux;
        return uy == ux ? x : uy.neg();
    }


    private static Term equalConstantReduce(Subterms x, UnifriedMap<Term, Term> remap) {
        assert(x.subs()==2);
        return equalConstantReduce(x.sub(0), x.sub(1), remap);
    }

    @Nullable
    private static Term equalConstantReduce(Term a, Term b, UnifriedMap<Term, Term> remap) {
        if (a.PROD() && b.PROD()) {
            var n = a.subs();
            if (n == b.subs())
                return equalConstantReduce((Compound)a, (Compound)b, remap, n);
        }

        return inlineAssignReturn(a, b, remap);
    }

    @Nullable
    private static Term equalConstantReduce(Compound a, Compound b, UnifriedMap<Term, Term> remap, int n) {
        var indeterminate = false;
        var eliminations = MetalBitSet.bits(n);

        for (var i = n - 1; i >= 0; i--) {
            switch (inlineAssign(a.sub(i), b.sub(i), remap)) {
                case -1 -> { return False; }
                case  0 -> indeterminate = true;
                case +1 -> eliminations.set(i);
            }
        }

        return indeterminate ?
            (eliminations.isEmpty() ? null : equalConstantReduceIndeterminate(a, b, remap, eliminations)) :
            True;
    }

    @Nullable private static Term inlineAssignReturn(Term a, Term b, UnifriedMap<Term, Term> remap) {
        return switch (inlineAssign(a, b, remap)) {
            case +1 -> True;
            case -1 -> False;
            default -> null;
        };
    }

    @Nullable private static Term equalConstantReduceIndeterminate(Compound a, Compound b, UnifriedMap<Term, Term> remap, MetalBitSet eliminations) {
        Term[] as = a.removing(eliminations), bs = b.removing(eliminations);
        var n = as.length;
        if (n!=bs.length)
            throw new UnsupportedOperationException();

        return (n == 1) ?
            equalConstantReduce(as[0], bs[0], remap)
            :
            equalConstantReduce($.pFast(as), $.pFast(bs), remap, n);
    }

    private static int inlineAssign(Term a, Term b, UnifriedMap<Term, Term> remap) {
        boolean aNeg = a instanceof Neg, bNeg = b instanceof Neg;
        if (aNeg == bNeg && a.equals(b))
            return +1;
        if (aNeg ^ bNeg && a.unneg().equals(b.unneg()))
            return -1;

        var v0 = (aNeg ? a.unneg() : a) instanceof Variable;
        var v1 = (bNeg ? b.unneg() : b) instanceof Variable;
        if (v0 || v1) {
            var ia = inlineAssign(a, b, remap, aNeg, bNeg, v0, v1);
            if (ia != -1) return ia;
        }

        return 0;
    }

    private static int inlineAssign(Term a, Term b, UnifriedMap<Term, Term> remap, boolean aNeg, boolean bNeg, boolean v0, boolean v1) {
        if (v0 && aNeg) {
            a = a.unneg();
            b = b.neg();
        } else if (v1 && bNeg) {
            b = b.unneg();
            a = a.neg();
        }
        if (v0 ^ v1) {
            var v = v0 ? b : a;
            if (!Functor.evalable(v))
                return inlineAssignConstant(v0 ? a : b, v, remap);
        }
        return -1;
    }

    private static int inlineAssignConstant(Term k, Term v, UnifriedMap<Term, Term> remap) {
        var vExist = remap.get(k);
        if (vExist!=null)
            return vExist.equals(v) ? +1 : -1;
        else {
            remap.put(k, v);
            return +1;
        }
    }


    @Nullable
    public final Term first(Compound x) {
        if (Functor.evalable(x)) {
            var f = new Evaluation.First();
            new Evaluation(this, f).eval(x);
            return f.the;
        }

        return null;
    }

    static final TermBuilder B =
        SimpleTermBuilder.the;

}