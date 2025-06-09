package jcog.lab;

import jcog.Log;
import jcog.Util;
import jcog.WTF;
import jcog.data.list.Lst;
import jcog.decision.TableDecisionTree;
import jcog.lab.var.FloatVar;
import jcog.optimize.BayesOptimizer;
import jcog.optimize.MyCMAESOptimizer;
import jcog.table.ARFF;
import jcog.table.DataTable;
import org.hipparchus.exception.MathIllegalStateException;
import org.hipparchus.optim.*;
import org.hipparchus.optim.nonlinear.scalar.GoalType;
import org.hipparchus.optim.nonlinear.scalar.ObjectiveFunction;
import org.hipparchus.optim.nonlinear.scalar.noderiv.MultiDirectionalSimplex;
import org.hipparchus.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.hipparchus.util.MathArrays;
import org.slf4j.Logger;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

import java.io.File;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

import static java.util.stream.Collectors.toList;

/**
 * procedure optimization context
 * sets controls, runs an experiment procedure, records the observation.
 *
 * @param S subject of the experiment
 * @param X experiment containing the subject
 *          S and X may be the same type if no distinction is necessary
 *          <p>
 *          goal (score) is in column 0 and it assumed to be maximized. for minimized, negate model's score function
 *          <p>
 *          TODO - max timeout parameter w/ timer that kills if time exceeded
 *          <p>
 *          https://ax.dev/docs/core.html
 */
public class Optimize<S, X> extends Lab<X> {

    private static final Logger logger = Log.log(Optimize.class);
    /**
     * active variables
     */
    public final List<Var<S, ?>> out;
    /**
     * objective
     */
    public final Goal<X> goal;
    private final Experiment.DataTableTarget dataTarget;
    private final Supplier<S> seed;
    private final Function<Supplier<S>, X> expBuilder;
    /**
     * active sensors
     */
    private final Lst<Sensor<X, ?>> in = new Lst();

    /**
     * internal feedback: out -> in
     */
    private final List<Sensor<S, ?>> varSensors;
    private final OptimizationStrategy strat;

    /**
     * history of experiments. TODO use ranking like TopN etc
     */
    public DataTable data = new ARFF();
    private double[] inc;
    private double[] min;
    private double[] max;
    public double[] mid;

    public Optimize(Supplier<S> seed,
                    Function<Supplier<S>, X> expBuilder,
                    Goal<X> g,
                    List<Var<S, ?>> v /* unsorted */,
                    List<Sensor<X, ?>> s,
                    OptimizationStrategy strat) {
        this.seed = seed;
        this.strat = strat;

        this.dataTarget = new Experiment.DataTableTarget(data);
        for (Var vv : v)
            vars.put(vv.id, vv);

        this.goal = g;

        this.out = new Lst<>(v).sortThis();

        this.varSensors = out.stream().map(Var::sense).collect(toList());

        this.in.addAll(s);
        this.in.sortThis();

        this.expBuilder = expBuilder;
    }

    public static TableDecisionTree tree(Table data, int discretization, int maxDepth, int predictColumn) {
        return data.isEmpty() ? null :
                new TableDecisionTree(data.copy(),
                        predictColumn /* score */, maxDepth, discretization);
    }

    /** runs iterations synchronously */
    public Optimize<S, X> run(@Deprecated int iters) {
        run(strat);
        return this;
    }

    private void run(OptimizationStrategy strategy) {

        //initialize numeric or numeric-able variables
        var numVars = out.size();

        mid = new double[numVars];
        min = new double[numVars];
        max = new double[numVars];
        inc = new double[numVars];

        var example = seed.get();
        var i = 0;
        for (var w : out) {
            var s = (FloatVar) w;

            var guess = s.get(example);

            var mi = min[i] = s.getMin();
            var ma = max[i] = s.getMax();
            var inc = this.inc[i] = s.getInc();

            if (guess != null && (mi != mi || ma != ma || inc != inc)) {
                var x = (float) guess;
                //HACK assumption
                mi = min[i] = x / 2;
                ma = max[i] = x * 2;
                this.inc[i] = x / 4;
            }

            mid[i] = guess != null ? Util.clamp((float) guess, mi, ma) : (mi + ma) / 2f;

            if ((mid[i] < min[i]) || (max[i] < mid[i])) throw new WTF();

            i++;
        }

        synchronized (data) {

            sense(goal);
            goal.register(dataTarget);

            for (var s : varSensors) {
                sense(s);
                s.register(dataTarget);
            }
            for (var s : in) {
                sense(s);
                s.register(dataTarget);
            }

            if (logger.isTraceEnabled()) {
                var s = data.columnNames().toString();
                logger.trace("{}", s.substring(1, s.length() - 1));
            }
        }

        strategy.run(this);

    }

    public double eval(double[] point) {

        try {
            Object[] subject = new Object[1], experiment = new Object[1];
            Supplier<X> y = () -> (X)(experiment[0] = expBuilder.apply(() -> {
                var s = subject(seed.get(), point);
                subject[0] = s; //for measurement
                return s;
            }));

            var score = score(y);

            var row = row((S) subject[0], (X) experiment[0], score);

            synchronized (data) {
                data.add(row);
            }

            if (logger.isTraceEnabled()) {
                var rs = Arrays.toString(row);
                logger.trace("{}", rs.substring(1, rs.length() - 1));
            }

            return score;

        } catch (RuntimeException t) {
            logger.error("", t);
            return Double.NaN;
        }

    }

    private double score(Supplier<X> y) {
        var s = goal.apply(y).doubleValue();
        if (s!=s) s = Double.NEGATIVE_INFINITY;
        return s;
    }

    private Object[] row(S x, X y, double score) {
        var row = new Object[1 + varSensors.size() + in.size()];
        var j = 0;
        row[j++] = score;
        for (Sensor v : varSensors) row[j++] = v.apply(x);
        for (Sensor s : in) row[j++] = s.apply(y);
        return row;
    }

    /**
     * builds an experiment subject (input)
     * TODO handle non-numeric point entries
     */
    private S subject(S x, double[] point) {
        for (int i = 0, dim = point.length; i < dim; i++)
            point[i] = ((Var<S, Float>) out.get(i)).set(x, (float) point[i]);
        return x;
    }

    public Row best() {
        return sorted().iterator().next();
    }

    public DataTable sorted() {
        return new DataTable(data.sortDescendingOn(data.column(0).name()));
    }

    public Optimize<S, X> print() {
        return print(System.out);
    }

    public Optimize<S, X> print(PrintStream out) {
        out.println(data.print(Integer.MAX_VALUE));
        return this;
    }

    public TableDecisionTree tree(int discretization, int maxDepth, int predictColumn) {
        return tree(data, discretization, maxDepth, predictColumn);
    }

    /**
     * string representing the variables manipulated in this experiment
     */
    public String varKey() {
        return out.toString();
    }

    public TableDecisionTree tree(int discretization, int maxDepth) {
        return tree(discretization, maxDepth, 0);
    }

    public Optimize report() {

        var e = this;
//        for (int d = 2; d < 7; d++) {
//            var t = e.tree(4, d);
//            if (t!=null) {
//                t.print();
//                t.printExplanations();
//                System.out.println();
//            }
//        }

        var data = e.data.sortDescendingOn(e.goal.id);

        var f = new File("/tmp/x." + System.currentTimeMillis() + ".csv");
        data.write().csv(f);
        System.out.println("written: " + f.getAbsolutePath());

        System.out.println(data.printAll());
        data.write().csv(System.out);
        return this;
    }

    public abstract static class OptimizationStrategy {
        public abstract void run(Optimize eOptimize);
    }

    private abstract static class ApacheCommonsMathOptimizationStrategy extends OptimizationStrategy {

        @Override
        public final void run(Optimize o) {
            run(o, new ObjectiveFunction(o::eval));
        }

        protected abstract void run(Optimize o, ObjectiveFunction func);
    }

    public static class SimplexOptimizationStrategy extends ApacheCommonsMathOptimizationStrategy {
        private final int maxIter;

        public SimplexOptimizationStrategy(int maxIter) {
            this.maxIter = maxIter;
        }

        @Override
        protected void run(Optimize o, ObjectiveFunction func) {

            var dim = o.inc.length;
            var step = new double[dim];
//            var init = new double[dim];
            for (var i = 0; i < dim; i++) {
                var min = o.min[i];
                var range = o.max[i] - min;
                step[i] = o.inc[i];
//                init[i] =
//                    Fuzzy.mean(o.min[i], o.max[i]);
//                    //min + (ThreadLocalRandom.current().nextFloat() * range);
            }
            var init = o.mid;

            var so = new SimplexOptimizer(
                NullConvergenceChecker
            );
            try {
                so.optimize(
                    func,
                    GoalType.MAXIMIZE,
                    new MultiDirectionalSimplex(step),
                        //new NelderMeadSimplex(step)
                    new InitialGuess(init),
                    new MaxEval(maxIter)
                );
            } catch (MathIllegalStateException e) {
                if (so.getEvaluations()>=maxIter) {
                    //done. ignore, this is apparently normal for Commons Math/Hipparchus's Optimizer API :(
                } else
                    throw new RuntimeException(e);
            }
        }

        @Deprecated private static final ConvergenceChecker<PointValuePair> NullConvergenceChecker = (iteration, previous, current) -> false;

    }

    /**
     * TODO rewrite using MyAsyncCMAESOptimizer's iterator() ?
     */
    public static class CMAESOptimizationStrategy extends ApacheCommonsMathOptimizationStrategy {
        private final int maxIter;

        public CMAESOptimizationStrategy(int maxIter) {
            this.maxIter = maxIter;
        }

        private static double[] sigma(Optimize o) {
            return MathArrays.scale(2, o.inc);
        }

        @Override
        protected void run(Optimize o, ObjectiveFunction func) {
            var popSize = population(o);
            var maxIter = iter(popSize);
            new MyCMAESOptimizer(maxIter, Double.NaN, popSize, sigma(o)).optimize(func,
                    GoalType.MAXIMIZE,
                    new InitialGuess(o.mid),
                    new SimpleBounds(o.min, o.max),
                    new MaxEval(maxIter)
            );
        }

        /**
         * Default population size:
         * Logarithmic Scaling: The logarithmic term log(n) helps in moderating the increase in population size as the dimensionality increases. This is based on the observation that the complexity of the search space increases exponentially with the number of dimensions, but the population does not need to grow at the same rate to effectively explore the space.
         * Constants: The constants 4 and 3 in the formula are empirically chosen to provide a balance between exploration (having a diverse set of candidate solutions) and exploitation (focusing on promising areas of the search space).
         */
        private int population(Optimize o) {
            var popSize =
                    4 + (int) (3 * Math.log(o.out.size()));
            //(int) Math.ceil(4 + 3 * Math.log(o.out.size()));

            return Math.min(popSize, maxIter);
        }

        /* must be perfectly divisible by population size */
        private int iter(int popSize) {
            return (int) Util.round(maxIter, popSize);
        }
    }

    public static class BayesianOptimizationStrategy extends OptimizationStrategy {

        private final int iterMax;

        public BayesianOptimizationStrategy(int iter) {
            super();
            this.iterMax = iter;
        }

        @Override
        public void run(Optimize o) {
            var b = new BayesOptimizer(Math.min(128, iterMax), o.min, o.max);
            for (int i = 0; i < iterMax; i++) {
                double[] x = b.next();
                b.put(x, o.eval(x));
                //if ((i + 1) % 1 == 0) System.out.printf("%5d:\t%.6f @ %s%n", i + 1, b.bestValue(), Str.n2(b.bestPoint()));
            }
        }
    }

//TODO
//    public static class GPOptimizationStrategy extends OptimizationStrategy {
//    }
//    public static class BayesianOptimizationStrategy extends OptimizationStrategy {
//    }


//    /**
//     * remove entries below a given percentile
//     */
//    public void cull(float minPct, float maxPct) {
//
//        int n = data.data.size();
//        if (n < 6)
//            return;
//
//        Quantiler q = new Quantiler((int) Math.ceil((n - 1) / 2f));
//        data.forEach(r -> {
//            q.addAt(((Number) r.get(0)).floatValue());
//        });
//        float minValue = q.quantile(minPct);
//        float maxValue = q.quantile(maxPct);
//        data.data.removeIf(r -> {
//            float v = ((Number) r.get(0)).floatValue();
//            return v <= maxValue && v >= minValue;
//        });
//    }

//    public List<DecisionTree> forest(int discretization, int maxDepth) {
//        if (data.isEmpty())
//            return null;
//
//        List<DecisionTree> l = new FasterList();
//        int attrCount = data.attrCount();
//        for (int i = 1; i < attrCount; i++) {
//            l.addAt(
//                    new RealDecisionTree(data.toFloatTable(0, i),
//                            0 /* score */, maxDepth, discretization));
//        }
//        return l;
//    }


//    /** repeats a function N times, and returns the mean of the finite-valued attempts */
//    public static <X> FloatFunction<Supplier<X>> repeat(FloatFunction<Supplier<X>> f, int repeats, boolean parallel) {
//        return X -> {
//            IntStream s = IntStream.range(0, repeats);
//            if (parallel) s = s.parallel();
//            return (float)(s.mapToDouble(i -> f.floatValueOf(X))
//                .filter(Double::isFinite)
//                .average().getAsDouble());
//        };
//    }

}
