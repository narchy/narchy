/*
 *
 *
 */
package alice.tuprolog;

import jcog.data.list.Lst;
import org.eclipse.collections.impl.list.mutable.primitive.BooleanArrayList;

import java.io.Serializable;
import java.util.List;

import static alice.tuprolog.PrologPrim.PREDICATE;

/**
 * @author Alex Benini
 * <p>
 * Core engine
 */
public class PrologRun implements Serializable, Runnable {

    public static final int HALT = -1;
    public static final int FALSE = 0;
    public static final int TRUE = 1;
    public static final int TRUE_CP = 2;

    static final State INIT = StateInit.the;
    static final State EXCEPTION = StateException.the;
    static final State GOAL_SELECTION = StateGoalSelection.the;
    static final State BACKTRACK = StateBacktrack.the;

    private final Lst<Solve> stackEnv = new Lst<>();
    Solution solution;

    Solve solve;
    Prolog prolog;
    private boolean relinkVar;
    private List<Term> bagOFres;
    private List<String> bagOFresString;
    private Term bagOFvarSet;
    private Term bagOfgoal;
    private Term bagOfBag;
    private Term query;
    private BooleanArrayList next;
    private int countNext;

    private Solve last_env;
    private String sinfoSetOf;

    static State endFalse() {
        return new Solution(FALSE);
    }

    static State endTrue() {
        return new Solution(TRUE);
    }

    static State endTrueCP() {
        return new Solution(TRUE_CP);
    }

    static State endHalt() {
        return new Solution(HALT);
    }

    public void initialize(Prolog vm) {
        prolog = vm;
        solution = null;
        next = new BooleanArrayList();
        countNext = 0;
    }

    private Solution solve() {
//        try {

        query.resolveTerm();

        prolog.libs.onSolveBegin(query);
        prolog.prims.identify(query, PREDICATE);

        freeze();

        var y = (solve = new Solve(this, query)).get();

        unfreeze();

        solution = y;
        if (sinfoSetOf != null)
            solution.setSetOfSolution(sinfoSetOf);
        if (!solution.hasOpenAlternatives())
            solveEnd();
        return solution;

//        } catch (Exception ex) {
//            ex.printStackTrace();
//            return new Solution(query);
//        }
    }


    public Solution solveNext() throws NoMoreSolutionException {
        if (!hasOpenAlternatives())
            throw new NoMoreSolutionException();

        freezeBacktrack();
        var y = solve.get();
        unfreeze();
        solution = y;

        if (this.sinfoSetOf != null)
            solution.setSetOfSolution(sinfoSetOf);

        if (!solution.hasOpenAlternatives())
            solveEnd();

        return solution;
    }

    private void freezeBacktrack() {
        freeze();
        solve = last_env;
        solve.next = BACKTRACK;
    }


    /**
     * Halts current solve computation
     */
    void solveHalt() {
        solve.mustStop();
        prolog.libs.onSolveHalt();
    }

    /**
     * Accepts current solution
     */
    void solveEnd() {
        prolog.libs.onSolveEnd();
    }


    private void freeze() {
        if (solve != null && (stackEnv.isEmpty() || stackEnv.getLast() != solve))
            stackEnv.add(solve);
    }

    private void unfreeze() {
        last_env = solve;
        var last = stackEnv.poll();
        if (last != null)
            solve = last;
    }


    void identify(Term t) {
        prolog.prims.identify(t, PREDICATE);
    }


    void pushSubGoal(SubGoalTree goals) {
        solve.context.goalsToEval.pushSubGoal(goals);
    }


    void cut() {
        solve.cut(solve.context.choicePointAfterCut);
    }


    /**
     * Asks for the presence of open alternatives to be explored
     * in current demostration process.
     *
     * @return true if open alternatives are present
     */
    public boolean hasOpenAlternatives() {
        var i = this.solution;
        return i != null && i.hasOpenAlternatives();
    }

    /**
     * Checks if the demonstration process was stopped by an halt command.
     *
     * @return true if the demonstration was stopped
     */
    boolean isHalted() {
        var i = this.solution;
        return i != null && i.isHalted();
    }


    @Override
    public void run() {
        if (solution == null)
            solve();

        try {
            while (hasOpenAlternatives())
                if (next.get(countNext))
                    solveNext();
        } catch (NoMoreSolutionException e) {
            e.printStackTrace();
        }
    }

    boolean getRelinkVar() {
        return this.relinkVar;
    }

    void setRelinkVar(boolean b) {
        this.relinkVar = b;
    }

    List<Term> getBagOFres() {
        return this.bagOFres;
    }

    void setBagOFres(List<Term> l) {
        this.bagOFres = l;
    }

    List<String> getBagOFresString() {
        return this.bagOFresString;
    }

    void setBagOFresString(List<String> l) {
        this.bagOFresString = l;
    }

    Term getBagOFvarSet() {
        return this.bagOFvarSet;
    }

    void setBagOFvarSet(Term l) {
        this.bagOFvarSet = l;
    }

    Term getBagOFgoal() {
        return this.bagOfgoal;
    }

    void setBagOFgoal(Term l) {
        this.bagOfgoal = l;
    }

    Term getBagOFBag() {
        return this.bagOfBag;
    }

    void setBagOFBag(Term l) {
        this.bagOfBag = l;
    }


    void setSetOfSolution(String s) {
        if (solution != null)
            solution.setSetOfSolution(s);
        this.sinfoSetOf = s;
    }

    void clearSinfoSetOf() {
        this.sinfoSetOf = null;
    }

    public final Solution solve(Term query) {
        this.query = query;
        return solve();
    }


}