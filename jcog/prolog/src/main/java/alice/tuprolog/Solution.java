/*
 * tuProlog - Copyright (C) 2001-2002  aliCE team at deis.unibo.it
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
package alice.tuprolog;

import jcog.data.list.Lst;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;
import java.util.stream.Collectors;

import static java.util.Collections.EMPTY_LIST;


/**
 * Solution, as an "End State"
 * <p>
 * SolveInfo class represents the result of a solve
 * request made to the engine, providing information
 * about the solution
 *
 * @author Alex Benini
 */
public class Solution extends State implements Serializable/*, ISolution<Term,Term,Term>*/ {

    public final int endState;
    public Struct goal;
    String setOfSolution;
    List<Var> vars;
    private Term query;
    private int setOfCounter;

    /**
     * @param initGoal
     * @param resultGoal
     * @param resultDemo
     * @param bindings
     */
    Solution(int endState) {
        this.endState = endState;
    }

    private static Term solve(Term a1, Object[] a, Term initGoalBag) {
        while (a1 instanceof Struct a1s && a1s.subs() > 0) {
            var a10 = a1s.sub(0);
            if (a10 instanceof Var) {
                initGoalBag = findVarName(a10, a, a1, 0);
            } else if (a10 instanceof Struct a10s) {
                a10 = solve(a10s.sub(0), a, a10);
                a1s.setSub(0, a10);
            }
            a1 = a1s.subs() > 1 ? a1s.sub(1) : null;
        }

        return a1 instanceof Var ?
            findVarName(a1, a, initGoalBag, 0) : initGoalBag;
    }

    private static Term findVarName(Term link, Object[] a, Term initGoalBag, int pos) {
        var findName = false;

        while (link instanceof Var vlink && !findName) {
            var y = 0;
            while (!findName && y < a.length) {
                var gVar = (Term) a[y];
                while (!findName && gVar instanceof Var gvar) {
                    if (gVar == link || gvar.name().equals((vlink).name())) {
                        ((Struct) initGoalBag).setSub(pos, new Var(((Var) a[y]).name()));
                        findName = true;
                    }
                    gVar = gvar.link();
                }
                y++;
            }
            link = vlink.link();
        }
        return initGoalBag;
    }

    private static Var varValue2(Var v) {
        Term l;
        while ((l = v.link()) != null && (l instanceof Var lv)) {
            v = lv;
        }
        return v;
    }

    private static Var structValue(Var v, int i) {
        structValue:
        while (true) {
            var vStruct = new Var();
            Term l;
            while ((l = v.link()) != null) {

                if (!(l instanceof Var lv)) {
                    if (l instanceof Struct ls) {
                        while (i > 0) {
                            var s1 = ls.sub(1);

                            if (s1 instanceof Struct s1s) {
                                ls = s1s;
                            } else if (s1 instanceof Var s1v) {
                                vStruct = s1v;
                                if (vStruct.link() != null) {
                                    i--;
                                    v = vStruct;
                                    continue structValue;
                                }
                                return vStruct;
                            }
                            i--;
                        }
                        vStruct = ((Var) ls.sub(0));
                    }
                    break;
                } else
                    v = lv;
            }

            return vStruct;
        }
    }

    private static void setStructValue(Var v, int i, Var v1) {
        Term l;
        while ((l = v.link()) != null) {

            if (l instanceof Var lv) {
                v = lv;
            } else if (l instanceof Struct ls) {
                while (i > 0) {

                    var s1 = ls.sub(1);
                    if (s1 instanceof Struct s1s)
                        ls = s1s;
                    else if (s1 instanceof Var s1v) {
                        v = s1v;
                        ls = ((Struct) v.link());
                    }
                    i--;
                }
                ls.setSub(0, v1);
                break;
            } else break;
        }

    }

    private static Lst<String> findVar(final Struct s, Lst<String> l) {
        var allVar = l;
        if (allVar == null) allVar = new Lst<>(0);
        var ss = s.subs();
        if (ss > 0) {
            var t = s.sub(0);
            if (ss > 1) {
                var tt = s.sub(1);
                if (tt instanceof Var ttv) {
                    allVar.add(ttv.name());
                } else if (tt instanceof Struct tts) {
                    findVar(tts, allVar);
                }
            }
            if (t instanceof Var tv) {
                allVar.add(tv.name());
            } else if (t instanceof Struct ts) {
                findVar(ts, allVar);
            }
        }
        return allVar;
    }

    private static Struct substituteVar(Struct s, Lst<String> lSol, Lst<String> lgoal) {
        var t = s.sub(0);

        Term tt = null;
        if (s.subs() > 1)
            tt = s.sub(1);

        if (tt instanceof Var ttv) {

            s.setSub(1, new Var(lgoal.get(lSol.indexOf(ttv.name()))));

            if (t instanceof Var tv) {
                s.setSub(0, new Var(lgoal.get(lSol.indexOf(tv.name()))));
            } else if (t instanceof Struct ts && ts.subs() > 0) {
                s.setSub(0, substituteVar(ts, lSol, lgoal));
            }
        } else {
            if (t instanceof Var tv) {
                s.setSub(0, new Var(lgoal.get(lSol.indexOf(tv.name()))));
            } else if (t instanceof Struct ts) {
                s.setSub(0, substituteVar(ts, lSol, lgoal));
            }
        }

        return s;
    }

    /**
     * Checks if the solve request was successful
     *
     * @return true if the solve was successful
     */
    public boolean isSuccess() {
        return endState > PrologRun.FALSE;
    }

    /**
     * Checks if the solve request was halted
     *
     * @return true if the solve was successful
     */
    public boolean isHalted() {
        return (endState == PrologRun.HALT);
    }

    /**
     * Checks if the solve request was halted
     *
     * @return true if the solve was successful
     */
    public boolean hasOpenAlternatives() {
        return (endState == PrologRun.TRUE_CP);
    }

    /**
     * Gets the query
     *
     * @return the query
     */
    public Term getQuery() {
        return query;
    }

//    @Override public String toString() {
//        return switch (endState) {
//            case PrologRun.FALSE -> "FALSE";
//            case PrologRun.TRUE -> "TRUE";
//            case PrologRun.TRUE_CP -> "TRUE_CP";
//            default -> "HALT";
//        };
//    }

    public void setSetOfSolution(String s) {
        setOfSolution = s;
    }

    /**
     * Gets the solution of the request
     *
     * @throws NoSolutionException if the solve request has not
     *                             solution
     */
    public Term getSolution() throws NoSolutionException {
        if (isSuccess())
            return goal;
        else
            throw new NoSolutionException();

    }

    public Term getSolutionOrNull() {
        if (isSuccess())
            return goal;
        else
            return null;
    }

    /**
     * Gets the list of the variables in the solution.
     *
     * @return the array of variables.
     * @throws NoSolutionException if current solve information
     *                             does not concern a successful
     */
    public List<Var> getBindingVars() throws NoSolutionException {
        if (isSuccess()) {
            return vars;
        } else {
            throw new NoSolutionException();
        }
    }

    /**
     * Gets the value of a variable in the substitution.
     *
     * @throws NoSolutionException     if the solve request has no solution
     * @throws Var.UnknownVarException if the variable does not appear in the substitution.
     */
    public Term getTerm(String varName) throws NoSolutionException, Var.UnknownVarException {
        var t = getVarValue(varName);
        if (t == null)
            throw new Var.UnknownVarException();
        return t;
    }

    /**
     * Gets the value of a variable in the substitution. Returns <code>null</code>
     * if the variable does not appear in the substitution.
     */
    public Term getVarValue(String varName) throws NoSolutionException {
        if (isSuccess()) {
            for (var v : vars) {
                if (v != null && v.name().equals(varName)) {
                    return v.term();
                }
            }
            return null;
        } else
            throw new NoSolutionException();
    }

    /**
     * Returns the string representation of the result of the demonstration.
     * <p>
     * For successful demonstration, the representation concerns
     * variables with bindings.  For failed demo, the method returns false string.
     */
    public String toString() {
        if (isSuccess()) {
            var st = new StringBuilder("yes");
            if (!vars.isEmpty()) {
                st.append(".\n");
            } else {
                st.append(". ");
            }
            for (var v : vars) {
                if (v != null && !v.isAnonymous() && v.isBound() &&
                        (!(v.term() instanceof Var) || (!((Var) (v.term())).name().startsWith("_")))) {
                    st.append(v);
                    st.append("  ");
                }
            }
            return st.toString().trim();
        } else {
            /*Castagna 06/2011*/
            return endState == PrologRun.HALT ? Term.HALT : Term.NO;
        }
    }

    public int result() {
        return endState;
    }

    @Override
    State run(Solve s) {
        this.query = s.query;
        var gv = s.goalVars.size();
        if (gv > 0) {

            var vars = new Lst<Var>(gv);
            goal = (Struct) s.startGoal.copyResult(s.goalVars, vars);
            this.vars = vars.isEmpty() ? EMPTY_LIST : vars;

            var end = this.endState;
            if (end == PrologRun.TRUE || end == PrologRun.TRUE_CP)
                if (s.run.prolog.relinkVar())
                    relinkVar(s);

        } else {
            vars = EMPTY_LIST;
            goal = s.startGoal;
        }

        return null;
    }

    private void relinkVar(Solve e) {
        var pParent = e.run.prolog;


        var bag = e.run.getBagOFres();
        var initBag = pParent.getBagOFbag();



        /* itero nel goal per cercare una eventuale struttura che deve fare match con la
         * result bag ESEMPIO setof(X,member(X,[V,U,f(U),f(V)]),[a,b,f(b),f(a)]).
         */
        var tgoal = pParent.getBagOFgoal();
        var a = (e.goalVars).toArray();


        var query = e.query;


        if (";".equals(((Struct) query).name())) {
            var query_temp = (Struct) ((Struct) query).sub(0);
            if ("setof".equals(query_temp.name()) && setOfCounter == 0) {
                query = query_temp;
                this.setOfCounter++;
            } else {
                query_temp = (Struct) ((Struct) query).sub(1);
                if ("setof".equals(query_temp.name()))
                    query = query_temp;
            }
        }

        if (((Struct) query).subs() > 2 && ((Struct) query).sub(2) instanceof Struct) {

            var findSamePredicateIndicator = false;
            var find = false;
            Term initGoalBag = null;

            Prolog p = null;

            while (tgoal instanceof Var tgoalv && tgoalv.link() != null) {
                tgoal = tgoalv.link();

                if (tgoal instanceof Struct tgoals) {
                    tgoal = tgoals.sub(1);

                    if (p == null) p = new Prolog();

                    if (tgoal.unify(p, ((Var) initBag).link())) {

                        initGoalBag = tgoal;
                        find = true;
                        findSamePredicateIndicator = true;
                        break;
                    } else if (((Var) initBag).link() instanceof Struct s) {

                        if (tgoal instanceof Struct tgoalss && s.key().compareTo(tgoalss.key()) == 0) {
                            findSamePredicateIndicator = true;
                            find = true;
                            initGoalBag = tgoal;
                        }
                    }

                    if (find || findSamePredicateIndicator && initGoalBag instanceof Struct) {

                        var a0 = ((Struct) initGoalBag).sub(0);
                        var a1 = ((Struct) initGoalBag).sub(1);
                        if (a0 instanceof Var) {
                            initGoalBag = findVarName(a0, a, initGoalBag, 0);
                        }
                        a1 = solve(a1, a, a1);
                        ((Struct) initGoalBag).setSub(1, a1);
                    }
                }
            }


            if (initGoalBag != null) {

                var initGoalBagList = new Lst<Term>();
                var initGoalBagTemp = (Struct) initGoalBag;
                while (initGoalBagTemp.subs() > 0) {
                    var t1 = initGoalBagTemp.sub(0);
                    initGoalBagList.add(t1);
                    var t2 = initGoalBagTemp.sub(1);
                    if (t2 instanceof Struct t2s) {
                        initGoalBagTemp = t2s;
                    }
                }


                var initGoalBagListOrdered = new Lst<Term>();
                if ("setof".equals(((Struct) query).name())) {
                    var initGoalBagListVar = initGoalBagList.stream().filter(anInitGoalBagList -> anInitGoalBagList instanceof Var).map(anInitGoalBagList -> ((Var) anInitGoalBagList).name()).collect(Collectors.toCollection(Lst::new));

                    var left = new Lst<Term>();
                    left.add(initGoalBagList.get(0));
                    List<Term> right = new Lst<>();
                    List<Term> right_temp = new Lst<>();

                    List<Term> left_temp = new Lst<>();
                    for (var m = 1; m < initGoalBagList.size(); m++) {
                        int k;
                        for (k = 0; k < left.size(); k++) {
                            if (initGoalBagList.get(m).isGreaterRelink(left.get(k), initGoalBagListVar)) {
                                left_temp.add(left.get(k));
                            } else {
                                left_temp.add(initGoalBagList.get(m));
                                break;
                            }
                        }
                        if (k == left.size())
                            left_temp.add(initGoalBagList.get(m));
                        for (var y = 0; y < left.size(); y++) {

                            var search = false;
                            for (var aLeft_temp : left_temp) {
                                if (aLeft_temp.toString().equals(left.get(y).toString()))
                                    search = true;
                            }
                            if (!search) {

                                right_temp.add(left.get(y));
                            }
                            left.remove(y);
                            y--;
                        }
                        for (var y = 0; y < right.size(); y++) {
                            right_temp.add(right.get(y));
                            right.remove(y);
                            y--;
                        }
                        right.addAll(right_temp);

                        right_temp.clear();
                        left.addAll(left_temp);

                        left_temp.clear();

                    }


                    initGoalBagListOrdered.addAll(left);
                    initGoalBagListOrdered.addAll(right);


                } else
                    initGoalBagListOrdered = initGoalBagList;

                initGoalBagTemp = (Struct) initGoalBag;

                Object[] t = initGoalBagListOrdered.toArray();
                var t1 = Arrays.stream(t).map(item -> (Term) item).toArray(Term[]::new);


                initGoalBag = new Struct(initGoalBagTemp.name(), t1);


                List<Term> initBagList = new Lst<>();
                var initBagTemp = (Struct) ((Var) initBag).link();
                while (initBagTemp.subs() > 0) {
                    var t0 = initBagTemp.sub(0);
                    initBagList.add(t0);
                    var t2 = initBagTemp.sub(1);
                    if (t2 instanceof Struct t2s)
                        initBagTemp = t2s;
                }

                var tNoOrd = initBagList.toArray();
                var termNoOrd = Arrays.stream(tNoOrd).map(o -> (Term) o).toArray(Term[]::new);


                initBag = new Struct(initGoalBagTemp.name(), termNoOrd);
            }


            if (findSamePredicateIndicator) {

//                if (p == null) p = new Prolog();

                if (!(find && initGoalBag.unify(p, initBag))) {

                    e.next = PrologRun.endFalse();

                    var prologRun = pParent.run;
                    var ss = prologRun.solution != null ? prologRun.solution.setOfSolution : null;
                    var s = ss != null ? ss + "\n\nfalse." : "null\n\nfalse.";
                    pParent.endFalse(s);

                    return;
                }
            }
        }
        /*
         * STEP1: dalla struttura risultato bagof (bag = (c.getEngineMan()).getBagOFres())
         * estraggo la lista di tutte le variabili
         * memorizzate nell'Lst<String> lSolVar
         * lSolVar = [H_e2301, H_e2302, H_e2303, H_e2304, H_e2305, H_e2306, H_e2307, H_e2308]
         */

        var lSolVar = new Lst<String>();

        /*NB lSolVar ha lunghezza multipla di lGoal var, se ho pi soluzioni si ripete
         * servirebbe esempio con 2 bag */
        var l_temp = new Lst<String>();
        for (var i = 0; i < bag.size(); i++) {
            var resVar = (Var) bag.get(i);

            var t = resVar.link();

            if (t != null) {
                if (t instanceof Struct t1) {


                    l_temp.clear();
                    l_temp = findVar(t1, l_temp);
                    for (var w = l_temp.size() - 1; w >= 0; w--) {
                        lSolVar.add(l_temp.get(w));
                    }
                } else if (t instanceof Var) {
                    while (t instanceof Var) {
                        resVar = (Var) t;

                        t = resVar.link();

                    }
                    lSolVar.add(resVar.name());
                    bag.set(i, resVar);
                }
            } else lSolVar.add(resVar.name());
        }

        /*
         * STEP2: dalla struttura goal bagof (goalBO = (Var)(c.getEngineMan()).getBagOFgoal())
         * estraggo la lista di tutte le variabili
         * memorizzate nell'Lst<String> lgoalBOVar
         * lgoalBOVar = [Z_e0, X_e73, Y_e74, V_e59, WithRespectTo_e31, U_e588, V_e59, H_e562, X_e73, Y_e74, F_e900]
         */

        var goalBO = (Var) pParent.getBagOFgoal();

        var lgoalBOVar = new Lst<String>();
        var goalBOvalue = goalBO.link();
        if (goalBOvalue instanceof Struct t1) {

            l_temp.clear();
            l_temp = findVar(t1, l_temp);
            for (var w = l_temp.size() - 1; w >= 0; w--) {
                lgoalBOVar.add(l_temp.get(w));
            }
        }


        /*
         * STEP3: prendere il set di variabili libere della bagof
         * fare il match con le variabili del goal in modo da avere i nomi del goal esterno
         * questo elenco ci servirˆ per eliminare le variabili in pi che abbiamo in lgoalBOVar
         * ovvero tutte le variabili associate al template
         * lGoalVar [Y_e74, U_e588, V_e59, X_e73, Y_e74, U_e588, F_e900]
         * mette quindi in lGoalVar le variabili che compaiono in goalVars e sono anche libere
         * per la bagof c.getEngineMan().getBagOFvarSet()
         */

        var v = (Var) pParent.getBagOFvarSet();
        var varList = (Struct) v.link();
        List<String> lGoalVar = new Lst<>();


        if (varList != null)
            for (Iterator<? extends Term> it = varList.listIterator(); it.hasNext(); ) {


                var var = it.next();
                for (var anA : a) {
                    var vv = (Var) anA;
                    var vLink = vv.link();
                    if (vLink != null && vLink.isEqual(var)/*&& !(var.toString().startsWith("_"))*/) {

                        lGoalVar.add(vv.name());
                    }
                }
            }


        /*
         * STEP4: pulisco lgoalBOVar lasciando solo i nomi che compaiono effettivamente in
         * lGoalVar (che  la rappresentazione con nomi esterni delle variabili libere nel
         * goal della bagof
         */
        lgoalBOVar.retainAll(lGoalVar);

        if (lGoalVar.size() > lgoalBOVar.size()) {

            for (var h = 0; h < lGoalVar.size(); h++)
                if (h >= lgoalBOVar.size()) {

                    lgoalBOVar.add(lGoalVar.get(h));
                }
        }
        /*
         * STEP5: sostituisco le variabili nel risultato (sia in goals che vars)
         * a) cerco l'indice della variabile in lSolVar
         * b) sostituisco con quella di stesso indice in lgoalBOVar
         */
        var goalSolution = new Var();

        if (!lSolVar.isEmpty() && !lgoalBOVar.isEmpty() && !varList.isGround() && !goalBO.isGround()) {
            String bagVarName = null;
            for (var i = 0; i < bag.size(); i++) {

                var resVar = (Var) bag.get(i);

                var t = resVar.term();
//                Term t = resVar.link();
//                if (t == null)
//                    t = resVar;

                bagVarName = null;
                for (var anA : a) {
                    var vv = (Var) anA;
                    var vv_link = structValue(vv, i);

                    if (vv_link.isEqual(t)) {


                        if (bagVarName == null) {
                            bagVarName = vv.getOriginalName();
                            goalSolution = vv;
                        }


                        var vll = vv_link.link();

                        if (vll instanceof Struct) {
                            var s = substituteVar((Struct) vll, lSolVar, lgoalBOVar);

                        } else {
                            setStructValue(vv, i, new Var(lgoalBOVar.get(lSolVar.indexOf(resVar.name()))));
                        }
                    }
                }

            }

            var n = vars.size();
            for (var j = 0; j < n; j++) {
                var vv = vars.get(j);
                var on = vv.getOriginalName();
                if (bagVarName.equals(on)) {
                    var solVar = varValue2(goalSolution);

                    solVar.setName(on);
                    solVar.rename(0, 0);

                    vars.set(j, solVar);
                    break;
                }
            }
        }

        /*
         * STEP6: gestisco caso particolare SETOF in cui non stampa la soluzione
         */
        var bagString = pParent.getBagOFresString();
        var i = 0;
        var s = "";

        var bs = bagString.size();
        for (var m = 0; m < bs; m++) {
            var bagResString = bag.get(m).toString();

            if (bag.get(m) instanceof Var && ((Var) bag.get(m)).link() != null && (((Var) bag.get(m)).link() instanceof Struct) && !((Var) bag.get(m)).link().isAtom() && bagResString.length() != bagString.get(m).length()) {

                var st = new StringTokenizer(bagString.get(m));
                var st1 = new StringTokenizer(bagResString);
                while (st.hasMoreTokens()) {
                    var t1 = st.nextToken(" /(),;");

                    var t2 = st1.nextToken(" /(),;");

                    if (t1.compareTo(t2) != 0 && !t2.contains("_")) {

                        s = s + lGoalVar.get(i) + '=' + t2 + ' ';

                        pParent.setSetOfSolution(s);
                        i++;
                    }
                }
            }
        }


        pParent.relinkVar(false);
        pParent.setBagOFres(null);
        pParent.setBagOFgoal(null);
        pParent.setBagOFvarSet(null);
        pParent.setBagOFbag(null);
    }


}