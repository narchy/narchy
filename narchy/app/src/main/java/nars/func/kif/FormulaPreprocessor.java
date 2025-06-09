/**
 * This code is copyright Articulate Software (c) 2017.  Some portions
 * copyright Teknowledge (c) 2003 and reused under the terms of the GNU license.
 * This software is released under the GNU Public License <http:
 * Users of this code also consent, by use of this code, to credit Articulate Software
 * and Teknowledge in any writings, briefings, publications, presentations, or
 * other representations of any software which incorporates, builds on, or uses this
 * code.  Please cite the following article in any publication with references:
 * <p>
 * Pease, A., (2003). The Sigma Ontology Development Environment,
 * in Working Notes of the IJCAI-2003 Workshop on Ontology and Distributed Systems,
 * August 9, Acapulco, Mexico. See also https:
 * <p>
 * This class expects the following to be in the ontology.
 * Their absence won't cause an exception, but will prevent correct behavior.
 * VariableArityRelation
 * subclass
 * instance
 * SetOrClass
 */
package nars.func.kif;

import com.google.common.collect.Sets;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public enum FormulaPreprocessor { ;

    /** ***************************************************************
     * For any given formula, stop generating new pred var instantiations
     * and row var expansions if this threshold value has been exceeded.
     * The default value is 2000.
     */
    private static final int AXIOM_EXPANSION_LIMIT = 2000;

    private static boolean debug = false;

    private static final boolean RowVarExpansion = false;

//    /** ***************************************************************
//     * A + is appended to the type if the parameter must be a class
//     *
//     * @return the type for each argument to the given predicate, where
//     * ArrayList element 0 is the result, if a function, 1 is the first
//     * argument, 2 is the second etc.
//     */
//    private static ArrayList<String> getTypeList(String pred, KB kb) {
//
//        return kb.kbCache.signatures.get(pred);
//    }

    /** ***************************************************************
     * Find the argument type restriction for a given predicate and
     * argument number that is inherited from one of its super-relations.
     * A "+" is appended to the type if the parameter must be a class,
     * meaning that a domainSubclass is defined for this argument in one
     * of the loaded .kif files.  Argument number 0 is used for the return
     * type of a Function.  Asking for a non-existent arg will return null;
     */
    public static String findType(int numarg, String pred, KB kb) {

        ArrayList<String> sig = null;


        if (kb.kbCache != null && kb.kbCache.signatures != null)
            sig = kb.kbCache.signatures.get(pred);
        if (sig == null) {


            return null;
        }
        if (numarg >= sig.size())
            return null;

        return sig.get(numarg);
    }

    /** ***************************************************************
     * This method tries to remove all but the most specific relevant
     * classes from a List of sortal classes.
     *
     * @param types A List of classes (class name Strings) that
     * constrain the value of a SUO-KIF variable.
     *
     * @param kb The KB used to determine if any of the classes in the
     * List types are redundant.
     *
     * @return void
     */
    public static void winnowTypeList(HashSet<String> types, KB kb) {

        long t1 = 0L;
        if (types.size() > 1) {
            Object[] valArr = types.toArray();
            String clX;
            String clY;
            for (int i = 0; i < valArr.length; i++) {
                boolean stop = false;
                for (int j = 0; j < valArr.length; j++) {
                    if (i != j) {
                        clX = (String) valArr[i];
                        clY = (String) valArr[j];
                        if (clX.equals(clY) || kb.isSubclass(clX, clY)) {
                            types.remove(clY);
                            if (types.size() < 2) {
                                stop = true;
                                break;
                            }
                        }
                    }
                }
                if (stop) break;
            }
        }
    }

    /** ***************************************************************
     * Add clauses for every variable in the antecedent to restrict its
     * type to the type restrictions defined on every relation in which
     * it appears.  For example
     * (=>
     *   (foo ?A B)
     *   (bar B ?A))
     *
     * (domain foo 1 Z)
     *
     * would result in
     *
     * (=>
     *   (instance ?A Z)
     *   (=>
     *     (foo ?A B)
     *     (bar B ?A)))
     */
    public static Formula addTypeRestrictions(Formula form, KB kb) {

        if (debug) System.out.println("addTypeRestrictions: form " + form);

        Map<String, HashSet<String>> varDomainTypes = computeVariableTypes(form, kb);

        HashMap<String, HashSet<String>> varExplicitTypes = findExplicitTypesClassesInAntecedent(kb, form);


        var varmap = new HashMap<>();
        for (Map.Entry<String, HashSet<String>> entry : varDomainTypes.entrySet()) {
            String var = entry.getKey();
            if (!varExplicitTypes.containsKey(var)) {

                varmap.put(var, entry.getValue());
            } else {
                 var domainTypes = entry.getValue();
                 var explicitTypes = varExplicitTypes.get(var);
                 var  types = domainTypes.stream().filter(dt -> dt.endsWith("+")).collect(Collectors.toCollection(HashSet::new));
                for (String et : explicitTypes) {
                    if (et.endsWith("+")) {
                        types.add(et);
                    }
                }
                varmap.put(var, types);
            }
        }


        List<List<String>> quantifiedUnquantifiedVariables =
                form.collectQuantifiedUnquantifiedVariables();
        Iterable<String> unquantifiedVariables = quantifiedUnquantifiedVariables.get(1);

        StringBuffer sb = new StringBuffer();
        boolean begin = true;
        for (String unquantifiedV : unquantifiedVariables) {
            var  types = (Set<String>) varmap.get(unquantifiedV);
            if (types != null && !types.isEmpty()) {
                for (String t : types) {
                    if (begin) {
                        sb.append("(=> \n  (and \n");
                        begin = false;
                    }
                    if (!t.endsWith("+"))
                        sb.append(" (instance ").append(unquantifiedV).append(' ').append(t).append(") ");
                    else
                        sb.append(" (subclass ").append(unquantifiedV).append(' ').append(t, 0, t.length() - 1).append(") ");
                }
            }
        }

        if (!begin)
            sb.append(")\n");
        if (debug) System.out.println("addTypeRestrictions: sb: " + sb);

        if ((form.theFormula.contains(Formula.EQUANT)) ||
                (form.theFormula.contains(Formula.UQUANT)))
            addTypeRestrictionsRecurse(kb, form, sb);

        if (!begin)
            sb.append(")\n");

        Formula f = new Formula();
        f.read(sb.toString());

        if (StringUtil.emptyString(f.theFormula) || f.empty())
            f.read(form.theFormula);
        if (debug) System.out.println("addTypeRestrictions: result: " + f);
        if (debug) System.out.println("addTypeRestrictions: form at end: " + form);
        if (debug) System.out.println("addTypeRestrictions: sb at end: '" + sb + '\'');
        return f;
    }

    /** ***************************************************************
     * Recursively add sortals for existentially quantified variables
     *
     * @param kb The KB used to add type restrictions.
     * @param f The formula in KIF syntax
     * @param sb A StringBuilder used to store the new formula with sortals
     */
    private static void addTypeRestrictionsRecurse(KB kb, Formula f, StringBuffer sb) {

        if (debug) System.out.println("addTypeRestrictionsRecurse: input: " + f);
        if (debug) System.out.println("addTypeRestrictionsRecurse: sb: " + sb);
        if (f == null || StringUtil.emptyString(f.theFormula) || f.empty())
            return;

        String carstr = f.car();
        if (debug) System.out.println("addTypeRestrictionsRecurse: carstr: " + carstr);
        if (Formula.atom(carstr) && (Formula.isLogicalOperator(carstr) || carstr.equals(Formula.EQUAL))) {
            sb.append('(').append(carstr).append(' ');
            if (debug) System.out.println("addTypeRestrictionsRecurse: interior sb: " + sb);
            if (carstr.equals(Formula.EQUANT) || carstr.equals(Formula.UQUANT)) {


                sb.append(f.getArgument(1)).append(' ');
                Collection<String> quantifiedVariables = collectVariables(f.getArgument(1));


                HashMap<String, HashSet<String>> varDomainTypes = computeVariableTypes(f, kb);
                Map<String, HashSet<String>> varExplicitTypes = findExplicitTypesClassesInAntecedent(kb, f);


                HashMap<String, HashSet<String>> varmap = (HashMap<String, HashSet<String>>) varDomainTypes.clone();
                if (varExplicitTypes != null) {
                    for (String v : varExplicitTypes.keySet())
                        varmap.remove(v);
                }

                boolean addSortals = quantifiedVariables.stream().map(varmap::get).anyMatch(strings -> strings != null && !strings.isEmpty());
                if (addSortals) {
                    switch (carstr) {
                        case Formula.EQUANT -> sb.append("(and ");
                        case Formula.UQUANT -> sb.append("(=> (and ");
                    }
                }

                for (String existentiallyQV : quantifiedVariables) {
                    Set<String> types = varmap.get(existentiallyQV);
                    if (types != null && !types.isEmpty()) {
                        for (String t : types) {
                            if (!t.endsWith("+"))
                                sb.append(" (instance ").append(existentiallyQV).append(' ').append(t).append(") ");
                            else
                                sb.append(" (subclass ").append(existentiallyQV).append(' ').append(t, 0, t.length() - 1).append(") ");
                        }
                    }
                }
                if (addSortals && carstr.equals(Formula.UQUANT))
                    sb.append(')');
                for (int i = 2; i < f.listLength(); i++)
                    addTypeRestrictionsRecurse(kb, new Formula(f.getArgument(i)), sb);
                if (addSortals)
                    sb.append(')');
            } else {
                if (debug) System.out.println("addTypeRestrictionsRecurse: input interior: " + f);
                if (debug) System.out.println("addTypeRestrictionsRecurse: args: " + f.complexArgumentsToArrayList(1));
                if (debug) System.out.println("addTypeRestrictionsRecurse: list length: " + f.listLength());

                if (debug)
                    for (int i = 1; i < f.listLength(); i++) {
                        Formula newF = new Formula(f.getArgument(i));
                        System.out.println(f.getArgument(i) + " : " + newF + " : " + newF.theFormula);
                    }

                for (int i = 1; i < f.listLength(); i++)
                    addTypeRestrictionsRecurse(kb, new Formula(f.getArgument(i)), sb);
            }
            sb.append(')');
        } else if (f.isSimpleClause(kb) || f.atom()) {
            if (debug) System.out.println("addTypeRestrictionsRecurse: here2");
            sb.append(f).append(" ");
        } else {
            if (debug) System.out.println("addTypeRestrictionsRecurse: here3");
            addTypeRestrictionsRecurse(kb, f.carAsFormula(), sb);
            addTypeRestrictionsRecurse(kb, f.cdrAsFormula(), sb);
        }
    }

    /** ***************************************************************
     * Collect variables from strings.
     *
     * For example,
     * Input = (?X ?Y ?Z)
     * Output = a list of ?X, ?Y and ?Z
     *
     * Input = ?X
     * Output = a list of ?X
     */
    private static ArrayList<String> collectVariables(String argstr) {

        ArrayList<String> arglist = new ArrayList<>();
        if (argstr.startsWith(Formula.V_PREF)) {
            arglist.add(argstr);
            return arglist;
        } else if (argstr.startsWith(Formula.LP)) {
            arglist = new ArrayList<>(Arrays.asList(argstr.substring(1, argstr.length() - 1).split(" ")));
            return arglist;
        } else {
            System.err.println("Errors in FormulaPreprocessor.collectVariables ...");
            return null;
        }
    }

    /** ************************************************************************
     * Get the most specific type for variables.
     *
     * @param kb The KB to be used for processing
     * @param types a list of sumo types for a sumo target/variable
     * @return the most specific sumo type for the target/variable
     *
     * For example
     * types of ?Writing = [Entity, Physical, Process, IntentionalProcess,
     *                      ContentDevelopment, Writing]
     * return the most specific type Writing
     */
    protected static String getMostRelevantType(KB kb, Set<String> types) {

        HashSet<String> insts = new HashSet<>();
        for (String type : types) {
            if (!type.endsWith("+"))
                insts.add(type);
            else
                insts.add(type.substring(0, type.length() - 1));
        }
        if (insts != null) {
            winnowTypeList(insts, kb);
            for (String inst : insts) {
                return inst;
            }
        }

        return null;
    }

    /*****************************************************************
     * Collect the types of any variables that are specifically defined
     * in the antecedent of a rule with an instance or subclass expression.
     * TODO: This may ultimately require CNF conversion and then checking negative
     * literals, but for now it's just a hack to grab preconditions.
     */
    public static HashMap<String, HashSet<String>> findExplicitTypesInAntecedent(KB kb, Formula form) {

        if (!form.isRule())

            return null;

        Formula f = new Formula();
        f.read(form.theFormula);
        Formula antecedent = f.cdrAsFormula().carAsFormula();

        return findExplicitTypes(kb, antecedent);
    }

    /*****************************************************************
     * Collect the types of any variables that are specifically defined
     * in the antecedent of a rule with an instance expression;
     * Collect the super classes of any variables that are specifically
     * defined in the antecedent of a rule with an subclass expression;
     */
    public static HashMap<String, HashSet<String>> findExplicitTypesClassesInAntecedent(KB kb, Formula form) {

        Formula f = new Formula();
        f.read(form.theFormula);
        Formula antecedent = findAntecedent(f);
        HashMap<String, HashSet<String>> varExplicitTypes = new HashMap<>();
        HashMap<String, HashSet<String>> varExplicitClasses = new HashMap<>();
        findExplicitTypesClasses(kb, antecedent, varExplicitTypes, varExplicitClasses);
        return varExplicitTypes;
    }

    /** ***************************************************************
     * Return a formula's antecedents
     */
    private static Formula findAntecedent(Formula f) {

        if (!f.theFormula.contains(Formula.IF) && !f.theFormula.contains(Formula.IFF))
            return f;
        String carstr = f.car();
        if (Formula.atom(carstr) && Formula.isLogicalOperator(carstr)) {
            if (carstr.equals(Formula.IF) || carstr.equals(Formula.IFF))
                return f.cdrAsFormula().carAsFormula();
            else
                return f;
        }
        return f;
    }

    /*****************************************************************
     * Collect variable names and their types from instance or subclass
     * expressions. subclass restrictions are marked with a '+'.
     *
     * @param form The formula in KIF syntax
     *
     * @return A map of variables paired with a set of sumo types collected
     * from instance and subclass expressions.
     *
     * TODO: This may ultimately require CNF conversion and then checking
     * negative literals, but for now it's just a hack to grab preconditions.
     */
    public static HashMap<String, HashSet<String>> findExplicitTypes(KB kb, Formula form) {

        HashMap<String, HashSet<String>> varExplicitTypes = new HashMap<>();
        HashMap<String, HashSet<String>> varExplicitClasses = new HashMap<>();
        findExplicitTypesRecurse(kb, form, false, varExplicitTypes, varExplicitClasses);

        varExplicitTypes.putAll(varExplicitClasses);
        return varExplicitTypes;
    }

    /*****************************************************************
     * Collect variable names and their types from instance or subclass
     * expressions.
     *
     * @param form The formula in KIF syntax
     * @param varExplicitTypes A map of variables paired with sumo types
     *                         collected from instance expressions
     * @param varExplicitClasses A map of variables paired with sumo types
     *                           collected from subclass expression
     */
    public static void findExplicitTypesClasses(KB kb, Formula form,
                                                HashMap<String, HashSet<String>> varExplicitTypes,
                                                HashMap<String, HashSet<String>> varExplicitClasses) {

        findExplicitTypesRecurse(kb, form, false, varExplicitTypes, varExplicitClasses);
    }

    /*****************************************************************
     * Recursively collect a variable name and its types.
     */
    public static void findExplicitTypesRecurse(KB kb, Formula form, boolean isNegativeLiteral,
                                                HashMap<String, HashSet<String>> varExplicitTypes,
                                                HashMap<String, HashSet<String>> varExplicitClasses) {

        if (form == null || StringUtil.emptyString(form.theFormula) || form.empty())
            return;

        String carstr = form.car();

        if (Formula.atom(carstr) && Formula.isLogicalOperator(carstr)) {
            switch (carstr) {
                case Formula.EQUANT:
                case Formula.UQUANT:
                    for (int i = 2; i < form.listLength(); i++)
                        findExplicitTypesRecurse(kb, new Formula(form.getArgument(i)), false, varExplicitTypes, varExplicitClasses);
                    break;
                case Formula.NOT:
                    for (int i = 1; i < form.listLength(); i++)
                        findExplicitTypesRecurse(kb, new Formula(form.getArgument(i)), true, varExplicitTypes, varExplicitClasses);
                    break;
                default:
                    for (int i = 1; i < form.listLength(); i++)
                        findExplicitTypesRecurse(kb, new Formula(form.getArgument(i)), false, varExplicitTypes, varExplicitClasses);
                    break;
            }
        } else if (form.isSimpleClause(kb)) {
            if (isNegativeLiteral)
                return;
            Pattern p = Pattern.compile("\\(instance (\\?[a-zA-Z0-9\\-_]+) ([?a-zA-Z0-9\\-_]+)");
            Matcher m = p.matcher(form.theFormula);
            while (m.find()) {
                String var = m.group(1);
                String cl = m.group(2);
                HashSet<String> hs = new HashSet<>();
                if (!cl.startsWith("?")) {
                    if (varExplicitTypes.containsKey(var))
                        hs = varExplicitTypes.get(var);
                    hs.add(cl);
                } else {
                    if (varExplicitTypes.containsKey(var))
                        hs = varExplicitTypes.get(var);
                }
                varExplicitTypes.put(var, hs);
            }

            p = Pattern.compile("\\(subclass (\\?[a-zA-Z0-9\\-_]+) ([?a-zA-Z0-9\\-]+)");
            m = p.matcher(form.theFormula);
            while (m.find()) {
                String var = m.group(1);
                String cl = m.group(2);
                HashSet<String> hs = new HashSet<>();
                if (!cl.startsWith("?")) {
                    if (varExplicitClasses.containsKey(var))
                        hs = varExplicitClasses.get(var);
                    hs.add(cl + '+');
                } else {
                    if (varExplicitClasses.containsKey(var))
                        hs = varExplicitClasses.get(var);
                }
                varExplicitClasses.put(var, hs);
            }
        } else {
            findExplicitTypesRecurse(kb, form.carAsFormula(), false, varExplicitTypes, varExplicitClasses);
            findExplicitTypesRecurse(kb, form.cdrAsFormula(), false, varExplicitTypes, varExplicitClasses);
        }
    }

    /** ***************************************************************
     * utility method to add a String element to a HashMap of String
     * keys and a value of an HashSet of Strings
     */
    private static void addToMap(HashMap<String, HashSet<String>> map, String key, String element) {

        HashSet<String> al = map.get(key);
        if (al == null)
            al = new HashSet<>();
        al.add(element);
        map.put(key, al);
    }

    /** ***************************************************************
     * utility method to merge two HashMaps of String keys and a values
     * of an HashSet of Strings
     */
    static HashMap<String, HashSet<String>> mergeToMap(HashMap<String, HashSet<String>> map1,
                                                       Map<String, HashSet<String>> map2, KB kb) {

        HashMap<String, HashSet<String>> result = new HashMap<>(map1);

        for (Map.Entry<String, HashSet<String>> entry : map2.entrySet()) {
            String key = entry.getKey();
            Set<String> value = new HashSet<>();
            if (result.containsKey(key)) {
                value = result.get(key);
            }
            value.addAll(entry.getValue());
            value = kb.removeSuperClasses(value);
            result.put(key, Sets.newHashSet(value));
        }
        return result;
    }

    /*****************************************************************
     * This method returns a HashMap that maps each String variable in
     * this the names of types (classes) of which the variable must be
     * an instance or the names of types of which the variable must be
     * a subclass. Note that this method does not capture explicit type
     * from assertions such as (=> (instance ?Foo Bar) ...). This method
     * just consider restrictions implicitly defined from the arg types
     * of relations.
     *
     * @param kb The KB to be used to compute the sortal constraints
     *           for each variable.
     * @return A HashMap of variable names and their types. Subclass
     *         restrictions are marked with a '+', meaning that a
     *         domainSubclass is defined for this argument in one of
     *         the loaded .kif files. Instance restrictions have no
     *         special mark.
     */
    public static HashMap<String, HashSet<String>> computeVariableTypes(Formula form, KB kb) {

        if (debug) System.out.println("INFO in FormulaPreprocessor.computeVariableTypes(): \n" + form);
        Formula f = new Formula();
        f.read(form.theFormula);
        HashMap<String, HashSet<String>> result = new HashMap<>();
        return computeVariableTypesRecurse(kb, form, result);
    }

    /** ***************************************************************
     */
    private static HashMap<String, HashSet<String>> computeVariableTypesRecurse(KB kb, Formula f,
                                                                                HashMap<String, HashSet<String>> input) {

        HashMap<String, HashSet<String>> result = new HashMap<>();
        if (f == null || StringUtil.emptyString(f.theFormula) || f.empty())
            return result;
        if (debug) System.out.println("INFO in FormulaPreprocessor.computeVariableTypesRecurse(): \n" + f);
        String carstr = f.car();
        if (Formula.atom(carstr) && Formula.isLogicalOperator(carstr)) {
            result.putAll(input);
            for (int i = 1; i < f.listLength(); i++)
                result = mergeToMap(result, computeVariableTypesRecurse(kb, new Formula(f.getArgument(i)), input), kb);
        } else if (f.isSimpleClause(kb)) {
            String pred = carstr;
            if (f.theFormula.contains("?") && !Formula.isVariable(pred)) {
                Formula newf = f.cdrAsFormula();
                int argnum = 1;
                while (!newf.empty()) {
                    String arg = newf.car();
                    if (Formula.isVariable(arg)) {
                        String cl = findType(argnum, pred, kb);

                        if (StringUtil.emptyString(cl)) {


                        } else
                            addToMap(result, arg, cl);
                    } else if (Formula.isFunctionalTerm(arg)) {
                        result = mergeToMap(result, computeVariableTypesRecurse(kb, new Formula(arg), input), kb);
                    }
                    newf = newf.cdrAsFormula();
                    argnum++;
                }
            }
        } else {
            result = mergeToMap(input, computeVariableTypesRecurse(kb, f.carAsFormula(), input), kb);
            result = mergeToMap(result, computeVariableTypesRecurse(kb, f.cdrAsFormula(), input), kb);
        }
        return result;
    }

    /** ***************************************************************
     * Pre-process a formula before sending it to the theorem prover.
     * This includes ignoring meta-knowledge like documentation strings,
     * translating mathematical operators, quoting higher-order formulas,
     * adding a numerical suffix to VariableArityRelations based on their count,
     * expanding row variables and prepending the 'holds__' predicate.
     * @return an ArrayList of Formula(s)
     */
    private static String preProcessRecurse(Formula f, String previousPred, boolean ignoreStrings,
                                     boolean translateIneq, boolean translateMath,
                                     KB kb) {


        StringBuilder result = new StringBuilder(1024);
        if (f.listP() && !f.empty()) {
            String prefix = "";
            String pred = f.car();
            if (Formula.isQuantifier(pred)) {

                result.append(' ');
                result.append(f.cadr());

                String next = f.caddr();
                Formula nextF = new Formula();
                nextF.read(next);
                result.append(' ');
                result.append(preProcessRecurse(nextF, "", ignoreStrings, translateIneq, translateMath, kb));
            } else {
                if (kb.isInstanceOf(pred, "VariableArityRelation")) {
                    int arity = f.complexArgumentsToArrayList(0).size() - 1;
                    String oldPred = pred;


                    pred = pred + '_' + arity;
                    kb.kbCache.copyNewPredFromVariableArity(pred, oldPred, arity);
                    if (debug) System.out.println("preProcessRecurse: pred: " + pred);
                }
                Formula restF = f.cdrAsFormula();

                int argCount = 1;
                while (!restF.empty()) {
                    argCount++;
                    String arg = restF.car();
                    Formula argF = new Formula();
                    argF.read(arg);
                    if (argF.listP()) {
                        String res = preProcessRecurse(argF, pred, ignoreStrings, translateIneq, translateMath, kb);
                        result.append(' ');
//                        if (!Formula.isLogicalOperator(pred) &&
//                                !Formula.isComparisonOperator(pred) &&
//                                !Formula.isMathFunction(pred) &&
//                                !argF.isFunctionalTerm()) {
//                            result.append('`');
//                        }
                        result.append(res);
                    } else
                        result.append(' ').append(arg);
                    restF.theFormula = restF.cdr();


                }
                if ("yes".equals(KBmanager.manager.getPref("holdsPrefix"))) {
                    if (!Formula.isLogicalOperator(pred) && !Formula.isQuantifierList(pred, previousPred))
                        prefix = "holds_";
                    if (f.isFunctionalTerm())
                        prefix = "apply_";
                    if ("holds".equals(pred)) {
                        pred = "";
                        argCount--;
                        prefix = prefix + argCount + "__ ";
                    } else {
                        if (!Formula.isLogicalOperator(pred) &&
                                !Formula.isQuantifierList(pred, previousPred) &&
                                !Formula.isMathFunction(pred) &&
                                !Formula.isComparisonOperator(pred)) {
                            prefix = prefix + argCount + "__ ";
                        } else
                            prefix = "";
                    }
                }
            }
            result.insert(0, pred);
            result.insert(0, prefix);
            result.insert(0, '(');
            result.append(')');

        }
        return result.toString();
    }

    /** ***************************************************************
     * Tries to successively instantiate predicate variables and then
     * expand row variables in this Formula, looping until no new
     * Formulae are generated.
     *
     * @param kb The KB to be used for processing this Formula
     *
     * @param addHoldsPrefix If true, predicate variables are not
     * instantiated
     *
     * @return an ArrayList of Formula(s), which could be empty.
     */
    private static Collection<Formula> replacePredVarsAndRowVars(Formula form, KB kb, boolean addHoldsPrefix) {

        Formula startF = new Formula();
        startF.read(form.theFormula);
        LinkedHashSet<Formula> accumulator = new LinkedHashSet<>();
        accumulator.add(startF);
        Collection<Formula> working = new ArrayList<>();
        int prevAccumulatorSize = 0;

        while (accumulator.size() != prevAccumulatorSize) {
            prevAccumulatorSize = accumulator.size();

            if (!addHoldsPrefix) {
                working.clear();
                working.addAll(accumulator);
                accumulator.clear();
                for (Formula f : working) {
                    Set<Formula> instantiations = PredVarInst.instantiatePredVars(f, kb);
                    form.errors.addAll(f.getErrors());

                    if (instantiations != null) {
                        if (instantiations.isEmpty()) {
                            accumulator.add(f);
                        } else {
                            accumulator.addAll(instantiations);
                        }
                    }
                }
            }


            if (!accumulator.isEmpty() && (accumulator.size() < AXIOM_EXPANSION_LIMIT)) {
                working.clear();
                working.addAll(accumulator);
                accumulator.clear();
                for (Formula f : working) {
                    if (RowVarExpansion) {
                        if (accumulator.addAll(RowVars.expandRowVars(kb, f)) && accumulator.size() > AXIOM_EXPANSION_LIMIT) {
                            System.out.println("  AXIOM_EXPANSION_LIMIT EXCEEDED: " + AXIOM_EXPANSION_LIMIT);
                            break;
                        }
                    } else {
                        accumulator.add(f);
                    }
                }
            }
        }
        return accumulator;
    }

//    /** ***************************************************************
//     * Returns true if this Formula appears not to have any of the
//     * characteristics that would cause it to be rejected during
//     * translation to TPTP form, or cause problems during inference.
//     * Otherwise, returns false.
//     *
//     * @param query true if this Formula represents a query, else
//     * false.
//     *
//     * @param kb The KB object to be used for evaluating the
//     * suitability of this Formula.
//     *
//     * @return boolean
//     */
//    private static boolean isOkForInference(Formula f, boolean query, KB kb) {
//
//
//        boolean pass = !(
//                StringUtil.containsNonAsciiChars(f.theFormula)
//
//
//                        || (!query
//                        && !Formula.isLogicalOperator(f.car())
//
//                        && (f.theFormula.indexOf('"') == -1)
//
//                        && f.theFormula.matches(".*\\?\\w+.*"))
//        );
//        return pass;
//    }

    /** ***************************************************************
     * Adds statements of the form (instance <Entity> <SetOrClass>) if
     * they are not already in the KB.
     *
     * @param kb The KB to be used for processing the input Formulae
     * in variableReplacements
     *
     * @param isQuery If true, this method just returns the initial
     * input List, variableReplacements, with no additions
     *
     * @param variableReplacements A List of Formulae in which
     * predicate variables and row variables have already been
     * replaced, and to which (instance <Entity> <SetOrClass>)
     * Formulae might be added
     *
     * @return an ArrayList of Formula(s), which could be larger than
     * the input List, variableReplacements, or could be empty.
     */
    private static ArrayList<Formula> addInstancesOfSetOrClass(Formula form, KB kb,
                                                               boolean isQuery, List<Formula> variableReplacements) {

        ArrayList<Formula> result = new ArrayList<>();
        if ((variableReplacements != null) && !variableReplacements.isEmpty()) {
            if (isQuery)
                result.addAll(variableReplacements);
            else {
                Set<Formula> formulae = new HashSet<>();
                String arg0;
                Formula f;
                for (Formula variableReplacement : variableReplacements) {
                    f = variableReplacement;
                    formulae.add(f);
                    if (f.listP() && !f.empty()) {
                        arg0 = f.car();
                        int start = switch (arg0) {
                            case "subclass" -> 0;
                            case "instance" -> 1;
                            default -> -1;
                        };
                        if (start > -1) {
                            ArrayList<String> args =
                                new ArrayList<>(Arrays.asList(f.getArgument(1), f.getArgument(2)));
                            int argslen = args.size();
                            String ioStr;
                            Formula ioF;
                            String arg;
                            for (int i = start; i < argslen; i++) {
                                arg = args.get(i);
                                if (!Formula.isVariable(arg) && !"SetOrClass".equals(arg) && Formula.atom(arg)) {
                                    StringBuilder sb = new StringBuilder();
                                    sb.setLength(0);
                                    sb.append("(instance ");
                                    sb.append(arg);
                                    sb.append(" SetOrClass)");
                                    ioF = new Formula();
                                    ioStr = sb.toString()/*.intern()*/;
                                    ioF.read(ioStr);
                                    ioF.sourceFile = form.sourceFile;
                                    if (!kb.formulaMap.containsKey(ioStr)) {
                                        formulae.add(ioF);
                                    }
                                }
                            }
                        }
                    }
                }
                result.addAll(formulae);
            }
        }
        return result;
    }

    /** ***************************************************************
     * Pre-process a formula before sending it to the theorem prover.
     * This includes ignoring meta-knowledge like documentation strings,
     * translating mathematical operators, quoting higher-order formulas,
     * expanding row variables and prepending the 'holds__' predicate.
     *
     * @param kb The KB to be used for processing this Formula
     *
     * @return an ArrayList of Formula(s), which could be empty.
     *
     */
    public static List<Formula> preProcess(Formula form, KB kb) {

        List<Formula> results = new ArrayList<>();
        if (!StringUtil.emptyString(form.theFormula)) {
            if (!form.isBalancedList()) {
                String errStr = "Unbalanced parentheses or quotes in: " + form.theFormula;
                form.errors.add(errStr);
                return results;
            }
            Formula f = new Formula();
            f.read(form.theFormula);
            if (StringUtil.containsNonAsciiChars(f.theFormula))
                f.theFormula = StringUtil.replaceNonAsciiChars(f.theFormula);

            KBmanager mgr = KBmanager.manager;
            boolean addHoldsPrefix = "yes".equalsIgnoreCase(mgr.getPref("holdsPrefix"));
            Collection<Formula> variableReplacements = replacePredVarsAndRowVars(form, kb, addHoldsPrefix);
            form.errors.addAll(f.getErrors());

            if (!variableReplacements.isEmpty()) {
                Formula fnew;
                String theNewFormula;
                boolean translateMath = true;
                boolean translateIneq = true;
                boolean ignoreStrings = false;
                for (Formula formula : variableReplacements) {
                    fnew = formula;
                    theNewFormula = FormulaPreprocessor.preProcessRecurse(fnew, "", ignoreStrings, translateIneq, translateMath, kb);
                    fnew.read(theNewFormula);

                    form.errors.addAll(fnew.getErrors());
                    fnew.sourceFile = form.sourceFile;
                    if (!StringUtil.emptyString(theNewFormula))
                        results.add(fnew);
//                    if (debug) System.out.println("preProcess: results: " + results);
                }
            }
        }
//        if (debug) System.out.println("INFO in FormulaPreprocessor.preProcess(): 1 result: " + results);
//        KBmanager mgr = KBmanager.manager;
//        boolean typePrefix = "yes".equalsIgnoreCase(mgr.getPref("typePrefix"));
//        if (typePrefix && !isQuery) {
//            for (Formula f : results) {
//                FormulaPreprocessor fp = new FormulaPreprocessor();
//                f.read(FormulaPreprocessor.addTypeRestrictions(f, kb).theFormula);
//            }
//        }
//        if (debug) System.out.println("INFO in FormulaPreprocessor.preProcess(): 2 result: " + results);
        return results;
    }

    /** ***************************************************************
     */
    public static void testFindTypes() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));

        System.out.println();
        String strf = "(=> (forall (?ELEMENT) (<=> (element ?ELEMENT ?SET1) " +
                "(element ?ELEMENT ?SET2))) (equal ?SET1 ?SET2))";
        Formula f = new Formula();
        f.read(strf);
        System.out.println("Formula: " + f);
        System.out.println("Var types: " + computeVariableTypes(f, kb));

        System.out.println();
        strf = "(=> (and (attribute ?AREA LowTerrain) (part ?ZONE ?AREA)" +
                " (slopeGradient ?ZONE ?SLOPE)) (greaterThan 0.03 ?SLOPE))";
        f.read(strf);
        System.out.println("Formula: " + f);
        System.out.println("Var types: " + computeVariableTypes(f, kb));

        System.out.println();
        strf = "(=> (and (typicalPart ?PART ?WHOLE) (instance ?X ?PART) " +
                "(equal ?PARTPROB (ProbabilityFn (exists (?Y) (and " +
                "(instance ?Y ?WHOLE) (part ?X ?Y))))) (equal ?NOTPARTPROB " +
                "(ProbabilityFn (not (exists (?Z) (and (instance ?Z ?WHOLE) " +
                "(part ?X ?Z))))))) (greaterThan ?PARTPROB ?NOTPARTPROB))";
        f.read(strf);
        System.out.println("Formula: " + f);
        System.out.println("Var types: " + computeVariableTypes(f, kb));

        System.out.println();
        strf = "(<=> (instance ?REL TransitiveRelation) " +
                "(forall (?INST1 ?INST2 ?INST3) " +
                "(=> (and (?REL ?INST1 ?INST2) " +
                "(?REL ?INST2 ?INST3)) (?REL ?INST1 ?INST3))))";
        f.read(strf);
        System.out.println("Formula: " + f);
        System.out.println("Var types: " + computeVariableTypes(f, kb));
        System.out.println("Explicit types: " + findExplicitTypesInAntecedent(kb, f));
    }

    /** ***************************************************************
     */
    public static void testFindExplicit() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));
        String formStr = "(<=> (instance ?REL TransitiveRelation) " +
                "(forall (?INST1 ?INST2 ?INST3) " +
                "(=> (and (?REL ?INST1 ?INST2) " +
                "(?REL ?INST2 ?INST3)) (?REL ?INST1 ?INST3))))";
        Formula f = new Formula(formStr);
//        FormulaPreprocessor fp = new FormulaPreprocessor();
        System.out.println("Formula: " + f);
        Pattern p = Pattern.compile("\\(instance (\\?[a-zA-Z0-9]+) ([a-zA-Z0-9\\-_]+)");
        Matcher m = p.matcher(formStr);
        m.find();
        String var = m.group(1);
        String cl = m.group(2);
        System.out.println("FormulaPreprocessor.testExplicit(): " + var + ' ' + cl);
        System.out.println("Explicit types: " + findExplicitTypesInAntecedent(kb, f));
    }

    /** ***************************************************************
     */
    public static void testAddTypes() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));

        System.out.println();
        String strf = "(=> (forall (?ELEMENT) (<=> (element ?ELEMENT ?SET1) " +
                "(element ?ELEMENT ?SET2))) (equal ?SET1 ?SET2))";
        Formula f = new Formula();
        f.read(strf);
//        FormulaPreprocessor fp = new FormulaPreprocessor();
        FormulaPreprocessor.debug = true;
        System.out.println(FormulaPreprocessor.addTypeRestrictions(f, kb));

        System.out.println();
        strf = "(=> (and (attribute ?AREA LowTerrain) (part ?ZONE ?AREA)" +
                " (slopeGradient ?ZONE ?SLOPE)) (greaterThan 0.03 ?SLOPE))";
        f.read(strf);
//        fp = new FormulaPreprocessor();
        System.out.println(FormulaPreprocessor.addTypeRestrictions(f, kb));

        System.out.println();
        strf = "(=> (and (typicalPart ?PART ?WHOLE) (instance ?X ?PART) " +
                "(equal ?PARTPROB (ProbabilityFn (exists (?Y) (and " +
                "(instance ?Y ?WHOLE) (part ?X ?Y))))) (equal ?NOTPARTPROB " +
                "(ProbabilityFn (not (exists (?Z) (and (instance ?Z ?WHOLE) " +
                "(part ?X ?Z))))))) (greaterThan ?PARTPROB ?NOTPARTPROB))";
        f.read(strf);
//        fp = new FormulaPreprocessor();
        System.out.println(FormulaPreprocessor.addTypeRestrictions(f, kb));
    }

    /** ***************************************************************
     */
    public static void testOne() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));

        System.out.println();
        System.out.println();
//        FormulaPreprocessor fp = new FormulaPreprocessor();
        FormulaPreprocessor.debug = true;
        String strf = "(=>\n" +
                "    (equal\n" +
                "        (GreatestCommonDivisorFn @ROW) ?NUMBER)\n" +
                "    (forall (?ELEMENT)\n" +
                "        (=>\n" +
                "            (inList ?ELEMENT\n" +
                "                (ListFn @ROW))\n" +
                "            (equal\n" +
                "                (RemainderFn ?ELEMENT ?NUMBER) 0))))";
        Formula f = new Formula();
        f.read(strf);
//        fp = new FormulaPreprocessor();

        System.out.println(preProcess(f, kb));
    }

    /** ***************************************************************
     */
    public static void testTwo() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));

        System.out.println();
        System.out.println();
//        FormulaPreprocessor fp = new FormulaPreprocessor();
        FormulaPreprocessor.debug = true;
        String strf = "(equal (AbsoluteValueFn ?NUMBER1) 2)";
        Formula f = new Formula();
        f.read(strf);
//        fp = new FormulaPreprocessor();
        System.out.println("testTwo(): equality: " + preProcess(f, kb));
    }

    /** ***************************************************************
     */
    public static void testThree() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));

        System.out.println();
        System.out.println();
//        FormulaPreprocessor fp = new FormulaPreprocessor();
        FormulaPreprocessor.debug = true;
        String strf = '\n' +
                "(<=>\n" +
                "    (and\n" +
                "        (equal\n" +
                "            (AbsoluteValueFn ?NUMBER1) ?NUMBER2)\n" +
                "        (instance ?NUMBER1 RealNumber)\n" +
                "        (instance ?NUMBER2 RealNumber))\n" +
                "    (or\n" +
                "        (and\n" +
                "            (instance ?NUMBER1 NonnegativeRealNumber)\n" +
                "            (equal ?NUMBER1 ?NUMBER2))\n" +
                "        (and\n" +
                "            (instance ?NUMBER1 NegativeRealNumber)\n" +
                "            (equal ?NUMBER2\n" +
                "                (SubtractionFn 0 ?NUMBER1)))))";
        Formula f = new Formula();
        f.read(strf);
//        fp = new FormulaPreprocessor();
        System.out.println("testThree(): " + preProcess(f, kb));
    }

    /** ***************************************************************
     */
    public static void testFour() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));

        System.out.println();
        System.out.println();
//        FormulaPreprocessor fp = new FormulaPreprocessor();
        FormulaPreprocessor.debug = true;
        String strf = "(forall (?NUMBER ?ELEMENT ?CLASS)\n" +
                "        (=>\n" +
                "          (equal ?ELEMENT\n" +
                "            (ListOrderFn\n" +
                "              (ListFn_1 ?FOO) ?NUMBER))\n" +
                "          (instance ?ELEMENT ?CLASS)))";
        Formula f = new Formula();
        f.read(strf);
//        fp = new FormulaPreprocessor();
        System.out.println("testFour() signature for ListFn: " + kb.kbCache.signatures.get("ListFn"));
        System.out.println("testFour() valence for ListFn: " + kb.kbCache.valences.get("ListFn"));
        System.out.println("testFour() signature for ListFn_1: " + kb.kbCache.signatures.get("ListFn_1"));
        System.out.println("testFour() valence for ListFn_1: " + kb.kbCache.valences.get("ListFn_1"));
        System.out.println("testFour(): " + FormulaPreprocessor.addTypeRestrictions(f, kb));
    }
//
//    /** ***************************************************************
//     */
//    public static void main(String[] args) {
//
//
//
//        testFour();
//
//
//
//
//    }

}