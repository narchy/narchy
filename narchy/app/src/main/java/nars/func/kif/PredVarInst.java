package nars.func.kif;

/**
 * This code is copyright Articulate Software (c) 2014.
 * This software is released under the GNU Public License <http:
 * Users of this code also consent, by use of this code, to credit Articulate Software
 * and Teknowledge in any writings, briefings, publications, presentations, or
 * other representations of any software which incorporates, builds on, or uses this
 * code.  Please cite the following article in any publication with references:
 * <p>
 * Pease, A., (2003). The Sigma Ontology Development Environment,
 * in Working Notes of the IJCAI-2003 Workshop on Ontology and Distributed Systems,
 * August 9, Acapulco, Mexico.  See also sigmakee.sourceforge.net
 */

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public enum PredVarInst {
    ;


    private static final Map<String, Integer> predVarArity = new HashMap<>();


    private static final HashMap<String, HashSet<String>> candidatePredicates = new HashMap<>();


    private static final List<String> logicalTerms = Arrays.asList("forall", "exists", "=>", "and", "or", "<=>", "not");

    /** ***************************************************************
     * There are two type conditions:
     * one type condition is extracted from domain expression;
     * second type condition is specifically define in the antecedent
     * of a rule with an instance or subclass expression;
     *
     * @param input formula
     * @param types type condition extracted from domain expression
     *
     * @return add explicit type condition into types
     */
    private static HashMap<String,HashSet<String>> addExplicitTypes(KB kb, Formula input, HashMap<String,HashSet<String>> types) {

        HashMap<String, HashSet<String>> explicit = FormulaPreprocessor.findExplicitTypesInAntecedent(kb, input);
        if (explicit == null || explicit.keySet() == null || explicit.keySet().isEmpty())
            return types;
        HashMap<String, HashSet<String>> result = new HashMap<>();
        for (Map.Entry<String, HashSet<String>> entry : explicit.entrySet()) {
            String var = entry.getKey();
            HashSet<String> hs = new HashSet<>();
            if (types.containsKey(var))
                hs = types.get(var);
            hs.addAll(entry.getValue());
            result.put(var, hs);
        }
        return result;
    }

    /** ***************************************************************
     * @param input formula
     * @param kb knowledge base
     * @return A list of formulas where predicate variables are instantiated;
     *         There are three possible returns:
     *         return null if input contains predicate variables but cannot be instantiated;
     *         return empty if input contains no predicate variables;
     *         return a list of instantiated formulas if the predicate variables are instantiated;
     */
    public static Set<Formula> instantiatePredVars(Formula input, KB kb) {

        HashSet<String> predVars = gatherPredVars(kb,input);
        if (predVars == null )
            return null;

        Set<Formula> result = new HashSet<>();
        if (predVars.isEmpty())
            return result;

        HashMap<String,HashSet<String>> varTypes = findPredVarTypes(input,kb);

        varTypes = addExplicitTypes(kb, input, varTypes);
        for (Map.Entry<String, HashSet<String>> entry : varTypes.entrySet()) {
            String var = entry.getKey();
            for (String rel : kb.kbCache.relations) {
                if (kb.kbCache.valences.get(rel).equals(predVarArity.get(var))) {
                    boolean ok = entry.getValue().stream().allMatch(varType -> kb.isInstanceOf(rel, varType));

                    if (ok) {
                        Formula f = input.deepCopy();
                        f = f.replaceVar(var, rel);
                        Formula f2 = input.deepCopy();
                        f2.theFormula = f.theFormula;
                        result.add(f);
                    }
                }
            }
        }
        if (result.isEmpty()) {
            String errStr = "No predicate instantiations for ";
            errStr += input.theFormula;
            input.errors.add(errStr);
            return null;
        }

        return result;
    }

    /** ***************************************************************
     *
     */
    private static String hasCorrectArityRecurse(Formula f, KB kb) throws IllegalArgumentException, TypeNotPresentException {

        if (f == null || StringUtil.emptyString(f.theFormula) || f.empty() ||
                Formula.atom(f.theFormula) || f.isVariable())
            return null;
        String rel = f.getArgument(0);
        List<String> l = f.complexArgumentsToArrayList(1);

        if (Formula.listP(rel)) {
            Formula p = new Formula();
            p.read(rel);
            String res = hasCorrectArityRecurse(p, kb);
            if (!StringUtil.emptyString(res))
                return res;
        } else {

            if (l == null) {
                return null;
            }
            Integer intval = null;
            if (kb.kbCache != null && kb.kbCache.valences != null)
                intval = kb.kbCache.valences.get(rel);
            int val = 0;
            if (intval != null)
                val = intval;
            else {
                if (!l.isEmpty() && !logicalTerms.contains(rel) && !rel.startsWith("?")) {
                    System.out.printf("INFO in PredVarInst.hasCorrectArityRecurse(): Predicate %s does not have an arity defined in KB, can't get the arity number!\n%s\n", rel, f, f.getSourceFile(), f.startLine);

                }
            }

            if (kb.kbCache == null || !kb.kbCache.transInstOf(rel, "VariableArityRelation")) {

                if (val > 0 && val != l.size()) {
                    System.out.println("Error in PredVarInst.hasCorrectArityRecurse() expected arity " +
                            val + " but found " + l.size() + " for relation " + rel + " in formula: " + f);
                    throw new IllegalArgumentException(rel);
                }
            }
            if (f.isSimpleClause(kb)) {

                for (String arg : l) {
                    if (kb.isFunction(arg)) {
                        String result = hasCorrectArityRecurse(new Formula(arg), kb);
                        if (!StringUtil.emptyString(result))
                            return result;
                    }
                }
            } else {

                if (Formula.isQuantifier(f.car()))
                    return hasCorrectArityRecurse(f.cddrAsFormula(), kb);
            }
        }
        if (l != null && !l.isEmpty()) {
            for (String k : l) {
                if (Formula.atom(k))
                    continue;
                Formula ff = new Formula();
                ff.read(k);
                String res = hasCorrectArityRecurse(ff, kb);
                if (!StringUtil.emptyString(res))
                    return res;
            }
        }
        return null;
    }

    /** ***************************************************************
     * If arity is correct, return null, otherwise, return the predicate
     * that has its arity violated in the given formula.
     */
    public static String hasCorrectArity(Formula f, KB kb) {
        String res = null;
        try {
            res = hasCorrectArityRecurse(f, kb);
        } catch (IllegalArgumentException e) {
            System.out.printf("FileName:%s\nLine number:%d\n", f.getSourceFile(), f.startLine);
            return e.getMessage();
        }
        return res;
    }

    /** ***************************************************************
     * This method returns an ArrayList of query answer literals.  The
     * first element is an ArrayList of query literals that might be
     * used to simplify the Formula to be instantiated.  The second
     * element is the query literal (ArrayList) that will be used as a
     * template for doing the variable substitutions.  All subsequent
     * elements are ground literals (ArrayLists).
     *
     * kb A KB to query for answers.
     *
     * queryLits A List of query literals.  The first item in
     * the list will be a SUO-KIF variable (String), which indexes the
     * list.  Each subsequent item is a query literal (List).
     *
     * @return An ArrayList of literals, or an empty ArrayList if no
     * query answers can be found.

    private static ArrayList computeSubstitutionTuples(KB kb, List queryLits) {

    ArrayList result = new ArrayList();
    if (kb != null && queryLits != null && !queryLits.isEmpty()) {
    String idxVar = (String) queryLits.get(0);
    int i = 0;
    int j = 0;


    ArrayList sortedQLits = new ArrayList(queryLits);
    sortedQLits.remove(0);
    if (sortedQLits.size() > 1) {
    Comparator comp = new Comparator() {
    public int compare(Object o1, Object o2) {
    Integer c1 = Integer.valueOf(getVarCount((List) o1));
    Integer c2 = Integer.valueOf(getVarCount((List) o2));
    return c1.compareTo(c2);
    }
    };
    Collections.sort(sortedQLits, Collections.reverseOrder(comp));
    }


    List tmplist = new ArrayList(sortedQLits);
    List ioLits = new ArrayList();
    sortedQLits.clear();
    List ql = null;
    for (Iterator iql = tmplist.iterator(); iql.hasNext();) {
    ql = (List) iql.next();
    if (((String)(ql.get(0))).equals("instance"))
    ioLits.addAt(ql);
    else
    sortedQLits.addAt(ql);
    }
    sortedQLits.addAll(ioLits);



    ArrayList simplificationLits = new ArrayList();




    List keyLit = null;




    ArrayList answers = null;

    Set working = new HashSet();
    ArrayList accumulator = null;

    boolean satisfiable = true;
    boolean tryNextQueryLiteral = true;


    for (i = 0; (i < sortedQLits.size()) && tryNextQueryLiteral; i++) {
    ql = (List) sortedQLits.get(i);
    accumulator = kb.askWithLiteral(ql);
    satisfiable = ((accumulator != null) && !accumulator.isEmpty());
    tryNextQueryLiteral = (satisfiable || (getVarCount(ql) > 1));

    if (satisfiable) {
    simplificationLits.addAt(ql);
    if (keyLit == null) {
    keyLit = ql;
    answers = KB.formulasToArrayLists(accumulator);
    }
    else {
    accumulator = KB.formulasToArrayLists(accumulator);


    working.clear();
    List ql2 = null;
    int varPos = ql.indexOf(idxVar);
    String target = null;
    for (j = 0; j < accumulator.size(); j++) {
    ql2 = (List) accumulator.get(j);
    target = (String) (ql2.get(varPos));

    working.addAt(target);

    }
    accumulator.clear();
    accumulator.addAll(answers);
    answers.clear();
    varPos = keyLit.indexOf(idxVar);
    for (j = 0; j < accumulator.size(); j++) {
    ql2 = (List) accumulator.get(j);
    target = (String) (ql2.get(varPos));
    if (working.contains(target))
    answers.addAt(ql2);
    }
    }
    }
    }
    if (satisfiable && (keyLit != null)) {
    result.addAt(simplificationLits);
    result.addAt(keyLit);
    result.addAll(answers);
    }
    else
    result.clear();
    }
    return result;
    }

    /** ***************************************************************
     * This method returns an ArrayList in which each element is
     * another ArrayList.  The head of each element is a variable.
     * The subsequent objects in each element are query literals
     * (ArrayLists).
     *
     * kb The KB to use for computing variable type signatures.
     *
     * varTypeMap A Map from variables to their types, as
     * explained in the javadoc entry for gatherPredVars(kb)
     *
     * Formula.gatherPredVars(KB kb)
     *
     * @return An ArrayList, or null if the input formula contains no
     * predicate variables.

    private static ArrayList prepareIndexedQueryLiterals(KB kb, Map varTypeMap) {

    ArrayList ans = new ArrayList();
    HashSet<String> varsWithTypes = ((varTypeMap instanceof Map)
    ? varTypeMap
    : gatherPredVars());
    if (!varsWithTypes.isEmpty()) {
    String yOrN = (String) varsWithTypes.get("arg0");

    if (!StringUtil.emptyString(yOrN) && yOrN.equalsIgnoreCase("yes")) {

    ArrayList varWithTypes = null;
    ArrayList indexedQueryLits = null;

    String var = null;
    for (Iterator it = varsWithTypes.keySet().iterator(); it.hasNext();) {
    var = (String) it.next();
    if (Formula.isVariable(var)) {
    varWithTypes = (ArrayList) varsWithTypes.get(var);
    indexedQueryLits = gatherPredVarQueryLits(kb, varWithTypes);
    if (!indexedQueryLits.isEmpty()) {
    ans.addAt(indexedQueryLits);
    }
    }
    }
    }
    }
    return ans;
    }

    /** ***************************************************************
     * Get a set of all the predicate variables in the formula
     */
    private static HashSet<String> gatherPredVarRecurse(KB kb, Formula f) {

        HashSet<String> ans = new HashSet<>();

        if (f == null || f.empty() || Formula.atom(f.theFormula) || f.isVariable())
            return ans;
        if (f.isSimpleClause(kb)) {
            String arg0 = f.getArgument(0);

            if (arg0.startsWith("?")) {
                List<String> arglist = f.complexArgumentsToArrayList(1);
                if (arglist != null && !arglist.isEmpty()) {
                    ans.add(arg0);
                    predVarArity.put(arg0, arglist.size());
                }
                else {

            }
            }
        }
        else if (Formula.isQuantifier(f.car())) {

            Formula f2 = f.cddrAsFormula();
            ans.addAll(gatherPredVarRecurse(kb, f2));
        } else {

            ans.addAll(gatherPredVarRecurse(kb, f.carAsFormula()));
            ans.addAll(gatherPredVarRecurse(kb, f.cdrAsFormula()));
        }

        return ans;
    }

    /** ***************************************************************
     * Add a key,value pair for a multiple value ArrayList

     private static HashMap<String,HashSet<String>>
     addToArrayList(HashMap<String,HashSet<String>> ar, String key, String value) {

     HashSet<String> val = ar.get(key);
     if (val == null)
     val = new HashSet<String>();
     val.addAt(value);
     ar.put(key, val);
     return ar;
     }

     /** ***************************************************************
     * Get a set of all the types for predicate variables in the formula.
     *
     * @return a HashMap in which the keys are predicate variables,
     * and the values are HashSets containing one or more class
     * names that indicate the type constraints that apply to the
     * variable.  If no predicate variables can be gathered from the
     * Formula, the HashMap will be empty.  Note that predicate variables
     * must logically be instances (of class Relation).
     */
    static HashMap<String, HashSet<String>> findPredVarTypes(Formula f, KB kb) {

        HashSet<String> predVars = gatherPredVars(kb,f);
        HashMap<String,HashSet<String>> typeMap = FormulaPreprocessor.computeVariableTypes(f, kb);
        HashMap<String, HashSet<String>> result = predVars.stream().filter(typeMap::containsKey).collect(Collectors.toMap(Function.identity(), typeMap::get, (a, b) -> b, HashMap::new));
        return result;
    }

    /** ***************************************************************
     * Collect and return all predicate variables for the given formula
     */
    protected static HashSet<String> gatherPredVars(KB kb, Formula f) {

        HashSet<String> varlist = null;
        HashMap<String,HashSet<String>> ans = new HashMap<>();
        if (!StringUtil.emptyString(f.theFormula)) {
            varlist = gatherPredVarRecurse(kb,f);
        }
        return varlist;
    }

    /** ***************************************************************
     * This method tries to remove literals from the Formula that
     * match litArr.  It is intended for use in simplification of this
     * Formula during predicate variable instantiation, and so only
     * attempts removals that are likely to be safe in that context.
     *
     * litArr A List object representing a SUO-KIF atomic
     * formula.
     *
     * @return A new Formula with at least some occurrences of litF
     * removed, or the original Formula if no removals are possible.

    private static Formula maybeRemoveMatchingLits(List litArr) {
    Formula f = KB.literalListToFormula(litArr);
    return maybeRemoveMatchingLits(_f,f);
    }

    /** ***************************************************************
     * This method tries to remove literals from the Formula that
     * match litF.  It is intended for use in simplification of this
     * Formula during predicate variable instantiation, and so only
     * attempts removals that are likely to be safe in that context.
     *
     * litF A SUO-KIF literal (atomic Formula).
     *
     * @return A new Formula with at least some occurrences of litF
     * removed, or the original Formula if no removals are possible.

    private static Formula maybeRemoveMatchingLits(Formula input, Formula litF) {

    Formula result = null;
    Formula f = input;
    if (f.listP() && !f.empty()) {
    StringBuilder litBuf = new StringBuilder();
    String arg0 = f.car();
    if (Arrays.asList(Formula.IF, Formula.IFF).contains(arg0)) {
    String arg1 = f.getArgument(1);
    String arg2 = f.getArgument(2);
    if (arg1.equals(litF.theFormula)) {
    Formula arg2F = new Formula();
    arg2F.read(arg2);
    litBuf.append(maybeRemoveMatchingLits(arg2F,litF).theFormula);
    }
    else if (arg2.equals(litF.theFormula)) {
    Formula arg1F = new Formula();
    arg1F.read(arg1);
    litBuf.append(maybeRemoveMatchingLits(arg1F,litF).theFormula);
    }
    else {
    Formula arg1F = new Formula();
    arg1F.read(arg1);
    Formula arg2F = new Formula();
    arg2F.read(arg2);
    litBuf.append("(" + arg0 + " "
    + maybeRemoveMatchingLits(arg1F,litF).theFormula + " "
    + maybeRemoveMatchingLits(arg2F,litF).theFormula + ")");
    }
    }
    else if (Formula.isQuantifier(arg0)
    || arg0.equals("holdsDuring")
    || arg0.equals("KappaFn")) {
    Formula arg2F = new Formula();
    arg2F.read(f.caddr());
    litBuf.append("(" + arg0 + " " + f.cadr() + " "
    + maybeRemoveMatchingLits(arg2F,litF).theFormula + ")");
    }
    else if (Formula.isCommutative(arg0)) {
    List litArr = f.literalToArrayList();
    if (litArr.contains(litF.theFormula))
    litArr.remove(litF.theFormula);
    String args = "";
    int len = litArr.size();
    for (int i = 1 ; i < len ; i++) {
    Formula argF = new Formula();
    argF.read((String) litArr.get(i));
    args += (" " + maybeRemoveMatchingLits(argF,litF).theFormula);
    }
    if (len > 2)
    args = ("(" + arg0 + args + ")");
    else
    args = args.trim();
    litBuf.append(args);
    }
    else {
    litBuf.append(f.theFormula);
    }
    Formula newF = new Formula();
    newF.read(litBuf.toString());
    result = newF;
    }
    if (result == null)
    result = input;
    return result;
    }

    /** ***************************************************************
     * Return true if the input predicate can take relation names a
     * arguments, else returns false.

    private static boolean isPossibleRelnArgQueryPred (KB kb, String predicate) {

    ArrayList<String> sig = kb.kbCache.signatures.get(predicate);
    for (int i = 1; i < sig.size(); i++) {
    String argType = sig.get(i);
    if (!argType.endsWith("+")) {
    HashSet<String> prents = kb.kbCache.getParentClasses(argType);
    if (prents != null && prents.contains("Relation"))
    return true;
    }
    }
    return false;
    }

    /** ***************************************************************
     * This method collects and returns literals likely to be of use
     * as templates for retrieving predicates to be substituted for
     * var.
     *
     * varWithTypes A List containing a variable followed,
     * optionally, by class names indicating the type of the variable.
     *
     * @return An ArrayList of literals (Lists) with var at the head.
     * The first element of the ArrayList is the variable (String).
     * Subsequent elements are Lists corresponding to SUO-KIF
     * formulas, which will be used as query templates.

    private static ArrayList gatherPredVarQueryLits(KB kb, List varWithTypes) {

    ArrayList ans = new ArrayList();
    String var = (String) varWithTypes.get(0);
    Set added = new HashSet();


    List clauses = _f.getClauses();
    Map varMap = _f.getVarMap();
    String qlString = null;
    ArrayList queryLit = null;

    if (clauses != null) {
    Iterator it2 = null;
    Formula f = null;
    Iterator it1 = clauses.iterator();
    while (it1.hasNext()) {
    List clause = (List) it1.next();
    List negLits = (List) clause.get(0);

    if (!negLits.isEmpty()) {
    int flen = -1;
    String arg = null;
    String arg0 = null;
    String target = null;
    String origVar = null;
    List lit = null;
    boolean working = true;
    for (int ci = 0;
    ci < 1;

    ci++) {

    lit = (List)(clause.get(ci));
    it2 = lit.iterator();

    while (it2.hasNext()) {
    f = (Formula) it2.next();
    if (f.theFormula.matches(".*SkFn\\s+\\d+.*")
    || f.theFormula.matches(".*Sk\\d+.*"))
    continue;
    flen = f.listLength();
    arg0 = f.getArgument(0);

    if (!StringUtil.emptyString(arg0)) {


    if (Formula.isVariable(arg0)) {
    origVar = Clausifier.getOriginalVar(arg0, varMap);
    if (origVar.equals(var)
    && !varWithTypes.contains("Predicate")) {
    varWithTypes.addAt("Predicate");
    }
    }
    else {
    queryLit = new ArrayList();
    queryLit.addAt(arg0);
    boolean foundVar = false;
    for (int i = 1; i < flen; i++) {
    arg = f.getArgument(i);
    if (!Formula.listP(arg)) {
    if (Formula.isVariable(arg)) {
    arg = Clausifier.getOriginalVar(arg, varMap);
    if (arg.equals(var))
    foundVar = true;
    }
    queryLit.addAt(arg);
    }
    }

    if (queryLit.size() != flen)
    continue;




    if (isPossibleRelnArgQueryPred(kb, arg0) && foundVar) {

    target = "";
    if (queryLit.size() > 2)
    target = (String) queryLit.get(2);
    if (!(arg0.equals("instance")
    && target.equals("Relation"))) {
    String queryLitStr = queryLit.toString().intern();
    if (!added.contains(queryLitStr)) {
    ans.addAt(queryLit);

    added.addAt(queryLitStr);
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }



    String argType = null;
    int vtLen = varWithTypes.size();
    if (vtLen > 1) {
    for (int j = 1 ; j < vtLen ; j++) {
    argType = (String) varWithTypes.get(j);
    if (!argType.equals("Relation")) {
    queryLit = new ArrayList();
    queryLit.addAt("instance");
    queryLit.addAt(var);
    queryLit.addAt(argType);
    qlString = queryLit.toString().intern();
    if (!added.contains(qlString)) {
    ans.addAt(queryLit);
    added.addAt(qlString);
    }
    }
    }
    }


    if (!ans.isEmpty())
    ans.addAt(0, var);
    return ans;
    }

    /** ***************************************************************
     */
    public static void arityTest() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));
        System.out.println("INFO in PredVarInst.test(): completed loading KBs");
        String formStr = "(=> " +
                "(and " +
                "(instance ?SOUND RadiatingSound) " +
                "(agent ?SOUND ?OBJ) " +
                "(attribute ?SOUND Audible)) " +
                "(exists (?HUMAN) " +
                "(and " +
                "(instance ?HUMAN Human) " +
                "(capability " +
                "(KappaFn ?HEAR " +
                "(and " +
                "(instance ?HEAR Hearing) " +
                "(agent ?HEAR ?HUMAN) " +
                "(destination ?HEAR ?HUMAN) " +
                "(origin ?HEAR ?OBJ))) agent ?HUMAN)))) ";
        Formula f = new Formula(formStr);
        System.out.println("INFO in PredVarInst.arityTest(): formula: " + f);
        System.out.println("INFO in PredVarInst.arityTest(): correct arity: " + hasCorrectArity(f, kb));
    }

    /** ***************************************************************
     */
    public static void test() {

        KBmanager.manager.initializeOnce();
        KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));
        System.out.println("INFO in PredVarInst.test(): completed loading KBs");
        if (kb.kbCache.transInstOf("exhaustiveAttribute", "VariableArityRelation")) {
            System.out.println("INFO in PredVarInst.test() variable arity: ");
        } else
            System.out.println("INFO in PredVarInst.test() not variable arity: ");
        System.out.println("INFO in PredVarInst.test(): " + kb.kbCache.instances.get("partition"));
        System.out.println("INFO in PredVarInst.test(): " + kb.kbCache.insts.contains("partition"));

        String formStr = "(<=> (instance ?REL TransitiveRelation) " +
                "(forall (?INST1 ?INST2 ?INST3) " +
                "(=> (and (?REL ?INST1 ?INST2) " +
                "(?REL ?INST2 ?INST3)) (?REL ?INST1 ?INST3))))";

        Formula f = new Formula(formStr);
        System.out.println("Formula: " + f);
        System.out.println("Pred vars: " + gatherPredVars(kb, f));
        System.out.println("Pred vars with types: " + findPredVarTypes(f, kb));
        System.out.println("Explicit types: " + FormulaPreprocessor.findExplicitTypesInAntecedent(kb, f));
        System.out.println("Instantiated: " + instantiatePredVars(f, kb));
        System.out.println();

        formStr = "(=> " +
                "(instance ?JURY Jury) " +
                "(holdsRight " +
                "(exists (?DECISION) " +
                "(and " +
                "(instance ?DECISION LegalDecision) " +
                "(agent ?DECISION ?JURY))) ?JURY))";

        f = new Formula(formStr);
        System.out.println("Formula: " + f);
        System.out.println("Pred vars: " + gatherPredVars(kb, f));
        System.out.println("Instantiated: " + instantiatePredVars(f, kb));

    }

    /** ***************************************************************
     */
    public static void main(String[] args) {


        test();
        /*
         KBmanager.getMgr().initializeOnce();
         KB kb = KBmanager.getMgr().getKB(KBmanager.getMgr().getPref("sumokbname"));
         String formStr = "(<=> (instance ?REL TransitiveRelation) " +
         "(forall (?INST1 ?INST2 ?INST3) " +
         "(=> (and (?REL ?INST1 ?INST2) " +
         "(?REL ?INST2 ?INST3)) (?REL ?INST1 ?INST3))))";
         formStr = "(=> (inverse ?REL1 ?REL2) (forall (?INST1 ?INST2) (<=> (?REL1 ?INST1 ?INST2) (?REL2 ?INST2 ?INST1))))";
         
         Formula f = kb.formulaMap.get(formStr);
         if (f == null) {
        	System.out.println("Error " + formStr + " not found.");
        	formStr = kb.formulas.get("ant-reflexiveOn").get(0);
        	f = kb.formulaMap.get(formStr);
         }
         
         System.out.println(instantiatePredVars(f,kb));
         */
    }
}