/* This code is copyright Articulate Software (c) 2003.  Some
portions copyright Teknowledge (c) 2003 and reused under the terms of
the GNU license.  This software is released under the GNU Public
License <http:
also consent, by use of this code, to credit Articulate Software and
Teknowledge in any writings, briefings, publications, presentations,
or other representations of any software which incorporates, builds
on, or uses this code.  Please cite the following article in any
publication with references:

Pease, A., (2003). The Sigma Ontology Development Environment, in
Working Notes of the IJCAI-2003 Workshop on Ontology and Distributed
Systems, August 9, Acapulco, Mexico. See also http:

Authors:
Adam Pease
Infosys LTD.

Formula is an important class that contains information and operations
about individual SUO-KIF formulas.
*/

package nars.func.kif;

import com.google.common.collect.Sets;
import jcog.data.list.Lst;
import org.eclipse.collections.impl.set.mutable.UnifiedSet;

import java.io.File;
import java.io.Serializable;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * ***********************************************************
 * Handle operations on an individual formula.  This includes
 * formatting for presentation as well as pre-processing for sending
 * to the inference engine.
 */
public class Formula implements Comparable, Serializable {

	public static final String AND = "and";
	public static final String OR = "or";
	public static final String NOT = "not";
	public static final String IF = "=>";
	public static final String IFF = "<=>";
	public static final String UQUANT = "forall";
	public static final String EQUANT = "exists";
	public static final String EQUAL = "equal";
	private static final String GT = "greaterThan";
	private static final String GTET = "greaterThanOrEqualTo";
	private static final String LT = "lessThan";
	private static final String LTET = "lessThanOrEqualTo";

	public static final String KAPPAFN = "KappaFn";
	private static final String PLUSFN = "AdditionFn";
	private static final String MINUSFN = "SubtractionFn";
	private static final String TIMESFN = "MultiplicationFn";
	private static final String DIVIDEFN = "DivisionFn";
	public static final String SKFN = "SkFn";
	public static final String SK_PREF = "Sk";
	public static final String FN_SUFF = "Fn";
	public static final String V_PREF = "?";
	public static final String R_PREF = "@";
	public static final String VX = "?X";
	public static final String VVAR = "?VAR";
	public static final String RVAR = "@ROW";

	public static final String LP = "(";
	public static final String RP = ")";
	public static final String SPACE = " ";

	private static final String LOG_TRUE = "True";
	public static final String LOG_FALSE = "False";

	/**
	 * The SUO-KIF logical operators.
	 */
	private static final List<String> LOGICAL_OPERATORS = Arrays.asList(UQUANT,
		EQUANT,
		AND,
		OR,
		NOT,
		IF,
		IFF);
	public static final List<String> DOC_PREDICATES = Arrays.asList("documentation",
		"comment",
		"format"

	);
	/**
	 * **************************************************************
	 * This constant indicates the maximum predicate arity supported
	 * by the current implementation of Sigma.
	 */
	static final int MAX_PREDICATE_ARITY = 7;
	private static final Set quoteChars = Set.of('"', '\'');
	/**
	 * SUO-KIF mathematical comparison predicates.
	 */
	private static final List<String> COMPARISON_OPERATORS = Arrays.asList(EQUAL,
		GT,
		GTET,
		LT,
		LTET);
	/**
	 * The SUO-KIF mathematical functions are implemented in Vampire, but not yet EProver.
	 */
	private static final List<String> MATH_FUNCTIONS = Arrays.asList(PLUSFN,
		MINUSFN,
		TIMESFN,
		DIVIDEFN);
	private static final Pattern empty = Pattern.compile("\\(\\s*\\)");
	private static final Pattern ATOM_pattern = Pattern.compile(".*\\s.*");
	//private static final Pattern TRIM_INNER = Pattern.compile("\\s+");
	private static final Pattern SKOLEMED = Pattern.compile("^.?" + SK_PREF + "\\S*\\s*\\d+");
	@Deprecated public final Collection<String> errors = new LinkedList<>();
	/**
	 * The source file in which the formula appears.
	 */
	public String sourceFile;
	/**
	 * The line in the file on which the formula starts.
	 */
	public int startLine;

//	/** Warnings found during execution. */
//    public TreeSet<String> warnings = new TreeSet<>();
	/**
	 * The line in the file on which the formula ends.
	 */
	public int endLine;

//    public static final String termMentionSuffix  = "__m";
//    public static final String classSymbolSuffix  = "__t";
//    public static final String termSymbolPrefix   = "s__";
//    public static final String termVariablePrefix = "V__";
//	/**
//	 * The length of the file in bytes at the position immediately
//	 * after the end of the formula.  This value is used only for
//	 * formulas entered via KB.tell().  In general, you should not
//	 * count on it being set to a value other than -1L.
//	 */
//	public long endFilePosition = -1L;
	/**
	 * The formula in textual forms.
	 */
	public String theFormula;
//	/**
//	 * **************************************************************
//	 * A list of TPTP formulas (Strings) that together constitute the
//	 * translation of theFormula.  This member is a List, because
//	 * predicate variable instantiation and row variable expansion
//	 * might cause theFormula to expand to several TPTP formulas.
//	 */
//	public ArrayList<String> theTptpFormulas = null;
	/**
	 * ****************************************************************
	 * A list of clausal (resolution) forms generated from this
	 * Formula.
	 */
	private ArrayList theClausalForm;

	/**
	 * ****************************************************************
	 * Constructor to build a formula from an existing formula.  This isn't
	 * a complete deepCopy() since it leaves out the errors and warnings
	 * variables
	 */
	public Formula(Formula f) {
		this.endLine = f.endLine;
		this.startLine = f.startLine;
		this.sourceFile = f.sourceFile;
		this.theFormula = f.theFormula;
	}

	/**
	 * ****************************************************************
	 */
	public Formula() {
	}

	/**
	 * ****************************************************************
	 * Just set the textual version of the formula
	 */
	public Formula(String f) {
		theFormula = f;
	}

	/**
	 * **************************************************************
	 * Test whether the String is a LISP atom.
	 */
	public static boolean atom(String s) {

		boolean ans = false;
		if (!StringUtil.emptyString(s)) {
			String str = s.trim();
			ans = StringUtil.isQuotedString(s) ||
				!str.contains(")") && !ATOM_pattern.matcher(str).matches();
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Test whether the String is an empty formula.  Not to be
	 * confused with a null string or empty string.  There must be
	 * parentheses with nothing or whitespace in the middle.
	 */
	public static boolean empty(String s) {
		return listP(s) && empty.matcher(s).matches();
	}

	/**
	 * **************************************************************
	 * Test whether the String is a list.
	 */
	public static boolean listP(String s) {

		if (!StringUtil.emptyString(s)) {
			String str = s.trim();
			if (str.isEmpty())
				return false;
			return str.startsWith("(") && str.endsWith(")");
		}
		return false;
	}

	/**
	 * **************************************************************
	 * Parse a String into an ArrayList of Formulas. The String must be
	 * a LISP-style list.
	 */
	private static List<Formula> parseList(String s) {

		Formula f = new Formula();
		f.read('(' + s + ')');

		if (f.empty())
			return List.of();
		List<Formula> result = new Lst<>();
		while (!f.empty()) {
			String car = f.car();
			f.read(f.cdr());
			Formula newForm = new Formula();
			newForm.read(car);
			result.add(newForm);
		}
		return result;
	}

	/**
	 * ****************************************************************
	 * Compares two formulae by recursively traversing its predicate
	 * structure and by building possible variable maps
	 * between the variables of the two formulae. If a complete mapping
	 * is possible, it is returned.
	 * Each recursive call returns a list of sets of variable pairs.
	 * Each pair is a variable from the first formula and
	 * its potential corresponding variable in the second formula. Each
	 * set is a potential complete mapping between all
	 * the variables in the first formula and the ones in the second. The
	 * returned list contains all possible sets, so
	 * in essence all possible valid mappings of variables between the two
	 * formulas. The method will reconcile the list
	 * returned by all one level deeper recursive calls and return the list
	 * of sets which offer no contradictions.
	 * <p>
	 * Note: for clauses with commutative
	 *
	 * @param f1
	 * @param f2
	 * @param kb
	 * @param memoMap a memo-ization mechanism designed to reduce the number
	 *                of recursive calls in "dynamic programming"
	 *                fashion
	 * @return null - if the formulas cannot be equals (due to having different predicates for example)
	 * empty list- formulas are equal, but there are no variables to map
	 * list 0f variable mapping sets the list of possible variable mapping sets which will make formulas equal
	 */
	private static List<Set<VariableMapping>> mapFormulaVariables(Formula f1, Formula f2, KB kb,
																  HashMap<FormulaUtil.FormulaMatchMemoMapKey, List<Set<VariableMapping>>> memoMap) {


		FormulaUtil.FormulaMatchMemoMapKey key = FormulaUtil.createFormulaMatchMemoMapKey(f1.theFormula, f2.theFormula);

		List<Set<VariableMapping>> k;
		if ((k = memoMap.get(key)) != null)
			return k;

//		if (f1 == null && f2 == null) {
//			List<Set<VariableMapping>> result = new FasterList<>(1);
//			result.add(new HashSet<>(0));
//			return result;
//		}


		if (f1.atom() && f2.atom())
			if (f1.isVariable() && f2.isVariable() || isSkolemTerm(f1.theFormula) && isSkolemTerm(f2.theFormula)) {
				Set<VariableMapping> set = new HashSet<>(1);
				set.add(new VariableMapping(f1.theFormula, f2.theFormula));
				List<Set<VariableMapping>> result = new Lst<>(1);
				result.add(set);
				return result;
			} else if (f1.theFormula.equals(f2.theFormula)) {
				List<Set<VariableMapping>> result = new Lst<>(1);
				result.add(new UnifiedSet<>(0));
				return result;
			} else return null;
		else if (f1.atom() || f2.atom()) return null;


		Formula head1 = new Formula();
		head1.read(f1.car());
		Formula head2 = new Formula();
		head2.read(f2.car());
		List<Set<VariableMapping>> headMaps = mapFormulaVariables(head1, head2, kb, memoMap);
		memoMap.put(FormulaUtil.createFormulaMatchMemoMapKey(head1.theFormula, head2.theFormula), headMaps);
		if (headMaps == null) return null;


		List<String> args1 = f1.complexArgumentsToArrayList(1);
		List<String> args2 = f2.complexArgumentsToArrayList(1);
		if (args1.size() != args2.size()) return null;

		if (!isCommutative(head1.theFormula) && !(kb != null && kb.isInstanceOf(head1.theFormula, "SymmetricRelation"))) {

			List<Set<VariableMapping>> runningMaps = headMaps;
			for (int i = 0; i < args1.size(); i++) {
				Formula parameter1 = new Formula();
				parameter1.read(args1.get(i));
				Formula parameter2 = new Formula();
				parameter2.read(args2.get(i));
				List<Set<VariableMapping>> parameterMaps = mapFormulaVariables(parameter1, parameter2, kb, memoMap);
				memoMap.put(FormulaUtil.createFormulaMatchMemoMapKey(parameter1.theFormula, parameter2.theFormula), parameterMaps);
				runningMaps = VariableMapping.intersect(runningMaps, parameterMaps);
				if (runningMaps == null) return null;
			}
			return runningMaps;
		} else {

			List<Set<VariableMapping>> unionMaps = new Lst<>();
			List<int[]> permutations = FormulaUtil.getPermutations(args1.size(),
				(a, b) -> mapFormulaVariables(new Formula(args1.get(a)), new Formula(args2.get(b)), kb, memoMap) != null);
			for (int[] perm : permutations) {
				List<Set<VariableMapping>> currentMaps = headMaps;
				boolean currentPairingValid = true;
				for (int i = 0; i < args1.size(); i++) {
					Formula parameter1 = new Formula();
					parameter1.read(args1.get(i));
					Formula parameter2 = new Formula();
					parameter2.read(args2.get(perm[i]));
					List<Set<VariableMapping>> parameterMaps = mapFormulaVariables(parameter1, parameter2, kb, memoMap);
					memoMap.put(FormulaUtil.createFormulaMatchMemoMapKey(parameter1.theFormula, parameter2.theFormula), parameterMaps);
					currentMaps = VariableMapping.intersect(currentMaps, parameterMaps);
					if (currentMaps == null) {
						currentPairingValid = false;
						break;
					}
				}
				if (currentPairingValid) unionMaps = VariableMapping.union(unionMaps, currentMaps);
			}
			if (unionMaps.isEmpty()) unionMaps = null;
			return unionMaps;
		}
	}

	/**
	 * ****************************************************************
	 *
	 * @param formula
	 * @param kb
	 * @param varPlaceholders
	 * @return
	 */
	private static String normalizeParameterOrder(String formula, KB kb, boolean varPlaceholders) {
		if (formula == null) return null;

		if (!listP(formula)) return varPlaceholders && isVariable(formula) ? "?XYZ" : formula;

		Formula f = new Formula();
		f.read(formula);


		List<String> args = f.complexArgumentsToArrayList(1);
		if (args == null || args.isEmpty()) return formula;
		List<String> orderedArgs = args.stream().map(s -> normalizeParameterOrder(s, kb, varPlaceholders)).toList();


		String head = f.car();
		if (isCommutative(head) || kb != null && kb.isInstanceOf(head, "SymmetricRelation"))
			Collections.sort(orderedArgs);


		StringBuilder result = new StringBuilder(LP);
		if (varPlaceholders && isSkolemTerm(head)) head = "?SknFn";
		result.append(head);
		result.append(SPACE);
		for (String arg : orderedArgs) {
			result.append(arg);
			result.append(SPACE);
		}
		result.deleteCharAt(result.length() - 1);
		result.append(RP);

		return result.toString();
	}

	/**
	 * **************************************************************
	 * Translate SUMO inequalities to the typical inequality symbols that
	 * some theorem provers require.
	 */
	private static String translateInequalities(String s) {
		if ("greaterThan".equalsIgnoreCase(s)) return ">";
		if ("greaterThanOrEqualTo".equalsIgnoreCase(s)) return ">=";
		if ("lessThan".equalsIgnoreCase(s)) return "<";
		if ("lessThanOrEqualTo".equalsIgnoreCase(s)) return "<=";
		return "";
	}

	/**
	 * **************************************************************
	 * Collect quantified and unquantified variables recursively
	 */
	private static void collectQuantifiedUnquantifiedVariablesRecurse(Formula f, HashMap<String, Boolean> varFlag,
																	  HashSet<String> unquantifiedVariables, HashSet<String> quantifiedVariables) {

		if (f == null || StringUtil.emptyString(f.theFormula) || f.empty())
			return;

		String carstr = f.car();
		int fLen = f.listLength();
		if (atom(carstr) && isLogicalOperator(carstr)) if (carstr.equals(EQUANT) || carstr.equals(UQUANT)) {
			String varString = f.getArgument(1);
			String[] varArray = varString.substring(1, varString.length() - 1).split(" ");
			quantifiedVariables.addAll(Arrays.asList(varArray));

			for (int i = 2; i < fLen; i++)
				collectQuantifiedUnquantifiedVariablesRecurse(new Formula(f.getArgument(i)),
					varFlag, unquantifiedVariables, quantifiedVariables);
		} else for (int i = 1; i < fLen; i++)
			collectQuantifiedUnquantifiedVariablesRecurse(new Formula(f.getArgument(i)),
				varFlag, unquantifiedVariables, quantifiedVariables);
		else for (int i = 0; i < fLen; i++) {
			String arg = f.getArgument(i);
			if (arg.startsWith("?") || arg.startsWith("@")) {
				if (!varFlag.containsKey(arg) && !quantifiedVariables.contains(arg)) {
					unquantifiedVariables.add(arg);
					varFlag.put(arg, false);
				}
			} else collectQuantifiedUnquantifiedVariablesRecurse(new Formula(arg),
				varFlag, unquantifiedVariables, quantifiedVariables);
		}
	}

	/**
	 * **************************************************************
	 * Test whether a Formula is a functional target
	 */
	public static boolean isFunctionalTerm(String s) {
		Formula f = new Formula();
		f.read(s);
		return f.isFunctionalTerm();
	}

	/**
	 * **************************************************************
	 * Test whether a String formula is a variable
	 */
	public static boolean isVariable(String term) {
		return !StringUtil.emptyString(term)
			&& (term.startsWith(V_PREF)
			|| term.startsWith(R_PREF));
	}

	/**
	 * **************************************************************
	 * Test whether a list with a predicate is a quantifier list
	 */
	public static boolean isQuantifierList(String listPred, String previousPred) {
		return (previousPred.equals(EQUANT) || previousPred.equals(UQUANT)) &&
			(listPred.startsWith(R_PREF) || listPred.startsWith(V_PREF));
	}

	/**
	 * **************************************************************
	 * Test whether a predicate is a logical quantifier
	 */
	public static boolean isQuantifier(String pred) {
		return !StringUtil.emptyString(pred)
			&& (pred.equals(EQUANT)
			|| pred.equals(UQUANT));
	}

	/**
	 * **************************************************************
	 * A static utility method.
	 *
	 * @param obj Any object, but should be a String.
	 * @return true if obj is a SUO-KIF commutative logical operator,
	 * else false.
	 */
	public static boolean isCommutative(String obj) {
		return !StringUtil.emptyString(obj)
			&& (obj.equals(AND)
			|| obj.equals(OR));
	}

	/**
	 * **************************************************************
	 * Returns the dual logical operator of op, or null if op is not
	 * an operator or has no dual.
	 *
	 * @param op A String, assumed to be a SUO-KIF logical operator
	 * @return A String, the dual operator of op, or null.
	 */
	protected static String getDualOperator(String op) {
		String ans = null;
		if (op != null) {
			String[][] duals = {{UQUANT, EQUANT},
				{EQUANT, UQUANT},
				{AND, OR},
				{OR, AND},
				{NOT, ""},
				{"", NOT},
				{LOG_TRUE, LOG_FALSE},
				{LOG_FALSE, LOG_TRUE}
			};
			for (String[] dual : duals) if (op.equals(dual[0])) ans = dual[1];
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Returns true if target is a standard FOL logical operator, else
	 * returns false.
	 *
	 * @param term A String, assumed to be an atomic SUO-KIF target.
	 */
	public static boolean isLogicalOperator(String term) {
		return LOGICAL_OPERATORS.contains(term);
	}

	/**
	 * **************************************************************
	 * Returns true if target is a valid SUO-KIF target, else
	 * returns false.
	 *
	 * @param term A String, assumed to be an atomic SUO-KIF target.
	 */
	public static boolean isTerm(String term) {

		if (!StringUtil.emptyString(term) && !listP(term) &&
			Character.isJavaIdentifierStart(term.charAt(0))) {
			return IntStream.range(0, term.length()).allMatch(i -> Character.isJavaIdentifierPart(term.charAt(i)));
		} else
			return false;
	}

	/**
	 * **************************************************************
	 * Returns true if target is a SUO-KIF predicate for comparing two
	 * (typically numeric) terms, else returns false.
	 *
	 * @param term A String.
	 */
	public static boolean isComparisonOperator(String term) {

		return COMPARISON_OPERATORS.contains(term);
	}

	/**
	 * **************************************************************
	 * Returns true if target is a SUO-KIF mathematical function, else
	 * returns false.
	 *
	 * @param term A String.
	 */
	public static boolean isMathFunction(String term) {

		return MATH_FUNCTIONS.contains(term);
	}

	/**
	 * **************************************************************
	 * Returns true if formula is a valid formula with no variables,
	 * else returns false.
	 */
	private static boolean isGround(String form) {

		if (StringUtil.emptyString(form))
			return false;
		if (!form.contains("\""))
			return !form.contains("?") && !form.contains("@");
		boolean inQuote = false;
		for (int i = 0; i < form.length(); i++) {
			if (form.charAt(i) == '"')
				inQuote = !inQuote;
			if ((form.charAt(i) == '?' || form.charAt(i) == '@') && !inQuote)
				return false;
		}
		return true;
	}

	/**
	 * **************************************************************
	 * Returns true if target is a SUO-KIF function, else returns false.
	 * Note that this test is purely syntactic, and could fail for
	 * functions that do not adhere to the convention of ending all
	 * functions with "Fn".
	 *
	 * @param term A String.
	 */
	public static boolean isFunction(String term) {

		System.out.println("Error in Formula.isFuction(): must use KB.isFunction() instead");
		return !StringUtil.emptyString(term) && term.endsWith(FN_SUFF);
	}

	/**
	 * **************************************************************
	 * Returns true if target is a SUO-KIF Skolem target, else returns false.
	 *
	 * @param term A String.
	 * @return true or false
	 */
	static boolean isSkolemTerm(String term) {
		return !StringUtil.emptyString(term) && SKOLEMED.matcher(term.trim()).matches();
	}

	/**
	 * **************************************************************
	 * Compare the given formula to the query and return whether
	 * they are the same.
	 */
	public static boolean isQuery(String query, String formula) {

		Formula f = new Formula();
		f.read(formula);
		return f.equals(query);
	}

	/**
	 * **************************************************************
	 * Compare the given formula to the negated query and return whether
	 * they are the same (minus the negation).
	 */
	public static boolean isNegatedQuery(String query, String formulaString) {
		boolean result = false;
		String fstr = formulaString.trim();
		if (fstr.startsWith("(not")) {
			Formula f = new Formula();
			f.read(fstr);
			result = query.equals(f.getArgument(1));
		}
		return result;
	}

//	/**
//	 * **************************************************************
//	 * Format a formula for text presentation.
//	 */
//	public static String textFormat(String input) {
//		Formula f = new Formula(input);
//		return f.format("", "  ", Character.valueOf((char) 10).toString());
//	}

//	/**
//	 * **************************************************************
//	 * A test method.
//	 */
//	public static void testCollectVariables() {
//
//		Formula f = new Formula();
//		f.read("(=> " +
//			"  (and " +
//			"    (attribute ?H Muslim) " +
//			"    (equal " +
//			"      (WealthFn ?H) ?W)) " +
//			"(modalAttribute " +
//			"  (exists (?Z ?T) " +
//			"    (and " +
//			"      (instance ?Z Zakat) " +
//			"      (instance ?Y Year) " +
//			"      (during ?Y " +
//			"        (WhenFn ?H)) " +
//			"      (holdsDuring ?Y " +
//			"        (attribute ?H FullyFormed)) " +
//			"      (agent ?Z ?H) " +
//			"      (patient ?Z ?T) " +
//			"      (monetaryValue ?T ?C) " +
//			"      (greaterThan ?C " +
//			"        (MultiplicationFn ?W 0.025)))) Obligation)) ");
//		System.out.println("Quantified variables: " + f.collectQuantifiedVariables());
//		System.out.println("All variables: " + f.collectAllVariables());
//		System.out.println("Unquantified variables: " + f.collectUnquantifiedVariables());
//		System.out.println("Terms: " + f.collectTerms());
//	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	public static void testIsSimpleClause() {

		KBmanager.manager.initializeOnce();
		KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));
		Formula f1 = new Formula();
		f1.read("(not (instance ?X Human))");
		System.out.println("Simple clause? : " + f1.isSimpleClause(kb) + '\n' + f1 + '\n');
		f1.read("(instance ?X Human)");
		System.out.println("Simple clause? : " + f1.isSimpleClause(kb) + '\n' + f1 + '\n');
		f1.read("(=> (attribute ?Agent Investor) (exists (?Investing) (agent ?Investing ?Agent)))");
		System.out.println("Simple clause? : " + f1.isSimpleClause(kb) + '\n' + f1 + '\n');
		f1.read("(member (SkFn 1 ?X3) ?X3)");
		System.out.println("Simple clause? : " + f1.isSimpleClause(kb) + '\n' + f1 + '\n');
		f1.read("(member ?VAR1 Org1-1)");
		System.out.println("Simple clause? : " + f1.isSimpleClause(kb) + '\n' + f1 + '\n');
		f1.read("(capability (KappaFn ?HEAR (and (instance ?HEAR Hearing) (agent ?HEAR ?HUMAN) " +
			"(destination ?HEAR ?HUMAN) (origin ?HEAR ?OBJ))) agent ?HUMAN)");
		System.out.println("Simple clause? : " + f1.isSimpleClause(kb) + '\n' + f1 + '\n');
	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	public static void testReplaceVar() {

		Formula f1 = new Formula();
		f1.read("(<=> (instance ?REL TransitiveRelation) (forall (?INST1 ?INST2 ?INST3) " +
			" (=> (and (?REL ?INST1 ?INST2) (?REL ?INST2 ?INST3)) (?REL ?INST1 ?INST3))))");
		System.out.println("Input: " + f1);
		System.out.println(f1.replaceVar("?REL", "part"));
	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	public static void testComplexArgs() {

		Formula f1 = new Formula();
		f1.read("(during ?Y (WhenFn ?H))");
		System.out.println("Input: " + f1);
		System.out.println(f1.complexArgumentsToArrayList(1));
	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	public static void testBigArgs() {

		Formula f1 = new Formula();
		f1.read("(=>   (instance ?AT AutomobileTransmission)  (hasPurpose ?AT    (exists (?C ?D ?A ?R1 ?N1 ?R2 ?R3 ?R4 ?N2 ?N3)      (and        (instance ?C Crankshaft)        (instance ?D Driveshaft)        (instance ?A Automobile)        (part ?D ?A)        (part ?AT ?A)        (part ?C ?A)        (connectedEngineeringComponents ?C ?AT)        (connectedEngineeringComponents ?D ?AT)        (instance ?R1 Rotating)        (instance ?R2 Rotating)               (instance ?R3 Rotating)        (instance ?R4 Rotating)        (patient ?R1 ?C)        (patient ?R2 ?C)        (patient ?R3 ?D)        (patient ?R4 ?D)        (causes ?R1 ?R3)        (causes ?R2 ?R4)        (not          (equal ?R1 ?R2))        (holdsDuring ?R1          (measure ?C (RotationFn ?N1 MinuteDuration)))        (holdsDuring ?R2          (measure ?C (RotationFn ?N1 MinuteDuration)))        (holdsDuring ?R3          (measure ?D (RotationFn ?N2 MinuteDuration)))        (holdsDuring ?R4          (measure ?D (RotationFn ?N3 MinuteDuration)))        (not          (equal ?N2 ?N3))))))");
		System.out.println("Input: " + f1);
		System.out.println(f1.validArgs());
	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	public static void testCar1() {

		Formula f1 = new Formula();
		f1.read("(=>   (instance ?AT AutomobileTransmission)  (hasPurpose ?AT    (exists (?C ?D ?A ?R1 ?N1 ?R2 ?R3 ?R4 ?N2 ?N3)      (and        (instance ?C Crankshaft)        (instance ?D Driveshaft)        (instance ?A Automobile)        (part ?D ?A)        (part ?AT ?A)        (part ?C ?A)        (connectedEngineeringComponents ?C ?AT)        (connectedEngineeringComponents ?D ?AT)        (instance ?R1 Rotating)        (instance ?R2 Rotating)               (instance ?R3 Rotating)        (instance ?R4 Rotating)        (patient ?R1 ?C)        (patient ?R2 ?C)        (patient ?R3 ?D)        (patient ?R4 ?D)        (causes ?R1 ?R3)        (causes ?R2 ?R4)        (not          (equal ?R1 ?R2))        (holdsDuring ?R1          (measure ?C (RotationFn ?N1 MinuteDuration)))        (holdsDuring ?R2          (measure ?C (RotationFn ?N1 MinuteDuration)))        (holdsDuring ?R3          (measure ?D (RotationFn ?N2 MinuteDuration)))        (holdsDuring ?R4          (measure ?D (RotationFn ?N3 MinuteDuration)))        (not          (equal ?N2 ?N3))))))");
		System.out.println("Input: " + f1);
		System.out.println(f1.validArgs());
	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	private static void testArg2ArrayList() {

		System.out.println("testArg2ArrayList(): ");
		String f = "(termFormat EnglishLanguage experimentalControlProcess \"experimental control (process)\")";
		Formula form = new Formula(f);
		System.out.println(form.argumentsToArrayList(0));
	}

	/**
	 * **************************************************************
	 * A test method.
	 */
	public static void main(String[] args) {


		testArg2ArrayList();
	}

	public String getSourceFile() {
		return this.sourceFile;
	}

	public void setSourceFile(String filename) {
		this.sourceFile = filename;
	}

	public Collection<String> getErrors() {
		return this.errors;
	}

//	/**
//	 * **************************************************************
//	 * Returns an ArrayList of the TPTP formulas (Strings) that
//	 * together constitute the TPTP translation of theFormula.
//	 *
//	 * @return An ArrayList of Strings, or an empty ArrayList if no
//	 * translations have been created or entered.
//	 */
//	public ArrayList<String> getTheTptpFormulas() {
//
//		if (theTptpFormulas == null)
//			theTptpFormulas = new FasterList<>();
//		return theTptpFormulas;
//	}

//	/**
//	 * **************************************************************
//	 * Clears theTptpFormulas if the ArrayList exists, else does
//	 * nothing.
//	 */
//	public void clearTheTptpFormulas() {
//
//		if (theTptpFormulas != null)
//			theTptpFormulas.clear();
//	}

//	/**
//	 * **************************************************************
//	 * Returns a List of the clauses that together constitute the
//	 * resolution form of this Formula.  The list could be empty if
//	 * the clausal form has not yet been computed.
//	 *
//	 * @return ArrayList
//	 */
//	private ArrayList getTheClausalForm() {
//
//		if (theClausalForm == null && !StringUtil.emptyString(theFormula))
//			theClausalForm = Clausifier.toNegAndPosLitsWithRenameInfo(this);
//		return theClausalForm;
//	}
//
//	/**
//	 * **************************************************************
//	 * This method clears the list of clauses that together constitute
//	 * the resolution form of this Formula, and can be used in
//	 * preparation for recomputing the clauses.
//	 */
//	public void clearTheClausalForm() {
//
//		if (theClausalForm != null)
//			theClausalForm.clear();
//		theClausalForm = null;
//	}

//	/**
//	 * **************************************************************
//	 * Returns a List of List objects.  Each such object contains, in
//	 * turn, a pair of List objects.  Each List object in a pair
//	 * contains Formula objects.  The Formula objects contained in the
//	 * first List object (0) of a pair represent negative literals
//	 * (antecedent conjuncts).  The Formula objects contained in the
//	 * second List object (1) of a pair represent positive literals
//	 * (consequent conjuncts).  Taken together, all of the clauses
//	 * constitute the resolution form of this Formula.
//	 *
//	 * @return A List of Lists.
//	 */
//	public ArrayList getClauses() {
//
//		ArrayList clausesWithVarMap = getTheClausalForm();
//		if (clausesWithVarMap == null || clausesWithVarMap.isEmpty())
//			return null;
//		return (ArrayList) clausesWithVarMap.get(0);
//	}

//	/**
//	 * **************************************************************
//	 * Returns a map of the variable renames that occurred during the
//	 * translation of this Formula into the clausal (resolution) form
//	 * accessible via this.getClauses().
//	 *
//	 * @return A Map of String (SUO-KIF variable) key-value pairs.
//	 */
//	public HashMap getVarMap() {
//
//		ArrayList clausesWithVarMap = getTheClausalForm();
//		if (clausesWithVarMap == null || clausesWithVarMap.size() < 3)
//			return null;
//		return (HashMap) clausesWithVarMap.get(2);
//	}

	/**
	 * **************************************************************
	 * Read a String into the variable 'theFormula'.
	 */
	public void read(String s) {
		theFormula = s;
	}

	/**
	 * **************************************************************
	 *
	 * @return a unique ID by appending the hashCode() of the
	 * formula String to the file name in which it appears
	 */
	public String createID() {

		String fname = sourceFile;
		if (!StringUtil.emptyString(fname) && fname.lastIndexOf(File.separator) > -1)
			fname = fname.substring(fname.lastIndexOf(File.separator) + 1);
		int hc = theFormula.hashCode();
		String result;
		result = hc < 0 ? 'N' + Integer.valueOf(hc).toString().substring(1) + fname : hc + fname;
		return result;
	}

	/**
	 * **************************************************************
	 * Copy the Formula.  This is in effect a deep copy although it ignores
	 * the errors and warnings variables.
	 */
	private Formula copy() {

		return new Formula(this);
	}

	/**
	 * **************************************************************
	 */
	public Formula deepCopy() {
		return copy();
	}

	/**
	 * **************************************************************
	 * Implement the Comparable interface by defining the compareTo
	 * method.  Formulas are equal if their formula strings are equal.
	 */
	public int compareTo(Object f) throws ClassCastException {

		if (f == null) {
			System.out.println("Error in Formula.compareTo(): null formula");
			throw new ClassCastException("Error in Formula.compareTo(): null formula");
		}
		if (!"com.articulate.sigma.Formula".equalsIgnoreCase(f.getClass().getName()))
			throw new ClassCastException("Error in Formula.compareTo(): "
				+ "Class cast exception for argument of class: "
				+ f.getClass().getName());
		return theFormula.compareTo(((Formula) f).theFormula);
	}

	/**
	 * **************************************************************
	 * Returns true if the Formula contains no unbalanced parentheses
	 * or unbalanced quote characters, otherwise returns false.
	 *
	 * @return boolean
	 */
	public boolean isBalancedList() {

		boolean ans = false;
		if (this.listP()) if (this.empty())
			ans = true;
		else {
			String input = this.theFormula.trim();

			int i = 0;
			int len = input.length();
			int end = len - 1;
			int pLevel = 0;
			int qLevel = 0;
			char prev = '0';
			char ch;
			boolean insideQuote = false;
			char quoteCharInForce = '0';
			while (i < len) {
				ch = input.charAt(i);
				if (!insideQuote) {
					if (ch == '(')
						pLevel++;
					else if (ch == ')')
						pLevel--;
					else if (quoteChars.contains(ch) && prev != '\\') {
						insideQuote = true;
						quoteCharInForce = ch;
						qLevel++;
					}
				} else if (quoteChars.contains(ch)
					&& ch == quoteCharInForce
					&& prev != '\\') {
					insideQuote = false;
					quoteCharInForce = '0';
					qLevel--;
				}
				prev = ch;
				i++;
			}
			ans = pLevel == 0 && qLevel == 0;
		}
		return ans;
	}

	/**
	 * **************************************************************
	 *
	 * @return the LISP 'car' of the formula as a String - the first
	 * element of the list. Note that this operation has no side
	 * effect on the Formula.
	 * <p>
	 * Currently (10/24/2007) this method returns the empty string
	 * ("") when invoked on an empty list.  Technically, this is
	 * wrong.  In most LISPS, the car of the empty list is the empty
	 * list (or nil).  But some parts of the Sigma code apparently
	 * expect this method to return the empty string when invoked on
	 * an empty list.
	 */
	public String car() {

		String ans = null;
		if (this.listP()) if (this.empty())

			ans = "";
		else {
			String input = this.theFormula.trim();
			StringBuilder sb = new StringBuilder(1024);

			int i = 1;
			int len = input.length();
			int end = len - 1;
			int level = 0;
			char prev = '0';
			char ch;
			boolean insideQuote = false;
			char quoteCharInForce = '0';
			while (i < end) {
				ch = input.charAt(i);
				if (!insideQuote) if (ch == '(') {
					sb.append('(');
					level++;
				} else if (ch == ')') {
					sb.append(')');
					level--;
					if (level <= 0)
						break;
				} else if (Character.isWhitespace(ch) && level <= 0) {
					if (!sb.isEmpty())
						break;
				} else if (quoteChars.contains(ch) && prev != '\\') {
					sb.append(ch);
					insideQuote = true;
					quoteCharInForce = ch;
				} else
					sb.append(ch);
				else if (quoteChars.contains(ch)
					&& ch == quoteCharInForce
					&& prev != '\\') {
					sb.append(ch);
					insideQuote = false;
					quoteCharInForce = '0';
					if (level <= 0)
						break;
				} else
					sb.append(ch);
				prev = ch;
				i++;
			}
			ans = sb.toString();
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Return the LISP 'cdr' of the formula - the rest of a list minus its
	 * first element.
	 * Note that this operation has no side effect on the Formula.
	 */
	public String cdr() {

		String ans = null;
		if (this.listP()) if (this.empty())
			ans = this.theFormula;
		else {
			String input = theFormula.trim();
			int i = 1;
			int len = input.length();
			int end = len - 1;
			int level = 0;
			char prev = '0';
			char ch;
			boolean insideQuote = false;
			char quoteCharInForce = '0';
			int carCount = 0;
			while (i < end) {
				ch = input.charAt(i);
				if (!insideQuote) if (ch == '(') {
					carCount++;
					level++;
				} else if (ch == ')') {
					carCount++;
					level--;
					if (level <= 0)
						break;
				} else if (Character.isWhitespace(ch) && level <= 0) {
					if (carCount > 0)
						break;
				} else if (quoteChars.contains(ch) && prev != '\\') {
					carCount++;
					insideQuote = true;
					quoteCharInForce = ch;
				} else
					carCount++;
				else if (quoteChars.contains(ch)
					&& ch == quoteCharInForce
					&& prev != '\\') {
					carCount++;
					insideQuote = false;
					quoteCharInForce = '0';
					if (level <= 0)
						break;
				} else
					carCount++;
				prev = ch;
				i++;
			}
			if (carCount > 0) {
				int j = i + 1;
				ans = j < end ? '(' + input.substring(j, end).trim() + ')' : "()";
			}
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Returns a new Formula which is the result of 'consing' a String
	 * into this Formula, similar to the LISP procedure of the same
	 * name.  This procedure is a little bit of a kluge, since this
	 * Formula is treated simply as a LISP object (presumably, a LISP
	 * list), and could be degenerate or malformed as a Formula.
	 * <p>
	 * Note that this operation has no side effect on the original Formula.
	 *
	 * @param obj The String object that will become the 'car' (or
	 *            head) of the resulting Formula (list).
	 * @return a new Formula, or the original Formula if the cons fails.
	 */
	public Formula cons(String obj) {

		Formula ans = this;
		String fStr = this.theFormula;
		if (!StringUtil.emptyString(obj) && !StringUtil.emptyString(fStr)) {
			String theNewFormula;
			if (this.listP())
				theNewFormula = this.empty() ? '(' + obj + ')' : '(' + obj + ' ' + fStr.substring(1, fStr.length() - 1) + ')';
			else


				theNewFormula = '(' + obj + " . " + fStr + ')';
			ans = new Formula();
			ans.read(theNewFormula);
		}
		return ans;
	}

	/**
	 * **************************************************************
	 *
	 * @return a new Formula, or the original Formula if the cons fails.
	 */
	public Formula cons(Formula f) {

		return cons(f.theFormula);
	}

	/**
	 * **************************************************************
	 * Returns the LISP 'cdr' of the formula as a new Formula, if
	 * possible, else returns null.
	 * <p>
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a Formula, or null.
	 */
	public Formula cdrAsFormula() {

		String thisCdr = this.cdr();
		if (listP(thisCdr)) {
			Formula f = new Formula();
			f.read(thisCdr);
			return f;
		}
		return null;
	}

	/**
	 * **************************************************************
	 * Returns the LISP 'car' of the formula as a new Formula, if
	 * possible, else returns null.
	 * <p>
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a Formula, or null.
	 */
	public Formula carAsFormula() {

		String thisCar = this.car();

		Formula f = new Formula();
		f.read(thisCar);
		return f;


	}

	/**
	 * **************************************************************
	 * Returns the LISP 'cadr' (the second list element) of the
	 * formula.
	 * <p>
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a String, or the empty string if the is no cadr.
	 */
	public String cadr() {

		return this.getArgument(1);
	}

	/**
	 * **************************************************************
	 * Returns the LISP 'cddr' of the formula - the rest of the rest,
	 * or the list minus its first two elements.
	 * <p>
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a String, or null.
	 */
	private String cddr() {

		Formula fCdr = this.cdrAsFormula();
		if (fCdr != null)
			return fCdr.cdr();
		return null;
	}

	/**
	 * **************************************************************
	 * Returns the LISP 'cddr' of the formula as a new Formula, if
	 * possible, else returns null.
	 * <p>
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a Formula, or null.
	 */
	public Formula cddrAsFormula() {

		String thisCddr = this.cddr();
		if (listP(thisCddr)) {
			Formula f = new Formula();
			f.read(thisCddr);
			return f;
		}
		return null;
	}

	/**
	 * **************************************************************
	 * Returns the LISP 'caddr' of the formula, which is the third
	 * list element of the formula.
	 * <p>
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a String, or the empty string if there is no caddr.
	 */
	public String caddr() {
		return this.getArgument(2);
	}

	/**
	 * **************************************************************
	 * Returns the LISP 'append' of the formulas
	 * Note that this operation has no side effect on the Formula.
	 *
	 * @return a Formula
	 */
	public Formula append(Formula f) {

		Formula newFormula = new Formula();
		newFormula.read(theFormula);
		if (newFormula.equals("") || newFormula.atom()) {
			System.out.println("Error in KB.append(): attempt to append to non-list: " + theFormula);
			return this;
		}
		if (f == null || f.theFormula == null || f.theFormula.isEmpty() || "()".equals(f.theFormula))
			return newFormula;
		f.theFormula = f.theFormula.trim();
		if (!f.atom())
			f.theFormula = f.theFormula.substring(1, f.theFormula.length() - 1);
		int lastParen = theFormula.lastIndexOf(')');
		String sep = "";
		if (lastParen > 1)
			sep = " ";
		newFormula.theFormula = newFormula.theFormula.substring(0, lastParen) + sep + f.theFormula + ')';
		return newFormula;
	}

	/**
	 * **************************************************************
	 * Test whether the Formula is a LISP atom.
	 */
	public boolean atom() {

		return atom(theFormula);
	}

	/**
	 * **************************************************************
	 * Test whether the Formula is an empty list.
	 */
	public boolean empty() {

		return empty(theFormula);
	}

	/**
	 * **************************************************************
	 * Test whether the Formula is a list.
	 */
	public boolean listP() {
		return listP(theFormula);
	}

	/**
	 * **************************************************************
	 *
	 * @see #validArgs() validArgs below for documentation
	 */
	private String validArgsRecurse(Formula f, String filename, Integer lineNo) {

		if (f.theFormula.isEmpty() || !f.listP() || f.atom() || f.empty())
			return "";
		String pred = f.car();
		String rest = f.cdr();
		Formula restF = new Formula();
		restF.read(rest);
		int argCount = 0;
		while (!restF.empty()) {
			argCount++;
			String arg = restF.car();
			Formula argF = new Formula();
			argF.read(arg);
			String result = validArgsRecurse(argF, filename, lineNo);
			if (!result.isEmpty())
				return result;
			restF.theFormula = restF.cdr();
		}
		String location = "";
		if (filename != null && lineNo != null)
			location = "near line " + lineNo + " in " + filename;
		if (pred.equals(AND) || pred.equals(OR)) {
			if (argCount < 2) {
				String errString = "Too few arguments for 'and' or 'or' at " + location + ": " + f;
				errors.add(errString);
				return errString;
			}
		} else if (pred.equals(UQUANT) || pred.equals(EQUANT)) if (argCount != 2) {
			String errString = "Wrong number of arguments for quantifer at " + location + ": " + f;
			errors.add(errString);
			return errString;
		} else {
			Formula quantF = new Formula();
			quantF.read(rest);
			if (!listP(quantF.car())) {
				String errString = "No var list for quantifier at " + location + ": " + f;
				errors.add(errString);
				return errString;
			}
		}
		else if (pred.equals(IFF) || pred.equals(IF)) {
			if (argCount != 2) {
				String errString = "Wrong number of arguments for '<=>' or '=>' at " + location + ": " + f;
				errors.add(errString);
				return errString;
			}
		} else if (pred.equals(EQUAL)) {
			if (argCount != 2) {
				String errString = "Wrong number of arguments for 'equals' at " + location + ": " + f;
				errors.add(errString);
				return errString;
			}
		} else if (!isVariable(pred) && argCount > MAX_PREDICATE_ARITY + 1) {


			String errString = "Maybe too many arguments at " + location + ": " + f;
			errors.add(errString);
			return errString;
		}
		return "";
	}

	/**
	 * **************************************************************
	 * Test whether the Formula uses logical operators and predicates
	 * with the correct number of arguments.  "equals", "<=>", and
	 * "=>" are strictly binary.  "or", and "and" are binary or
	 * greater. "not" is unary.  "forall" and "exists" are unary with
	 * an argument list.  Warn if we encounter a formula that has more
	 * arguments than MAX_PREDICATE_ARITY.
	 *
	 * @param filename If not null, denotes the name of the file being
	 *                 parsed.
	 * @param lineNo   If not null, indicates the location of the
	 *                 expression (formula) being parsed in the file being read.
	 * @return an empty String if there are no problems or an error message
	 * if there are.
	 */
	private String validArgs(String filename, Integer lineNo) {

		if (theFormula == null || theFormula.isEmpty())
			return "";
		Formula f = new Formula();
		f.read(theFormula);
		return validArgsRecurse(f, filename, lineNo);
	}

	/**
	 * **************************************************************
	 * Test whether the Formula uses logical operators and predicates
	 * with the correct number of arguments.  "equals", "<=>", and
	 * "=>" are strictly binary.  "or", and "and" are binary or
	 * greater. "not" is unary.  "forall" and "exists" are unary with
	 * an argument list.  Warn if we encounter a formula that has more
	 * arguments than MAX_PREDICATE_ARITY.
	 *
	 * @return an empty String if there are no problems or an error message
	 * if there are.
	 */
	private String validArgs() {
		return this.validArgs(null, null);
	}

	/**
	 * **************************************************************
	 * Compare two lists of formulas, testing whether they are equal,
	 * without regard to order.  (B A C) will be equal to (C B A). The
	 * method iterates through one list, trying to find a match in the other
	 * and removing it if a match is found.  If the lists are equal, the
	 * second list should be empty once the iteration is complete.
	 * Note that the formulas being compared must be lists, not atoms, and
	 * not a set of formulas unenclosed by parentheses.  So, "(A B C)"
	 * and "(A)" are valid, but "A" is not, nor is "A B C".
	 */
	private boolean compareFormulaSets(String s) {

		List<Formula> thisList = parseList(this.theFormula.substring(1, this.theFormula.length() - 1));
		List<Formula> sList = parseList(s.substring(1, s.length() - 1));
		int k = thisList.size();
		if (k != sList.size())
			return false;
		if (k > 0) {
			for (Formula formula : thisList)
				for (int j = 0; j < sList.size(); j++)
					if (formula.logicallyEquals(sList.get(j).theFormula)) {
						sList.remove(j);
						j = sList.size();
					}
			return sList.isEmpty();
		} else
			return false;
	}

	/**
	 * **************************************************************
	 * Test if the contents of the formula are equal to the argument
	 * at a deeper level than a simple string equals.  The only logical
	 * manipulation is to treat conjunctions and disjunctions as unordered
	 * bags of clauses. So (and A B C) will be logicallyEqual(s) for example,
	 * to (and B A C).  Note that this is a fairly time-consuming operation
	 * and should not generally be used for comparing large sets of formulas.
	 */
	private boolean logicallyEquals(String s) {

		if (this.equals(s))
			return true;
		if (atom(s) && s.compareTo(theFormula) != 0)
			return false;

		Formula form = new Formula();
		form.read(this.theFormula);
		Formula sform = new Formula();
		sform.read(s);

		switch (form.car()) {
			case "and", "or" -> {
				if (!sform.car()/*.intern()*/.equals(sform.car()/*.intern()*/))
					return false;
				form.read(form.cdr());
				sform.read(sform.cdr());
				return form.compareFormulaSets(sform.theFormula);
			}
			default -> {
				Formula newForm = new Formula();
				newForm.read(form.car());
				Formula newSform = new Formula();
				newSform.read(sform.cdr());
				return newForm.logicallyEquals(sform.car()) &&
						newSform.logicallyEquals(form.cdr());
			}
		}
	}

	/**
	 * **************************************************************
	 * If equals is overridden, hashCode must use the same
	 * "significant" fields.
	 */
	public int hashCode() {
		return theFormula.hashCode();
//		String thisString = Clausifier.normalizeVariables(this.theFormula).trim();
//		return thisString.hashCode();
	}

	/**
	 * **************************************************************
	 * Test if the contents of the formula are equal to the
	 * argument. Normalize all variables so that formulas can be equal
	 * independent of their variable names, which have no semantics.
	 */
	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;
		if (!(o instanceof Formula f)) return false;

        return theFormula.equals(f.theFormula);
//		if (theFormula == f.theFormula)
//			return true;
//		return f.theFormula == null ? this.theFormula == null : TRIM_INNER.matcher(Clausifier.normalizeVariables(this.theFormula).trim()).replaceAll(" ")
//			.equals(TRIM_INNER.matcher(Clausifier.normalizeVariables(f.theFormula).trim()).replaceAll(" "));
	}

	/**
	 * **************************************************************
	 * Test if the contents of the formula are equal to the String argument.
	 * Normalize all variables.
	 */
	private boolean equals(String s) {

		if (s == null) return false;

//		String f = theFormula;
		return theFormula.equals(s);
//		Formula form = new Formula();
//		Formula sform = new Formula();
//
//		form.theFormula = f;
//		s = Clausifier.normalizeVariables(s)/*.intern()*/;
//		sform.read(s);
//		s = sform.toString().trim()/*.intern()*/;
//
//		form.theFormula = Clausifier.normalizeVariables(theFormula);
//		return form.toString().trim()/*.intern()*/
//			.equals(s);
	}

//	/**
//	 * **************************************************************
//	 * Tests if this is logically equal with the parameter formula. It
//	 * employs three equality tests starting with the
//	 * fastest and finishing with the slowest:
//	 * <p>
//	 * - string comparisons: if the strings of the two formulae are
//	 * equal return true as the formulae are also equal,
//	 * otherwise try comparing them by more complex means
//	 * <p>
//	 * - compare the predicate structure of the formulae (deepEquals(...)):
//	 * this comparison only checks if the two formulae
//	 * have an equal structure of predicates disregarding variable
//	 * equivalence. Example:
//	 * (and (instance ?A Human) (instance ?A Mushroom)) according
//	 * to deepEquals(...) would be equal to
//	 * (and (instance ?A Human) (instance ?B Mushroom)) even though
//	 * the first formula refers only one variable
//	 * but the second one refers two, and as such they are not logically
//	 * equal. This method generates false positives, but
//	 * only true negatives. If the result of the comparison is false,
//	 * we return false, otherwise keep trying.
//	 * <p>
//	 * - try to logically unify the formulae by matching the predicates
//	 * and the variables
//	 *
//	 * @param f
//	 * @return
//	 */
//	public boolean logicallyEquals(Formula f) {
//		boolean equalStrings = this.equals(f);
//		if (equalStrings) return true;
//		else if (!this.deepEquals(f)) return false;
//		else return this.unifyWith(f);
//	}
//
//	/**
//	 * ****************************************************************
//	 * Compares this formula with the parameter by trying to compare the
//	 * predicate structure of th two and logically
//	 * unify their variables. The helper method mapFormulaVariables(....)
//	 * returns a logical mapping between the variables
//	 * of two formulae of one exists.
//	 *
//	 * @param f
//	 * @return
//	 */
//	public boolean unifyWith(Formula f) {
//		if (f == this)
//			return true;
//		Formula f1 = Clausifier.clausify(this);
//		Formula f2 = Clausifier.clausify(f);
//
//
//		KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));
//
//		HashMap<FormulaUtil.FormulaMatchMemoMapKey, List<Set<VariableMapping>>> memoMap = new HashMap<>();
//		List<Set<VariableMapping>> result = mapFormulaVariables(f1, f2, kb, memoMap);
//		return result != null;
//	}

//	/**
//	 * **************************************************************
//	 * Test if the contents of the formula are equal to the argument.
//	 */
//	public boolean deepEquals(Formula f) {
//
//
//		if (f == null) return false;
//
//		boolean stringsEqual = Objects.equals(this.theFormula, f.theFormula);
//		if (stringsEqual || this.theFormula == null || f.theFormula == null) return stringsEqual;
//
//		Formula f1 = Clausifier.clausify(this);
//		Formula f2 = Clausifier.clausify(f);
//
//
//		KB kb = KBmanager.manager.getKB(KBmanager.manager.getPref("sumokbname"));
//
//		String normalized1 = normalizeParameterOrder(f1.theFormula, kb, true);
//		String normalized2 = normalizeParameterOrder(f2.theFormula, kb, true);
//
//		return normalized1.equals(normalized2);
//	}

	/**
	 * **************************************************************
	 * Return the numbered argument of the given formula.  The first
	 * element of a formula (i.e. the predicate position) is number 0.
	 * Returns the empty string if there is no such argument position.
	 */
	public String getArgument(int argnum) {

		Formula form = new Formula();
		form.read(theFormula);
		String ans = "";
		for (int i = 0; form.listP(); i++) {
			ans = form.car();
			if (i == argnum) break;
			form.read(form.cdr());
		}
		return ans == null ? "" : ans;
	}

	/**
	 * **************************************************************
	 * Returns a non-negative int value indicating the top-level list
	 * length of this Formula if it is a proper listP(), else returns
	 * -1.  One caveat: This method assumes that neither null nor the
	 * empty string are legitimate list members in a wff.  The return
	 * value is likely to be wrong if this assumption is mistaken.
	 *
	 * @return A non-negative int, or -1.
	 */
	public int listLength() {

		int ans = -1;
		if (this.listP()) {
			ans = 0;
			while (!StringUtil.emptyString(this.getArgument(ans)))
				++ans;
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Return all the arguments in a simple formula as a list, starting
	 * at the given argument.  If formula is complex (i.e. an argument
	 * is a function or sentence), then return null.  If the starting
	 * argument is greater than the number of arguments, also return
	 * null.
	 */
	public List<String> argumentsToArrayList(int start) {

		if (theFormula.indexOf('(', 1) != -1) {
			List<String> erList = complexArgumentsToArrayList(0);
			for (String s : erList)
				if (s.indexOf('(') != -1 && !StringUtil.quoted(s)) {
					String err = "Error in Formula.argumentsToArrayList() complex formula: " + this;
					errors.add(err);
					System.out.println(err);
					return null;
				}
		}
		int index = start;
		List<String> result = new Lst<>();
		String arg = getArgument(index);
		while (arg != null && !arg.isEmpty()) {
			result.add(arg/*.intern()*/);
			index++;
			arg = getArgument(index);
		}
		if (index == start)
			return null;
		return result;
	}

	/**
	 * **************************************************************
	 * Return all the arguments in a formula as a list, starting
	 * at the given argument.  If the starting
	 * argument is greater than the number of arguments, return null.
	 */
	public List<String> complexArgumentsToArrayList(int start) {

		int index = start;
		List<String> result = new Lst<>();
		String arg = getArgument(index);
		while (arg != null && !arg.isEmpty()) {
			result.add(arg/*.intern()*/);
			index++;
			arg = getArgument(index);
		}
		if (index == start)
			return null;
		return result;
	}

	/**
	 * **************************************************************
	 * Collects all variables in this Formula.  Returns an ArrayList
	 * containing a pair of ArrayLists.  The first contains all
	 * explicitly quantified variables in the Formula.  The second
	 * contains all variables in Formula that are not within the scope
	 * of some explicit quantifier.
	 *
	 * @return An ArrayList containing two ArrayLists, each of which
	 * could be empty
	 */
	public List<List<String>> collectVariables() {

		List<List<String>> ans = new Lst<>();
		ans.add(new Lst());
		ans.add(new Lst());
		Collection<String> unquantified = new HashSet<>(collectAllVariables());
		Collection<String> quantified = new HashSet<>(collectQuantifiedVariables());
		unquantified.removeAll(quantified);
		ans.get(0).addAll(quantified);
		ans.get(1).addAll(unquantified);
		return ans;
	}

	/**
	 * **************************************************************
	 * A new method to collect all quantified and unquantified variables
	 * in this Formula. Return an ArrayList containing a pair of ArrayLists.
	 * The first contains all explicitly quantified varialbles.
	 * The second contains all variables that are not within the scope of
	 * some explicit quantifiers.
	 * <p>
	 * This function is different from the old version collectVariables()
	 * in that it can keep track of some bad axioms where a variable is both
	 * in quantified list and unquantified list;
	 *
	 * @return An ArrayList containing two ArrayLists, each of which could be empty.
	 */
	public List<List<String>> collectQuantifiedUnquantifiedVariables() {

		HashSet<String> unquantifiedVariables = new HashSet<>();
		HashSet<String> quantifiedVariables = new HashSet<>();
		HashMap<String, Boolean> varFlag = new HashMap<>();
		collectQuantifiedUnquantifiedVariablesRecurse
			(this, varFlag, unquantifiedVariables, quantifiedVariables);

		Set<String> intersections = Sets.intersection(quantifiedVariables, unquantifiedVariables);
		if (!intersections.isEmpty())
			System.out.println("Error in Formula.collectQuantifiedUnquantifiedVariables(): Some variables (" +
				intersections
				+ ") are both quantified (" + quantifiedVariables
				+ ") and unquantified (" + unquantifiedVariables + ") in formula \n" + theFormula);

		List<List<String>> quantifiedUnquantifiedVariables = new Lst<>();
		quantifiedUnquantifiedVariables.add(new Lst(quantifiedVariables));
		quantifiedUnquantifiedVariables.add(new Lst(unquantifiedVariables));

		return quantifiedUnquantifiedVariables;
	}

	/**
	 * **************************************************************
	 * Collects all variables in this Formula.  Returns an Set
	 * of String variable names (with initial '?').  Note that
	 * duplicates are not removed.
	 *
	 * @return An ArrayList of String variable names
	 */
	private Set<String> collectAllVariables() {


		HashSet<String> resultSet = new HashSet<>();
		if (listLength() < 1)
			return resultSet;
		Formula fcar = new Formula();
		fcar.read(this.car());
		if (fcar.isVariable())
			resultSet.add(fcar.theFormula);
		else if (fcar.listP())
			resultSet.addAll(fcar.collectAllVariables());
		Formula fcdr = new Formula();
		fcdr.read(this.cdr());
		if (fcdr.isVariable())
			resultSet.add(fcdr.theFormula);
		else if (fcdr.listP())
			resultSet.addAll(fcdr.collectAllVariables());

		return resultSet;
	}
//
//	/**
//	 * **************************************************************
//	 * Collects all quantified variables in this Formula.  Returns an ArrayList
//	 * of String variable names (with initial '?').  Note that
//	 * duplicates are not removed.
//	 *
//	 * @return An ArrayList of String variable names
//	 */
//	public ArrayList<String> collectExistentiallyQuantifiedVariables() {
//
//		ArrayList<String> result = new FasterList<>();
//		if (listLength() < 1)
//			return result;
//		Formula fcar = new Formula();
//		fcar.read(this.car());
//		Collection<String> resultSet = new HashSet<>();
//		if (fcar.theFormula.equals(EQUANT)) {
//			Formula remainder = new Formula();
//			remainder.read(this.cdr());
//			if (!remainder.listP()) {
//				System.out.println("Error in Formula.collectQuantifiedVariables(): incorrect quantification: " + this);
//				return result;
//			}
//			Formula varlist = new Formula();
//			varlist.read(remainder.car());
//			resultSet.addAll(varlist.collectAllVariables());
//			resultSet.addAll(remainder.cdrAsFormula().collectExistentiallyQuantifiedVariables());
//		} else {
//			if (fcar.listP())
//				resultSet.addAll(fcar.collectExistentiallyQuantifiedVariables());
//			resultSet.addAll(this.cdrAsFormula().collectExistentiallyQuantifiedVariables());
//		}
//		result.addAll(resultSet);
//		return result;
//	}

	/**
	 * **************************************************************
	 * Collects all quantified variables in this Formula.  Returns an ArrayList
	 * of String variable names (with initial '?').  Note that
	 * duplicates are not removed.
	 *
	 * @return An ArrayList of String variable names
	 */
	private List<String> collectQuantifiedVariables() {

		List<String> result = new Lst<>();
		if (listLength() < 1)
			return result;
		Formula fcar = new Formula();
		fcar.read(this.car());
		Collection<String> resultSet = new HashSet<>();
		if (fcar.theFormula.equals(UQUANT) || fcar.theFormula.equals(EQUANT)) {
			Formula remainder = new Formula();
			remainder.read(this.cdr());
			if (!remainder.listP()) {
				System.out.println("Error in Formula.collectQuantifiedVariables(): incorrect quantification: " + this);
				return result;
			}
			Formula varlist = new Formula();
			varlist.read(remainder.car());
			resultSet.addAll(varlist.collectAllVariables());
			resultSet.addAll(remainder.cdrAsFormula().collectQuantifiedVariables());
		} else {
			if (fcar.listP())
				resultSet.addAll(fcar.collectQuantifiedVariables());
			resultSet.addAll(this.cdrAsFormula().collectQuantifiedVariables());
		}
		result.addAll(resultSet);
		return result;
	}

	/**
	 * **************************************************************
	 * Collect all the unquantified variables in a formula
	 */
	public List<String> collectUnquantifiedVariables() {
		return collectVariables().get(1);
	}

	/**
	 * **************************************************************
	 * Collect all the terms in a formula
	 */
	private Set<String> collectTerms() {

		if (this.theFormula == null || this.theFormula.isEmpty()) {
			System.out.println("Error in Formula.collectTerms(): " +
				"No formula to collect terms from: " + this);
			return null;
		}

		HashSet<String> resultSet = new HashSet<>();
		if (this.empty())
			return resultSet;

		if (this.atom())
			resultSet.add(theFormula);
		else {
			Formula f = new Formula();
			f.read(theFormula);
			while (!f.empty() && f.theFormula != null && !f.theFormula.isEmpty()) {
				Formula f2 = new Formula();
				f2.read(f.car());
				resultSet.addAll(f2.collectTerms());
				f.read(f.cdr());
			}
		}

		return resultSet;
	}

	/**
	 * **************************************************************
	 * Replace variables with a value as given by the map argument
	 */
	public Formula substituteVariables(Map<String, String> m) {

		Formula newFormula = new Formula();
		newFormula.read("()");
		if (atom()) {
			if (m.containsKey(theFormula)) {
				theFormula = m.get(theFormula);
				if (this.listP())
					theFormula = '(' + theFormula + ')';
			}
			return this;
		}
		if (!empty()) {
			Formula f1 = new Formula();
			f1.read(this.car());
			newFormula = f1.listP() ? newFormula.cons(f1.substituteVariables(m)) : newFormula.append(f1.substituteVariables(m));
			Formula f2 = new Formula();
			f2.read(this.cdr());
			newFormula = newFormula.append(f2.substituteVariables(m));
		}
		return newFormula;
	}

	/**
	 * **************************************************************
	 * Makes implicit quantification explicit.
	 *
	 * @param query controls whether to add universal or existential
	 *              quantification.  If true, add existential.
	 * @result the formula as a String, with explicit quantification
	 */
	public String makeQuantifiersExplicit(boolean query) {

		String result = this.theFormula;
		String arg0 = this.car();
		List<List<String>> vpair = collectVariables();
		List<String> quantVariables = vpair.get(0);
		List<String> unquantVariables = vpair.get(1);

		if (!unquantVariables.isEmpty()) {
			StringBuilder sb = new StringBuilder();
			sb.append(query ? "(exists (" : "(forall (");
			boolean afterTheFirst = false;
			for (String unquantVariable : unquantVariables) {
				if (afterTheFirst) sb.append(' ');
				sb.append(unquantVariable);
				afterTheFirst = true;
			}
			sb.append(") ");
			sb.append(this.theFormula);
			sb.append(')');
			result = sb.toString();
		}
		return result;
	}

	/**
	 * **************************************************************
	 *
	 * @param kb          - The KB used to compute variable arity relations.
	 * @param relationMap is a Map of String keys and values where
	 *                    the key is the renamed relation and the
	 *                    value is the original name.  This is setAt
	 *                    as a side effect of this method.
	 * @return A new version of the Formula in which every
	 * VariableArityRelation has been renamed to include a numeric
	 * suffix corresponding to the actual number of arguments in the
	 * Formula.
	 */
	private Formula renameVariableArityRelations(KB kb, TreeMap<String, String> relationMap) {

		Formula result = this;
		if (this.listP()) {
			StringBuilder sb = new StringBuilder();
			Formula f = new Formula();
			f.read(this.theFormula);
			int flen = f.listLength();
			sb.append('(');
			String arg;
			String suffix = "_" + (flen - 1);
			for (int i = 0; i < flen; i++) {
				arg = f.getArgument(i);
				if (i > 0)
					sb.append(' ');
				if (i == 0 && kb.kbCache.transInstOf(arg, "VariableArityRelation") && !arg.endsWith(suffix)) {
					relationMap.put(arg + suffix, arg);
					arg += suffix;
				} else if (listP(arg)) {
					Formula argF = new Formula();
					argF.read(arg);
					arg = argF.renameVariableArityRelations(kb, relationMap).theFormula;
				}
				sb.append(arg);
			}
			sb.append(')');
			f = new Formula();
			f.read(sb.toString());
			result = f;
		}
		return result;
	}

//	/**
//	 * **************************************************************
//	 * Returns a HashMap in which the keys are the Relation constants
//	 * gathered from this Formula, and the values are ArrayLists in
//	 * which the ordinal positions 0 - n are occupied by the names of
//	 * the corresponding argument types.  n should never be greater
//	 * than the value of Formula.MAX_PREDICATE_ARITY.  For each
//	 * Predicate key, the length of its ArrayList should be equal to
//	 * the predicate's valence + 1.  For each Function, the length of
//	 * its ArrayList should be equal to its valence.  Only Functions
//	 * will have argument types in the 0th position of the ArrayList,
//	 * since this position contains a function's range type.  This
//	 * means that all Predicate ArrayLists will contain at least one
//	 * null value.  A null value will also be added to the nth
//	 * position of an ArrayList when no value can be obtained for that
//	 * position.
//	 *
//	 * @return A HashMap that maps every Relation occurring in this
//	 * Formula to an ArrayList indicating the Relation's argument
//	 * types.  Some HashMap keys may map to null values or empty
//	 * ArrayLists, and most ArrayLists will contain some null values.
//	 */
//	public Map<String, ? extends List> gatherRelationsWithArgTypes(KB kb) {
//
//		Map<String, List> argtypemap = new HashMap<>();
//		Set<String> relations = gatherRelationConstants();
//
//		for (String r : relations) {
//			int atlen = MAX_PREDICATE_ARITY + 1;
//			var argtypes = IntStream.range(0, atlen).mapToObj(i -> kb.getArgType(r, i)).collect(Collectors.toCollection(ArrayList::new));
//			argtypemap.put(r, argtypes);
//		}
//		return argtypemap;
//	}

//	/**
//	 * **************************************************************
//	 * Returns a HashSet of all atomic KIF Relation constants that
//	 * occur as Predicates or Functions (argument 0 terms) in this
//	 * Formula.
//	 *
//	 * @return a HashSet containing the String constants that denote
//	 * KIF Relations in this Formula, or an empty HashSet.
//	 */
//	public HashSet<String> gatherRelationConstants() {
//
//		Collection<String> accumulator = new HashSet<>();
//		if (this.listP() && !this.empty())
//			accumulator.add(this.theFormula);
//		Collection<String> kifLists = new FasterList<>();
//		Formula f;
//		HashSet<String> relations = new HashSet<>();
//		while (!accumulator.isEmpty()) {
//			kifLists.clear();
//			kifLists.addAll(accumulator);
//			accumulator.clear();
//			String klist;
//			for (String kifList : kifLists) {
//				klist = kifList;
//				if (listP(klist)) {
//					f = new Formula();
//					f.read(klist);
//					for (int i = 0; !f.empty(); i++) {
//						String arg = f.car();
//						if (listP(arg)) {
//							if (!empty(arg)) accumulator.add(arg);
//						} else if (isQuantifier(arg)) {
//							accumulator.add(f.getArgument(2));
//							break;
//						} else if (i == 0
//							&& !isVariable(arg)
//							&& !isLogicalOperator(arg)
//							&& !arg.equals(SKFN)
//							&& !StringUtil.isQuotedString(arg)
//							&& !arg.matches(".*\\s.*")) relations.add(arg);
//						f = f.cdrAsFormula();
//					}
//				}
//			}
//		}
//		return relations;
//	}

	/**
	 * **************************************************************
	 * Test whether a Formula is a functional target.  Note this assumes
	 * the textual convention of all functions ending with "Fn".
	 */
	public boolean isFunctionalTerm() {

		boolean ans = false;
		if (this.listP()) {
			String pred = this.car();
			ans = pred.length() > 2 && pred.endsWith(FN_SUFF);
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Test whether a Formula contains a Formula as an argument to
	 * other than a logical operator.
	 */
	private boolean isHigherOrder() {

		if (this.listP()) {
			String pred = this.car();
			boolean logop = isLogicalOperator(pred);
			List<String> al = literalToArrayList();
			for (int i = 1; i < al.size(); i++) {
				String arg = al.get(i);
				Formula f = new Formula();
				f.read(arg);
				if (!atom(arg) && !f.isFunctionalTerm()) if (logop) {
					if (f.isHigherOrder())
						return true;
				} else
					return true;
			}
		}
		return false;
	}

	/**
	 * **************************************************************
	 * Test whether the Formula is a variable
	 */
	public boolean isVariable() {
		return isVariable(theFormula);
	}

	/**
	 * **************************************************************
	 * Test whether the Formula is automatically created by caching
	 */
	public boolean isCached() {
		return sourceFile.contains(KB._cacheFileSuffix);
	}

	/**
	 * **************************************************************
	 * Returns true only if this Formula, explicitly quantified or
	 * not, starts with "=>" or "<=>", else returns false.  It would
	 * be better to test for the occurrence of at least one positive
	 * literal with one or more negative literals, but this test would
	 * require converting the Formula to clausal form.
	 */
	public boolean isRule() {

		boolean ans = false;
		if (this.listP()) {
			String arg0 = this.car();
			if (isQuantifier(arg0)) {
				String arg2 = this.getArgument(2);
				if (listP(arg2)) {
					Formula newF = new Formula();
					newF.read(arg2);
					ans = newF.isRule();
				}
			} else ans = Arrays.asList(IF, IFF).contains(arg0);
		}
		return ans;
	}

	/**
	 * **************************************************************
	 * Returns true only if this Formula, is a horn clause or is simply
	 * modified to be horn by breaking out a conjunctive conclusion.
	 */
	public boolean isHorn(KB kb) {

		if (!isRule()) {
			System.out.println("Error in Formula.isHorn(): Formula is not a rule: " + this);
			return false;
		}
		if (isHigherOrder())
			return false;
		if (theFormula.contains("exists") || theFormula.contains("forall"))
			return false;

		Formula antecedent = cdrAsFormula().carAsFormula();
		if (!antecedent.isSimpleClause(kb) && !"and".equals(antecedent.car()))
			return false;
		Formula consequent = cdrAsFormula().cdrAsFormula().carAsFormula();
		return consequent.isSimpleClause(kb) || "and".equals(consequent.car());
	}

	/**
	 * **************************************************************
	 * Test whether a Formula is a simple list of terms (including
	 * functional terms).
	 */
	public boolean isSimpleClause(KB kb) {

		if (!listP(theFormula))
			return false;
		if (!atom(car()))
			return false;
		String arg;
		int argnum = 1;
		do {
			arg = getArgument(argnum);
			argnum++;
			if (listP(arg)) {
				Formula f = new Formula(arg);
				if (!kb.isFunction(f.car()))
					return false;
			}
		} while (!StringUtil.emptyString(arg));
		return true;
	}
//
//    /** ***************************************************************
//     * Remove the 'holds' prefix wherever it appears.
//     */
//    public static String postProcess(String s) {
//
//        s = s.replaceAll("holds_\\d+__ ","");
//        s = s.replaceAll("apply_\\d+__ ","");
//        return s;
//    }

	/**
	 * **************************************************************
	 * Test whether a Formula is a simple clause wrapped in a
	 * negation.
	 */
	public boolean isSimpleNegatedClause(KB kb) {

		if (!listP(theFormula))
			return false;
		Formula f = new Formula();
		f.read(theFormula);
		if ("not".equals(f.car())) {
			f.read(f.cdr());
			if (empty(f.cdr())) {
				f.read(f.car());
				return f.isSimpleClause(kb);
			} else
				return false;
		} else
			return false;
	}

	/**
	 * ****************************************************************
	 * Tests if this formula is an existentially quantified formula
	 *
	 * @return
	 */
	public boolean isExistentiallyQuantified() {

		return EQUANT.equals(car());
	}

	/**
	 * ****************************************************************
	 * Tests if this formula is an universally quantified formula
	 *
	 * @return
	 */
	public boolean isUniversallyQuantified() {

		return UQUANT.equals(car());
	}

	/**
	 * **************************************************************
	 * Returns true if formula does not have variables, else returns false.
	 */
	public boolean isGround() {
		return isGround(theFormula);
	}

	/**
	 * **************************************************************
	 * Returns true if formula is a simple binary relation (note
	 * that because the argument list includes the predicate, which is
	 * argument 0, there will be three elements)
	 */
	public boolean isBinary() {

		List<String> l = argumentsToArrayList(0);
		return l != null && l.size() == 3;
	}

	/**
	 * **************************************************************
	 *
	 * @return An ArrayList (ordered tuple) representation of the
	 * Formula, in which each top-level element of the Formula is
	 * either an atom (String) or another list.
	 */
	public List<String> literalToArrayList() {

		List<String> tuple = new Lst<>();
		Formula f = this;
		if (f.listP()) while (!f.empty()) {
			tuple.add(f.car());
			f = f.cdrAsFormula();
		}
		return tuple;
	}

	/**
	 * **************************************************************
	 * Replace v with target.
	 * TODO: See if a regex replace is faster (commented out buggy code below)
	 */
	public Formula replaceVar(String v, String term) {


		if (StringUtil.emptyString(theFormula) || empty())
			return this;
		Formula newFormula = new Formula();
		newFormula.read("()");
		if (isVariable()) {
			if (theFormula.equals(v))
				theFormula = term;
			return this;
		}
		if (atom())
			return this;
		if (!empty()) {
			Formula f1 = new Formula();
			f1.read(car());

			newFormula = f1.listP() ? newFormula.cons(f1.replaceVar(v, term)) : newFormula.append(f1.replaceVar(v, term));
			Formula f2 = new Formula();
			f2.read(cdr());

			newFormula = newFormula.append(f2.replaceVar(v, term));
		}
		return newFormula;
	}
//
//	/**
//	 * ****************************************************************
//	 *
//	 * @param quantifier
//	 * @param vars
//	 * @return
//	 * @throws Exception
//	 */
//	public Formula replaceQuantifierVars(String quantifier, List<String> vars) throws Exception {
//
//		if (!quantifier.equals(car())) throw new Exception("The formula is not properly quantified: " + this);
//
//		Formula param = new Formula();
//		param.read(cadr());
//		ArrayList<String> existVars = param.complexArgumentsToArrayList(0);
//
//		if (existVars.size() != vars.size())
//			throw new Exception("Wrong number of variables: " + vars + " to substitute in existentially quantified formula: " + this);
//
//		Formula result = this;
//		for (int i = 0; i < existVars.size(); i++) result = result.replaceVar(existVars.get(i), vars.get(i));
//
//		return result;
//	}

	/**
	 * **************************************************************
	 * Format a formula for either text or HTML presentation by inserting
	 * the proper hyperlink code, characters for indentation and end of line.
	 * A standard LISP-style pretty printing is employed where an open
	 * parenthesis triggers a new line and added indentation.
	 *
	 * @param hyperlink   - the URL to be referenced to a hyperlinked target.
	 * @param indentChars - the proper characters for indenting text.
	 * @param eolChars    - the proper character for end of line.
	 */
	private String format(String hyperlink, String indentChars, String eolChars) {

		if (theFormula == null)
			return "";
		if (!StringUtil.emptyString(theFormula))
			theFormula = theFormula.trim();
		if (atom())
			return theFormula;
		String legalTermChars = "-:";
		String varStartChars = "?@";
		String quantifiers = "forall|exists";
		StringBuilder token = new StringBuilder();
		StringBuilder formatted = new StringBuilder();
		int indentLevel = 0;
		boolean inQuantifier = false;
		boolean inToken = false;
		boolean inVariable = false;
		boolean inVarlist = false;
		boolean inComment = false;

		int flen = theFormula.length();
		char pch = '0';
		char ch;
		for (int i = 0; i < flen; i++) {

			ch = theFormula.charAt(i);
			if (inComment) {
				formatted.append(ch);
				if (i > 70 && ch == '/')
					formatted.append(' ');
				if (ch == '"')
					inComment = false;
			} else {
				if (ch == '('
					&& !inQuantifier
					&& (indentLevel != 0 || i > 1)) {
					if (Character.isWhitespace(pch))
						formatted = formatted.deleteCharAt(formatted.length() - 1);
					formatted.append(eolChars);
					formatted.append(String.valueOf(indentChars).repeat(Math.max(0, indentLevel)));
				}
				if (i == 0 && indentLevel == 0 && ch == '(')
					formatted.append(ch);
				if (!inToken && !inVariable && Character.isJavaIdentifierStart(ch)) {
					token = new StringBuilder(ch);
					inToken = true;
				}
				if (inToken && (Character.isJavaIdentifierPart(ch)
					|| legalTermChars.indexOf(ch) > -1))
					token.append(ch);
				if (ch == '(') if (inQuantifier) {
					inQuantifier = false;
					inVarlist = true;
					token = new StringBuilder();
				} else
					indentLevel++;
				if (ch == '"')
					inComment = true;
				if (ch == ')') if (!inVarlist)
					indentLevel--;
				else
					inVarlist = false;
				if (token.indexOf("forall") > -1 || token.indexOf("exists") > -1)
					inQuantifier = true;
				if (inVariable
					&& !Character.isJavaIdentifierPart(ch)
					&& legalTermChars.indexOf(ch) == -1)
					inVariable = false;
				if (varStartChars.indexOf(ch) > -1)
					inVariable = true;
				if (inToken
					&& !Character.isJavaIdentifierPart(ch)
					&& legalTermChars.indexOf(ch) == -1) {
					inToken = false;
					if (StringUtil.isNonEmptyString(hyperlink)) {
						formatted.append("<a href=\"");
						formatted.append(hyperlink);
						formatted.append("&target=");
						formatted.append(token);
						formatted.append("\">");
						formatted.append(token);
						formatted.append("</a>");
					} else
						formatted.append(token);
					token = new StringBuilder();
				}
				if (i > 0 && !inToken && !(Character.isWhitespace(ch) && pch == '(')) if (Character.isWhitespace(ch)) {
					if (!Character.isWhitespace(pch))
						formatted.append(' ');
				} else
					formatted.append(ch);
			}
			pch = ch;
		}
		if (inToken) if (StringUtil.isNonEmptyString(hyperlink)) {
			formatted.append("<a href=\"").append(hyperlink).append("&target=").append(token).append("\">").append(token).append("</a>");
		} else
			formatted.append(token);
		return formatted.toString();
	}

	/**
	 * **************************************************************
	 * Format a formula for text presentation.
	 */
	public String toString() {
		return format("", "  ", Character.valueOf((char) 10).toString());
	}

//	/**
//	 * **************************************************************
//	 * Format a formula for text presentation include file and line#.
//	 */
//	public String toStringMeta() {
//		return format("", "  ", Character.valueOf((char) 10).toString()) +
//			'[' + sourceFile + ' ' + startLine + '-' + endLine + ']';
//	}

//	/**
//	 * **************************************************************
//	 * Format a formula for HTML presentation.
//	 */
//	public String htmlFormat(String html) {
//		return format(html, "&nbsp;&nbsp;&nbsp;&nbsp;", "<br>\n");
//	}


//	public String htmlFormat(KB kb, String href) {
//		String kbHref = href + "/sigma/Browse.jsp?kb=" + kb.name;
//		return format(kbHref, "&nbsp;&nbsp;&nbsp;&nbsp;", "<br>\n");
//	}

	/**
	 * **************************************************************
	 * Format a formula as a prolog statement.  Note that only tuples
	 * are converted properly at this time.  Statements with any embedded
	 * formulas or functions will be rejected with a null return.
	 */
	public String toProlog() {

		System.out.println("INFO in Formula.toProlog(): formula: " + theFormula);
		if (!listP()) {
			System.out.println("Error in Formula.toProlog(): Not a formula: " + theFormula);
			return null;
		}
		if (empty()) {
			System.out.println("Error in Formula.toProlog(): Empty formula: " + theFormula);
			return null;
		}
		String relation = car();
		Formula f = new Formula();
		if (!atom(relation)) {
			//System.out.println("Error in Formula.toProlog(): Relation not an atom: " + relation);
			//return null;
			var l = literalToArrayList();
			relation = l.remove(0);
			//f.theFormula(Formula.list)
			f.theFormula = cdr();
		} else {
			f.theFormula = cdr();
		}

		StringBuilder result = new StringBuilder();
		result.append(relation).append('(');
		//System.out.println("INFO in Formula.toProlog(): result so far: " + result);
		//System.out.println("INFO in Formula.toProlog(): remaining formula: " + f);
		int l = f.listLength();
		for (int i = 0; i < l; i++) {
			String arg = f.getArgument(i);
			//while (!f.empty()) {
			//String arg = f.car();
			//System.out.println("INFO in Formula.toProlog(): argForm: " + arg);
			if (!atom(arg)) {
				result.append(new Formula(arg).toProlog());
			} else if (isVariable(arg))
				result.append(Character.toUpperCase(arg.charAt(1))).append(arg.substring(2));
			else if (StringUtil.isQuotedString(arg))
				result.append(arg);
			else
				result.append('\'').append(arg).append('\'');

			result.append(i == l-1 ? ')' : ',');

			//f.theFormula = f.cdr();
		}
		return result.toString();
	}

	/**
	 * **************************************************************
	 * Replace term2 with term1
	 */
	public Formula rename(String term2, String term1) {

		Formula newFormula = new Formula();
		newFormula.read("()");
		if (atom()) {
			if (theFormula.equals(term2))
				theFormula = term1;
			return this;
		}
		if (!empty()) {
			Formula f1 = new Formula();
			f1.read(car());
			newFormula = f1.listP() ? newFormula.cons(f1.rename(term2, term1)) : newFormula.append(f1.rename(term2, term1));
			Formula f2 = new Formula();
			f2.read(cdr());
			newFormula = newFormula.append(f2.rename(term2, term1));
		}
		return newFormula;
	}


	public Formula negate() {
		return new Formula("(not " + theFormula + ')');
	}


	private static class VariableMapping {

		final String var1;
		final String var2;

		VariableMapping(String v1, String v2) {
			var1 = v1;
			var2 = v2;
		}

		static List<Set<VariableMapping>> intersect(Iterable<Set<VariableMapping>> mapList1,
													Iterable<Set<VariableMapping>> mapList2) {

			if (mapList1 == null || mapList2 == null) return null;
			List<Set<VariableMapping>> intersection = new LinkedList<>();
			for (Set<VariableMapping> set1 : mapList1)
				for (Set<VariableMapping> set2 : mapList2) {
					Set<VariableMapping> newSet = unify(set1, set2);
					if (newSet != null && !intersection.contains(newSet)) intersection.add(newSet);
				}
			if (intersection.isEmpty()) intersection = null;
			return intersection;
		}

		static List<Set<VariableMapping>> union(Collection<Set<VariableMapping>> mapList1,
												Iterable<Set<VariableMapping>> mapList2) {

			List<Set<VariableMapping>> union = new LinkedList<>();
			if (mapList1 != null) union.addAll(mapList1);
			if (mapList2 != null) for (Set<VariableMapping> set2 : mapList2) if (!union.contains(set2)) union.add(set2);
			return union;
		}

		private static Set<VariableMapping> unify(Collection<VariableMapping> set1, Iterable<VariableMapping> set2) {

			Set<VariableMapping> result = new HashSet<>(set1);
			for (VariableMapping element : set2) {

				for (VariableMapping e : result) {
					boolean leftVarsEqual = e.var1.equals(element.var1);
					boolean rightVarsEqual = e.var2.equals(element.var2);
					if (leftVarsEqual != rightVarsEqual) return null;
				}
				result.add(element);
			}
			return result;
		}

		@Override
		public boolean equals(Object o) {
			if (this == o) return true;
			if (o == null || getClass() != o.getClass()) return false;

			VariableMapping that = (VariableMapping) o;

			if (!Objects.equals(var1, that.var1)) return false;
			return Objects.equals(var2, that.var2);
		}

		@Override
		public int hashCode() {
			int result = var1 != null ? var1.hashCode() : 0;
			result = 31 * result + (var2 != null ? var2.hashCode() : 0);
			return result;
		}

		@Override
		public String toString() {
			return "VariableMapping{" +
				"var1='" + var1 + '\'' +
				", var2='" + var2 + '\'' +
				'}';
		}
	}
}