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


import jcog.data.map.ObjIntHashMap;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.NoSuchElementException;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * This class defines a parser of prolog terms and sentences.
 * <p/>
 * BNF part 2: Parser
 * term ::= exprA(1200)
 * exprA(n) ::= exprB(n) { op(yfx,n) exprA(n-1) |
 * op(yf,n) }*
 * exprB(n) ::= exprC(n-1) { op(xfx,n) exprA(n-1) |
 * op(xfy,n) exprA(n) |
 * op(xf,n) }*
 * <p>
 * exprC(n) ::= '-' integer | '-' float |
 * op( fx,n ) exprA(n-1) |
 * op( fy,n ) exprA(n) |
 * exprA(n)
 * exprA(0) ::= integer |
 * float |
 * atom |
 * variable |
 * atom'(' exprA(1200) { ',' exprA(1200) }* ')' |
 * '[' [ exprA(1200) { ',' exprA(1200) }* [ '|' exprA(1200) ] ] ']' |
 * '(' { exprA(1200) }* ')'
 * '{' { exprA(1200) }* '}'
 * op(type,n) ::= atom | { symbol }+
 */
public class PrologParser {
    private static final Predicate<String> atomMatches = Pattern.compile("(!|[a-z][a-zA-Z_0-9]*)").asMatchPredicate();
    private final Tokenizer tokenizer;
    private final ObjIntHashMap<Term> offsetsMap;
    private PrologOperators ops = PrologOperators.DefaultOps.defaultOps;
    private int tokenStart;

    /**
     * creating a Parser specifing how to handle operators
     * and what text to parse
     */
    public PrologParser(String theoryText, PrologOperators op, ObjIntHashMap<Term> mapping) {
        this(theoryText, mapping);
        if (op != null)
            ops = op;
    }


    /**
     * creating a Parser specifing how to handle operators
     * and what text to parse
     */
    public PrologParser(String theoryText, PrologOperators op) {
        this(theoryText, op, null);
    }

    /*Castagna 06/2011*/

    /**
     * creating a parser with default operator interpretation
     */
    public PrologParser(String theoryText, ObjIntHashMap<Term> mapping) {
        tokenizer = new Tokenizer(theoryText);
        offsetsMap = mapping;
    }
    /**/

    public PrologParser(String theoryText) {
        this(theoryText, PrologOperators.DefaultOps.defaultOps);
    }

    /**
     * Static service to get a term from its string representation
     */
    public static Term parseSingleTerm(String st) throws InvalidTermException {
        return parseSingleTerm(st, null);
    }

    /**
     * Static service to get a term from its string representation,
     * providing a specific operator manager
     */
    public static Term parseSingleTerm(String st, PrologOperators op) throws InvalidTermException {
        try {
            var p = new PrologParser(st, op);
            var t = p.tokenizer.readToken();
            if (t.isEOF())
                throw new InvalidTermException("Term starts with EOF");

            p.tokenizer.unreadToken(t);
            var term = p.expr(false);
            if (term == null)
                throw new InvalidTermException("Term is null");
            if (!p.tokenizer.readToken().isEOF())
                throw new InvalidTermException("The entire string could not be read as one term");
            term.resolveTerm();
            return term;
        } catch (IOException ex) {
            throw new InvalidTermException("An I/O error occured");
        }
    }

    static NumberTerm parseInteger(String s) {
        var num = Long.parseLong(s);
        return num > Integer.MIN_VALUE && num < Integer.MAX_VALUE ? new NumberTerm.Int((int) num) : new NumberTerm.Long(num);
    }

    static NumberTerm.Double parseFloat(String s) {
        return new NumberTerm.Double(Double.parseDouble(s));
    }

    static NumberTerm createNumber(String s) {
        try {
            return parseInteger(s);
        } catch (RuntimeException e) {
            return parseFloat(s);
        }
    }

    /**
     * @return true if the String could be a prolog atom
     */
    public static boolean isAtom(String s) {
        return atomMatches.test(s);
    }

    public Iterator<Term> iterator() {
        return new TermIterator(this);
    }

    /**
     * Parses next term from the stream built on string.
     *
     * @param endNeeded <tt>true</tt> if it is required to parse the end token
     *                  (a period), <tt>false</tt> otherwise.
     * @throws InvalidTermException if a syntax error is found.
     */
    public Term nextTerm(boolean endNeeded) {
        try {
            var t = tokenizer.readToken();
            if (t.isEOF())
                return null;

            tokenizer.unreadToken(t);
            var term = expr(false);
            if (term == null)
                throw invalidTermException("The parser is unable to finish.");

            if (endNeeded && tokenizer.readToken().getType() != Tokenizer.END)
                throw invalidTermException("The term '" + term + "' is not ended with a period.");

            term.resolveTerm();
            return term;
        } catch (IOException ex) {
            throw invalidTermException("I/O error");
        }
    }

    private InvalidTermException invalidTermException(String s) {
        var rc = tokenizer.offsetToRowColumn(getCurrentOffset());
        return new InvalidTermException(s, rc[0], rc[1] - 1);
    }

    private Term expr(boolean commaIsEndMarker) throws InvalidTermException, IOException {
        return exprA(PrologOperators.OP_HIGH, commaIsEndMarker).result;
    }

    private IdentifiedTerm exprA(int maxPriority, boolean commaIsEndMarker) throws InvalidTermException, IOException {

        var leftSide = exprB(maxPriority, commaIsEndMarker);

        var t = tokenizer.readToken();
        for (; t.isOperator(commaIsEndMarker); t = tokenizer.readToken()) {

            var YFX = ops.opPrio(t.seq, "yfx");
            var YF = ops.opPrio(t.seq, "yf");


            if (YF < leftSide.priority || YF > maxPriority) YF = -1;

            if (YFX < leftSide.priority || YFX > maxPriority) YFX = -1;

            if (YFX >= YF && YFX >= PrologOperators.OP_LOW) {
                var ta = exprA(YFX - 1, commaIsEndMarker);
                if (ta != null) {
                    /*Castagna 06/2011*/

                    leftSide = identifyTerm(YFX, new Struct(t.seq, leftSide.result, ta.result), tokenStart);
                    /**/
                    continue;
                }
            }

            if (YF >= PrologOperators.OP_LOW) {
                /*Castagna 06/2011*/

                leftSide = identifyTerm(YF, new Struct(t.seq, leftSide.result), tokenStart);
                /**/
                continue;
            }
            break;
        }
        tokenizer.unreadToken(t);
        return leftSide;
    }

    private IdentifiedTerm exprB(int maxPriority, boolean commaIsEndMarker) throws InvalidTermException, IOException {
        var left = parseLeftSide(commaIsEndMarker, maxPriority);

        var operator = tokenizer.readToken();
        for (; operator.isOperator(commaIsEndMarker); operator = tokenizer.readToken()) {
            var XFX = ops.opPrio(operator.seq, "xfx");
            var XFY = ops.opPrio(operator.seq, "xfy");
            var XF = ops.opPrio(operator.seq, "xf");


            if (XFX > maxPriority || XFX < PrologOperators.OP_LOW) XFX = -1;
            if (XFY > maxPriority || XFY < PrologOperators.OP_LOW) XFY = -1;
            if (XF > maxPriority || XF < PrologOperators.OP_LOW) XF = -1;


            var haveAttemptedXFX = false;
            if (XFX >= XFY && XFX >= XF && XFX >= left.priority) {
                var found = exprA(XFX - 1, commaIsEndMarker);
                if (found != null) {
                    /*Castagna 06/2011*/


                    left = identifyTerm(XFX, new Struct(operator.seq, left.result, found.result), tokenStart);
                    /**/
                    continue;
                } else
                    haveAttemptedXFX = true;
            }

            if (XFY >= XF && XFY >= left.priority) {
                var found = exprA(XFY, commaIsEndMarker);
                if (found != null) {
                    /*Castagna 06/2011*/


                    left = identifyTerm(XFY, new Struct(operator.seq, left.result, found.result), tokenStart);
                    /**/
                    continue;
                }
            }

            if (XF >= left.priority)
                /*Castagna 06/2011*/

                return identifyTerm(XF, new Struct(operator.seq, left.result), tokenStart);
            /**/


            if (!haveAttemptedXFX && XFX >= left.priority) {
                var found = exprA(XFX - 1, commaIsEndMarker);
                if (found != null) {
                    /*Castagna 06/2011*/


                    left = identifyTerm(XFX, new Struct(operator.seq, left.result, found.result), tokenStart);
                    /**/
                    continue;
                }
            }
            break;
        }
        tokenizer.unreadToken(operator);
        return left;
    }

    /**
     * Parses and returns a valid 'leftside' of an expression.
     * If the left side starts with a prefix, it consumes other expressions with a lower priority than itself.
     * If the left side does not have a prefix it must be an expr0.
     *
     * @param commaIsEndMarker used when the leftside is part of and argument list of expressions
     * @param maxPriority      operators with a higher priority than this will effectivly end the expression
     * @return a wrapper of: 1. term correctly structured and 2. the priority of its root operator
     * @throws InvalidTermException
     */
    private IdentifiedTerm parseLeftSide(boolean commaIsEndMarker, int maxPriority) throws InvalidTermException, IOException {

        var f = tokenizer.readToken();
        if (f.isOperator(commaIsEndMarker)) {
            var FX = ops.opPrio(f.seq, "fx");
            var FY = ops.opPrio(f.seq, "fy");

            if ("-".equals(f.seq)) {
                var t = tokenizer.readToken();
                if (t.isNumber())
                    /*Michele Castagna 06/2011*/

                    return identifyTerm(0, createNumber('-' + t.seq), tokenStart);
                    /**/
                else
                    tokenizer.unreadToken(t);
            }


            if (FY > maxPriority) FY = -1;
            if (FX > maxPriority) FX = -1;


            var haveAttemptedFX = false;
            if (FX >= FY && FX >= PrologOperators.OP_LOW) {
                var found = exprA(FX - 1, commaIsEndMarker);
                if (found != null)
                    /*Castagna 06/2011*/

                    return identifyTerm(FX, new Struct(f.seq, found.result), tokenStart);
                    /**/
                else
                    haveAttemptedFX = true;
            }

            if (FY >= PrologOperators.OP_LOW) {
                var found = exprA(FY, commaIsEndMarker);
                if (found != null)
                    /*Castagna 06/2011*/

                    return identifyTerm(FY, new Struct(f.seq, found.result), tokenStart);
                /**/
            }

            if (!haveAttemptedFX && FX >= PrologOperators.OP_LOW) {
                var found = exprA(FX - 1, commaIsEndMarker);
                if (found != null)
                    /*Castagna 06/2011*/

                    return identifyTerm(FX, new Struct(f.seq, found.result), tokenStart);
                /**/
            }
        }
        tokenizer.unreadToken(f);

        return new IdentifiedTerm(0, expr0());
    }

    /**
     * exprA(0) ::= integer |
     * float |
     * variable |
     * atom |
     * atom( exprA(1200) { , exprA(1200) }* ) |
     * '[' exprA(1200) { , exprA(1200) }* [ | exprA(1200) ] ']' |
     * '{' [ exprA(1200) ] '}' |
     * '(' exprA(1200) ')'
     */
    private Term expr0() throws InvalidTermException, IOException {
        var t1 = tokenizer.readToken();

        /*Castagna 06/2011*/
		/*
		if (t1.isType(Tokenizer.INTEGER))
			return Parser.parseInteger(t1.seq);

		if (t1.isType(Tokenizer.FLOAT))
			return Parser.parseFloat(t1.seq);

		if (t1.isType(Tokenizer.VARIABLE))
			return new Var(t1.seq);
		*/

        var tempStart = tokenizer.tokenStart();

        if (t1.isType(Tokenizer.INTEGER)) {
            Term i = parseInteger(t1.seq);
            map(i, tokenizer.tokenStart());
            return i;
        }

        if (t1.isType(Tokenizer.FLOAT)) {
            Term f = parseFloat(t1.seq);
            map(f, tokenizer.tokenStart());
            return f;
        }

        if (t1.isType(Tokenizer.VARIABLE)) {
            Term v = new Var(t1.seq);
            map(v, tokenizer.tokenStart());
            return v;
        }
        /**/

        var result = IntStream.of(Tokenizer.ATOM, Tokenizer.SQ_SEQUENCE, Tokenizer.DQ_SEQUENCE).anyMatch(t1::isType);
        if (result) {
            if (!t1.isFunctor())
                /*Castagna 06/2011*/ {

                Term f = new Struct(t1.seq);
                map(f, tokenizer.tokenStart());
                return f;
            }
            /**/

            var functor = t1.seq;
            var t2 = tokenizer.readToken();
            if (!t2.isType(Tokenizer.LPAR))
                throw new InvalidTermException("Something identified as functor misses its first left parenthesis");
            var a = expr0_arglist();
            var t3 = tokenizer.readToken();
            if (t3.isType(Tokenizer.RPAR))
                /*Castagna 06/2011*/ {

                Term c = new Struct(functor, a);
                map(c, tempStart);
                return c;
            }
            /**/
            /*Castagna 06/2011*/

            throw invalidTermException("Missing right parenthesis '(" + a + "' -> here <-");
            /**/
        }

        if (t1.isType(Tokenizer.LPAR)) {
            var term = expr(false);
            if (tokenizer.readToken().isType(Tokenizer.RPAR))
                return term;
            /*Castagna 06/2011*/

            throw invalidTermException("Missing right parenthesis '(" + term + "' -> here <-");
            /**/
        }

        if (t1.isType(Tokenizer.LBRA)) {
            var t2 = tokenizer.readToken();
            if (t2.isType(Tokenizer.RBRA))
                return Struct.emptyList();

            tokenizer.unreadToken(t2);
            var term = expr0_list();
            if (tokenizer.readToken().isType(Tokenizer.RBRA))
                return term;
            /*Castagna 06/2011*/

            throw invalidTermException("Missing right bracket '[" + term + " ->' here <-");
            /**/
        }

        if (t1.isType(Tokenizer.LBRA2)) {
            var t2 = tokenizer.readToken();
            if (t2.isType(Tokenizer.RBRA2))
                /*Castagna 06/2011*/ {

                Term b = new Struct("{}");
                map(b, tempStart);
                return b;
            }
            /**/
            tokenizer.unreadToken(t2);
            var arg = expr(false);
            t2 = tokenizer.readToken();
            if (t2.isType(Tokenizer.RBRA2))
                /*Castagna 06/2011*/ {

                Term b = new Struct("{}", arg);
                map(b, tempStart);
                return b;
            }
            /*Castagna 06/2011*/

            throw invalidTermException("Missing right braces '{" + arg + "' -> here <-");
            /**/
        }
        /*Castagna 06/2011*/

        throw invalidTermException("Unexpected token '" + t1.seq + '\'');
        /**/
    }

    private Term expr0_list() throws InvalidTermException, IOException {
        var head = expr(true);
        var t = tokenizer.readToken();
        switch (t.seq) {
            case "," -> {
                return new Struct(head, expr0_list());
            }
            case "|" -> {
                return new Struct(head, expr(true));
            }
            case "]" -> {
                tokenizer.unreadToken(t);
                return new Struct(head, Struct.emptyList());
            }
            case null, default -> {
            }
        }
        /*Castagna 06/2011*/

        throw invalidTermException("The expression '" + head + "' is not followed by either a ',' or '|'  or ']'.");
        /**/

    }

    /*Castagna 06/2011*/
    /*
     * Francesco Fabbri
     * 18/04/2011
     * Mapping terms on text
     */

    private LinkedList<Term> expr0_arglist() throws InvalidTermException, IOException {
        var head = expr(true);
        var t = tokenizer.readToken();
        var ts = t.seq;
        if (ts.length() == 1) {
            var ts0 = ts.charAt(0);
            if (ts0 == ',') {
                var l = expr0_arglist();
                l.addFirst(head);
                return l;
            }
            if (ts0 == ')') {
                tokenizer.unreadToken(t);
                var l = new LinkedList<Term>();
                l.add(head);
                return l;
            }
        }
        /*Castagna 06/2011*/

        /*Castagna 06/2011*/

        throw invalidTermException("The argument '" + head + "' is not followed by either a ',' or ')'.");
        /**/
    }

    private IdentifiedTerm identifyTerm(int priority, Term term, int offset) {
        map(term, offset);
        return new IdentifiedTerm(priority, term);
    }

//    public HashMap<Term, Integer> getTextMapping() {
//    	return offsetsMap;
//    }

    /*
     * Francesco Fabbri
     * 19/04/2011
     * Offset / line tracking
     */

    private void map(Term term, int offset) {
        if (offsetsMap != null)
            offsetsMap.put(term, offset);
    }

    /*Castagna 06/2011*/

    /**/
    public int getCurrentLine() {
        return tokenizer.lineno();
    }


//	public int[] offsetToRowColumn(int offset) {
//    	return tokenizer.offsetToRowColumn(offset);
//	}
    /**/

    public int getCurrentOffset() {
        return tokenizer.tokenOffset();
    }

    private record IdentifiedTerm(int priority, Term result) {
    }

    /**
     * This class represents an iterator of terms from Prolog text embedded
     * in a parser. Note that this class resembles more a generator than an
     * iterator type. In fact, both {@link TermIterator#next()} and
     * {@link TermIterator#hasNext()} throws {@link InvalidTermException} if
     * the next term they are trying to return or check for contains a syntax
     * error; this is due to both methods trying to generate the next term
     * instead of just returning it or checking for its existence from a pool
     * of already produced terms.
     */
    static class TermIterator implements Iterator<Term> {

        private final PrologParser parser;
        private boolean hasNext;
        private Term next;

        TermIterator(PrologParser p) {
            parser = p;
            //next = parser.nextTerm(true);
            //hasNext = (next != null);
        }

        @Override
        public Term next() {
            if (hasNext) {
                if (next == null) {
                    next = parser.nextTerm(true);
                    if (next == null)
                        throw new NoSuchElementException();
                }
                hasNext = false;
                var temp = next;
                next = null;
                return temp;
            } else if (hasNext()) {
                hasNext = false;
                var temp = next;
                next = null;
                return temp;
            }
            throw new NoSuchElementException();
        }

        /**
         * @throws InvalidTermException if, while the parser checks for the
         *                              existence of the next term, a syntax error is encountered.
         */
        @Override
        public boolean hasNext() {
            if (!hasNext) {
                next = parser.nextTerm(true);
                if (next != null)
                    hasNext = true;
            }
            return hasNext;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }

    }
}