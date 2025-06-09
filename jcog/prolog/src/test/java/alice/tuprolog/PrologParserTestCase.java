package alice.tuprolog;


import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PrologParserTestCase {
	
	@Test
	void testReadingTerms() throws InvalidTermException {
		PrologParser p = new PrologParser("hello.");
		Struct result = new Struct("hello");
		assertEquals(result, p.nextTerm(true));
	}
	
	@Test
	void testReadingEOF() throws InvalidTermException {
		PrologParser p = new PrologParser("");
		assertNull(p.nextTerm(false));
	}
	
	@Test
	void testUnaryPlusOperator() {
		PrologParser p = new PrologParser("n(+100).\n");
        
		
		
		try {
			assertNotNull(p.nextTerm(true));
			fail("");
		} catch (InvalidTermException e) {}
	}
	
	@Test
	void testUnaryMinusOperator() throws InvalidTermException {
		PrologParser p = new PrologParser("n(-100).\n");
		
		
		
		
		Struct result = new Struct("n", new NumberTerm.Int(-100));
		result.resolveTerm();
		assertEquals(result, p.nextTerm(true));
	}
	
	@Test
	void testBinaryMinusOperator() throws InvalidTermException {
		String s = "abs(3-11)";
		PrologParser p = new PrologParser(s);
		Struct result = new Struct("abs", new Struct("-", new NumberTerm.Int(3), new NumberTerm.Int(11)));
		assertEquals(result, p.nextTerm(false));
	}
	
	@Test
	void testListWithTail() throws InvalidTermException {
		PrologParser p = new PrologParser("[p|Y]");
		Struct a = new Struct(new Struct("p"), new Var("Y"));
		
		Term b = p.nextTerm(false);
		a.resolveTerm(((Var)((Struct)b).sub(1)).timestamp);
		assertEquals(a, b);
	}
	
	@Test
	void testBraces() throws InvalidTermException {
		String s = "{a,b,[3,{4,c},5],{a,b}}";
		PrologParser parser = new PrologParser(s);
		assertEquals(s, parser.nextTerm(false).toString());
	}
	
	@Test
	void testUnivOperator() throws InvalidTermException {
		PrologParser p = new PrologParser("p =.. q.");
		Struct result = new Struct("=..", new Struct("p"), new Struct("q"));
		assertEquals(result, p.nextTerm(true));
	}
	
	@Test
	void testDotOperator() throws InvalidTermException {
        PrologOperators.DefaultOps om = PrologOperators.DefaultOps.defaultOps;
		om.opNew(".", "xfx", 600);
        String s = "class('java.lang.Integer').'MAX_VALUE'";
        PrologParser p = new PrologParser(s, om);
		Struct result = new Struct(".", new Struct("class", new Struct("java.lang.Integer")),
				                        new Struct("MAX_VALUE"));
		assertEquals(result, p.nextTerm(false));
	}
	
	@Test
	void testBracketedOperatorAsTerm() throws InvalidTermException {
        PrologOperators.DefaultOps om = PrologOperators.DefaultOps.defaultOps;
		om.opNew("u", "fx", 200);
		om.opNew("b1", "yfx", 400);
		om.opNew("b2", "yfx", 500);
		om.opNew("b3", "yfx", 300);
        String s = "u (b1) b2 (b3)";
        PrologParser p = new PrologParser(s, om);
		Struct result = new Struct("b2", new Struct("u", new Struct("b1")), new Struct("b3"));
		assertEquals(result, p.nextTerm(false));
	}
	
	@Test
	void testBracketedOperatorAsTerm2() throws InvalidTermException {
        PrologOperators.DefaultOps om = PrologOperators.DefaultOps.defaultOps;
		om.opNew("u", "fx", 200);
		om.opNew("b1", "yfx", 400);
		om.opNew("b2", "yfx", 500);
		om.opNew("b3", "yfx", 300);
        String s = "(u) b1 (b2) b3 a";
        PrologParser p = new PrologParser(s, om);
		Struct result = new Struct("b1", new Struct("u"), new Struct("b3", new Struct("b2"), new Struct("a")));
		assertEquals(result, p.nextTerm(false));
	}
	
	@Test
	void testIntegerBinaryRepresentation() throws InvalidTermException {
		String n = "0b101101";
		PrologParser p = new PrologParser(n);
		NumberTerm result = new NumberTerm.Int(45);
		assertEquals(result, p.nextTerm(false));
        try {
            String invalid = "0b101201";
            new PrologParser(invalid).nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
	}
	
	@Test
	void testIntegerOctalRepresentation() throws InvalidTermException {
		String n = "0o77351";
		PrologParser p = new PrologParser(n);
		NumberTerm result = new NumberTerm.Int(32489);
		assertEquals(result, p.nextTerm(false));
        try {
            String invalid = "0o78351";
            new PrologParser(invalid).nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
	}
	
	@Test
	void testIntegerHexadecimalRepresentation() throws InvalidTermException {
		String n = "0xDECAF";
		PrologParser p = new PrologParser(n);
		NumberTerm result = new NumberTerm.Int(912559);
		assertEquals(result, p.nextTerm(false));
        try {
            String invalid = "0xG";
            new PrologParser(invalid).nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
	}
	
	@Test
	void testEmptyDCGAction() throws InvalidTermException {
		String s = "{}";
		PrologParser p = new PrologParser(s);
		Struct result = new Struct("{}");
		assertEquals(result, p.nextTerm(false));
	}
	
	@Test
	void testSingleDCGAction() throws InvalidTermException {
		String s = "{hello}";
		PrologParser p = new PrologParser(s);
		Struct result = new Struct("{}", new Struct("hello"));
		assertEquals(result, p.nextTerm(false));
	}
	
	@Test
	void testMultipleDCGAction() throws InvalidTermException {
		String s = "{a, b, c}";
		PrologParser p = new PrologParser(s);
		Struct result = new Struct("{}",
                                   new Struct(",", new Struct("a"),
                                       new Struct(",", new Struct("b"), new Struct("c"))));
		assertEquals(result, p.nextTerm(false));
	}
	
	 
	@Test
	void testDCGActionWithOperators() {
        Struct result = new Struct("{}",
                            new Struct(",", new Struct("=..", new Var("A"), new Var("B")),
                                new Struct(",", new Struct("hotel"), new NumberTerm.Int(2))));
        result.resolveTerm();
        String input = "{A =.. B, hotel, 2}";
        PrologParser p = new PrologParser(input);
        
		assertEquals(result.toString(), p.nextTerm(false).toString());
	}
	
	@Test
	void testMissingDCGActionElement() {
		String s = "{1, 2, , 4}";
		PrologParser p = new PrologParser(s);
		try {
			p.nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
	}
	
	@Test
	void testDCGActionCommaAsAnotherSymbol() {
		String s = "{1 @ 2 @ 4}";
		PrologParser p = new PrologParser(s);
		try {
			p.nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
	}
	
	@Test
	void testUncompleteDCGAction() {
		String s = "{1, 2,}";
		PrologParser p = new PrologParser(s);
		try {
			p.nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
		
		s = "{1, 2";
		p = new PrologParser(s);
		try {
			p.nextTerm(false);
			fail("");
		} catch (InvalidTermException expected) {}
	}

	@Test
	void testMultilineComments() throws InvalidTermException {
		String theory = String.join("\n", "t1.", "/*", "t2", "*/", "t3.") + '\n';
		PrologParser p = new PrologParser(theory);
		assertEquals(new Struct("t1"), p.nextTerm(true));
		assertEquals(new Struct("t3"), p.nextTerm(true));
	}
	
	@Test
	void testSingleQuotedTermWithInvalidLineBreaks() {
		String s = """
				out('can_do(X).
				can_do(Y).
				').""";
		PrologParser p = new PrologParser(s);
		try {
			p.nextTerm(true);
			fail("");
		} catch (InvalidTermException expected) {}
	}
	
	
	
	
	
	
	
	
	
	

}
