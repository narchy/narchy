package alice.tuprolog;

public class ParsingSpeedTest {
	
	public static void main(String... args) throws InvalidTermException {
		int repetitions = 1000;
		long start = System.currentTimeMillis();
		PrologOperators om = new Prolog().ops;
		for (int i = 0; i < repetitions; i++)
			PrologParser.parseSingleTerm("A ; B :- A =.. ['->', C, T], !, (C, !, T ; B)", om);
		long time = System.currentTimeMillis() - start;
		System.out.println("Time parsing " + repetitions + " terms: " + time + " milliseconds.");
	}

}
