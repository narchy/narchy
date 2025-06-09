package nars;

import jcog.Util;
import jcog.util.ArrayUtil;
import nars.concept.Concept;
import nars.concept.TaskConcept;
import nars.memory.CaffeineMemory;
import nars.table.BeliefTable;
import nars.table.TaskTable;
import nars.term.Termed;
import nars.time.clock.RealTime;

import java.net.MalformedURLException;
import java.net.URL;

import static nars.Op.*;

public enum N { ;

	static final NAR global = new NARS()
		.memory(CaffeineMemory.soft())
		.time(new RealTime.MS())
		.get();

	static final ThreadLocal<NAR> nar = ThreadLocal.withInitial(()->global);

	public static  Termed $(String term) {
		return () -> $.$$(term);
	}

	private static Termed resolve(Object x) {
		return x instanceof String ? $((String)x) : $.the(x);
	}

	public static Termed $(Op o, Object... args) {
		return args instanceof Term[] ? $(o, ((Term[])args)) :
			$(o, Util.map(N::resolve, new Termed[args.length], args));
	}

	public static Termed $(Object... x) {
		if (x.length > 1 && x[0] instanceof Op) {
			return $((Op)x[0], ArrayUtil.subarray(x, 1, x.length));
		} else {
			throw new UnsupportedOperationException();
		}
	}

	@SafeVarargs
	public static <T extends Termed> Termed $(Op o, T... args) {
		return () -> {
			Term[] a;
			if (args instanceof Term[]) {
				a = (Term[]) args;
			} else {
				//TODO repeats only need .term() once
				a = Util.map(Termed::term, new Term[args.length], args);
			}
			return o.build(terms, a);
		};
	}

	public static Concept the(String term) {
		return the($(term));
	}

	public static Concept the(Object... x) {
		return the($(x));
	}

	public static Concept the(Termed id) {
		return the(id.term());
	}

	public static Concept the(Term id) {
		return nar.get().conceptualize(id.concept());
	}

	public static <T extends TaskTable> T tasks(byte punc, Object... x) {
		return (T) ((TaskConcept)the(x)).table(punc);
	}

	public static BeliefTable beliefs(Object... x) { return tasks(BELIEF, x); }
	public static TaskTable goals(Object... x) { return tasks(GOAL, x); }
	public static TaskTable questions(Object... x) { return tasks(QUESTION, x); }
	public static TaskTable quests(Object... x) { return tasks(QUEST, x); }

	public static void main(String[] args) throws MalformedURLException {
		beliefs(IMPL, "x", "y").print();
		questions(IMPL, "x", $(CONJ, "y", "z")).taskStream().forEach(System.out::println);
		the(INH, "x", $(PROD, +1, "?1", new URL("http://localhost"))).print();
	}
}