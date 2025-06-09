package jcog.decision.feature;

import tech.tablesaw.api.Row;

import java.util.function.Function;
import java.util.stream.Stream;

public abstract class DiscreteFeature<X> {
	public final String name;
	public final int id;

	protected DiscreteFeature(int id, String name) {
		this.id = id; this.name = name;
	}

	@Override
	public String toString() {
		return name;
	}

	public abstract void learn(X x);

	public abstract void learn(Row r);

	public abstract Stream<Function<Function,Object>> classifiers();

	/** perform any updates before use in learning */
	public void commit() {

	}

}
