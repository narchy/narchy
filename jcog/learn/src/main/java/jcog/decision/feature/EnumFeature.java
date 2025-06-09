package jcog.decision.feature;

import jcog.TODO;
import jcog.data.map.ObjIntHashMap;
import tech.tablesaw.api.Row;

import java.util.function.Function;
import java.util.stream.Stream;

public class EnumFeature extends DiscreteFeature<String> implements Function<Function,Object> {

	final ObjIntHashMap<String> values = new ObjIntHashMap<>();

	public EnumFeature(int c, String name) {
		super(c, name);
	}

	@Override
	public void learn(String s) {
		values.getIfAbsentPut(s, values::size);
	}

	@Override
	public void learn(Row r) {
        throw new TODO();
	}

	@Override
	public Stream<Function<Function, Object>> classifiers() {
		return Stream.of(this);
	}

	@Override
	public Object apply(Function function) {
		return function.apply(id); //assert: in values
	}

}
