package jcog.func;

@FunctionalInterface
public interface ObjectObjectToDoubleFunction<X, Y> {
    double doubleValueOf(X x, Y y);
}
