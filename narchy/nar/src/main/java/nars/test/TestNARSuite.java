package nars.test;

import jcog.data.list.Lst;
import nars.NAR;
import org.eclipse.collections.api.block.function.primitive.DoubleFunction;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

@Deprecated public class TestNARSuite extends Lst<TestNARSuite.MyTestNAR> {

    private final Supplier<NAR> narBuilder;
    private final Stream<Method> testMethods;

    @SafeVarargs
    public TestNARSuite(Supplier<NAR> narBuilder, Class<? extends NALTest>... testClasses) {
        this(narBuilder, tests(testClasses));
    }

    public TestNARSuite(Supplier<NAR> narBuilder, Stream<Method> testMethods) {
        this.narBuilder = narBuilder;
        this.testMethods = testMethods;
    }

    @SafeVarargs
    public static Stream<Method> tests(Class<? extends NALTest>... c) {
        List<Method> list = new ArrayList<>();
        for (Class<? extends NALTest> cc : c) {
            for (Method x : cc.getDeclaredMethods()) {
                if (x.getAnnotation(Test.class) != null) {
                    x.trySetAccessible();
                    list.add(x);
                }
            }
        }
        return list.stream();
    }

    public void run(boolean parallel) {
        run(parallel, 1);
    }

    public TestNARSuite run(boolean parallel, int iterations) {



        List<Method> mm = testMethods.toList();

        for (int i = 0; i < iterations; i++) {


            (parallel ? mm.stream().parallel() : mm.stream()).forEach(m -> {
                String testName = m.getDeclaringClass().getName() + ' ' + m.getName();
                MyTestNAR t = new MyTestNAR(narBuilder.get(), testName);
                synchronized (TestNARSuite.this) {
                    add(t); //allow repeats
                }

                try {
                    NALTest.test(t, m);
                } catch (RuntimeException e) {
                    e.printStackTrace();
                }
            });
        }
        return this;
    }


    public double sum(Function<NAR,Number> n) {
        return doubleStream(n).sum();
    }
    public double sum(DoubleFunction<NAR> n) {
        return doubleStream(n).sum();
    }

    public DoubleStream doubleStream(DoubleFunction<NAR> n) {
        return narStream().mapToDouble(n);
    }

    public Stream<NAR> narStream() {
        return stream().map(x -> x.nar);
    }

    public DoubleStream doubleStream(Function<NAR,Number> n) {
        return narStream().map(n).mapToDouble(Number::doubleValue);
    }

    static class MyTestNAR extends TestNAR {
        public final String name;

        MyTestNAR(NAR nar, String testName) {
            super(nar);
            this.name = testName;
        }
    }

    /** summary */
    public void print() {
        forEach(x -> System.out.println(x.name + ' ' + x.score));
    }

    public double score(/* scoring mode */) {
        return this.stream().mapToDouble(x -> x.score).sum();
    }
}