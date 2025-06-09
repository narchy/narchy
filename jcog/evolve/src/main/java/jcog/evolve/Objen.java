package jcog.evolve;

import io.github.classgraph.ClassGraph;
import io.github.classgraph.ScanResult;
import jcog.TODO;
import jcog.util.ArrayUtil;
import org.jetbrains.annotations.Nullable;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.lang.reflect.Parameter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Objen {

    private final Map<Class, List<Impl>> impls = new ConcurrentHashMap<>();
    private final Set<Class> stack = new HashSet<>();


    public <T> DNA<T> evolve(Class<T> type, Function<T, Double> fitness) {
        return evolve(type, fitness, new GeneticOptimizer.GeneticConfig());
    }

    public <T> DNA<T> evolve(Class<T> type, Function<T, Double> fitness, GeneticOptimizer.GeneticConfig config) {
        return evolve(type, fitness, new GeneticOptimizer(), config);
    }

    private <T, C extends OptimizationConfig> DNA<T> evolve(Class<T> type, Function<T, Double> fitness, Optimizer<C> o, C oc) {
        if (!impls.containsKey(type)) {
            //any(type);
            all(type);
        }

        return o.optimize(type, fitness, oc);
    }

    private static void bindValidate(Class type, Class impl) {
        Objects.requireNonNull(type, "Type cannot be null");
        Objects.requireNonNull(impl, "Implementation cannot be null");
        if (!type.isAssignableFrom(impl)) {
            throw new IllegalArgumentException(
                    String.format("Implementation %s must be assignable to %s",
                            impl.getName(), type.getName())
            );
        }
    }


    private <T> void implAdd(Class<T> type, Impl<? extends T> impl) {
        impls.computeIfAbsent(type, k-> new ArrayList<>()).add(impl);
    }

    private <T> List<Impl> impls(Class<T> type) {
        return impls.get(type);
    }


    /** bind the implementation type */
    public final <T> Objen any(Class<T> type) {
        return any(type, type);
    }



    /**
     * Binds one or more specific implementation class to an interface or abstract class.
     *
     * @param type the interface or abstract class
     * @param typeImpl the concrete implementation class
     */
    @SafeVarargs public final <T> Objen any(Class<T> type, Class<? extends T>... typeImpls) {
        if (typeImpls.length==0)
            throw new UnsupportedOperationException("no typeImpl provided");

        var existingImpls = this.impls.putIfAbsent(type, Stream.of(typeImpls)
            .peek(t -> bindValidate(type, t))
            .map(CtorImpl::new)
            .collect(Collectors.toList()));

        if (existingImpls!=null)
            throw new UnsupportedOperationException(type + " Implementations already bound: " + existingImpls);

        return this;
    }

    /** Bind the implementation to all known implementing classes that are currently loaded.
     *  This finds all non-abstract classes that implement/extend the given type
     *  across all packages. */
    public final <T> Objen all(Class<T> type) {
        return all(type, ClassLoader.getSystemClassLoader());
    }

    public final <T> Objen all(Class<T> t, ClassLoader cl) {
        var tInstantiable = !t.isInterface() && !Modifier.isAbstract(t.getModifiers());

        var implementations = subclasses(t);

        if (implementations.isEmpty() && !tInstantiable)
            throw new IllegalArgumentException(
                "No implementations found for " + t.getName());

        var ii = implementations.toArray(new Class[0]);

        return any(t, tInstantiable ? ArrayUtil.add(ii, t) : ii);
    }


    /** instantiates a type */
    public <T> T get(Class<T> type) {
        synchronized(stack) {
            if (!stack.add(type))
                throw new RuntimeException("Cyclic dependency detected: " + type);
            try {
                return _get(type);
            } finally {
                stack.remove(type);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private <T> T _get(Class<T> type) {
        var impls = this.impls.computeIfAbsent(type, t -> {
            if (t.isInterface() || Modifier.isAbstract(t.getModifiers()))
                throw new RuntimeException("No implementation bound for: " + t);

            any(type, type);
            return this.impls.get(t);
        });
        return ((Impl<T>) impls.getFirst()).create(this, new HashMap<>());
    }


    private Random rng() {
        return ThreadLocalRandom.current();
    }






    private enum NullEnum {}

    @Retention(RetentionPolicy.RUNTIME)
    @Target({ElementType.PARAMETER, ElementType.METHOD})
    public @interface Range {
        double min() default Double.NEGATIVE_INFINITY;
        double max() default Double.POSITIVE_INFINITY;
        double step() default 0;
        Class<? extends Enum> enumClass() default NullEnum.class;
    }

    interface ParamSpace {
        Value random(RandomGenerator r);
        Value convert(double raw);
    }

    record Value(double raw, Object converted) {
    }

    private static class Bounds {
        final double min, max, step;
        final boolean integer;
        @Nullable final Class<? extends Enum> enumClass;

        Bounds(Parameter p, Range r) {
            this.min = r != null ? r.min() : Double.NEGATIVE_INFINITY;
            this.max = r != null ? r.max() : Double.POSITIVE_INFINITY;
            this.step = r != null ? r.step() : 0;
            this.integer = p.getType()==int.class; //or Integer.class?
            this.enumClass = (r != null && r.enumClass() != NullEnum.class) ? r.enumClass() : null;
        }

        Value generate(RandomGenerator r) {
            double raw;
            if (step > 0) {
                var steps = (int)((max - min) / step) + 1;
                if (steps <= 0)
                    throw new IllegalArgumentException("Step size is too large for the given range.");
                raw = min + (r.nextInt(steps) * step);
            } else {
                raw = min + r.nextDouble() * (max - min);
            }

            if (integer)
                raw = Math.round(raw);

            return convert(raw);
        }

        Value convert(double raw) {
            Object converted;
            if (enumClass != null) {
                Object[] constants = enumClass.getEnumConstants();
                converted = constants[Math.min(Math.max(0, (int)raw), constants.length - 1)];
            } else if (integer) {
                converted = (int)raw;
            } else {
                converted = raw;
            }
            return new Value(raw, converted);
        }
    }

    private static class DefaultParamSpace implements ParamSpace {

        private final Parameter param;
        private final Bounds space;

        private DefaultParamSpace(Parameter p) {
            this.param = p;
            this.space = new Bounds(p, p.getAnnotation(Range.class));
        }

        @Override
        public Value random(RandomGenerator r) {
            return space.generate(r);
        }

        @Override
        public Value convert(double raw) {
            return space.convert(raw);
        }
    }

    interface Impl<T> {
        T create(Objen o, Map<Parameter, Value> vals);
    }

    private static final Map<Parameter, ParamSpace> parameterSpaceCache = new ConcurrentHashMap<>();
    private static ParamSpace paramSpace(Parameter parameter) {
        return parameterSpaceCache.computeIfAbsent(parameter, DefaultParamSpace::new);
    }

    private record CtorImpl<T>(Class<T> type, Constructor<T> ctor,
                               List<? extends ParamSpace> params) implements Impl<T> {

        public CtorImpl(Class<T> type) {
            this(type, ctor(type));
        }

        public CtorImpl(Class<T> type, Constructor<T> ctor) {
            this(type, ctor, params(ctor));
        }

        static <T> List<? extends ParamSpace> params(Constructor<T> ctor) {
            return Stream.of(ctor.getParameters())
                    .filter(p -> p.isAnnotationPresent(Range.class))
                    .map(Objen::paramSpace)
                    .toList();
        }


        @SuppressWarnings("unchecked")
        private static <T> Constructor<T> ctor(Class<T> c) {
            return (Constructor<T>) Stream.of(c.getConstructors())
                    .max(Comparator.comparingInt(Constructor::getParameterCount))
                    .orElseThrow();
        }

        @Override
        public T create(Objen o, Map<Parameter, Value> vals) {
            try {
                return ctor.newInstance(Stream.of(ctor.getParameters())
                    .map(p -> p.isAnnotationPresent(Range.class)
                        ? vals.getOrDefault(p, paramSpace(p).random(o.rng())).converted
                        : o.get(p.getType()))
                    .toArray());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public class DNA<T> implements Supplier<T> {
        private final Class<T> type;
        private final Map<Class, Impl> impls;
        private final Map<Parameter, Value> vals;

        /** NaN if unscored */
        private final double score;

        DNA(Class<T> type) {
            this(type, new HashMap<>(), new HashMap<>(), Double.NaN);
        }

        private DNA(Class<T> type, Map<Class, Impl> impls, Map<Parameter, Value> vals, double score) {
            this.type = type;
            this.impls = impls;
            this.vals = vals;
            this.score = score;
        }

        public DNA(DNA<T> d, Function<T, Double> score) {
            var instance = d.get();
            this(d, instance, score.apply(instance));
        }

        /** TODO decide whether to store instance */
        public DNA(DNA<T> d, T instance, double score) {
            this(d, score);
        }

        private DNA(DNA<T> d, double score) {
            this(d.type, d.impls, d.vals, score);
        }

        private DNA<T> mix(DNA<T> other, double rate) {
            if (type!=other.type)
                throw new UnsupportedOperationException("type mismatch");
            return new DNA<>(
                type,
                newImpls(other, rate),
                mixValues(other, rate),
                Double.NEGATIVE_INFINITY);
        }

        private HashMap<Parameter, Value> mixValues(DNA<T> other, double rate) {
            var y = new HashMap<>(vals);
            vals.forEach((p, value) -> {
                var rng = rng();
                if (rng.nextDouble() < rate) {
                    y.put(p, paramSpace(p).random(rng));
                } else if (rng.nextBoolean()) {
                    y.put(p, other.vals.getOrDefault(p, value));
                }
            });
            return y;
        }

        private HashMap<Class, Impl> newImpls(DNA<T> other, double rate) {
            var newImpls = new HashMap<>(impls);
            impls.forEach((type, impl) -> {
                List<Impl> options = impls(type);
                if (options != null && !options.isEmpty()) {
                    var rng = rng();
                    if (rng.nextDouble() < rate) {
                        // Random implementation when mutating
                        newImpls.put(type, options.get(rng.nextInt(options.size())));
                    } else if (rng.nextBoolean()) {
                        // Take implementation from other parent when crossing over
                        var otherImpl = other.impls.get(type);
                        if (otherImpl != null)
                            newImpls.put(type, otherImpl);
                    }
                }
            });
            return newImpls;
        }

        void setImpl(Class type) {
            var options = Objen.this.impls.get(type);
            if (options != null && !options.isEmpty())
                impls.put(type, options.get(rng().nextInt(options.size()))); // Randomly select initial implementation
        }

        /** create instance */
        @Override public T get() {
            return impl(type).create(Objen.this, vals);
        }

        @SuppressWarnings("unchecked")
        private <T> Impl<T> impl(Class<T> type) {
            return (Impl<T>) impls.computeIfAbsent(type, t -> {
                if (t.isInterface() || Modifier.isAbstract(t.getModifiers()))
                    throw new RuntimeException("No implementation bound for: " + t);

                any(type, type);
                return Objen.this.impls.get(t).getFirst();
            });
        }


        public DNA<T> score(Function<T, Double> score) {
            return new DNA<>(this, score);
        }
    }

    /** Base interface for optimization configurations */
    private interface OptimizationConfig {
    }

    /** Interface for optimization algorithms */
    private interface Optimizer<C extends OptimizationConfig> {
        <T> DNA<T> optimize(Class<T> type, Function<T, Double> score, C config);
    }

    /** Genetic algorithm implementation */
    public class GeneticOptimizer implements Optimizer<GeneticOptimizer.GeneticConfig> {

        @Override
        public <T> DNA<T> optimize(Class<T> type, Function<T, Double> score, GeneticConfig config) {
            // initial population
            var pop = Stream.generate(()->new DNA<>(type)).map(dna -> {
                        dna.setImpl(type);
                        return dna.score(score);
                    })
                .limit(config.populationSize())
                .toList();

            // Evolution loop
            var generations = config.generations();
            for (var g = 0; g < generations; g++) {
                var popPrev = pop;
                pop = Stream.concat(
                    elites(config, popPrev),
                    births(type, score, config, popPrev)
                ).toList();
            }

            return pop.stream()
                    .max(Comparator.comparingDouble(d -> d.score))
                    .orElseThrow();
        }

        /** Create new individuals */
        private <T> Stream<DNA<T>> births(Class<T> type, Function<T, Double> score, GeneticConfig cfg, List<DNA<T>> pop) {
            var tournamentSize = cfg.tournamentSize();
            return Stream.generate(() -> tournament(pop, tournamentSize)
                    .mix(tournament(pop, tournamentSize), cfg.mutationRate()))
                    .limit(cfg.populationSize() - cfg.eliteCount())
                    .map(dna -> dna.score(score));
        }

        private <T> Stream<DNA<T>> elites(GeneticConfig cfg, Collection<DNA<T>> pop) {
            return pop.stream()
                    .sorted(Comparator.comparingDouble(d -> -d.score))
                    .limit(cfg.eliteCount());
        }

        private <T> DNA<T> tournament(List<DNA<T>> pop, int size) {
            return Stream.generate(() ->
                    pop.get(rng().nextInt(pop.size()))
                )
                .limit(size)
                .max(Comparator.comparingDouble(d -> d.score))
                .orElseThrow();
        }

        /** Configuration for genetic algorithm optimization */
        public record GeneticConfig(
                int populationSize,
                int generations,
                double mutationRate,
                int tournamentSize,
                int eliteCount
        ) implements OptimizationConfig {
            public GeneticConfig() {
                this(100, 50, 0.1, 10, 5);
            }
        }
    }

    /** CMA-ES implementation */
    public class CMAESOptimizer implements Optimizer<CMAESOptimizer.CMAESConfig> {

        @Override
        public <T> DNA<T> optimize(Class<T> type, Function<T, Double> fitnessFunction, CMAESConfig config) {
            throw new TODO();
        }

        /** Configuration for CMA-ES optimization */
        public record CMAESConfig(
                int populationSize,
                int maxIterations,
                double initialStepSize,
                double targetFitness
        ) implements OptimizationConfig {
            public CMAESConfig() {
                this(100, 50, 0.5, 1e-10);
            }
        }
    }

    /** shared, thread-safe */
    private static ScanResult classGraph = null;

    private static List<Class<?>> subclasses(Class<?> superClass) {
        synchronized (Objen.class) {
            if (classGraph == null)
                classGraph = new ClassGraph().enableClassInfo().scan();
        }

        var cl =superClass.isInterface() ?
                classGraph.getClassesImplementing(superClass)
                :
                classGraph.getSubclasses(superClass);
        return cl.loadClasses();
    }
}
