# Gen Framework Cheatsheet 🧬
*Evolution-powered Dependency Injection that writes code for you*

## Quick Start (2 minutes)
```java
// 1. Create container
var o = new Gen();

// 2. Let it evolve optimal implementations & parameters
MyService optimal = o.evolve(MyService.class);
```

## Why Gen?
- Automatically finds optimal implementations & parameters
- Zero configuration needed for simple cases
- Evolves both architecture and parameters
- Works with existing code - just add annotations
- Makes hard optimization problems easy

## Core Features

### 1. Implementation Evolution
```java
Algorithm best = o.evolve(Algorithm.class);
```

### 2. Parameter Evolution
 * @Range supports: double, int enum

```java
class Optimizer {
    public Optimizer(
        @Range(min=0, max=100) double learningRate,
        @Range(min=1, max=10) int batchSize,
        @Range(min=-1, max=1, step=0.1) double momentum
    ) {
        // Gen will find optimal values
    }
}
```

### 3. Scoring (fitness) Functions
```java
Algorithm best = o.evolve(Algorithm.class, 
    algo -> /* scoring calculation */);
```

### 4. Advanced Configuration
```java
var cfg = new Gen.Config(
    pop = 200,     // Population size
    gens = 100,    // Generations
    rate = 0.15,   // Mutation rate
    tourney = 4,   // Tournament size
    elite = 3,     // Elite count
    parallel = true // Parallel evolution
);

var optimal = o.evolve(System.class, System::score, cfg);
```

## Real-World Examples

### Machine Learning Hyperparameter Tuning
```java
@Impl({SGD.class, Adam.class, RMSProp.class})
interface Optimizer {
    void optimize(Model model);
}

class Adam implements Optimizer {
    public Adam(
        @Range(min=1e-5, max=1e-2) double learningRate,
        @Range(min=0.9, max=0.999) double beta1,
        @Range(min=0.9, max=0.999) double beta2
    ) {}
    
    @Evolve
    public double fitness() {
        return model.evaluate();
    }
}
```

### Game AI Evolution
```java
@Impl({MinMax.class, MonteCarlo.class, Neural.class})
interface GameAI {
    Move findBestMove(GameState state);
}

// Evolve tournament
GameAI champion = o.evolve(GameAI.class, 
    ai -> playTournament(ai));
```

### Web Service Optimization
```java
class WebServer {
    public WebServer(
        @Range(min=1, max=32) int threads,
        @Range(min=100, max=10000) int queueSize,
        @Range(min=1000, max=60000) int timeout
    ) {}
    
    @Evolve
    double measurePerformance() {
        return throughput / latency;
    }
}
```

## Pro Tips 💡

1. **Auto-Discovery**
```java
// Scan package for @Impl & @Evolve annotations
o.scan("com.example");
```

2. **Combine with Regular DI**
```java
// Normal DI
o.bind(Service.class, ServiceImpl.class);

// With evolution
o.any(Service.class, FastImpl.class, ReliableImpl.class);
```

3. **Multi-Objective Optimization**
```java
o.evolve(System.class, sys -> 
    sys.performance() * 0.6 + 
    sys.reliability() * 0.4
);
```

4. **Staged Evolution**
```java
// First evolve architecture
var arch = o.evolve(System.class, System::structuralFitness);

// Then fine-tune parameters
var tuned = o.evolve(arch.getClass(), System::performanceFitness);
```

## Common Use Cases

- 🎯 Algorithm Selection & Tuning
- 🔧 System Configuration Optimization
- 🧮 Machine Learning Architecture Search
- 🎮 Game AI Development
- ⚡ Performance Optimization
- 🔄 Load Balancer Tuning
- 📊 Trading Strategy Evolution

## Quick Reference

### Annotations
- `@Impl({A.class, B.class})` - Possible implementations
- `@Range(min, max, step)` - Parameter bounds
- `@Evolve(fitness="method")` - Fitness method

### Key Methods
- `o.evolve(Class)` - Evolve using @Evolve
- `o.evolve(Class, fitness)` - Custom fitness
- `o.evolve(Class, fitness, cfg)` - Full control
- `o.scan(package)` - Auto-discovery
- `o.bind(iface, impl)` - Regular DI
- `o.any(type, impls...)` - Evolution candidates

Remember: Gen works best when you:
1. Define clear parameter ranges
2. Provide meaningful fitness functions
3. Let it explore multiple implementations
4. Use parallel evolution for large searches

Happy Evolving!🧬

----

# TODO


### 1. Error Handling and Validation
- No validation of configuration parameters (population size, generations, etc.)
- Missing checks for invalid range configurations (min > max, negative step values)
- No custom exceptions for different error cases
- Silent failures possible in parameter conversion

### 2. Thread Safety
- Shared Random instance (`random`) could cause issues in parallel evolution
- `cache` and `stack` fields are not thread-safe
- No synchronization in `evaluate` method for parallel fitness evaluations

### 3. API Design Issues
- `Gen.Range` annotation placement limited to parameters
- No way to specify custom crossover or mutation operators
- Cannot customize selection strategy
- No progress monitoring or early stopping
- Missing builder pattern for configuration

### 4. Implementation Limitations
- Single-objective optimization only (multi-objective requires manual weighting)
- No support for constraints beyond simple ranges
- Fixed tournament selection without alternatives
- No support for adaptive mutation rates
- Missing support for parallel fitness evaluation

### 5. Documentation Gaps
- No JavaDoc on public methods
- Missing documentation about evolution parameters' impact
- No examples of common patterns or best practices
- Unclear failure modes and error handling

## Recommendations

### 1. Enhanced Error Handling
```java
public class GenException extends RuntimeException {
    public GenException(String message) { super(message); }
}

public class ValidationException extends GenException {}
public class EvolutionException extends GenException {}
public class ConfigurationException extends GenException {}
```

### 2. Improved Configuration API
```java
public class GenConfig {
    public static class Builder {
        private int populationSize = 100;
        private int generations = 50;
        
        public Builder populationSize(int size) {
            if (size <= 0) throw new ConfigurationException("Population size must be positive");
            this.populationSize = size;
            return this;
        }
        
        // Additional builder methods...
        
        public GenConfig build() {
            validate();
            return new GenConfig(this);
        }
    }
}
```

### 3. Extended Range Support
```java
@Target({ElementType.PARAMETER, ElementType.FIELD, ElementType.METHOD})
public @interface Range {
    // Existing parameters...
    
    Class<? extends Validator> validator() default NoValidator.class;
    String[] constraints() default {};
}

public interface Validator {
    boolean isValid(double value);
}
```

### 4. Evolution Monitoring
```java
public interface EvolutionListener {
    void onGenerationComplete(EvolutionStats stats);
    void onNewBestSolution(Solution solution);
    boolean shouldTerminate(EvolutionStats stats);
}

public class EvolutionStats {
    private final int generation;
    private final double bestFitness;
    private final double averageFitness;
    private final double diversity;
}
```

### 5. Multi-Objective Support
```java
public interface MultiObjectiveFitness {
    double[] evaluate();
    String[] getObjectiveNames();
}

public class ParetoSolution<T> {
    private final T solution;
    private final double[] objectives;
}
```

### 6. Parallel Evolution Support
```java
public class ParallelGen extends Gen {
    private final ExecutorService executor;
    
    @Override
    protected List<DNA> evaluatePopulation(List<DNA> population) {
        return population.parallelStream()
                .map(this::evaluateIndividual)
                .collect(Collectors.toList());
    }
}
```

## Usage Improvements

### Advanced Parameter Control
```java
@Parameter(name = "learningRate")
@Range(min = 0.0001, max = 0.1)
@LogScale  // New annotation for logarithmic scaling
private double learningRate;

@Parameter(name = "batchSize")
@Range(min = 1, max = 32)
@PowerOfTwo  // New annotation for power-of-two values
private int batchSize;
```

### Constraint Support
```java
@Constraints({
    @Constraint("batchSize <= maxBatchSize"),
    @Constraint("learningRate * momentum < 1.0")
})
public class Optimizer {
    // Parameters...
}

@Target({ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@interface Constraint {
    String expression(); // e.g. "width * height <= 1000"
}
class Rectangle {
    public Rectangle(
            @Range(min=1, max=100) @Constraint("width * height <= 1000") int width,
            @Range(min=1, max=100) @Constraint("width * height <= 1000") int height
    ) {
        // ...
    }
}

```

# Gen Framework Enhancement Proposal 🧬++

## Time-Aware Evolution

Support for evolving systems that need to adapt to changing conditions:

```java
public interface TimeAwareEvolution<T> {
    // Evolve system with awareness of time-varying fitness landscape
    T evolveWithTimeAwareness(
        TimeSeries<Environment> environmentHistory,
        TimeWindow predictionWindow
    );
}

@TimeDependent
class AdaptiveSystem {
    @Range(timeVariant = true)
    double adaptiveParameter;
    
    @EnvironmentDependent
    double calculateFitness(Environment current) {
        return performance * environmentFit;
    }
}
```

## 4. Expression Evolution and Constraints

Enable evolution of mathematical expressions and enforce constraint expressions

```java
@EvolveSymbolic
interface PricingRule {
    double calculatePrice(Product product, Market market);
}

// Gen will evolve expressions like:
// price = basePrice * (1 + marketDemand) * seasonalFactor
//         + max(competitorPrice * 0.9, minimumMargin)

@Constraints({
    @Constraint("result >= minimumPrice"),
    @Constraint("result <= maximumPrice"),
    @Constraint("complexity <= 10")  // Limit expression complexity
})
class EvolvingFormula {
    @SymbolicOperators({"+", "-", "*", "/", "max", "min"})
    @SymbolicVariables({"basePrice", "demand", "competition"})
    String evolvedExpression;
}


```

## 5. Multi-Objective Evolution with Interactive Selection

Support for problems with multiple competing objectives:

```java
public interface MultiObjectiveGen<T> {
    ParetoCurve<T> evolvePareto(
        List<ObjectiveFunction<T>> objectives,
        InteractiveSelector<T> selector
    );
}

// Example usage:
var pareto = o.evolvePareto(
    List.of(
        System::performance,
        System::reliability,
        System::cost
    ),
    // Interactive selection during evolution
    solutions -> SwingUI.showSelectionDialog(solutions)
);
```

## 6. Meta-Evolution

Evolution of evolution parameters:

```java
@MetaEvolution
class EvolutionStrategy {
    @Range(min = 0.01, max = 0.5)
    double mutationRate;
    
    @Range(min = 10, max = 1000)
    int populationSize;
    
    @Range(enumClass = SelectionStrategy.class)
    SelectionStrategy selection;
    
    @MetaFitness
    double evaluateStrategy(Problem problem) {
        return new Gen(this).solve(problem).convergenceSpeed;
    }
}
```

## 7. Quantum-Inspired Evolution

Leverage quantum computing concepts for enhanced exploration:

```java
public interface QuantumGen {
    // Quantum-inspired binary encoding
    interface QuantumChromosome {
        double[] amplitudes();
        void rotate(double theta);
        void interfere(QuantumChromosome other);
    }
    
    @QuantumEncoding
    class QuantumParameter {
        @Superposition
        @Range(min = -1, max = 1)
        double value;
        
        @Entangle(with = "otherParam")
        void maintainCorrelation();
    }
}
```

## 8. Interactive Evolution with Explainability

Provide insights into the evolution process:

```java
public interface ExplainableGen<T> {
    record Explanation(
        DecisionTree<T> evolutionTree,
        FeatureImportance importance,
        SolutionLineage lineage
    ) {}
    
    // Get explanation for evolved solution
    Explanation explain(T solution);
    
    // Visual evolution tracking
    void attachVisualizer(EvolutionVisualizer viz);
}

@Explain
class ExplainableSystem {
    @TrackFeature(name = "Architecture Decisions")
    @Range(enumClass = ArchType.class)
    ArchType architecture;
    
    @TrackFeature(name = "Performance Impact")
    @Range(min = 0, max = 100)
    double performanceParam;
}
```

## 9. Constraint Evolution

Evolve not just solutions but constraints themselves:

```java
@ConstraintEvolution
interface SystemConstraints {
    @EvolvingConstraint
    boolean isValid(SystemState state);
    
    @ConstraintFitness
    double evaluateConstraint(
        List<SystemState> valid,
        List<SystemState> invalid
    );
}
```

## 10. Self-Adaptive Operators

Evolution operators that adapt based on their effectiveness:

```java
public interface AdaptiveOperator<T> {
    void updateEffectiveness(double improvement);
    double getSelectionProbability();
}

@AdaptiveEvolution
class AdaptiveSystem {
    @AdaptiveMutation(
        operators = {
            GaussianMutation.class,
            UniformMutation.class,
            PolynomialMutation.class
        }
    )
    double parameter;
    
    @AdaptiveCrossover(
        operators = {
            SinglePoint.class,
            Uniform.class,
            Arithmetic.class
        }
    )
    void crossover(AdaptiveSystem other);
}
```

These enhancements would significantly expand Gen's capabilities while maintaining its core simplicity and ease of use. Each feature can be implemented modularly, allowing users to opt-in to advanced functionality as needed.