# Gen Framework 🧬
*A framework for evolutionary optimization of software components and parameters.*

The Gen Framework offers a novel approach to software development by leveraging evolutionary algorithms to automatically discover optimal implementations and parameter configurations for your Java classes and interfaces. By defining a search space of possibilities (different algorithms, settings, etc.) and providing a fitness function to measure success, Gen can explore this space to find solutions that best meet your criteria. This is particularly useful for complex problems where manual optimization is difficult or time-consuming.

## Quick Start (2 minutes)
```java
// 1. Create a Gen container instance
var o = new Gen();

// 2. Define an interface or class for Gen to evolve
//    Use annotations like @Impl to specify possible implementations
//    and @Range for parameter tuning.
//    (See examples below or in the jcog documentation)

// 3. Let Gen evolve the optimal implementation and its parameters
MyService optimalService = o.evolve(MyService.class);

// optimalService is now an instance of a class implementing MyService,
// with its parameters tuned by the evolutionary process.
```

## Why Gen?
- **Automated Optimization**: Automatically discovers effective implementations and fine-tunes their parameters.
- **Simplified Configuration**: Minimal setup needed for basic use cases; sensible defaults are provided.
- **Flexible Evolution**: Can evolve both the structural aspects (which implementation to use) and numerical/categorical parameters.
- **Integrates with Existing Code**: Annotations allow Gen to work with your current codebase with minor additions.
- **Tackles Complexity**: Makes challenging optimization tasks more approachable by automating the search for solutions.

## Core Features

### 1. Implementation Evolution
Gen can select the best class from a set of candidates that implement a given interface.
```java
// Assuming Algorithm is an interface and FastAlgo, PreciseAlgo are implementations
// @Impl({FastAlgo.class, PreciseAlgo.class}) on Algorithm interface or supplied elsewhere
Algorithm bestAlgorithm = o.evolve(Algorithm.class);
```

### 2. Parameter Evolution
Gen tunes parameters of constructors or methods annotated with `@Range`. This supports `double`, `int`, and `enum` types.
```java
class MyOptimizer {
    public MyOptimizer(
        @Range(min=0.01, max=0.5) double learningRate,
        @Range(min=16, max=128) int batchSize,
        @Range(min=-0.9, max=0.9, step=0.1) double momentum // Step for discrete numeric ranges
    ) {
        // Gen will inject optimized values for learningRate, batchSize, and momentum
    }
}
```

### 3. Custom Scoring (Fitness) Functions
You provide a fitness function that Gen uses to evaluate how "good" a particular evolved instance is. Higher scores are better.
```java
// Evolve the Algorithm class, scoring each candidate instance
Algorithm bestAlgo = o.evolve(Algorithm.class,
    algorithmCandidate -> {
        // Example: run the algorithm on a benchmark and return its performance score
        double score = benchmark(algorithmCandidate);
        return score;
    }
);
```

### 4. Advanced Configuration
For more control over the evolutionary process, you can provide a `Gen.Config` object.
```java
var config = new Gen.Config(
    /*popSize =*/ 200,     // Population size for each generation
    /*gens =*/ 100,        // Number of generations to run
    /*mutRate =*/ 0.15,   // Mutation rate (probability of change)
    /*tournamentSize =*/ 4,   // Size of selection tournament
    /*eliteCount =*/ 3,     // Number of best individuals to carry to next gen
    /*parallel =*/ true    // Enable parallel evolution if possible
);

// Evolve an instance of MySystem, using a method System::score for fitness, with custom config
MySystem optimalSystem = o.evolve(MySystem.class, MySystem::score, config);
```

## Real-World Examples

### Machine Learning Hyperparameter Tuning
Automatically find the best optimizer and its hyperparameters for a machine learning model.
```java
// Define an interface for optimizers
@Impl({SGD.class, Adam.class, RMSProp.class}) // SGD, Adam, RMSProp are example implementations
interface Optimizer {
    void optimize(Model model); // Model is your ML model class
}

// Example Adam optimizer implementation with tunable parameters
class Adam implements Optimizer {
    final double learningRate, beta1, beta2;
    public Adam(
        @Range(min=1e-5, max=1e-2, scale="log") double learningRate, // Log scale for learning rate
        @Range(min=0.8, max=0.999) double beta1,
        @Range(min=0.8, max=0.999) double beta2
    ) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }
    
    @Evolve // Annotation to mark the fitness method
    public double fitness(Model model) { // Assumes Model class is injectable or passed
        // Train or evaluate the model using these Adam parameters
        // Return a score, e.g., accuracy or negated loss
        return model.evaluate(this);
    }
}
```

### Game AI Evolution
Evolve a competitive AI for a game by playing different AI implementations or parameter sets against each other.
```java
@Impl({MinMaxAI.class, MonteCarloAI.class, NeuralNetAI.class}) // Example AI strategies
interface GameAI {
    Move findBestMove(GameState currentState);
}

// Evolve by running a tournament
GameAI championAI = o.evolve(GameAI.class,
    aiPlayer -> playTournament(aiPlayer) // playTournament returns a score for the AI
);
```

### Web Service Optimization
Tune web server parameters like thread count, queue size, and timeouts for optimal performance.
```java
class WebServer {
    final int threadCount;
    final int queueCapacity;
    final int requestTimeoutMillis;

    public WebServer(
        @Range(min=4, max=64) int threads,
        @Range(min=100, max=20000) int queueSize,
        @Range(min=500, max=30000) int timeout
    ) {
        this.threadCount = threads;
        this.queueCapacity = queueSize;
        this.requestTimeoutMillis = timeout;
    }
    
    @Evolve
    double measurePerformance() {
        // Simulate load or use live metrics
        // Return a score, e.g., (requestsPerSecond / averageLatency)
        double throughput = getSimulatedThroughput(this);
        double latency = getSimulatedLatency(this);
        return (latency == 0) ? 0 : throughput / latency;
    }
}
```

## Pro Tips 💡

1.  **Auto-Discovery of Implementations and Fitness Methods**:
    Gen can scan a specified package for classes annotated with `@Impl` (for implementations) and methods annotated with `@Evolve` (for fitness evaluation), simplifying setup.
    ```java
    // Scan "com.example.myproject" for relevant Gen annotations
    o.scan("com.example.myproject");
    ```

2.  **Combine with Standard Dependency Injection (DI)**:
    Use Gen for parts of your system that benefit from evolutionary optimization, and standard DI for other parts. Gen can act as a DI container itself.
    ```java
    // Standard DI binding
    o.bind(DatabaseService.class, ProductionDBServiceImpl.class);

    // Specify candidates for evolutionary selection for another service
    o.any(CacheService.class, FastMemoryCacheImpl.class, DistributedCacheImpl.class);
    ```

3.  **Multi-Objective Optimization**:
    If you have multiple criteria for success, combine them into a single fitness score.
    ```java
    o.evolve(MyComplexSystem.class, system -> {
        double performanceScore = system.calculatePerformance();
        double stabilityScore = system.assessStability();
        // Weighted sum for multi-objective optimization
        return (performanceScore * 0.7) + (stabilityScore * 0.3);
    });
    ```

4.  **Staged Evolution for Complex Systems**:
    Break down the evolution process. First, evolve broader architectural choices, then fine-tune parameters of the chosen architecture.
    ```java
    // Stage 1: Evolve the best general architecture (e.g., choice of core components)
    var bestArchitecture = o.evolve(SystemArchitecture.class, SystemArchitecture::calculateStructuralFitness);

    // Stage 2: Fine-tune the parameters of the chosen architecture
    // Note: You might need to instantiate and then evolve, or have specific mechanisms for this
    var finelyTunedSystem = o.evolve(bestArchitecture.getClass(), SystemArchitecture::calculatePerformanceFitness);
    ```

## Common Use Cases

- 🎯 **Algorithm Selection & Hyperparameter Tuning**: Finding the best algorithm and its settings for tasks like sorting, searching, or machine learning.
- 🔧 **System Configuration Optimization**: Optimizing settings for databases, web servers, or other complex software systems.
- 🧮 **Machine Learning Architecture Search (NAS)**: Evolving neural network architectures or other ML model structures.
- 🎮 **Game AI Development**: Creating adaptive and challenging game opponents or behaviors.
- ⚡ **Performance Optimization**: Tuning software to run faster or use fewer resources.
- 🔄 **Load Balancer Configuration**: Finding optimal routing rules and settings for load balancers.
- 📊 **Financial Trading Strategy Evolution**: Developing and optimizing automated trading strategies.

## Quick Reference

### Key Annotations:
-   `@Impl({ImplementationA.class, ImplementationB.class, ...})`: Placed on an interface or abstract class to specify potential concrete implementations Gen can choose from.
-   `@Range(min, max, step, scale)`: Used on constructor parameters or fields to define the search space for numeric or enum values.
    -   `min`, `max`: Define the lower and upper bounds.
    -   `step` (optional): Defines the increment for discrete numeric values.
    -   `scale` (optional): Can be "log" for logarithmic scaling of the search space, useful for parameters like learning rates.
-   `@Evolve(fitness="methodName")` or just `@Evolve`: Marks a method within a class as the fitness function. If `fitness="methodName"` is used, it specifies a method in the current class or an injectable component to call for fitness. If just `@Evolve`, the annotated method itself is the fitness function.

### Key Methods in `Gen` Class:
-   `o.evolve(Class<T> type)`: Evolves an instance of `type`. Requires an `@Evolve` annotation in the target class or its implementations to define the fitness function.
-   `o.evolve(Class<T> type, Function<T, Double> fitnessFunction)`: Evolves an instance of `type` using the provided `fitnessFunction` to score candidates.
-   `o.evolve(Class<T> type, Function<T, Double> fitnessFunction, Gen.Config config)`: Evolves with full control over the process via a `Gen.Config` object.
-   `o.scan(String packageName)`: Scans the given package for Gen-related annotations to auto-configure implementations and fitness methods.
-   `o.bind(Class<T> interfaceType, Class<? extends T> implementationType)`: Standard dependency injection binding.
-   `o.any(Class<T> type, Class<? extends T>... implementations)`: Registers multiple implementations for `type` that Gen can choose from during evolution (alternative to `@Impl`).

**Tips for Effective Evolution with Gen:**
1.  **Define Clear Parameter Ranges**: Ensure `@Range` annotations accurately reflect meaningful boundaries for your parameters.
2.  **Provide Meaningful Fitness Functions**: The fitness function is crucial; it must accurately guide the evolution towards desirable solutions.
3.  **Explore Diverse Implementations**: Offer Gen a variety of implementations (via `@Impl` or `o.any`) to allow for architectural evolution.
4.  **Utilize Parallel Evolution**: For computationally intensive fitness evaluations or large search spaces, enable `parallel = true` in `Gen.Config` to speed up the process.

Happy Evolving!🧬

## Future Development

For a detailed list of planned enhancements, ideas under consideration, and known areas for improvement, please see the [FUTURE_FEATURES.md](./FUTURE_FEATURES.md) document.