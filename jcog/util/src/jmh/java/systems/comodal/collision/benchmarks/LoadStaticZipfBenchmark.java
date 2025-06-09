package systems.comodal.collision.benchmarks;

import com.github.benmanes.caffeine.cache.LoadingCache;
import org.cache2k.Cache;
import org.cache2k.Cache2kBuilder;
import org.cache2k.integration.CacheLoader;
import systems.comodal.collision.cache.CollisionBuilder;
import systems.comodal.collision.cache.CollisionCache;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

@State(Scope.Benchmark)
@Threads(Threads.MAX)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 1, time = 1)
@Measurement(iterations = 1, time = 2)
@Fork(1)
public class LoadStaticZipfBenchmark {

  static final int SIZE = 1 << 12;
  static final int MASK = SIZE - 1;
  static final int ITEMS = SIZE / 3;
  private static final int CAPACITY = 1 << 17;
  // have to sleep for at least 1ms, so amortize 10 microsecond disk calls,
  // by sleeping (10 / 1000.0)% of calls.
  private static final double SLEEP_RAND = 10 / 1000.0;
  private static final UnaryOperator<Long> LOADER = num -> {
    amortizedSleep();
    return punishMiss(num);
  };
  @Param({
      "Cache2k",
      "Caffeine",
      "Collision",
      "Collision_Packed",
      "Collision_Aggressive"
  })
  private BenchmarkFunctionFactory cacheType;
  private UnaryOperator<Long> benchmarkFunction;
  private Long[] keys = new Long[SIZE];

  private static void amortizedSleep() {
    try {
      if (Math.random() < SLEEP_RAND) {
        Thread.sleep(1);
      }
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private static Long punishMiss(final Long num) {
    final double cubed = Math.pow(num, 3);
    return (long) Math.cbrt(cubed);
  }

  private static CollisionBuilder.LoadingCollisionBuilder<Long, Long, Long> startCollision() {
    return CollisionCache
        .withCapacity(CAPACITY, Long.class)
        .setStrictCapacity(true)
        .setLoader(
            key -> {
              amortizedSleep();
              return key;
            }, (key, num) -> punishMiss(num));
  }

  public static void main(final String[] args) {
    if (args.length == 0) {
      runForMemProfiler("Collision");
    }
    for (final String arg : args) {
      runForMemProfiler(arg);
    }
  }

  private static double memEstimate(final Runtime rt) {
    return (rt.totalMemory() - rt.freeMemory()) / 1048576.0;
  }

  private static void runForMemProfiler(final String cacheType) {
    final BenchmarkFunctionFactory cacheFactory = BenchmarkFunctionFactory.valueOf(cacheType);
    final Long[] keys = new Long[SIZE];
    final ScrambledZipfGenerator generator = new ScrambledZipfGenerator(ITEMS);
    for (int i = 0; i < keys.length; i++) {
      keys[i] = generator.nextValue();
    }
    final Runtime rt = Runtime.getRuntime();
    rt.gc();
    Thread.yield();
    System.out.println("Estimating memory usage for " + cacheType);
    final double baseUsage = memEstimate(rt);
    System.out.format("%.2fmB base usage.%n", baseUsage);
    final UnaryOperator<Long> benchmarkFunction = cacheFactory.create();
    for (final Long key : keys) {
      if (!key.equals(benchmarkFunction.apply(key))) {
        throw new IllegalStateException(cacheType + " returned invalid value.");
      }
    }

    for (; ; ) {
      System.out.format("%.2fmB%n", memEstimate(rt) - baseUsage);
      final Long key = keys[(int) (Math.random() * keys.length)];
      System.out.println(key + " -> " + benchmarkFunction.apply(key));
      try {
        rt.gc();
        Thread.sleep(2000);
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
    }
  }

  @Setup
  public void setup() {
    this.benchmarkFunction = cacheType.create();
    final ScrambledZipfGenerator generator = new ScrambledZipfGenerator(ITEMS);
    IntStream.range(0, keys.length).parallel().forEach(i -> {
      final Long key = generator.nextValue();
      keys[i] = key;
      if (!key.equals(benchmarkFunction.apply(key))) {
        throw new IllegalStateException(cacheType + " returned invalid value.");
      }
    });
  }

  @Benchmark
  public Long getSpread(final ThreadState threadState) {
    return benchmarkFunction.apply(keys[threadState.index++ & MASK]);
  }

  public enum BenchmarkFunctionFactory {
    Cache2k {
      @Override
      public UnaryOperator<Long> create() {
        final Cache<Long, Long> cache = Cache2kBuilder
            .of(Long.class, Long.class)
            .disableStatistics(true)
            .entryCapacity(CAPACITY)
            .loader(new CacheLoader<>() {
              public Long load(final Long key) throws Exception {
                amortizedSleep();
                return punishMiss(key);
              }
            }).build();
        System.out.println(cache);
        return cache::get;
      }
    },
    Caffeine {
      @Override
      public UnaryOperator<Long> create() {
        final LoadingCache<Long, Long> cache = com.github.benmanes.caffeine.cache.Caffeine
            .newBuilder()
            .initialCapacity(CAPACITY)
            .maximumSize(CAPACITY)
            .build(LOADER::apply);
        return cache::get;
      }
    },
    Collision {
      @Override
      public UnaryOperator<Long> create() {
        final CollisionCache<Long, Long> cache = startCollision()
            .buildSparse(5.0);
        System.out.println(cache);
        return key -> cache.get(key, LOADER);
      }
    },
    Collision_Packed {
      @Override
      public UnaryOperator<Long> create() {
        final CollisionCache<Long, Long> cache = startCollision()
                .buildPacked();
        System.out.println(cache);
        return key -> cache.get(key, LOADER);
      }
    },
    Collision_Aggressive {
      @Override
      public UnaryOperator<Long> create() {
        final CollisionCache<Long, Long> cache = startCollision()
            .buildSparse(3.0);
        System.out.println(cache);
        return cache::getAggressive;
      }
    };

    public abstract UnaryOperator<Long> create();
  }

  @State(Scope.Thread)
  public static class ThreadState {

    int index = ThreadLocalRandom.current().nextInt();
  }
}
