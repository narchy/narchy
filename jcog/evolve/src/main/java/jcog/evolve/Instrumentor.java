package jcog.evolve;

import org.objectweb.asm.*;
import org.objectweb.asm.commons.AdviceAdapter;

import java.net.URL;
import java.net.URLClassLoader;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;

import static org.objectweb.asm.Opcodes.*;

public final class Instrumentor {

    public record MethodId(String className, String methodName, String descriptor, int access) {
        @Override public String toString() { return String.format("%s#%s%s", className, methodName, descriptor); }
    }

    public record Pointcut(Predicate<MethodId> methodPredicate, Advice advice) {}

    public interface AdviceContext {
        MethodId getMethodId();
        AdviceAdapter getAdapter();
        void loadThis();
        void loadArgument(int index);
        void setAdviceState(Runnable valueLoader);
        void loadAdviceState();
    }

    public interface Advice {
        default void onEnter(AdviceContext context) {}
        default void onExit(AdviceContext context) {}
        default void onThrow(AdviceContext context) {}
        default boolean requiresAdviceState() { return false; }
    }

    public static final class InstrumentationPlan {
        private final byte[] originalClassBytes;
        private final List<Pointcut> pointcuts = new ArrayList<>();

        private InstrumentationPlan(byte[] classBytes) { this.originalClassBytes = classBytes; }
        public static InstrumentationPlan forClass(byte[] classBytes) { return new InstrumentationPlan(classBytes); }
        
        public InstrumentationPlan addAdvice(Predicate<MethodId> predicate, Advice advice) {
            pointcuts.add(new Pointcut(predicate, advice));
            return this;
        }

        public byte[] apply() {
            var reader = new ClassReader(originalClassBytes);
            var writer = new ClassWriter(reader, ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
            var visitor = new AdvisingClassVisitor(writer, pointcuts);
            reader.accept(visitor, ClassReader.EXPAND_FRAMES);
            return writer.toByteArray();
        }
    }

    private static class AdvisingClassVisitor extends ClassVisitor {
        private final List<Pointcut> pointcuts;
        private String internalClassName;

        AdvisingClassVisitor(ClassVisitor cv, List<Pointcut> pointcuts) { super(ASM9, cv); this.pointcuts = pointcuts; }

        @Override
        public void visit(int v, int a, String name, String s, String sn, String[] i) {
            this.internalClassName = name;
            super.visit(v, a, name, s, sn, i);
        }

        @Override
        public MethodVisitor visitMethod(int access, String name, String desc, String sig, String[] ex) {
            var mv = super.visitMethod(access, name, desc, sig, ex);
            if ((access & (Opcodes.ACC_ABSTRACT | Opcodes.ACC_NATIVE | Opcodes.ACC_BRIDGE)) != 0) return mv;

            var methodId = new MethodId(internalClassName.replace('/', '.'), name, desc, access);
            var applicableAdvice = pointcuts.stream().filter(p -> p.methodPredicate().test(methodId)).map(Pointcut::advice).toList();

            return applicableAdvice.isEmpty() ? mv : new AdviceApplicator(api, mv, access, name, desc, methodId, applicableAdvice);
        }
    }

    private static class AdviceApplicator extends AdviceAdapter implements AdviceContext {
        private final MethodId methodId;
        private final List<Advice> advices;
        private final Label tryBlockStart = new Label();
        private final Label catchBlockStart = new Label();
        private Integer stateVar;

        AdviceApplicator(int api, MethodVisitor mv, int acc, String name, String desc, MethodId id, List<Advice> advices) {
            super(api, mv, acc, name, desc);
            this.methodId = id;
            this.advices = advices;
        }

        public MethodId getMethodId() { return methodId; }
        public AdviceAdapter getAdapter() { return this; }

        @Override
        public void loadArgument(int index) {

        }

        public void setAdviceState(Runnable valueLoader) {
            if (stateVar == null) stateVar = newLocal(Type.getType(Object.class));
            valueLoader.run();
            box(Type.getArgumentTypes(methodDesc)[0]);
            storeLocal(stateVar);
        }
        
        public void loadAdviceState() { loadLocal(stateVar); }

        @Override
        protected void onMethodEnter() {
            visitLabel(tryBlockStart);
            if (advices.stream().anyMatch(Advice::requiresAdviceState)) stateVar = newLocal(Type.getType(Object.class));
            advices.forEach(advice -> advice.onEnter(this));
        }

        @Override
        protected void onMethodExit(int opcode) {
            if (opcode != ATHROW) advices.forEach(advice -> advice.onExit(this));
        }

        @Override
        public void visitMaxs(int maxStack, int maxLocals) {
            visitTryCatchBlock(tryBlockStart, catchBlockStart, catchBlockStart, "java/lang/Throwable");
            visitLabel(catchBlockStart);
            final var exceptionVar = newLocal(Type.getType(Throwable.class));
            storeLocal(exceptionVar);
            advices.forEach(advice -> advice.onThrow(this));
            loadLocal(exceptionVar);
            visitInsn(ATHROW);
            super.visitMaxs(maxStack, maxLocals);
        }
    }

    public static class SystemOutLogger {
        public static void log(String context, String message) { System.out.printf("  [LOG::%s] %s%n", context, message); }
        public static void log(String context, String message, Throwable t) { System.out.printf("  [LOG::%s] %s - Exception: %s%n", context, message, t); }
    }
    
    public static final class PerformanceTelemetry {
        public record Metrics(LongSummaryStatistics stats) {
            @Override public String toString() { return String.format("invokes: %,d, total: %,.3f ms, avg: %,.3f µs", stats.getCount(), stats.getSum() / 1e6, stats.getAverage() / 1e3); }
        }
        private static final ConcurrentHashMap<String, Metrics> registry = new ConcurrentHashMap<>();
        
        public static void record(String methodId, long startTimeNanos) {
            var duration = System.nanoTime() - startTimeNanos;
            registry.computeIfAbsent(methodId, k -> new Metrics(new LongSummaryStatistics())).stats.accept(duration);
        }
        
        public static Map<String, Metrics> getReport() { return Map.copyOf(registry); }
        public static void reset() { registry.clear(); }

        public static final Advice ADVICE = new Advice() {
            @Override public boolean requiresAdviceState() { return true; }
            @Override public void onEnter(AdviceContext ctx) { ctx.setAdviceState(() -> ctx.getAdapter().invokeStatic(Type.getType(System.class), org.objectweb.asm.commons.Method.getMethod("long nanoTime()"))); }
            @Override public void onExit(AdviceContext ctx) {
                var aa = ctx.getAdapter();
                aa.visitLdcInsn(ctx.getMethodId().toString());
                ctx.loadAdviceState();
                aa.unbox(Type.LONG_TYPE);
                aa.invokeStatic(Type.getType(PerformanceTelemetry.class), org.objectweb.asm.commons.Method.getMethod("void record(java.lang.String, long)"));
            }
        };
    }

    // --- Demo Application ---
    public static class Workload {
        public void entrypoint() {
            hotMethod(10);
            coldMethod(100);
            try { errorMethod("test-input"); } catch (IllegalArgumentException e) { /* expected */ }
        }
        
        public void hotMethod(int iterations) {
            for (var i = 0; i < iterations; i++) {
                try { Thread.sleep(1); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            }
        }
        
        private void coldMethod(int multiplier) {
            var result = new Random().nextInt() * (long)multiplier;
        }

        public void errorMethod(String input) {
            if (input.equals("test-input")) throw new IllegalArgumentException("Invalid input received");
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("--- AOP Instrumentation Framework Demo ---\n");
        var targetClassName = Workload.class.getName();
        var resourceName = targetClassName.replace('.', '/') + ".class";
        byte[] originalClassBytes;
        try (var is = Instrumentor.class.getClassLoader().getResourceAsStream(resourceName)) {
            originalClassBytes = Objects.requireNonNull(is).readAllBytes();
        }

        System.out.println("--- Round 1: Instrument with performance telemetry on all public methods. ---");
        Predicate<MethodId> allPublicMethods = id -> (id.access() & Opcodes.ACC_PUBLIC) != 0;
        var plan1 = InstrumentationPlan.forClass(originalClassBytes)
            .addAdvice(allPublicMethods, PerformanceTelemetry.ADVICE);
        
        runWorkload(targetClassName, plan1.apply());

        var telemetry = PerformanceTelemetry.getReport();
        telemetry.entrySet().stream()
            .sorted(Map.Entry.<String, PerformanceTelemetry.Metrics>comparingByValue(Comparator.comparingLong(m -> m.stats().getSum())).reversed())
            .forEach(e -> System.out.printf("  - %-70s %s\n", e.getKey(), e.getValue()));

        var hotMethodId = telemetry.entrySet().stream().max(Map.Entry.comparingByValue(Comparator.comparingLong(m -> m.stats().getSum()))).map(Map.Entry::getKey).orElseThrow();
        System.out.printf("\nAnalysis: Hottest public method is '%s'.\n", hotMethodId);

        System.out.println("\n--- Round 2: Re-instrument. Keep telemetry, add arg logging to hot method, and exception auditing to error method. ---");
        Predicate<MethodId> isHotMethod = id -> id.toString().equals(hotMethodId);
        Predicate<MethodId> isErrorMethod = id -> id.methodName().equals("errorMethod");

        var plan2 = InstrumentationPlan.forClass(originalClassBytes)
            .addAdvice(allPublicMethods, PerformanceTelemetry.ADVICE)
            .addAdvice(isHotMethod, new ArgumentLoggerAdvice())
            .addAdvice(isErrorMethod, new ExceptionAuditorAdvice());
        
        runWorkload(targetClassName, plan2.apply());
        System.out.println("\n--- Demo Complete ---");
    }

    private static void runWorkload(String className, byte[] classBytes) throws Exception {
        var loader = new URLClassLoader(new URL[0], Instrumentor.class.getClassLoader());
        var defineClass = URLClassLoader.class.getDeclaredMethod("defineClass", String.class, byte[].class, int.class, int.class);
        defineClass.setAccessible(true);
        var loadedClass = (Class<?>) defineClass.invoke(loader, className, classBytes, 0, classBytes.length);
        var instance = loadedClass.getDeclaredConstructor().newInstance();
        loadedClass.getMethod("entrypoint").invoke(instance);
    }
    
    private static class ArgumentLoggerAdvice implements Advice {
        @Override public void onEnter(AdviceContext ctx) {
            var aa = ctx.getAdapter();
            aa.visitLdcInsn("ARGS");
            aa.visitTypeInsn(NEW, "java/lang/StringBuilder"); aa.visitInsn(DUP); aa.visitLdcInsn(ctx.getMethodId().methodName() + " args: "); aa.visitMethodInsn(INVOKESPECIAL, "java/lang/StringBuilder", "<init>", "(Ljava/lang/String;)V", false);
            var argTypes = Type.getArgumentTypes(ctx.getMethodId().descriptor());
            for (var i = 0; i < argTypes.length; i++) {
                ctx.loadArgument(i);
                aa.box(argTypes[i]);
                aa.visitMethodInsn(INVOKEVIRTUAL, "java/lang/StringBuilder", "append", "(Ljava/lang/Object;)Ljava/lang/StringBuilder;", false);
                if (i < argTypes.length - 1) aa.visitLdcInsn(", ");
            }
            aa.visitMethodInsn(INVOKEVIRTUAL, "java/lang/StringBuilder", "toString", "()Ljava/lang/String;", false);
            aa.invokeStatic(Type.getType(SystemOutLogger.class), org.objectweb.asm.commons.Method.getMethod("void log(java.lang.String, java.lang.String)"));
        }
    }
    
    private static class ExceptionAuditorAdvice implements Advice {
        @Override public void onThrow(AdviceContext ctx) {
            var aa = ctx.getAdapter();
            var exceptionVar = aa.newLocal(Type.getType(Throwable.class));
            aa.storeLocal(exceptionVar);
            aa.visitLdcInsn("AUDIT");
            aa.visitLdcInsn("Exception caught in " + ctx.getMethodId().methodName());
            aa.loadLocal(exceptionVar);
            aa.invokeStatic(Type.getType(SystemOutLogger.class), org.objectweb.asm.commons.Method.getMethod("void log(java.lang.String, java.lang.String, java.lang.Throwable)"));
            aa.loadLocal(exceptionVar);
        }
    }
}