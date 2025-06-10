//import org.mvel2.MVEL;
//import org.mvel2.integration.VariableResolver;
//import org.mvel2.integration.impl.MapVariableResolverFactory;
//
//import java.io.IOException;
//import java.lang.reflect.Modifier;
//import java.nio.file.Files;
//import java.nio.file.Path;
//import java.util.*;
//import java.util.concurrent.TimeUnit;
//import java.util.logging.Level;
//import java.util.logging.Logger;
//import java.util.regex.Matcher;
//import java.util.regex.Pattern;
//import java.util.stream.Collectors;
//
//public class HyperparameterSearch {
//    private static final Logger LOGGER = Logger.getLogger(HyperparameterSearch.class.getName());
//    static final long DEFAULT_TIMEOUT_MILLIS = TimeUnit.MINUTES.toMillis(10);
//    static final Pattern BETWEEN_PATTERN = Pattern.compile("between\\((.+?)\\)");
//    static final Pattern CHOICE_PATTERN = Pattern.compile("choice\\((.+?)\\)");
//    static final Pattern CLASS_PATTERN = Pattern.compile("class\\(\"(.+?)\"\\)");
//
//    public static void main(String[] args) {
//        try {
//            String script = args.length > 0 ? args[0] : getDefaultScript();
//            new HyperparameterSearch(script).run(100);
//        } catch (Exception e) {
//            LOGGER.log(Level.SEVERE, "Failed to run experiments", e);
//        }
//    }
//
//    private static String getDefaultScript() {
//        return """
//           x.agent=choice(SimpleAgent,AdvancedAgent)
//           x.environment=class("com.example.*Environment")
//           x.learningRate=between(1.0..10.0)
//           x.discountFactor=between(0.1..0.9)
//           """;
//    }
//
//    private final ExperimentRunner runner;
//    private final Experiment experiment;
//    private final VariableResolver variableResolver;
//
//    public HyperparameterSearch(String script) {
//        Map<String, Object> variables = parseScript(script);
//        variableResolver = new MapVariableResolverFactory(variables);
//        experiment = new Experiment(variableResolver);
//        runner = new ExperimentRunner();
//    }
//
//    private Map<String, Object> parseScript(String script) {
//        Map<String, Object> variables = new HashMap<>();
//        String[] lines = script.split("\\r?\\n");
//        for (String line : lines) {
//            if (!line.trim().isEmpty()) {
//                String[] parts = line.split("=", 2);
//                String variable = parts[0].trim();
//                String expression = parts[1].trim();
//                try {
//                    Object value = evaluateExpression(expression, variables);
//                    variables.put(variable, value);
//                } catch (Exception e) {
//                    LOGGER.log(Level.SEVERE, "Failed to evaluate expression: " + expression, e);
//                }
//            }
//        }
//        return variables;
//    }
//
//    private Object evaluateExpression(String expression, Map<String, Object> variables) {
//        Matcher betweenMatcher = BETWEEN_PATTERN.matcher(expression);
//        if (betweenMatcher.find()) {
//            String[] range = betweenMatcher.group(1).split("\\.\\.");
//            double min = Double.parseDouble(range[0]);
//            double max = Double.parseDouble(range[1]);
//            return getRandomValue(min, max);
//        }
//
//        Matcher choiceMatcher = CHOICE_PATTERN.matcher(expression);
//        if (choiceMatcher.find()) {
//            String[] choices = choiceMatcher.group(1).split(",");
//            return getRandomChoice(choices);
//        }
//
//        Matcher classMatcher = CLASS_PATTERN.matcher(expression);
//        if (classMatcher.find()) {
//            String classPattern = classMatcher.group(1);
//            return getRandomClass(classPattern);
//        }
//
//        try {
//            return MVEL.eval(expression, new MapVariableResolverFactory(variables));
//        } catch (Exception e) {
//            throw new RuntimeException("Failed to evaluate expression: " + expression, e);
//        }
//    }
//
//    private double getRandomValue(double min, double max) {
//        Random random = new Random();
//        return min + random.nextDouble() * (max - min);
//    }
//
//    private Object getRandomChoice(String[] choices) {
//        Random random = new Random();
//        return choices[random.nextInt(choices.size())];
//    }
//
//    private Object getRandomClass(String classPattern) {
//        try {
//            List<Class<?>> classes = ClassFinder.findClasses(classPattern);
//            Random random = new Random();
//            Class<?> randomClass = classes.get(random.nextInt(classes.size()));
//            return randomClass.getConstructor().newInstance();
//        } catch (Exception e) {
//            throw new RuntimeException("Failed to load class: " + classPattern, e);
//        }
//    }
//
//    public void run(int numExperiments) {
//        runner.runExperiments(experiment, numExperiments);
//    }
//}
//
//class ExperimentRunner {
//    private final Path outputDir = Path.of(System.getProperty("user.home"), "experiment_results");
//
//    void runExperiments(Experiment experiment, int numExperiments) {
//        try {
//            Files.createDirectories(outputDir);
//        } catch (IOException e) {
//            System.err.println("Failed to create output directory: " + e.getMessage());
//            return;
//        }
//
//        for (int i = 0; i < numExperiments; i++) {
//            double meanReward = experiment.run();
//            long wallTime = experiment.getWallTimeMillis();
//            saveExperimentResult(i, meanReward, wallTime);
//        }
//    }
//
//    private void saveExperimentResult(int index, double meanReward, long wallTimeMillis) {
//        String fileName = "experiment_%05d.txt".formatted(index);
//        Path filePath = outputDir.resolve(fileName);
//
//        try {
//            Files.write(filePath, "Mean Reward: %.4f\nWall Time (ms): %d".formatted(meanReward, wallTimeMillis).getBytes());
//        } catch (IOException e) {
//            System.err.println("Failed to save experiment result: " + e.getMessage());
//        }
//    }
//}
//
//class Experiment {
//    private final List<Object> implementationValues;
//    private final List<Double> parameterValues;
//    private final long timeoutMillis;
//    private final long startTimeMillis = System.currentTimeMillis();
//    private final VariableResolver resolver;
//
//    Experiment(VariableResolver resolver) {
//        this.resolver = resolver;
//        implementationValues = parseImplementationChoices();
//        parameterValues = parseParameterChoices();
//        timeoutMillis = HyperparameterSearch.DEFAULT_TIMEOUT_MILLIS;
//    }
//
//    private List<Object> parseImplementationChoices() {
//        return resolver.getKeys().stream()
//                .filter(key -> isImplementationChoice(resolver.getValue(key)))
//                .map(key -> getRandomImplementationChoice(key))
//                .toList();
//    }
//
//    private List<Double> parseParameterChoices() {
//        return resolver.getKeys().stream()
//                .filter(key -> isParameterChoice(resolver.getValue(key)))
//                .map(key -> getRandomParameterChoice(key))
//                .toList();
//    }
//
//    private boolean isImplementationChoice(Object value) {
//        return value instanceof String str && CHOICE_PATTERN.matcher(str).find() || CLASS_PATTERN.matcher(str).find();
//    }
//
//    private boolean isParameterChoice(Object value) {
//        return value instanceof String str && HyperparameterSearch.BETWEEN_PATTERN.matcher(str).find();
//    }
//
//    private Object getRandomImplementationChoice(String key) {
//        String expression = (String) resolver.getValue(key);
//        Matcher choiceMatcher = HyperparameterSearch.CHOICE_PATTERN.matcher(expression);
//        if (choiceMatcher.find()) {
//            String[] choices = choiceMatcher.group(1).split(",");
//            return getRandomChoice(choices);
//        }
//
//        Matcher classMatcher = HyperparameterSearch.CLASS_PATTERN.matcher(expression);
//        if (classMatcher.find()) {
//            String classPattern = classMatcher.group(1);
//            return getRandomClass(classPattern);
//        }
//        throw new IllegalArgumentException("Invalid implementation choice expression: " + expression);
//    }
//
//    private double getRandomParameterChoice(String key) {
//        String expression = (String) resolver.getValue(key);
//        Matcher betweenMatcher = HyperparameterSearch.BETWEEN_PATTERN.matcher(expression);
//        if (betweenMatcher.find()) {
//            String[] range = betweenMatcher.group(1).split("\\.\\.");
//            double min = Double.parseDouble(range[0]);
//            double max = Double.parseDouble(range[1]);
//            return getRandomValue(min, max);
//        }
//
//        throw new IllegalArgumentException("Invalid parameter choice expression: " + expression);
//    }
//
//    private double getRandomValue(double min, double max) {
//        return min + new Random().nextDouble() * (max - min);
//    }
//
//    private Object getRandomChoice(String[] choices) {
//        return choices[new Random().nextInt(choices.length)];
//    }
//
//    private Object getRandomClass(String classPattern) throws IOException, ClassNotFoundException {
//        List<Class<?>> classes = ClassFinder.findClasses(classPattern);
//        Class<?> randomClass = classes.get(new Random().nextInt(classes.size()));
//        try {
//            return randomClass.getConstructor().newInstance();
//        } catch (Exception e) {
//            throw new RuntimeException("Failed to instantiate class: " + randomClass.getName(), e);
//        }
//    }
//
//    double run() {
//        configureExperiment();
//        return runExperimentWithTimeLimit();
//    }
//
//    private void configureExperiment() {
//        for (Object value : implementationValues) {
//            if (value instanceof ImplementationChoice.Value implValue)
//                resolver.setValue(implValue.variable, implValue.choice.getTypeName());
//        }
//
//        for (Double value : parameterValues)
//            resolver.setValue("x.parameter", value);
//    }
//
//    private double runExperimentWithTimeLimit() {
//        double totalReward = 0;
//        int numEpisodes = 0;
//        long elapsedMillis = 0;
//
//        while (elapsedMillis < timeoutMillis) {
//            totalReward += runEpisode();
//            numEpisodes++;
//            elapsedMillis = System.currentTimeMillis() - startTimeMillis;
//        }
//
//        return totalReward / numEpisodes;
//    }
//
//    private double runEpisode() {
//        return 0.0; // Replace with actual episode implementation
//    }
//
//    long getWallTimeMillis() {
//        return System.currentTimeMillis() - startTimeMillis;
//    }
//}
//class ImplementationChoice {
//    private final List<Class<?>> choices;
//    private final Random random = new Random();
//    private final String variable;
//    ImplementationChoice(String expression) {
//        variable = parseVariable(expression);
//        choices = parseChoices(expression);
//    }
//
//    private String parseVariable(String expression) {
//        return expression.split("=")[0].trim();
//    }
//
//    private List<Class<?>> parseChoices(String expression) {
//        String choicesString = expression.split("=")[1].trim();
//        String[] classNames = choicesString.split(",");
//        return Arrays.stream(classNames)
//                .map(ImplementationChoice::loadClass)
//                .filter(clazz -> !Modifier.isAbstract(clazz.getModifiers()))
//                .toList();
//    }
//
//    private static Class<?> loadClass(String className) {
//        try {
//            return Class.forName(className.trim());
//        } catch (ClassNotFoundException e) {
//            throw new RuntimeException("Failed to load class: " + className, e);
//        }
//    }
//
//    Value getRandomValue() {
//        return new Value(variable, choices.get(random.nextInt(choices.size())));
//    }
//
//    static class Value {
//        final String variable;
//        final Class<?> choice;
//
//        Value(String variable, Class<?> choice) {
//            this.variable = variable;
//            this.choice = choice;
//        }
//    }
//}
//class ClassFinder {
//    public static List<Class<?>> findClasses(String classPattern) throws IOException, ClassNotFoundException {
//        String packageName = classPattern.replaceAll("\.\*", "");
//        String folderPath = packageName.replace(".", "/");
//        List<String> classNames = new ArrayList<>();
//        try (var walker = Files.walk(Path.of(System.getProperty("java.class.path")))) {
//            walker.filter(path -> path.toString().endsWith(".class"))
//                    .filter(path -> path.toString().contains(folderPath))
//                    .forEach(path -> {
//                        String className = path.toString()
//                                .replace(System.getProperty("java.class.path") + "/", "")
//                                .replace("/", ".")
//                                .replace(".class", "");
//                        if (className.matches(classPattern.replace("*", ".+"))) {
//                            classNames.add(className);
//                        }
//                    });
//        }
//
//        return classNames.stream()
//                .map(Class::forName)
//                .toList();
//    }
//}