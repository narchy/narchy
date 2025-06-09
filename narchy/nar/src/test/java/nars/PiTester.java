//package nars;
//
//import jcog.Util;
//import jcog.data.list.FasterList;
//import nars.nal.nal1.NAL1MultistepTest;
//import nars.nal.nal1.NAL1Test;
//import org.pitest.classinfo.ClassInfo;
//import org.pitest.classpath.ClassloaderByteArraySource;
//import org.pitest.classpath.CodeSource;
//import org.pitest.classpath.PathFilter;
//import org.pitest.classpath.ProjectClassPaths;
//import org.pitest.coverage.CoverageDatabase;
//import org.pitest.coverage.CoverageGenerator;
//import org.pitest.coverage.CoverageSummary;
//import org.pitest.coverage.execute.CoverageOptions;
//import org.pitest.coverage.execute.DefaultCoverageGenerator;
//import org.pitest.coverage.export.DefaultCoverageExporter;
//import org.pitest.functional.FCollection;
//import org.pitest.functional.prelude.Prelude;
//import org.pitest.mutationtest.ClassMutationResults;
//import org.pitest.mutationtest.EngineArguments;
//import org.pitest.mutationtest.MutationConfig;
//import org.pitest.mutationtest.MutationResult;
//import org.pitest.mutationtest.build.*;
//import org.pitest.mutationtest.config.*;
//import org.pitest.mutationtest.engine.MutationDetails;
//import org.pitest.mutationtest.engine.MutationEngine;
//import org.pitest.mutationtest.engine.gregor.config.GregorEngineFactory;
//import org.pitest.mutationtest.execute.MutationAnalysisExecutor;
//import org.pitest.mutationtest.incremental.DefaultCodeHistory;
//import org.pitest.mutationtest.incremental.IncrementalAnalyser;
//import org.pitest.mutationtest.incremental.ObjectOutputStreamHistoryStore;
//import org.pitest.mutationtest.statistics.MutationStatisticsListener;
//import org.pitest.mutationtest.tooling.JarCreatingJarFinder;
//import org.pitest.process.DefaultJavaExecutableLocator;
//import org.pitest.process.JavaAgent;
//import org.pitest.process.LaunchOptions;
//import org.pitest.util.IsolationUtils;
//import org.pitest.util.Timings;
//
//import java.util.*;
//import java.util.function.Predicate;
//
///**
// * executes pitest
// * https:
// */
//public class PiTester {
//
//    public static void main(String[] args) {
//
//
//        Predicate<String> testFilter =
//                t ->
//                        t.equals(NAL1Test.class.getName())
//                                || t.equals(NAL1MultistepTest.class.getName())
////                                || t.equals(AnonTest.class.getName())
////                                t.equals(BoolTest.class.getName()) ||
////                                t.equals(ConjTest.class.getName()) ||
////                                t.equals(VariableTest.class.getName()
//                                ;
//
//        run(
//                //CachedCompound.SimpleCachedCompound.class,
//                Set.of("*Compound*", "*Conj"),
//                //Set.of("nars.*"),
//                testFilter);
//
//    }
//
//
//    private static void run(final Class mutee, final Predicate<String> testFilter) {
//        final Set<String> mutees = Collections.singleton(mutee.getName() + '*');
//        run(mutees, testFilter);
//    }
//
//    private static void run(final Set<String> mutees, final Predicate<String> testFilter) {
//
//        int concurrency = Util.concurrencyExcept(1);
//
//
//        final Collection<MutationResult> results = new FasterList().asSynchronized();
//        MutationStatisticsListener stats = new MutationStatisticsListener() {
//            @Override
//            public void handleMutationResult(ClassMutationResults r) {
//                results.addAll(r.getMutations());
//                super.handleMutationResult(r);
//            }
//        };
//
//        MutationAnalysisExecutor mae = new MutationAnalysisExecutor(concurrency,
//                List.of(
//                        stats
//                )
//        );
//
//
//        final ReportOptions data = new ReportOptions();
//
//        data.setVerbose(true);
//        data.setNumberOfThreads(concurrency);
//        data.setReportDir("/tmp/pitest");
//        data.setExportLineCoverage(true);
//
//
//        data.setTargetTests(Set.of(testFilter));
//        data.setDependencyAnalysisMaxDistance(-1);
//
//        data.setTargetClasses(mutees);
//
//        data.setTimeoutConstant(PercentAndConstantTimeoutStrategy.DEFAULT_CONSTANT);
//        data.setTimeoutFactor(PercentAndConstantTimeoutStrategy.DEFAULT_FACTOR);
//
//        final JavaAgent agent = new JarCreatingJarFinder();
//
//        try {
//
//            TestPluginArguments config = TestPluginArguments.
//                    defaults().withTestPlugin("junit5");
//
//            CoverageOptions covopt = new CoverageOptions(
//                    data.getTargetClasses(),
//                    data.getExcludedClasses(),
//                    config,
//                    data.isVerbose(), data.getDependencyAnalysisMaxDistance());
//
//
//            final LaunchOptions launchOptions = new LaunchOptions(agent,
//                    new DefaultJavaExecutableLocator(), data.getJvmArgs(),
//                    new HashMap<>());
//
//            final PathFilter pf = new PathFilter(
//                    Prelude.not(new DefaultDependencyPathPredicate()),
//                    Prelude.not(new DefaultDependencyPathPredicate()));
//            final ProjectClassPaths cps = new ProjectClassPaths(data.getClassPath(),
//                    data.createClassesFilter(), pf);
//
//            final Timings timings = new Timings();
//            final CodeSource code = new CodeSource(cps);
//
//            final CoverageGenerator coverageGenerator = new DefaultCoverageGenerator(
//                    null, covopt, launchOptions, code,
//                    new DefaultCoverageExporter(
//                            new DirectoryResultOutputStrategy(
//                                    "/tmp/pitest",
//                                    new UndatedReportDirCreationStrategy()
//                            )
//                    ),
//
//                    timings, true);
//
//            final CoverageDatabase coverageData = coverageGenerator.calculateCoverage();
//
//            final EngineArguments arguments = EngineArguments.arguments()
//                    .withMutators(null);
//
//            final MutationEngine engine = new GregorEngineFactory()
//                    .createEngine(arguments);
//
//            final MutationConfig mutationConfig = new MutationConfig(engine,
//                    launchOptions);
//
//
//            final MutationTestBuilder builder = new MutationTestBuilder(
//                    new WorkerFactory(null,
//                            covopt.getPitConfig(),
//                            mutationConfig, arguments,
//                            new PercentAndConstantTimeoutStrategy(
//                                    data.getTimeoutFactor(),
//                                    data.getTimeoutConstant()),
//                            data.isVerbose(), true, data.getClassPath()
//                            .getLocalClassPath()),
//                    new IncrementalAnalyser(new DefaultCodeHistory(code,
//                            new ObjectOutputStreamHistoryStore(
//                                    data.createHistoryWriter(),
//                                    data.createHistoryReader())),
//                            coverageData),
//                    new MutationSource(
//                            mutationConfig,
//                            new DefaultTestPrioritiser(coverageData),
//                            new ClassloaderByteArraySource(
//                                    IsolationUtils.getContextClassLoader()),
//                            CompoundMutationInterceptor.nullInterceptor()),
//                    new DefaultGrouper(0));
//
//            mae.run(builder.createMutationTestUnits(FCollection.map(code.getCode(),
//                    ClassInfo.toClassName())));
//
//
//
//            results.forEach(m -> {
//
//                MutationDetails d = m.getDetails();
//                System.out.println(m.getStatusDescription() + ' ' + d.getFilename() + ' ' +
//                        d.getMethod() + ' ' + " line " + d.getLineNumber());
//                System.out.print("\t");
//                System.out.println(m);
//            });
//            stats.getStatistics().getScores().forEach(s -> {
//                s.report(System.out);
//                System.out.println();
//            });
//            stats.getStatistics().report(System.out);
//
//            CoverageSummary cs = coverageData.createSummary();
//            System.out.println("coverage=" + cs.getCoverage() + "% (" + cs.getNumberOfCoveredLines() + '/' + cs.getNumberOfLines() + ')');
//
//        } finally {
//            agent.close();
//        }
//    }
//
//
//}
