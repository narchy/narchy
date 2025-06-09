/*
 * Copyright (C) 2015 Machine Learning Lab - University of Trieste, 
 * Italy (http:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http:
 */
package jcog.grammar.evolve.configuration;

import jcog.data.list.Lst;
import jcog.grammar.evolve.evaluators.CachedTreeEvaluator;
import jcog.grammar.evolve.evaluators.TreeEvaluator;
import jcog.grammar.evolve.generations.InitialPopulationBuilder;
import jcog.grammar.evolve.generations.TokenizedContextPopulationBuilder;
import jcog.grammar.evolve.objective.Objective;
import jcog.grammar.evolve.objective.PrecisionCharmaskLengthObjective;
import jcog.grammar.evolve.postprocessing.BasicPostprocessor;
import jcog.grammar.evolve.postprocessing.Postprocessor;
import jcog.grammar.evolve.selections.best.BasicLearningBestSelector;
import jcog.grammar.evolve.selections.best.BestSelector;
import jcog.grammar.evolve.strategy.ExecutionStrategy;
import jcog.grammar.evolve.strategy.impl.CombinedMultithreadStrategy;
import jcog.grammar.evolve.terminalsets.TerminalSetBuilder;
import jcog.grammar.evolve.terminalsets.TokenizedContextTerminalSetBuilder;
import jcog.grammar.evolve.tree.*;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author MaleLabTs
 */
public class Configuration {

    private static final Logger LOG = Logger.getLogger(Configuration.class.getName());    
  
    /**
     * Initializes with default values, parameters and operators.
     */
    public Configuration() {
        this.evolutionParameters = new EvolutionParameters();
        this.evolutionParameters.setGenerations(1000);
        this.evolutionParameters.setPopulationSize(500);
        
        
        this.initialSeed = 0;
        this.jobId = 0;
        this.jobs = 4;
        this.objective = new PrecisionCharmaskLengthObjective() ;
        
        this.constants = Arrays.asList("\\d",
            "\\w",
            "\\.",":",",",";",
            "_","=","\"","'",
            "\\\\",
            "/",    
            "\\?","\\!",
            "\\}","\\{","\\(","\\)","\\[","\\]","<",">",
            "@","#"," "," ");

        this.ranges = new Lst<>();
        this.operators = new Lst(new String[]{
                "jcog.grammar.evolve.tree.operator.Group",
                "jcog.grammar.evolve.tree.operator.NonCapturingGroup",
                "jcog.grammar.evolve.tree.operator.ListMatch",
                "jcog.grammar.evolve.tree.operator.ListNotMatch",
                "jcog.grammar.evolve.tree.operator.MatchOneOrMore",

                "jcog.grammar.evolve.tree.operator.MatchZeroOrMore",

                "jcog.grammar.evolve.tree.operator.MatchZeroOrOne",

                "jcog.grammar.evolve.tree.operator.MatchMinMax"
        });

        
        Collections.addAll(operators,
            "jcog.grammar.evolve.tree.operator.PositiveLookbehind",
            "jcog.grammar.evolve.tree.operator.NegativeLookbehind",
            "jcog.grammar.evolve.tree.operator.PositiveLookahead",
            "jcog.grammar.evolve.tree.operator.NegativeLookahead");
        
      
        this.initNodeFactory(); 

        List<Leaf> terminalSet = this.nodeFactory.getTerminalSet();

        
        Collections.addAll(terminalSet,
            new RegexRange("A-Z"),
            new RegexRange("a-z"),
            new RegexRange("A-Za-z")
        );
        
        this.evaluator = new CachedTreeEvaluator();
        this.evaluator.setup(Collections.EMPTY_MAP);
        
        this.outputFolderName = ".";
        
        this.strategyParameters = new HashMap<>();

        this.strategyParameters.put("runStrategy","jcog.grammar.evolve.strategy.impl.SeparateAndConquerStrategy");
        
        this.strategyParameters.put("runStrategy2","jcog.grammar.evolve.strategy.impl.DiversityElitarismStrategy");
        
        this.strategyParameters.put("objective2","jcog.grammar.evolve.objective.CharmaskMatchLengthObjective");
        
        
        this.strategyParameters.put("threads","2");
        this.strategy = new CombinedMultithreadStrategy(); 
        
        
        this.terminalSetBuilderParameters = new HashMap<>();
        this.terminalSetBuilderParameters.put("tokenThreashold","80.0");
        this.terminalSetBuilder = new TokenizedContextTerminalSetBuilder();
        
        this.populationBuilderParameters = new HashMap<>();
        this.populationBuilderParameters.put("tokenThreashold","80.0");     
        this.populationBuilder = new TokenizedContextPopulationBuilder();
        
        this.postprocessorParameters = new HashMap<>();
        this.postprocessor = new BasicPostprocessor();
        this.postprocessor.setup(Collections.EMPTY_MAP);
         
        this.bestSelectorParameters = new HashMap<>();
        this.bestSelector = new BasicLearningBestSelector();
        this.bestSelector.setup(Collections.EMPTY_MAP);
    }      
    
    /**
     * Updates dataset and datasetCotainer stats and structures, and initializes terminalSetBuilder and populationBuilder.
     * You should invoke this method when the original Dataset/DatasetContainer is modified.
     */
    public void setup(){
        if (this.datasetContainer == null)
            this.datasetContainer = new DatasetContainer();
        this.datasetContainer.update();
        this.terminalSetBuilder.setup(this);
        this.populationBuilder.setup(this); 
    }

    public Configuration(Configuration cc) {
        this.evolutionParameters = cc.getEvolutionParameters();
        this.initialSeed = cc.getInitialSeed();
        this.jobId = cc.getJobId();
        this.jobs = cc.getJobs();
        this.objective = cc.getObjective();
        this.evaluator = cc.getEvaluator();
        this.outputFolder = cc.getOutputFolder();
        this.outputFolderName = cc.getOutputFolderName();
        this.strategy = cc.getStrategy();
        this.strategyParameters = new LinkedHashMap<>(cc.getStrategyParameters()); 
        this.configName = cc.getConfigName();
        this.populationBuilder = cc.getPopulationBuilder();
        this.terminalSetBuilderParameters = cc.getTerminalSetBuilderParameters();
        this.terminalSetBuilder = cc.getTerminalSetBuilder();
        this.populationBuilderParameters = cc.getPopulationBuilderParameters();
        this.datasetContainer = cc.getDatasetContainer();
        this.postprocessor = cc.getPostProcessor();
        this.postprocessorParameters = cc.getPostprocessorParameters();
        
        this.bestSelector = cc.getBestSelector();
        this.bestSelectorParameters = cc.getBestSelectorParameters();
        this.constants = cc.constants;
        this.ranges = cc.ranges;
        this.operators = cc.operators;
        this.isFlagging = cc.isIsFlagging();
        this.initNodeFactory(); 
    }
    
    
    private EvolutionParameters evolutionParameters;
    private long initialSeed;
    private int jobs;
    private int jobId;
    private transient File outputFolder;
    private String outputFolderName;
    private transient Objective objective;
    private transient TreeEvaluator evaluator;
    private transient ExecutionStrategy strategy;    
    private Map<String, String> strategyParameters;  
    private String configName;
    private transient NodeFactory nodeFactory;
    private transient InitialPopulationBuilder populationBuilder;
    private final Map<String, String> populationBuilderParameters;
    private Map<String, String> terminalSetBuilderParameters;
    private transient TerminalSetBuilder terminalSetBuilder;
    private DatasetContainer datasetContainer;
    private transient Postprocessor postprocessor;
    private Map<String, String> postprocessorParameters;
    private List<String> constants;
    private List<String> ranges;
    private final List<String> operators;
    private transient BestSelector bestSelector;
    private Map<String, String> bestSelectorParameters;
    private boolean  isFlagging = false;

    public NodeFactory getNodeFactory() {
        return nodeFactory;
    }

    public void setNodeFactory(NodeFactory nodeFactory) {
        this.nodeFactory = nodeFactory;
    }
    
    public String getConfigName() {
        return configName;
    }

    public void setConfigName(String configName) {
        this.configName = configName;
    }

    public EvolutionParameters getEvolutionParameters() {
        return evolutionParameters;
    }

    public void setEvolutionParameters(EvolutionParameters evolutionParameters) {
        this.evolutionParameters = evolutionParameters;
    }

    public long getInitialSeed() {
        return initialSeed;
    }

    public void setInitialSeed(long initialSeed) {
        this.initialSeed = initialSeed;
    }

    public Map<String, String> getPostprocessorParameters() {
        return postprocessorParameters;
    }

    public void setPostprocessorParameters(Map<String, String> postprocessorParameters) {
        this.postprocessorParameters = postprocessorParameters;
    }

    public Map<String, String> getTerminalSetBuilderParameters() {
        return terminalSetBuilderParameters;
    }

    public void setTerminalSetBuilderParameters(Map<String, String> terminalSetBuilderParameters) {
        this.terminalSetBuilderParameters = terminalSetBuilderParameters;
    }

    public TerminalSetBuilder getTerminalSetBuilder() {
        return terminalSetBuilder;
    }

    public void setTerminalSetBuilder(TerminalSetBuilder terminalSetBuilder) {
        this.terminalSetBuilder = terminalSetBuilder;
    }

    public List<String> getConstants() {
        return constants;
    }

    public void setConstants(List<String> constants) {
        this.constants = constants;
    }

    public List<String> getRanges() {
        return ranges;
    }

    public void setRanges(List<String> ranges) {
        this.ranges = ranges;
    }

    public List<String> getOperators() {
        return operators;
    }





    public int getJobId() {
        return jobId;
    }

    public void setJobId(int jobId) {
        this.jobId = jobId;
    }

    /**
     * Returns a clone of the current objective, the strategies should get the objective once and cache the instance.
     * There should be and instance per strategy (and one instance per job).
     * Calling the objective a lot of times is going to instantiate a lot of instances. 
     * @return
     */
    public Objective getObjective() {
        return objective.cloneObjective();
    }

    public void setObjective(Objective objective) {
        this.objective = objective;
    }

    public TreeEvaluator getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(TreeEvaluator evaluator) {
        this.evaluator = evaluator;
    }

    public File getOutputFolder() {
        return outputFolder;
    }

    public void setOutputFolder(File outputFolder) {
        this.outputFolder = outputFolder;
    }

    public int getJobs() {
        return jobs;
    }

    public void setJobs(int jobs) {
        this.jobs = jobs;
    }

    public ExecutionStrategy getStrategy() {
        return strategy;
    }

    public void setStrategy(ExecutionStrategy strategy) {
        this.strategy = strategy;
    }

    public Map<String, String> getStrategyParameters() {
        return strategyParameters;
    }

    public void setStrategyParameters(Map<String, String> strategyParameters) {
        this.strategyParameters = strategyParameters;
    }

    public InitialPopulationBuilder getPopulationBuilder() {
        return populationBuilder;
    }

    public void setPopulationBuilder(InitialPopulationBuilder populationBuilder) {
        this.populationBuilder = populationBuilder;
    }

    public Postprocessor getPostProcessor() {
        return postprocessor;
    }

    public void setPostProcessor(Postprocessor postprocessor) {
        this.postprocessor = postprocessor;
    }   
    
    public BestSelector getBestSelector() {
        return bestSelector;
    }

    public void setBestSelector(BestSelector bestSelector) {
        this.bestSelector = bestSelector;
        this.bestSelector.setup(Collections.EMPTY_MAP);
    }
    
    public Map<String, String> getBestSelectorParameters() {
        return bestSelectorParameters;
    }

    public boolean isIsFlagging() {
        return isFlagging;
    }

    public void setIsFlagging(boolean isFlagging) {
        this.isFlagging = isFlagging;
    }

    public void setBestSelectorParameters(Map<String, String> bestSelectorParameters) {
        this.bestSelectorParameters = bestSelectorParameters;
    }

    public String getOutputFolderName() {
        return outputFolderName;
    }

    public void setOutputFolderName(String outputFolderName) {
        this.outputFolderName = outputFolderName;
        this.outputFolder = new File(this.outputFolderName);
        checkOutputFolder(this.outputFolder);
    }
   
    private static void checkOutputFolder(File outputFolder) throws ConfigurationException {
        if (outputFolder == null) {
            throw new IllegalArgumentException("The output folder must be setAt");
        }
        if (!outputFolder.isDirectory()) {
            if (!outputFolder.mkdirs()) {
                throw new ConfigurationException("Unable to create output folder \""+outputFolder+ '"');
            }
        }
    }   
    
    public DatasetContainer getDatasetContainer() {
        return datasetContainer;
    }

    /**
     * Sets the new datasetContainer.
     * you need to call the the setup command, in order to initialize the datasetContainer and 
     * in order to generate terminalSets and initial populations,
     * @param datasetContainer
     */
    public void setDatasetContainer(DatasetContainer datasetContainer) {
        this.datasetContainer = datasetContainer;
    }
    
    public Map<String, String> getPopulationBuilderParameters() {
        return populationBuilderParameters;
    }    
    
    public final void initNodeFactory() {
        NodeFactory factory = new NodeFactory();
        List<Leaf> terminals = factory.getTerminalSet();

        for (String c : constants) {            
            terminals.add(new Constant(c));
        }

        for (String s : ranges) {
            terminals.add(new RegexRange(s));
        }

        List<Node> functions = factory.getFunctionSet();
        for (String o : operators) {
            try {
                functions.add(buildOperatorInstance(o));
            } catch (ClassNotFoundException | IllegalAccessException | InstantiationException | NoSuchMethodException | InvocationTargetException ex) {
                LOG.log(Level.SEVERE, "Unable to create required operator: " + o, ex);
                System.exit(1);
            }
        }
        this.nodeFactory = factory;
    }
    
    private static Node buildOperatorInstance(String o) throws ClassNotFoundException, InstantiationException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        return Class.forName(o).asSubclass(Node.class).getConstructor().newInstance();
    }
    
    /**
     * Changes the configured fitness with the java class objectiveClass
     * @param objectiveClass
     * @return
     */
    public void updateObjective(String objectiveClass) {
        try {
            Class<? extends Objective> operatorClass = Class.forName(objectiveClass).asSubclass(Objective.class);
            Objective operator = operatorClass.getConstructor().newInstance();
            this.objective = operator;
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException ex) {
            LOG.log(Level.SEVERE, "Unable to create required objective: " + objectiveClass, ex);
            System.exit(1);
        } catch (NoSuchMethodException | InvocationTargetException e) {
            e.printStackTrace();
        }

    }
}
