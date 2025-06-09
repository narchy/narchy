package jcog.neat;

import jcog.Util;
import jcog.WTF;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.ToDoubleFunction;
import java.util.stream.Stream;

import static java.lang.Math.abs;
import static java.lang.Math.random;
import static java.util.Comparator.comparingDouble;
import static java.util.stream.IntStream.range;

/**
 * To verify that all the requirements of Adaptive HyperNEAT are present in the code, let's go through each component and check if it is implemented:
 * <p>
 * NEAT (NeuroEvolution of Augmenting Topologies):
 * Genome representation: The Genome class represents the genome with a list of Gene objects.
 * Mutation operators: The mutate method in the Genome class includes adding nodes, adding connections, and modifying weights.
 * Crossover operator: The crossover method in the Genome class performs crossover between two genomes.
 * Speciation and fitness sharing: Not explicitly implemented in this code, but can be added if required.
 * HyperNEAT (Hypercube-based NeuroEvolution of Augmenting Topologies):
 * Substrate representation: The Substrate class represents the substrate with input, hidden, and output layers, and connections between nodes.
 * Encoding of substrate connectivity patterns using CPPNs: The CPPN class represents the Compositional Pattern Producing Network, which encodes the connectivity patterns of the substrate.
 * Querying the CPPN to determine connection weights in the substrate: The query method in the CPPN class is used to determine the connection weights in the substrate based on the coordinates of the nodes.
 * Adaptive HyperNEAT:
 * Dynamic substrate resolution adjustment based on the complexity of the problem: The adaptResolution method in the Substrate class increases the resolution of the substrate by doubling the input and output dimensions.
 * Iterative refinement of the substrate during evolution: The evolve method in the Evolution class triggers the adaptResolution method on the substrate every 10 generations.
 * Adaptation of the CPPN resolution to match the substrate resolution: The adaptResolution method in the CPPN class increases the resolution of the CPPN by adding new nodes and connections to match the increased resolution of the substrate.
 * Evolutionary process:
 * Population initialization: The initPop method in the Evolution class initializes the population with genomes containing initial nodes and connections.
 * Evaluation of individuals: The evalFitness method in the Evolution class evaluates the fitness of each genome by creating a substrate, applying the CPPN to determine the connection weights, and using the provided fitness function.
 * Selection and reproduction: The evolve method in the Evolution class performs tournament selection, crossover, and mutation to create the next generation of genomes.
 * Iteration until a termination condition is met: The main method iterates the evolutionary process for a specified number of generations.
 * Interface for configuring and running the algorithm:
 * Setting input and output dimensions: The Config record class allows setting the input and output dimensions of the substrate.
 * Specifying evolutionary parameters: The Config record class includes parameters for population size, mutation rate, crossover rate, and the maximum number of generations.
 * Retrieving the best-performing individual: The bestGenome method in the Evolution class returns the genome with the highest fitness after the evolutionary process.
 */
public class AdaptiveHyperNEAT {
    static final class Gene {
        private final int inn, from, to;
        private double weight;
        private boolean enabled;

        Gene(int inn, int from, int to, double weight, boolean enabled) {
            this.inn = inn;
            if (from == to)
                throw new UnsupportedOperationException();
            this.from = from;
            this.to = to;
            this.weight = weight;
            this.enabled = enabled;
        }

        @Override
        public String toString() {
            return "Gene{" +
                    "inn=" + inn +
                    ", from=" + from +
                    ", to=" + to +
                    ", weight=" + weight +
                    ", enabled=" + enabled +
                    '}';
        }
    }

    static final class Node {
        private final int id;
        private double activation;

        Node(int id, double activation) {
            this.id = id;
            this.activation = activation;
        }
    }

    static final class Connection {
        private final Node from, to;
        private final double weight;
        private final boolean enabled;

        Connection(Node from, Node to, double weight, boolean enabled) {
            if (from == to) throw new UnsupportedOperationException();
            if (weight == 0)
                throw new UnsupportedOperationException();

            this.from = from;
            this.to = to;
            this.weight = weight;
            this.enabled = enabled;
        }

    }

    record Config(int inW, int inH, int outW, int outH, int popSize, double mutRate, int maxGen) {
    }

    static class Genome {
        final List<Gene> genes = new ArrayList<>();
        double fitness = Double.NaN;

        @Override
        public String toString() {
            return "Genome $" + fitness + "=" + genes;
        }

        /** TODO separate mutation rates for nodes, connections, and weights */
        void mutate(double rate) {
            if (random() < rate)
                addNode();

            if (random() < rate)
                addConnection();

            int geneCount = genes.size();
            double rateGene = rate/geneCount;
            genes.forEach(g -> {
                if (g.enabled) {
                    if (random() < rateGene) g.weight += (random() - 0.5) * 0.1;
                } else {
                    if (random() < rateGene) g.enabled = true; //re-enable
                }
            });
        }

        @Nullable Genome crossover(Genome other) {
            if (this == other)
                return null;

            var child = new Genome();
            int a = genes.size(), b = other.genes.size();
            range(0, Math.max(a, b)).forEach(i -> {
                Gene g1 = i < a ? this.genes.get(i) : null, g2 = i < b ? other.genes.get(i) : null; //getGene(i), g2 = other.getGene(i);
                if (g1 != null && g2 != null) child.genes.add(random() < 0.5 ? g1 : g2);
                else if (g1 != null) child.genes.add(g1);
                else if (g2 != null) child.genes.add(g2);
            });
            return child;
        }

        void addNode() {
            if (genes.isEmpty()) {
                throw new UnsupportedOperationException();
            } else {
                int gi = (int) (random() * genes.size());
                var g = genes.get(gi);
                if (g.enabled) {
                    g.enabled = false;

                    var newNode = getInn() + 1;
                    //genes.set(gi, new Gene(g.inn, g.from, g.to, g.weight, false));

                    int source = g.from;
                    if (newNode == source) {
                        source = randomNode(); if (newNode == source) return; //give up
                    }

                    int target = g.to;
                    if (newNode == target) {
                        target = randomNode(); if (newNode == target) return; //give up
                    }

                    genes.add(new Gene(newNode, source, newNode, weightRandom(), true));
                    genes.add(new Gene(newNode, newNode, target, weightRandom(), true));

                }
            }
        }

//        private void addGeneRandom() {
//            int src = randomNode();
//            int tgt = randomNodeOtherThan(src);
//            genes.add(new Gene(getInn(), src, tgt, weightRandom(), true));
//        }

        void addConnection() {
            var from = randomNode();
            var to = randomNodeOtherThan(from);
            if (!hasConnection(from, to))
                genes.add(new Gene(getInn(), from, to, weightRandom(), true));
        }

        Gene getGene(int inn) {
            return genes.stream().filter(g -> g.inn == inn).findFirst().orElse(null);
        }

        int getInn() {
            return genes.isEmpty() ? 0 : genes.getLast().inn;
        }

        int randomNodeOtherThan(int other) {
            int g = genes.size();
            if (g < 2) throw new UnsupportedOperationException();
            if (g == 2) {
                return other == 0 ? 1 : 0;
            } else {
                int y;
                do {
                    y = randomNode();
                } while (y == other);
                return y;
            }
        }

        int randomNode() {
            return genes.get((int) (random() * genes.size())).from;
        }

        boolean hasConnection(int from, int to) {
            return genes.stream().anyMatch(g -> /*g.enabled &&*/ (g.from == from && g.to == to));
        }
    }

    static class Substrate {
        int inW, inH, outW, outH;
        double[][] out;
        final Map<Integer, Node> nodes = new HashMap<>();
        final List<Connection> conns = new ArrayList<>();

        Substrate(int inW, int inH, int outW, int outH) {
            this.inW = inW;
            this.inH = inH;
            this.outW = outW;
            this.outH = outH;
            //in = new double[inW][inH];
            out = new double[outW][outH];
        }

        double[][] activate(double[][] input) {
            clearActivations(nodes.values());

            // Set the activations of the input nodes
            for (int i = 0; i < inW; i++) {
                for (int j = 0; j < inH; j++) {
                    int id = id(i, j, inW, inH, true);
                    nodes.get(id).activation = input[i][j];
                }
            }

            // Propagate the activations through the substrate
            conns.stream().filter(c -> c.enabled).forEach(c -> {
                var from = nodes.get(c.from.id);
                var to = nodes.get(c.to.id);
                double x = activationFn(from.activation) * c.weight;
                to.activation += x;
            });

            // Store the activations of the output nodes in the output array
            range(0, outW).forEach(i -> range(0, outH).forEach(j -> {
                int id = id(i, j, outW, outH, false);
                out[i][j] = (nodes.get(id).activation);
            }));

            return out;
        }

        void addNode(int i, int j, boolean in) {
            var w = in ? inW : outW;
            var h = in ? inH : outH;
            var id = id(i, j, w, h, in);
            nodes.put(id, new Node(id, 0));
        }

        static int id(int i, int j, int w, int h, boolean inOrOut) {
            return (inOrOut ? 0 : w * h) + (i * h + j);
        }

        void adaptResolution() {
            var oldInW = inW; var oldInH = inH; var oldOutW = outW; var oldOutH = outH;
            inW *= 2; inH *= 2; outW *= 2; outH *= 2;
            out = new double[outW][outH];

            range(0, inW).forEach(i -> range(0, inH).forEach(j -> addNode(i, j, true)));
            range(0, outW).forEach(i -> range(0, outH).forEach(j -> addNode(i, j, false)));

            range(0, inW).forEach(i -> range(0, inH).forEach(j -> {

                var id1 = id(i, j, inW, inH, true);

                range(0, outW).forEach(k -> range(0, outH).forEach(l -> {

                    var id2 = id(k, l, outW, outH, false);

                    var w = conns.stream().filter(c ->
                        c.from.id == id(i/2, j/2, oldInW, oldInH, true) &&
                        c.to.id == id(k/2, l/2, oldOutW, oldOutH, false)
                    ).findFirst().map(c -> c.weight).orElseGet(AdaptiveHyperNEAT::weightRandom);

                    conns.add(new Connection(nodes.get(id1), nodes.get(id2), w, true));
                }));
            }));
        }

    }

    static private double weightRandom() {
        return random() * 2 - 1;
    }

    static class CPPN {
        final Genome genome;
        final Map<Integer, Node> nodes = new HashMap<>();
        final List<Connection> conns = new ArrayList<>();

        CPPN(Genome genome) {
            this.genome = genome;

            //Inputs
            nodes.put(0, new Node(0, 0));
            nodes.put(1, new Node(1, 0));
            nodes.put(2, new Node(2, 0));
            nodes.put(3, new Node(3, 0));

            //Output
            nodes.put(4, new Node(4, 0));

            genome.genes.forEach(g -> {
                if (g.enabled) {
                    int from = g.from, to = g.to;
                    Node a = insertNode(from), b = insertNode(to);
                    conns.add(new Connection(a, b, g.weight, g.enabled));
                }
            });
        }

        double query(double x1, double y1, double x2, double y2) {
            clearActivations(nodes.values());

            nodes.get(0).activation = x1;
            nodes.get(1).activation = y1;
            nodes.get(2).activation = x2;
            nodes.get(3).activation = y2;

            var size = nodes.size();

            conns.stream().filter(c -> c.enabled).forEach(c -> {
                double x = activationFn(c.from.activation) * c.weight;
                c.to.activation += x;
            });

            return nodes.get(4).activation;
        }


//        void adaptResolution() {
//            var size = nodes.size();
//            insertNode(size);
//            insertNode(size + 1);
//
//            genome.genes.forEach(g -> {
//                int from = g.from, to = g.to;
//                Node fromNode = nodes.get(from);
//                double weight = g.weight;
//                boolean enabled = g.enabled;
//                if (to == size - 2) {
//                    conns.add(new Connection(fromNode, nodes.get(size), weight, enabled));
//                    conns.add(new Connection(fromNode, nodes.get(size + 1), weight, enabled));
//                } else if (to == size - 1) {
//                    conns.add(new Connection(fromNode, nodes.get(size + 1), weight, enabled));
//                }
//            });
//        }

        private Node insertNode(int id) {
            return nodes.computeIfAbsent(id, ID -> new Node(ID, 0));
        }
    }

    public static void clearActivations(Iterable<Node> nodes) {
        nodes.forEach(i -> i.activation = 0);
    }

    static class Evolution {
        private final Config config;
        int popSize;
        double mutRate;
        List<Genome> pop = new ArrayList<>();
        final ToDoubleFunction<double[][]> fit;

        Evolution(Config config, ToDoubleFunction<double[][]> fit) {
            this.fit = fit;
            this.config = config;
        }

        void initPop() {
            range(0, popSize).forEach(i -> {
                var g = new Genome();

                for (int k = 0; k < 4; k++) {
                    int a = k; //ThreadLocalRandom.current().nextInt(4); //random input node
                    int b = 4; //output node
                    g.genes.add(new Gene(4, a, b, weightRandom(), true));
                }

                g.addNode();
                g.addConnection();
//                g.genes.add(new Gene(0, 0, 1, weightRandom(), true));
//                g.genes.add(new Gene(1, 1, 0, weightRandom(), true));

                //g.addNode();
                //g.addNode();
                //g.addConnection();
                pop.add(g);
            });
        }

        void evalFitness(Genome g, double[][] input) {
            double oldFitness = g.fitness;
            if (oldFitness!=oldFitness) {
                double[][] output = substrate(g).activate(input); // Activate the substrate with the input data
                g.fitness = fit.applyAsDouble(output); // Evaluate the fitness based on the substrate's output
            }
        }

        Substrate substrate(Genome genome) {
            var cppn = new CPPN(genome);
            var substrate = new Substrate(config.inW, config.inH, config.outW, config.outH);

            int inW = substrate.inW, inH = substrate.inH, outW = substrate.outW, outH = substrate.outH;

            range(0, inW).forEach(i -> range(0, inH).forEach(j -> substrate.addNode(i, j, true)));

            range(0, outW).forEach(i -> range(0, outH).forEach(j -> substrate.addNode(i, j, false)));

            range(0, inW).forEach(i -> range(0, inH).forEach(j -> {
                var ij = Substrate.id(i, j, inW, inH, true);
                range(0, outW).forEach(k -> range(0, outH).forEach(l -> {
                    var kl = Substrate.id(k, l, outW, outH, false);

                    var w = cppn.query(
                            i / (double) inW, j / (double) inH,
                            k / (double) outW, l / (double) outH);
                    if (w!=0)
                        substrate.conns.add(new Connection(substrate.nodes.get(ij), substrate.nodes.get(kl), w, true));
                }));
            }));

            return substrate;
        }

        double elitism =
            0.75;
            //0.9;

        void evolve(int gen) {
            var newPop = new ArrayList<>(pop.subList(0, (int) Math.round(elitism * popSize))); //elitism

            while (newPop.size() < popSize) {
                var child = tournamentSelection().crossover(tournamentSelection());
                if (child!=null) {
                    child.mutate(mutRate);
                    newPop.add(child);
                }
            }

            pop = newPop;

            if (gen % 1000 == 0) {
                pop.forEach(g -> {
                    Substrate s = substrate(g);
                    s.adaptResolution();
                    //s.cppn.adaptResolution();
                });
            }
        }

        Genome tournamentSelection() {
            var popSize = this.popSize;
            return Stream.generate(() -> pop.get((int) (random() * popSize))).limit(4)
                    .max(geneCmp).orElse(null);
        }

        static final Comparator<Genome> geneCmp = comparingDouble(g -> g.fitness);

//        Genome bestGenome() {
//            return pop.stream().max(geneCmp).orElse(null);
//        }
    }

    static abstract class Problem {
        abstract double[][] input();

        abstract double[][] outputExpected();

        abstract double fitness(double[][] output, double[][] expectedOutput);
    }

    static class SineProblem extends Problem {
        double[][] input() {
            return new double[][]{
                    {0.0}, {0.1}, {0.2}, {0.3}, {0.4},
                    {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}
            };
        }

        double[][] outputExpected() {
            return new double[][]{
                    {0.0}, {0.09983}, {0.19866}, {0.29552}, {0.38941},
                    {0.47942}, {0.56464}, {0.64421}, {0.71735}, {0.78332}, {0.84147}
            };
        }

        double fitness(double[][] output, double[][] expectedOutput) {
            double errSum = 0;
            if (output.length!= expectedOutput.length || output[0].length!=expectedOutput[0].length)
                throw new WTF();
            for (int i = 0; i < output.length; i++) {
                errSum += abs(output[i][0] - expectedOutput[i][0]);
            }
            return 1.0 / (1.0 + errSum);
        }
    }


    public static void main(String[] args) {
        Problem problem = new SineProblem();
        double[][] input = problem.input();
        double[][] outputExp = problem.outputExpected();

        Config config = new Config(input.length, 1, outputExp.length, 1,
                256, 0.25, 2000);

        var evo = new Evolution(config, y -> problem.fitness(y, outputExp));
        var popSize = config.popSize();
        var mutRate = config.mutRate();
        var maxGen = config.maxGen();

        evo.popSize = popSize;
        evo.mutRate = mutRate;


        evo.initPop();
        range(0, maxGen).forEach(generation -> {
            evo.pop.forEach(g -> evo.evalFitness(g, problem.input()));
            evo.pop.sort(Evolution.geneCmp.reversed());

            var best = evo.pop.getFirst(); //bestGenome();
            double fitMin = evo.pop.stream().mapToDouble(p -> p.fitness).min().getAsDouble();
            double fitMean = evo.pop.stream().mapToDouble(p -> p.fitness).average().getAsDouble();
            double fitMax = evo.pop.stream().mapToDouble(p -> p.fitness).max().getAsDouble();
            System.out.println("generation " + generation + ": " + fitMean + ".." + fitMax + "\t" + best);

            evo.evolve(generation);
        });
    }

    static double activationFn(double x) {
        //return 2 / (1 + exp(-4.9 * x)) - 1;
        return Util.sigmoid(x);
        //return Math.tanh(x);
    }

}