













package jcog.grammar.synthesize;

import jcog.grammar.synthesize.util.GrammarUtils;
import jcog.grammar.synthesize.util.GrammarUtils.*;
import jcog.grammar.synthesize.util.ParseTreeUtils;
import jcog.grammar.synthesize.util.ParseTreeUtils.ParseTreeMultiConstantNode;
import jcog.grammar.synthesize.util.ParseTreeUtils.ParseTreeNode;
import jcog.grammar.synthesize.util.ParseTreeUtils.ParseTreeRepetitionNode;

import java.util.*;
import java.util.stream.IntStream;

public enum GrammarFuzzer {
	;

	public static class SampleParameters {
        private final double[] pRepetition;
        private final double pRecursion;
        private final double pAllCharacters;
        private final int boxSize;

        public SampleParameters(double[] pRepetition, double pRecursion, double pAllCharacters, int boxSize) {
            this.pRepetition = pRepetition;
            this.pRecursion = pRecursion;
            this.pAllCharacters = pAllCharacters;
            this.boxSize = boxSize;
        }

        public boolean randRecursion(Random random) {
            return this.pRecursion >= random.nextDouble();
        }

        public boolean randAllCharacters(Random random) {
            return this.pAllCharacters >= random.nextDouble();
        }

        public int randRepetition(Random random) {
            double sample = random.nextDouble();
            double sum = 0.0;
            for (int i = 0; i < this.pRepetition.length; i++) {
                sum += this.pRepetition[i];
                if (sum >= sample) {
                    return i;
                }
            }
            return this.pRepetition.length;
        }

        public static int randAlternation(Random random) {
            return random.nextInt(3);
        }

        public static int randMultiAlternation(Random random, int numChoices) {
            return random.nextInt(numChoices);
        }

        public int getBoxSize() {
            return this.boxSize;
        }
    }

    public static class IntBox {
        private int value;

        public IntBox(int value) {
            this.value = value;
        }

        public void decrement() {
            this.value--;
        }

        public int value() {
            return this.value;
        }
    }

    private static ParseTreeNode sampleHelper(Node grammar, NodeMerges recursiveNodes, SampleParameters parameters, Random random, Map<Node, ParseTreeNode> backup, IntBox length) {
        if (length.value() == 0) {
            return backup.get(grammar);
        }
        length.decrement();
        if (!recursiveNodes.get(grammar).isEmpty() && parameters.randRecursion(random)) {
            int choice = SampleParameters.randMultiAlternation(random, recursiveNodes.get(grammar).size());
            return sampleHelper(new ArrayList<>(recursiveNodes.get(grammar)).get(choice), recursiveNodes, parameters, random, backup, length);
        } else if (grammar instanceof MultiAlternationNode) {
            int choice = SampleParameters.randMultiAlternation(random, grammar.getChildren().size());
            return sampleHelper(grammar.getChildren().get(choice), recursiveNodes, parameters, random, backup, length);
        } else if (grammar instanceof RepetitionNode) {
            ParseTreeNode start = sampleHelper(((RepetitionNode) grammar).start, recursiveNodes, parameters, random, backup, length);
            int reps = parameters.randRepetition(random);
            List<ParseTreeNode> rep = IntStream.range(0, reps).mapToObj(i -> sampleHelper(((RepetitionNode) grammar).rep, recursiveNodes, parameters, random, backup, length)).toList();
            ParseTreeNode end = sampleHelper(((RepetitionNode) grammar).end, recursiveNodes, parameters, random, backup, length);
            return new ParseTreeRepetitionNode((RepetitionNode) grammar, start, rep, end);
        } else if (grammar instanceof MultiConstantNode mconstNode) {
            StringBuilder sb = new StringBuilder();
            boolean useAllCharacters = parameters.randAllCharacters(random);
            for (Set<Character> characterOption : useAllCharacters ? mconstNode.characterOptions : mconstNode.characterChecks) {
                List<Character> characterOptionList = new ArrayList<>(characterOption);
                int choice = SampleParameters.randMultiAlternation(random, characterOptionList.size());
                sb.append(characterOptionList.get(choice));
            }
            return new ParseTreeMultiConstantNode(mconstNode, sb.toString());
        } else {
            throw new RuntimeException("Invalid node type: " + grammar.getClass().getName());
        }
    }

    private static void getBackup(ParseTreeNode node, Map<Node, ParseTreeNode> backup) {
        backup.put(node.getNode(), node);
        for (ParseTreeNode child : node.getChildren()) {
            getBackup(child, backup);
        }
    }

    public static ParseTreeNode sample(Node program, Grammar grammar, SampleParameters parameters, Random random) {
        Map<Node, ParseTreeNode> backup = new HashMap<>();
        if (grammar.node instanceof MultiAlternationNode man) {
            for (var parseTree : ParseTreeUtils.getParseTreeAlt(man))
                getBackup(parseTree, backup);

        } else {
            getBackup(ParseTreeUtils.getParseTree(grammar.node), backup);
        }
        for (Node node : grammar.merges.keySet()) {
            if (!backup.containsKey(node))
                throw new RuntimeException("Invalid node: " + node);

            for (Node merge : grammar.merges.get(node)) {
                if (!backup.containsKey(merge))
                    throw new RuntimeException("Invalid node: " + node);
            }
        }
        for (Node descendant : GrammarUtils.getDescendants(program)) {
            if (!backup.containsKey(descendant))
                throw new RuntimeException("Invalid node: " + descendant);
        }
        return sampleHelper(program, grammar.merges, parameters, random, backup, new IntBox(parameters.getBoxSize()));
    }

    public static class GrammarSampler implements Iterator<String>, Iterable<String> {
        private final Grammar grammar;
        private final SampleParameters parameters;
        private final Random random;

        public GrammarSampler(Grammar grammar, SampleParameters parameters, Random random) {
            this.grammar = grammar;
            this.parameters = parameters;
            this.random = random;
        }

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public String next() {
            return sample(this.grammar.node, this.grammar, this.parameters, this.random).getExample();
        }

        @Override
        public Iterator<String> iterator() {
            return this;
        }

        @Override
        public void remove() {
            throw new RuntimeException("Remove not supported!");
        }
    }

    public static class GrammarMutationSampler implements Iterator<String>, Iterable<String> {
        private final Grammar grammar;
        private final SampleParameters parameters;
        private final int maxLength;
        private final int numMutations;
        private final Random random;

        public GrammarMutationSampler(Grammar grammar, SampleParameters parameters, int maxLength, int numMutations, Random random) {
            this.grammar = grammar;
            this.parameters = parameters;
            this.maxLength = maxLength;
            this.numMutations = numMutations;
            this.random = random;
        }

        private ParseTreeNode sampleHelper(ParseTreeNode seed) {
            List<ParseTreeNode>[] descendants = ParseTreeUtils.getDescendantsByType(seed);
            int isMultiConstant = descendants[1].isEmpty() || (!descendants[0].isEmpty() && this.random.nextBoolean()) ? 0 : 1;
            int choice = this.random.nextInt(descendants[isMultiConstant].size());
            ParseTreeNode cur = descendants[isMultiConstant].get(choice);
            ParseTreeNode sub = GrammarFuzzer.sample(cur.getNode(), this.grammar, this.parameters, this.random);
            return ParseTreeUtils.getSubstitute(seed, cur, sub);
        }

        private ParseTreeNode sample(ParseTreeNode seed) {
            while (true) {
                ParseTreeNode result = sampleHelper(seed);
                if (result.getExample().length() <= this.maxLength) {
                    return result;
                }
            }
        }

        public String sampleOne(Node node) {
            ParseTreeNode cur = ParseTreeUtils.getParseTree(node);
            int choice = this.random.nextInt(this.numMutations);
            for (int i = 0; i < choice; i++) {
                cur = this.sample(cur);
            }
            return cur.getExample();
        }

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public String next() {
            Node node = this.grammar.node;
            if (node instanceof MultiAlternationNode) {
                List<Node> children = node.getChildren();
                int choice = this.random.nextInt(children.size());
                return this.sampleOne(children.get(choice));
            } else {
                return this.sampleOne(node);
            }
        }

        @Override
        public Iterator<String> iterator() {
            return this;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    public static class CombinedMutationSampler implements Iterator<String>, Iterable<String> {
        private final Iterator<String> sampler;
        private final int numMutations;
        private final Random random;

        public CombinedMutationSampler(Iterable<String> sampler, int numMutations, Random random) {
            this.sampler = sampler.iterator();
            this.numMutations = numMutations;
            this.random = random;
        }

        @Override
        public boolean hasNext() {
            return this.sampler.hasNext();
        }

        @Override
        public String next() {
            String sample = this.sampler.next();
            if (sample == null) {
                return null;
            }
            return this.random.nextBoolean() ? sample : nextStringMutant(sample, this.random.nextInt(this.numMutations), this.random);
        }

        @Override
        public Iterator<String> iterator() {
            return this;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    private static char nextChar(Random random) {
        return (char) random.nextInt(128);
    }

    
    private static String nextStringMutant(String string, Random random) {
        if (string.isEmpty()) {
            return String.valueOf(nextChar(random));
        } else {
            int randIndex = random.nextInt(string.length());
            String head = string.substring(0, randIndex);
            String tail = string.substring(randIndex);

            
            if (random.nextBoolean()) {
                return head + nextChar(random) + tail;
            } else {
                return tail.isEmpty() ? head : (head + tail.substring(1));
            }
        }
    }

    private static String nextStringMutant(String string, int numMutantions, Random random) {
        for (int i = 0; i < numMutantions; i++) {
            string = nextStringMutant(string, random);
        }
        return string;
    }
}