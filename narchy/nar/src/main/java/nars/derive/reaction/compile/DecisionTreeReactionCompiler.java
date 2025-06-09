package nars.derive.reaction.compile;

import jcog.data.list.Lst;
import jcog.data.set.ArrayHashSet;
import jcog.tree.perfect.TrieNode;
import nars.NAR;
import nars.action.Action;
import nars.derive.Deriver;
import nars.derive.reaction.Reaction;
import nars.derive.reaction.ReactionModel;
import nars.derive.reaction.memoize.MemoizePredicate;
import nars.term.control.*;
import nars.term.util.map.TermPerfectTrie;
import org.eclipse.collections.api.tuple.primitive.ObjectDoublePair;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.tuple.primitive.PrimitiveTuples;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

import static nars.term.control.PREDICATE.TRUE;

/**
 * An optimized reaction compiler that constructs a trie of conditions and actions,
 * using predicate ordering heuristics reminiscent of decision-tree splits.
 */
public class DecisionTreeReactionCompiler extends ReactionCompiler {

    static final Function<PREDICATE, IntArrayList> BRANCH_INDEX_CREATOR = k -> new IntArrayList(1);
    private static final boolean ENABLE_LAMBDA_OPTIMIZATION = false;
    private static final boolean ENABLE_MEMOIZATION = true;

    /* Core compilation methods */

    @Override
    public ReactionModel compile(ArrayHashSet<Reaction<Deriver>> reactions, NAR nar) {
        Lst<Reaction<Deriver>> reactionList = reactions.list;
        int reactionCount = reactionList.size();
        if (reactionCount == 0) {
            throw new IllegalStateException("No reactions to compile");
        }

        Action[] actions = new Action[reactionCount];
        TermPerfectTrie<PREDICATE<Deriver>, Action> predicateTrie = new TermPerfectTrie<>(reactionCount);

        // Build a queue of "causes" for each Reaction
        var causeQueue = nar.control.newCauses(reactionList.stream());

        // Gather predicate statistics
        Map<PREDICATE<Deriver>, PredicateStats<Deriver>> predicateStatsMap = collectPredicateStats(reactionList);

        // Choose a predicate ordering strategy based on information gain
        Comparator<PREDICATE<Deriver>> predicateComparator =
                new InformationGainStrategy<>(predicateStatsMap, reactionCount).get();

        // Insert each Reaction's preconditions into the trie
        int index = 0;
        for (Reaction<Deriver> reaction : reactionList) {
            Lst<PREDICATE<Deriver>> sortedPredicates = new Lst<>(reaction.pre);
            // Sort using the chosen strategy
            sortedPredicates.sort(predicateComparator);

            // Add action as final predicate
            Action action = reaction.action(causeQueue.poll());
            sortedPredicates.add(action);

            // Insert into trie
            Action existingAction = predicateTrie.put(sortedPredicates, action);
            if (existingAction != null && existingAction != action) {
                throw new IllegalStateException("Duplicate reaction detected:\n" + sortedPredicates +
                        "\nExisting action: " + existingAction);
            }
            actions[index++] = action;
        }

        // Compile the trie into a single optimized PREDICATE
        PREDICATE<Deriver> compiledPredicate = compileTrie(predicateTrie);

        return new ReactionModel(compiledPredicate, actions);
    }

    static <X> PREDICATE<X> compileTrie(TermPerfectTrie<PREDICATE<X>, Action> trie) {
        // Convert from Trie to raw PREDICATE structure
        PREDICATE<X> rawPredicate = compileTrieNode(trie.root);

        // Optimize the structure
        PREDICATE optimizedPredicate = optimizePredicate(rawPredicate);

        // Apply optional memoization
        PREDICATE<X> finalPredicate = ENABLE_MEMOIZATION
                ? MemoizePredicate.memoize(optimizedPredicate)
                : optimizedPredicate;

        // Optionally wrap in lambda-based compiler for performance
        if (ENABLE_LAMBDA_OPTIMIZATION) {
            return new LambdaPREDICATECompiler.MHPredicate<>(finalPredicate, finalPredicate.mh());
        }
        return finalPredicate;
    }

    /* Node compilation methods */

    static <X> PREDICATE<X> compileTrieNode(TrieNode<List<PREDICATE<X>>, Action> node) {
        if (!node.hasChildren()) {
            return compileLeafNode(node);
        }

        // For an internal node with children, create a FORK
        Lst<PREDICATE<X>> branches = new Lst<>(node.size());
        for (TrieNode<List<PREDICATE<X>>, Action> child : node) {
            branches.add(compileSequenceWithChild(child));
        }
        return FORK.fork(branches);
    }

    static <X> PREDICATE<X> compileLeafNode(TrieNode<List<PREDICATE<X>>, Action> node) {
        List<PREDICATE<X>> conditions = node.seq();
        return conditions != null ? AND.the(conditions) : TRUE;
    }

    static <X> PREDICATE<X> compileSequenceWithChild(TrieNode<List<PREDICATE<X>>, Action> node) {
        PREDICATE<X> childResult = node.hasChildren() ? compileTrieNode(node) : TRUE;
        int startIndex = node.start(), endIndex = node.end();

        // Create a list with all conditions plus the child result
        Lst<PREDICATE<X>> sequenceWithChild = new Lst<>((endIndex - startIndex) + 1);
        sequenceWithChild.addAll(node.seq().subList(startIndex, endIndex));
        sequenceWithChild.add(childResult);

        return AND.the(sequenceWithChild);
    }

    /* Optimization methods */

    static <X> PREDICATE<X> optimizePredicate(PREDICATE<X> predicate) {
        if (predicate == null) return null;

        if (predicate instanceof FORK<X> fork) {
            return optimizeFork(fork);
        }
        else if (predicate instanceof AND<X> ||
                predicate instanceof NOT<X> ||
                predicate instanceof IF<X>) {
            // Recursively optimize children, then perform local optimization
            return predicate.transform(DecisionTreeReactionCompiler::optimizePredicate);
        }
        else {
            // Base case: atomic predicates (ATOMIC, ACTION, etc.)
            return predicate;
        }
    }

    static <X> PREDICATE<X> optimizeFork(FORK<X> fork) {
        // First, recursively optimize each branch
        FORK<X> optimizedFork = (FORK<X>) fork.transform(DecisionTreeReactionCompiler::optimizePredicate);
        PREDICATE<X>[] branches = optimizedFork.branch;
        int branchCount = branches.length;

        switch (branchCount) {
            case 0 -> {
                return TRUE;
            }
            case 1 -> {
                return branches[0];
            }
            default -> {
                // Try to find common predicates across branches
                Set<PREDICATE<X>> commonPredicates = findCommonPredicates(branches);

                if (commonPredicates.isEmpty()) {
                    // No common predicates, try converting to IF structure
                    Lst<PREDICATE<X>> branchList = new Lst<>(branchCount, branches.clone());
                    PREDICATE<X> ifPredicate = tryConvertToIfStructure(branchList);
                    return (ifPredicate != null) ? ifPredicate : FORK.fork(branchList);
                } else {
                    // Factor out common predicates for optimization
                    return factorOutCommonPredicates(branches, commonPredicates);
                }
            }
        }
    }

    /* Common predicate extraction and handling */

    static <X> Set<PREDICATE<X>> findCommonPredicates(PREDICATE<X>[] branches) {
        int branchCount = branches.length;
        if (branchCount == 0) {
            return Collections.emptySet();
        }

        // Start with all predicates from first branch
        Set<PREDICATE<X>> commonPredicates = extractConjuncts(branches[0]);

        // Retain only those present in all branches
        for (int i = 1; i < branchCount && !commonPredicates.isEmpty(); i++) {
            commonPredicates.retainAll(extractConjuncts(branches[i]));
        }

        return commonPredicates;
    }

    static <X> Set<PREDICATE<X>> extractConjuncts(PREDICATE<X> predicate) {
        Set<PREDICATE<X>> conjunctSet = new HashSet<>();
        extractConjunctsRecursive(predicate, conjunctSet);
        return conjunctSet;
    }

    private static <X> void extractConjunctsRecursive(PREDICATE<X> predicate, Set<PREDICATE<X>> results) {
        if (predicate instanceof AND<X> and) {
            for (PREDICATE<X> conjunct : and.conditions()) {
                extractConjunctsRecursive(conjunct, results);
            }
        }
        else if (!(predicate instanceof FORK<X> || predicate instanceof IF<X>)) {
            // Add only if not a complex control structure
            results.add(predicate);
        }
    }

    private static <X> PREDICATE<X> factorOutCommonPredicates(PREDICATE<X>[] branches, Set<PREDICATE<X>> commonPredicates) {
        Lst<PREDICATE<X>> remainingPredicates = new Lst<>(branches.length);

        // Remove common predicates from each branch
        for (PREDICATE<X> branch : branches) {
            remainingPredicates.add(removePredicates(branch, commonPredicates));
        }

        // Create AND of common predicates with FORK of remaining predicates
        Lst<PREDICATE<X>> factored = new Lst<>(commonPredicates);
        factored.add(FORK.fork(remainingPredicates));

        return AND.the(factored);
    }

    private static <X> PREDICATE<X> removePredicates(PREDICATE<X> source, Collection<PREDICATE<X>> predicatesToRemove) {
        if (source instanceof AND<X> and) {
            Lst<PREDICATE<X>> remaining = null;

            for (PREDICATE<X> conjunct : and.conditions()) {
                if (!predicatesToRemove.contains(conjunct)) {
                    if (remaining == null) {
                        remaining = new Lst<>(1);
                    }
                    remaining.add(conjunct);
                }
            }

            return (remaining == null) ? TRUE : AND.the(remaining);
        }
        else {
            // Single predicate - remove if in the removal set
            return predicatesToRemove.contains(source) ? TRUE : source;
        }
    }

    /* IF structure conversion */

    @Nullable
    private static <X> PREDICATE<X> tryConvertToIfStructure(List<PREDICATE<X>> branches) {
        // Analyze positive and negative occurrences of predicates
        Map<PREDICATE<X>, IntArrayList> positiveOccurrences = new HashMap<>();
        Map<PREDICATE<X>, IntArrayList> negativeOccurrences = new HashMap<>();
        collectPredicateOccurrences(branches, positiveOccurrences, negativeOccurrences);

        // Find candidate predicates for IF conversion
        List<ObjectDoublePair<PREDICATE<X>>> candidates = findIfCandidates(branches, positiveOccurrences, negativeOccurrences);
        if (candidates == null) return null;

        // Sort candidates by score (highest first)
        if (candidates.size() > 1) {
            candidates.sort(Comparator.comparingDouble((ObjectDoublePair<PREDICATE<X>> z) -> z.getTwo()).reversed());
        }

        // Use best candidate to create IF structure
        PREDICATE<X> bestCandidate = candidates.get(0).getOne();
        return createIfStructure(bestCandidate, branches, positiveOccurrences, negativeOccurrences);
    }

    private static <X> void collectPredicateOccurrences(
            List<PREDICATE<X>> branches,
            Map<PREDICATE<X>, IntArrayList> positiveOccurrences,
            Map<PREDICATE<X>, IntArrayList> negativeOccurrences)
    {
        int branchCount = branches.size();
        for (int i = 0; i < branchCount; i++) {
            for (PREDICATE<X> predicate : extractConjuncts(branches.get(i))) {
                if (predicate instanceof NOT<X> negation) {
                    negativeOccurrences.computeIfAbsent(negation.cond, BRANCH_INDEX_CREATOR).add(i);
                }
                else {
                    positiveOccurrences.computeIfAbsent(predicate, BRANCH_INDEX_CREATOR).add(i);
                }
            }
        }
    }

    @Nullable
    private static <X> List<ObjectDoublePair<PREDICATE<X>>> findIfCandidates(
            List<PREDICATE<X>> branches,
            Map<PREDICATE<X>, IntArrayList> positiveOccurrences,
            Map<PREDICATE<X>, IntArrayList> negativeOccurrences)
    {
        List<ObjectDoublePair<PREDICATE<X>>> candidates = null;

        for (Map.Entry<PREDICATE<X>, IntArrayList> entry : positiveOccurrences.entrySet()) {
            PREDICATE<X> predicate = entry.getKey();
            IntArrayList positiveBranches = entry.getValue();

            // Skip if no positive occurrences or no negative counterpart
            if (positiveBranches == null) continue;

            IntArrayList negativeBranches = negativeOccurrences.get(predicate);
            if (negativeBranches == null) continue;

            // Branches must be disjoint for a clean split
            if (areDisjoint(positiveBranches, negativeBranches)) {
                double score = calculateIfScore(branches, positiveBranches, negativeBranches, predicate);
                if (score > 0) {
                    if (candidates == null) {
                        candidates = new ArrayList<>(4);
                    }
                    candidates.add(PrimitiveTuples.pair(predicate, score));
                }
            }
        }

        return candidates;
    }

    private static <X> PREDICATE<X> createIfStructure(
            PREDICATE<X> splitPredicate,
            List<PREDICATE<X>> branches,
            Map<PREDICATE<X>, IntArrayList> positiveOccurrences,
            Map<PREDICATE<X>, IntArrayList> negativeOccurrences)
    {
        int branchCount = branches.size();

        // Build true branch (branches containing splitPredicate)
        Lst<PREDICATE<X>> trueBranches = new Lst<>();
        IntArrayList trueIndices = collectTrueBranches(splitPredicate, branches, positiveOccurrences, trueBranches);

        // Build false branch (branches containing NOT splitPredicate)
        Lst<PREDICATE<X>> falseBranches = new Lst<>();
        IntArrayList falseIndices = collectFalseBranches(splitPredicate, branches, negativeOccurrences, falseBranches);

        // Collect remaining branches that don't fit cleanly into true/false paths
        Lst<PREDICATE<X>> otherBranches = new Lst<>(branchCount + 1);
        for (int i = 0; i < branchCount; i++) {
            boolean inTrueBranch = (trueIndices != null && trueIndices.contains(i));
            boolean inFalseBranch = (falseIndices != null && falseIndices.contains(i));

            // Add to others if in both or neither branch
            if ((inTrueBranch && inFalseBranch) || (!inTrueBranch && !inFalseBranch)) {
                otherBranches.add(branches.get(i));
            }
        }

        // Construct final FORK combining IF with other branches
        otherBranches.add(IF.the(splitPredicate, FORK.fork(trueBranches), FORK.fork(falseBranches)));
        return FORK.fork(otherBranches);
    }

    private static <X> IntArrayList collectTrueBranches(
            PREDICATE<X> predicate,
            List<PREDICATE<X>> branches,
            Map<PREDICATE<X>, IntArrayList> occurrences,
            List<PREDICATE<X>> targetBranches)
    {
        IntArrayList branchIndices = occurrences.getOrDefault(predicate, null);
        if (branchIndices != null) {
            branchIndices.forEach(i -> {
                // Remove the tested predicate since it's guaranteed by the IF condition
                PREDICATE<X> simplifiedBranch = removePredicates(branches.get(i), Set.of(predicate));
                targetBranches.add(simplifiedBranch);
            });
        }
        return branchIndices;
    }

    private static <X> IntArrayList collectFalseBranches(
            PREDICATE<X> predicate,
            List<PREDICATE<X>> branches,
            Map<PREDICATE<X>, IntArrayList> occurrences,
            List<PREDICATE<X>> targetBranches)
    {
        IntArrayList branchIndices = occurrences.getOrDefault(predicate, null);
        if (branchIndices != null) {
            branchIndices.forEach(i -> {
                // Determine which predicate to remove from the false branch
                PREDICATE<X> predicateToRemove;
                if (predicate instanceof NOT<X> negation) {
                    // If testing NOT(q), the ELSE branch guarantees q
                    predicateToRemove = negation.cond;
                } else {
                    // If testing p, the ELSE branch guarantees NOT(p)
                    predicateToRemove = predicate.neg();
                }

                PREDICATE<X> simplifiedBranch = removePredicates(branches.get(i), Set.of(predicateToRemove));
                targetBranches.add(simplifiedBranch);
            });
        }
        return branchIndices;
    }

    /* Utility methods */

    private static boolean areDisjoint(IntArrayList first, IntArrayList second) {
        IntArrayList smaller = (first.size() < second.size()) ? first : second;
        IntArrayList larger = (smaller == first) ? second : first;
        return !larger.containsAny(smaller);
    }

    private static <X> double calculateIfScore(
            Collection<PREDICATE<X>> branches,
            IntArrayList positiveBranches,
            IntArrayList negativeBranches,
            PREDICATE<X> predicate)
    {
        final double positiveCount = (positiveBranches != null) ? positiveBranches.size() : 0;
        final double negativeCount = (negativeBranches != null) ? negativeBranches.size() : 0;
        double totalCovered = positiveCount + negativeCount;

        if (totalCovered == 0) return 0;

        double coverageRatio = totalCovered / branches.size();
        double balanceFactor = Math.min(positiveCount, negativeCount) / Math.max(positiveCount, negativeCount);
        double costEfficiency = 1 / (1.0 + predicate.cost());

        return coverageRatio * balanceFactor * costEfficiency;
    }

    /* Predicate statistics collection */

    private static <X> Map<PREDICATE<X>, PredicateStats<X>> collectPredicateStats(Iterable<Reaction<X>> reactions) {
        Map<PREDICATE<X>, PredicateStats<X>> statsMap = new HashMap<>();

        for (Reaction<X> reaction : reactions) {
            for (PREDICATE<X> predicate : reaction.pre) {
                // Use un-negated form as the key
                PREDICATE<X> basePredicate = (predicate instanceof NOT<X> negation) ? negation.cond : predicate;
                statsMap.computeIfAbsent(basePredicate, PredicateStats::new)
                        .incrementOccurrence(predicate instanceof NOT<X>);
            }
        }

        return statsMap;
    }

    /* Predicate statistics and ordering strategies */

    private static class PredicateStats<X> {
        final PREDICATE<X> predicate;
        int totalOccurrences, positiveOccurrences, negativeOccurrences;

        PredicateStats(PREDICATE<X> predicate) {
            this.predicate = predicate;
        }

        void incrementOccurrence(boolean isNegated) {
            totalOccurrences++;
            if (isNegated) negativeOccurrences++; else positiveOccurrences++;
        }

        double getFrequency() {
            return totalOccurrences;
        }

        double getBalanceFactor() {
            if (totalOccurrences == 0) return 0;
            final double min = Math.min(positiveOccurrences, negativeOccurrences);
            final double max = Math.max(positiveOccurrences, negativeOccurrences);
            return (max == 0) ? 0 : (min / max);
        }

        double getCoverageRatio(int totalReactions) {
            return getFrequency() / totalReactions;
        }

        double getCostEfficiency() {
            return 1 / (1.0 + predicate.cost());
        }

        double getWeightedScore(int totalReactions) {
            return (1 + getCoverageRatio(totalReactions) * 4) *
                    (1 + getBalanceFactor() / 10f) *
                    getCostEfficiency();
        }

        double getInformationGainEstimate(int totalReactions) {
            return getCoverageRatio(totalReactions) * getBalanceFactor();
        }
    }

    static class InformationGainStrategy<X> implements Supplier<Comparator<PREDICATE<X>>> {
        private final Map<PREDICATE<X>, PredicateStats<X>> statsMap;
        private final int totalReactions;

        InformationGainStrategy(Map<PREDICATE<X>, PredicateStats<X>> statsMap, int totalReactions) {
            this.statsMap = statsMap;
            this.totalReactions = totalReactions;
        }

        @Override
        public Comparator<PREDICATE<X>> get() {
            return Comparator
                    .comparingDouble((PREDICATE<X> p) -> {
                        var stats = getPredicateStats(p);
                        double cost = p.cost();
                        return stats.getFrequency() / ( 1 + cost * 2);
                        //return stats.getFrequency() / ( 1 + cost * 1.5f);
                        //return stats.getFrequency() / ( 1 + cost);
                        //return stats.getFrequency() / ( 1 + cost / 2);
                        //return stats.getFrequency() / ( 1 + cost / 10);
                        //double gain = stats.getInformationGainEstimate(totalReactions);
                        //return (1 + gain) / (1 + cost); // Higher values indicate better gain-to-cost ratio
                        //return gain;
                        //return gain / (1 + cost); // Higher values indicate better gain-to-cost ratio
                    })
                    .reversed()
                    .thenComparing(PREDICATE::term);
        }

        private PredicateStats<X> getPredicateStats(PREDICATE<X> predicate) {
            PREDICATE<X> baseForm = predicate instanceof NOT ? ((NOT<X>)predicate).cond : predicate;
            return statsMap.getOrDefault(baseForm, new PredicateStats<>(baseForm));
        }
    }
}