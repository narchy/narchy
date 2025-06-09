package jcog.decision.impurity;


import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static com.google.common.math.DoubleMath.log2;

/**
 * Entropy calculator. -p log2 p - (1 - p)log2(1 - p) - this is the expected information, in bits, conveyed by somebody
 * telling you the class of a randomly drawn example; the purer the set of examples, the more predictable this message
 * becomes and the smaller the expected information.
 *
 * @author Ignas
 */
public class EntropyCalculator implements ImpurityCalculator {

    /**
     * {@inheritDoc}
     */
    @Override
    public <K, V> double impurity(K value, Supplier<Stream<Function<K, V>>> splitData) {
        List<V> labels = splitData.get().map(x -> x.apply(value)).distinct().toList();
        int s = labels.size();
        if (s > 1) {
            // TODO this can be done faster by comparing each all at once
            return labels.stream().mapToDouble(l -> {
                double p = ImpurityCalculator.empiricalProb(value, splitData.get(), l);
                return -1 * p * log2(p) - ((1 - p) * log2(1 - p));
            }).sum();
        } else if (s == 1)
            return 0;
        else
            throw new IllegalStateException();
    }

}