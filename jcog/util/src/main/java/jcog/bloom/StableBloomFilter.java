package jcog.bloom;

import jcog.bloom.hash.Hasher;
import jcog.random.RandomBits;

import java.util.Random;

/**
 * Stable Bloom Filters continuously "reset" random fields in the filter.
 * Deng and Rafiei have shown that by doing this, the FPR can be stabilised [1].
 * The disadvantage of this approach is that it introduces false negatives.
 * <p>
 * Created by jeff on 14/05/16.
 */
public class StableBloomFilter<E> extends MetalBloomFilter<E> implements CountingLeakySet<E> {

    private final RandomBits rng;

    /**
     * Set the unlearning rate to make the {@link LeakySet} stable. The unlearning rate represents
     * a percentage of filter cells that will be "unlearned" with each write operation.
     *
     * @param unlearningRate Must be between 0.0 and 1.0.
     * @return {@link BloomFilterBuilder} For chaining.
     */
    private final int forgetRate;

    public StableBloomFilter(int numberOfCells,
                             int numberOfHashes,
                             float forgetRate,
                             Random rng,
                             Hasher<? super E> hasher) {
        super(hasher, numberOfCells, numberOfHashes);
        this.forgetRate = (int) Math.ceil(this.cap * forgetRate);
        this.rng = new RandomBits(rng);
    }

    /**
     * if the element isnt contained, add it. return true if added, false if possibly present.
     */
    public boolean addIfMissing(E element) {
        return addIfMissing(element, 1);
    }

    public boolean addIfMissing(E element, float unlearnIfNew) {
        int[] hash = hash(element);
        if (contains(hash))
            return false;

        if (unlearnIfNew > 0)
            forget(unlearnIfNew);

        add(hash);
        return true;
    }

    @Override
    public void remove(E element) {
        remove(hash(element));
    }

    public void remove(int[] indices) {
        for (var i : indices)
            decrement(i);
    }

    public void forget(float forgetFactor) {
        double nForget = Math.ceil(forgetRate * forgetFactor);
        for (int i = 0; i < nForget; i++)
            decrement(rng.nextInt(cap));
    }


    private void decrement(int idx) {
        byte[] c = this.cells;
        if (c[idx] > 0)
            c[idx] -= 1;
    }

}
