package jcog.decide;

import jcog.Is;

import java.util.random.RandomGenerator;


@Is("Fitness_proportionate_selection")
public class DecideRoulette implements Decide {

    private final RandomGenerator rngFloat;

    public DecideRoulette(RandomGenerator rng) {
        this.rngFloat = rng;
    }

    @Override
    public int applyAsInt(float[] value) {
        return Roulette.selectRoulette(rngFloat, value);
    }

}