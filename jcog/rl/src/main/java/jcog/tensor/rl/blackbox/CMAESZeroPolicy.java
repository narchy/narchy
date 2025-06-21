package jcog.tensor.rl.blackbox;

import jcog.tensor.rl.blackbox.PopulationPolicy.CMAESPopulation;

import java.util.Random;

/** for learning constant default parameter values.
 *  chooses outputs directly, ignores inputs.
 *
 *  TODO correctly apply actual, not suggested (by ES), action vectors (mean) as ES feedback
 * */
public class CMAESZeroPolicy extends BlackboxPolicy {

    private CMAESPopulation population;

    public CMAESZeroPolicy(int numOutputs, int populationSize, int periodMin, int periodMax) {
        super(numOutputs, populationSize, periodMin, periodMax);
    }

    @Override
    public void clear(Random rng) {
        if (population==null)
            population = new CMAESPopulation();
        population.initialized = false;
        population.init(outputs, populationSize, 0, 1);
        super.clear(rng);
    }


    @Override
    protected double[] next() {
        return population.get(currentIndividual);
    }

    @Override
    protected double[] best() {
        return population.best();
    }

    @Override
    protected void commitPopulation(double[] rewards) {
        population.commit(rewards);
    }


    @Override
    protected void commitIndividual(double[] actualActions, double reward) {
        population.set(actions, currentIndividual);
    }

}