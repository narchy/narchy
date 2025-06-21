package jcog.tensor.rl.dqn.replay;

import jcog.tensor.rl.dqn.PolicyAgent;

import java.util.Random;
import java.util.random.RandomGenerator;

/**
 * experience replay buffer
 */
public abstract class Replay {
    private static final boolean ageDecay = false;
    float replayAlpha;
    float rememberProb;
    int trainIters;
    public long t;

    protected Replay(float replayAlpha, float rememberProb, int trainIters) {
        this.replayAlpha = replayAlpha;
        this.rememberProb = rememberProb;
        this.trainIters = trainIters;
    }

    protected void playback(PolicyAgent agent) {

        int trainIters = this.trainIters;
        double alpha = replayAlpha();
        int capacity = capacity();
        double dur = capacity / rememberProb;

//        float memoryUsage = ((float)size()) / capacity;
//        float memoryUsageFactor = sqr(memoryUsage);

        //trainIters = Math.min(size(), trainIters); //(int) Math.ceil(trainIters * memoryUsage);

        for (int i = 0; i < trainIters; i++) {
            Experience m = sample(agent.rng);
            float pri;
            if (ageDecay) {
                long age = t - m.t;
                double ageDiscount = Math.exp(-age / dur);
                pri = (float) (alpha * ageDiscount);
            } else {
                pri = (float) alpha;
            }

//            //normalize by memory utilization
//            pri *= memoryUsageFactor;

            rerun(m, pri, agent);
        }

    }

    protected void rerun(Experience m, float pri, PolicyAgent agent) {
        agent.run(m, null, pri);
    }

    protected abstract Experience sample(Random rng);

    abstract public int capacity();

    abstract public int size();

    float replayAlpha() {
        return this.replayAlpha;
        //* ((float)memory.size())/capacity; //discount for incompletely filled memory
    }

    public void run(PolicyAgent agent, double[] actionPrev, float reward, double[] x, double[] xPrev, double[] actionNext) {
        if (size() > 0)
            playback(agent);

        tryRemember(agent, xPrev, actionPrev, reward, x, actionNext);

        t++;
    }

    private void tryRemember(PolicyAgent agent, double[] xPrev, double[] actionPrev, float reward, double[] x, double[] actionNext) {
        int s = size(), c = capacity();
        if (s < c || agent.rng.nextFloat() <= rememberProb) {

            if (s >= c)
                pop(agent.RNG);

            //TODO clone() e
            add(new Experience(t, xPrev.clone(), actionPrev.clone(), reward, x.clone()), actionNext);
        }
    }

    protected abstract void add(Experience m, double[] qNext);

    @Deprecated protected abstract void pop(RandomGenerator rng);


}