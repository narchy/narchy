package jcog.tensor.rl.dqn.replay;

import jcog.tensor.rl.dqn.Policy;

/** a replay experience */
public class Experience {
    /** time */
    public final long t;

    /** reward */
    public final float r;

    /** input state */
    public final double[] x;

    /** previous input state */
    public final double[] x0;

    /** previous action */
    public final double[] a;

    /**
     * SARS
     * https://cs.stanford.edu/people/karpathy/reinforcejs/lib/rl.js
     * learnFromTuple: function(s0, a0, r0, s1, a1)
     */
    public Experience(long t, double[] x0, double[] a, float r, double[] x) {
        this.x0 = x0;
        this.a = a;
        this.r = r;
        this.x = x;
        this.t = t;
    }

    public double[] learn(Policy policy, float alpha) {
        return policy.learn(x0, a, r, x, alpha);
    }

    @Override public int hashCode() {
        return Long.hashCode(t);
    }

//    @Override public boolean equals(Object x) {
//        return this==x;
//    }

}