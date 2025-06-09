package jcog.tensor.rl.misc;

import jcog.Fuzzy;
import jcog.Util;
import jcog.nn.spiking0.build.RandomWiring;
import jcog.nn.spiking0.critterding.CritterdingBrain;
import jcog.tensor.rl.dqn.Policy;
import org.jetbrains.annotations.Nullable;

import java.util.Random;

public class SpikingPolicy implements Policy {
    public CritterdingBrain b = null;
    int internalIterations =
        //1;
        4;

    private boolean autoClearPotential =
        true;
        //false;

    public SpikingPolicy() {

    }

    @Override
    public void clear(Random rng) {
        b = null;
    }

    @Override
    public double[] learn(@Nullable double[] xPrev, double[] actionPrev, double reward, double[] x, float pri) {
        int inputs = x.length, outputs = actionPrev.length;
        if (b == null) {
            b = new CritterdingBrain(inputs, outputs);
            int hidden =
                    //inputs * outputs;
                    (int)Util.mean(inputs,outputs)*2;
            new RandomWiring(hidden,
                    3, 12,
                    //3, hidden/8,
                    0.5f, 0.25f).accept(b);
        }

        if (autoClearPotential) {
            for (var i : b.getInter())
                i.clear();
        }

        for (int i = 0; i < inputs; i++)
            b.getInput(i).setInput(Fuzzy.polarize(x[i])/internalIterations);

        b.alpha = reward; //reward modulated learning

        double[] out = new double[outputs];
        for (int i = 0; i < internalIterations; i++) {
            b.forward();
            for (int o = 0; o < outputs; o++)
                out[o] += b.getOutput(o).getOutput();
        }

        //Util.normalize(out, out.length);
//        for (int o = 0; o < outputs; o++)
//            out[o] /= internalIterations;
        double mo = Math.max(Math.abs(Util.min(out)), Math.abs(Util.max(out))); if (mo!=0) Util.mul(out, 1/mo);

//        System.out.println(n2(out));

        //Util.mul(1f/internalIterations, out);
        for (int o = 0; o < outputs; o++)
            out[o] = Fuzzy.unpolarize(Util.clampSafe(out[o], -1, +1));

        return out;
    }
}
