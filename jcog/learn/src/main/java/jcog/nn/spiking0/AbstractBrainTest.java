package jcog.nn.spiking0;

import jcog.nn.spiking0.build.RandomWiring;
import jcog.nn.spiking0.critterding.CritterdingBrain;
import jcog.nn.spiking0.learn.STDPSynapseLearning;
import jcog.nn.spiking0.neuron.IzhikevichNeuron;
import jcog.nn.spiking0.neuron.RealtimeBrain;
import jcog.random.XoRoShiRo128PlusRandom;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static jcog.Str.n4;
import static org.junit.jupiter.api.Assertions.assertEquals;

class AbstractBrainTest {

    @Test
    void izhikevich() {
        RealtimeBrain b = new RealtimeBrain();

        int N = 100;
        int S = N * N / 4;
        var rng = new XoRoShiRo128PlusRandom();

        for (int i = 0; i < N; i++)
            b.neurons.add(new IzhikevichNeuron().regular());

        for (int i = 0; i < S; i++) {
            int from = rng.nextInt(N);
            int to = rng.nextInt(N);
            var F = b.neurons.get(from);
            var T = b.neurons.get(from);
            if (T instanceof IzhikevichNeuron I)
                I.synapses.add(new AbstractSynapse<>(F, T, rng.nextFloat()));
        }

        var s = new STDPSynapseLearning();

        int iters = 100;

        for (int i = 0; i < iters; i++) {
            b.forward();
        }
    }

    @Test
    void critterding() {
        int inputs = 16;
        int outputs = 16;

        int neurons = 256;
        int minSynapses = 8;
        int maxSynapses = 12;

        CritterdingBrain b = new CritterdingBrain(inputs, outputs);

        new RandomWiring(
            neurons, minSynapses, maxSynapses, 0.25, 0.25
            //10000, 2, 200, 0.25f, 0.1f
        ).accept(b);

        assertEquals(b.neuronCount(), inputs + outputs + neurons);

        Random rng = new XoRoShiRo128PlusRandom();

        for (int i = 0; i < 100; i++) {
            if (i == 0) {
                for (InputNeuron sn : b.getInputs()) {
                    sn.setInput(rng.nextDouble(-1, +1));
                }
            }
            b.forward();
//            for (int j = 0; j < outputs; j++)
//                System.out.print(b.motor.get(j).getOutput() + " ");
//            System.out.println();

        }

    }

    /** TODO finish */
    @Test void testXOR() {
        int inputs = 2;
        int outputs = 1;

        int neurons = 16;
        int minSynapses = 2;
        int maxSynapses = 8;

        CritterdingBrain b = new CritterdingBrain(inputs, outputs);

        new RandomWiring(
                neurons, minSynapses, maxSynapses, 0.5, 0.25
                //10000, 2, 200, 0.25f, 0.1f
        ).accept(b);

        Random rng = new XoRoShiRo128PlusRandom();

        var ins = b.getInputs();
        var outs = b.getOutputs();
        ins.get(0).setInput(-1);
        ins.get(1).setInput(-1);
        for (int i = 0; i < 100; i++) {

            b.forward();

//            for (var s : b.synapses)
//                System.out.print(n4(s.weight) + " ");
//            System.out.println();

            for (var k : b.neuron)
                System.out.print(n4(k.potential) + "|" + n4(k.getOutput()) + " ");
            System.out.println();

            for (int j = 0; j < outputs; j++)
                System.out.print(b.motor.get(j).getOutput() + " ");
            System.out.println();

        }

    }
}