/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jcog.nndepr.spiking0.build;


import jcog.nndepr.spiking0.AbstractNeuron;
import jcog.nndepr.spiking0.InterNeuron;
import jcog.nndepr.spiking0.critterding.CritterdingBrain;
import jcog.nndepr.spiking0.critterding.CritterdingNeuron;
import jcog.random.XoRoShiRo128PlusRandom;

import java.util.*;

/**
 * @author Sue
 */
public class RandomWiring implements BrainWiring {
    final Random rng = new XoRoShiRo128PlusRandom();

    private final int numNeurons;
    private final int minSynapses;
    private final int maxSynapses;
    private final double percentChanceMotorNeuron;
    private final double percentChanceSensorySynapse;
    double defaultPotentialDecay =
            //1;
            0.99f;

    public RandomWiring(int numNeurons, int minSynapses, int maxSynapses, double percentChanceSensorySynapse, double percentChanceMotorNeuron) {
        this.numNeurons = numNeurons;
        this.minSynapses = minSynapses;
        this.maxSynapses = maxSynapses;
        this.percentChanceMotorNeuron = percentChanceMotorNeuron;
        this.percentChanceSensorySynapse = percentChanceSensorySynapse;

    }


    @Override
    public void accept(CritterdingBrain b) {
        //b.wireRandomly(numNeurons, minSynapses, maxSynapses, percentChanceSensorySynapse, percentChanceMotorNeuron, potentialDecay)
        //}
        //public CritterdingBrain wireRandomly(int numNeurons, int minSynapses, int maxSynapses, double percentChanceSensorySynapse, double percentChanceMotorNeuron, double potentialDecay) {
        final List<NeuronBuilder> neuronBuilders = new ArrayList();

        // determine number of neurons this brain will start with
        //int numNeurons = (int) Math.round(Maths.random(minNeuronsAtBuildtime, maxNeuronsAtBuildtime));


        // create the architectural neurons
        for (int i = 0; i < numNeurons; i++) {
            b.newRandomNeuronBuilder(neuronBuilders, percentChanceMotorNeuron, rng);
        }

        // create architectural synapses
        for (NeuronBuilder n : neuronBuilders) {
            // determine amount of synapses this neuron will start with
            int SynapseAmount = rng.nextInt(minSynapses, maxSynapses);

            // create the architectural neurons
            for (int j = 0; j < SynapseAmount; j++) {
                b.newRandomSynapseBuilder(n, percentChanceSensorySynapse);
            }
        }

        Map<InterNeuron, List<SynapseBuilder>> built = new HashMap();

        // create all runtime neurons
        for (NeuronBuilder nb : neuronBuilders) {
            CritterdingNeuron n = nb.newNeuron();
            b.addNeuron(n);
            built.put(n, nb.synapseBuilders);
        }

        // create their synapses & link them to their inputneurons
        for (Map.Entry<InterNeuron, List<SynapseBuilder>> entry : built.entrySet()) {

            for (SynapseBuilder sb : entry.getValue()) {
                AbstractNeuron i;

                if (sb.isSensorNeuron) {
                    // sensor neuron id synapse is connected to
                    i = b.randomSenseNeuron(rng);
                } else {
                    // if not determine inter neuron id
                    // as in real life, neurons *CAN* connect to themselves
                    i = b.randomInterNeuron(rng);
                }


                b.newSynapse(i, entry.getKey(), sb.weight);
            }


//            if (n.getMotor() != null) {
//                b.newSynapse(n, n.getMotor(), 1.0);
//            }
        }

//	//cerr << "total neurons: " << totalNeurons << "total synapses: " << totalSynapses << endl;

    }


}
