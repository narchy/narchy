/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jcog.nndepr.spiking0.critterding;

import jcog.TODO;
import jcog.data.list.Lst;
import jcog.nndepr.spiking0.*;
import jcog.nndepr.spiking0.build.NeuronBuilder;
import jcog.nndepr.spiking0.build.SynapseBuilder;

import java.util.*;

/**
 * java port of critterding's BRAINZ system
 *+
 * https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/64482/eth-6334-01.pdf;jsessionid=6FB71F0B03B0A4391CD8AF71897F78D6?sequence=1
 */
@Deprecated public class CritterdingBrain extends AbstractBrain {

    @Deprecated public final Map<AbstractNeuron, AbstractSynapse<AbstractNeuron>[]> neuronSynapses = new HashMap();

    public final List<AbstractSynapse<AbstractNeuron>> synapses = new Lst<>();
    public final List<CritterdingNeuron> neuron = new Lst<>();

    public double alpha = 1;

    double percentChanceInhibitoryNeuron;      // percent chance that when adding a new random neuron, it's inhibitory
    double percentChanceInhibitorySynapses;    //    // percent chance that when adding a new random neuron, it has inhibitory synapses
    double percentChancePlasticNeuron; //    // percent chance that when adding a new random neuron, it is has synaptic plasticity

    double plasticityStrengthen = 0.1f;    //    // min/max synaptic plasticity strengthening factor

    /** FORGETTING */
    double plasticityWeaken =
        //0;
        0.005f;

    double potentialDecay = 0.01f;

    double firingThreshold = 1;

    public CritterdingBrain(int numInputs, int numOutputs) {
        this();
        for (int i = 0; i < numInputs; i++) newInput();
        for (int i = 0; i < numOutputs; i++) newOutput();
    }

    private CritterdingBrain() {
        super();

        percentChanceInhibitoryNeuron =
                0.5;
                //0.25f;
        percentChanceInhibitorySynapses =
                //0;
                0.5;
                //0.25f;

        percentChancePlasticNeuron = 1;
    }

    @Deprecated private AbstractSynapse<AbstractNeuron>[] incomingSynapses(final InterNeuron n) {
        return neuronSynapses.get(n);
    }

    @Override public void forward() {

        for (var m : motor)
            m.clear();

        for (var s : synapses)
            s.invalid = true;

        double potentialFactor = 1 - potentialDecay;
        double plasticityWeakenFactor = 1 - plasticityWeaken;
        //double plasticityStrengthenFactor = 1 + plasticityStrengthen;
        for (var n : neuron)
            n.forward(incomingSynapses(n),
                alpha, potentialFactor,
                plasticityStrengthen,
                firingThreshold);

        if (plasticityWeaken > 0) {
            double synapseDecay = 1 - plasticityWeaken;

//            double l1 = synapseWeightsL1() / synapses.size();
//            double synapseDecay = //(1 - plasticityWeaken) / (wEpsilon + l1);
//                    1 / (1 + plasticityWeaken * l1);
            //System.out.println(n4(l1) + " -> " + n4(synapseDecay));
            for (var s : synapses)
                s.weight *= synapseDecay;
        }

        // commit outputs at the end
        for (var n : neuron) {
            double o = n.output = n.nextOutput;
            OutputNeuron mn = n.motor;
            if (mn!=null)
                mn.activate(o);
        }

    }

    private double synapseWeightsL1() {
        double sum = 0;
        for (var s : synapses)
            sum += Math.abs(s.weight);
        return sum;
    }

    public OutputNeuron randomMotorNeuron(Random rng) {
        return motor.get(rng.nextInt(motor.size()));
    }

    public InterNeuron randomInterNeuron(Random rng) {
        return neuron.get(rng.nextInt(neuron.size()));
    }

    public InputNeuron randomSenseNeuron(Random rng) {
        return sense.get(rng.nextInt(sense.size()));
    }

    // build time functions
    public void newRandomNeuronBuilder(final List<NeuronBuilder> b, double percentChanceMotorNeuron, Random rng) {
        // new architectural neuron
        NeuronBuilder n = new NeuronBuilder();

        if (rng.nextFloat() <= percentChanceMotorNeuron)
            n.motor = randomMotorNeuron(rng);

        n.isInhibitory = rng.nextFloat() <= percentChanceInhibitoryNeuron;
        n.isPlastic = Math.random() <= percentChancePlasticNeuron;

        b.add(n);
    }

    public void newRandomSynapseBuilder(NeuronBuilder bn, double percentChanceSensorySynapse) {
        float weight =
                (Math.random() <= percentChanceInhibitorySynapses) ? -1 : +1;

        // new architectural synapse
        //  is it connected to a sensor neuron ?
        //  < 2 because if only 1 archneuron, it can't connect to other one

        bn.synapseBuilders.add(
            new SynapseBuilder(weight,
            Math.random() <= percentChanceSensorySynapse));
    }

    public List<CritterdingNeuron> getInter() {
        return neuron;
    }

    public int neuronCount() {
        return interNeuronCount() + sense.size() + motor.size();
    }

    public int interNeuronCount() {
        return neuron.size();
    }

    public int synapseCount() {
        return synapses.size();
    }

    public void removeSynapse(AbstractSynapse<AbstractNeuron> s) {
        throw new TODO();
//        if (synapses.remove(s)) {
//            List<CritterdingSynapse> incoming = neuronSynapses.get(s.target);
//            if (!incoming.remove(s)) {
//                System.err.println("Error removing " + s);
//            }
//        }
    }

    public AbstractSynapse<AbstractNeuron> newSynapse(AbstractNeuron from, AbstractNeuron to, double weight) {
        AbstractSynapse<AbstractNeuron> s = new AbstractSynapse(from, to, weight);
        synapses.add(s);

        AbstractSynapse[] incoming = neuronSynapses.get(to);
        if (incoming == null) {
            incoming = new AbstractSynapse[]{s};
            neuronSynapses.put(to, incoming);
        } else {
            incoming = Arrays.copyOf(incoming, incoming.length + 1);
            incoming[incoming.length - 1] = s;
            neuronSynapses.put(to, incoming);
        }

        return s;
    }

//		// load save architecture (serialize)
//			void			setArch(string* content);
//			string*			getArch();
//
//    // build commands
//    // functions
//    void copyFrom(const   Brainz& otherBrain);
//			void			mergeFrom(const Brainz& otherBrain1, const Brainz& otherBrain2);
//    private void newMotorSynapse(InterNeuron from, MotorNeuron to) {
//        add(new MotorSynapse(from, to));
//    }
//    public void forwardUntilAnswer() {
//        //		neuronsFired = 0;
//
//		// clear Motor Outputs
//		for ( unsigned int i=0; i < numberOfOutputs; i++ )
//			Outputs[i].output = false;
//
//		// clear Neurons
//		for ( unsigned int i=0; i < totalNeurons; i++ )
//		{
//			Neurons[i].output = 0;
//			Neurons[i].potential = 0.0f;
//		}
//
//		unsigned int counter = 0;
//		bool motorFired = false;
//
//		while ( counter < 1000 && !motorFired )
//		{
//			for ( unsigned int i=0; i < totalNeurons; i++ )
//			{
//				NeuronInterz* n = &Neurons[i];
//
//				n->process();
//
//				// if neuron fires
//				if ( n->waitoutput != 0 )
//				{
//					neuronsFired++;
//
//					// motor neuron check & exec
//					if ( n->isMotor )
//					{
//						motorFired = true;
//						*Outputs[n->motorFunc].output = true;
//						//cerr << "neuron " << i << " fired, motor is " << Neurons[i]->MotorFunc << " total now " << Outputs[Neurons[i]->MotorFunc]->output << endl;
//					}
//				}
//			}
//			// commit outputs at the end
//			for ( unsigned int i=0; i < totalNeurons; i++ ) Neurons[i].output = Neurons[i].waitoutput;
//
//			counter++;
//		}
//    }
//    public void removeObsoleteMotorsAndSensors() {
//		for ( int i = 0; i < (int)ArchNeurons.size(); i++ )
//		{
//			ArchNeuronz* an = &ArchNeurons[i];
//			// disable motor neurons
//			if ( an->isMotor )
//			{
//				if ( findMotorNeuron( an->motorID ) == -1 )
//				{
//					an->isMotor = false;
//				}
//			}
//
//			// disable sensor inputs
//			for ( int j = 0; j < (int)an->ArchSynapses.size(); j++ )
//			{
//				ArchSynapse* as = &an->ArchSynapses[j];
//				if ( as->isSensorNeuron )
//				{
//					if ( findSensorNeuron( as->neuronID ) == -1 )
//					{
//						an->ArchSynapses.erase( an->ArchSynapses.begin()+j );
//						j--;
//					}
//				}
//			}
//		}
//    }

    public void addNeuron(CritterdingNeuron n) {
        neuron.add(n);
    }

    public void removeNeuron(CritterdingNeuron n) {
        neuron.remove(n);
        this.neuronSynapses.remove(n);
    }

    public AbstractSynapse<AbstractNeuron> getWeakestSynapse() {
        double minWeight = 0;
        AbstractSynapse<AbstractNeuron> m = null;
        for (AbstractSynapse<AbstractNeuron> s : synapses) {
            if ((Math.abs(s.weight) < minWeight) || (m == null)) {
                minWeight = s.weight;
                m = s;
            }
        }
        return m;
    }

    public void removeDisconnectedNeurons() {
        List<CritterdingNeuron> rem = new LinkedList();
        for (CritterdingNeuron i : neuron) {
            AbstractSynapse<AbstractNeuron>[] ll = incomingSynapses(i);
            if (ll != null)
                if (ll.length > 0)
                    continue;
            rem.add(i);
        }
        for (CritterdingNeuron in : rem) {
            removeNeuron(in);
        }
    }

    public void removeWeakestSynapses(int deadSynapsesRemoved) {
        int toRemove = Math.min(synapseCount(), deadSynapsesRemoved);
        for (int i = 0; i < toRemove; i++) {
            removeSynapse(getWeakestSynapse());
        }
    }


}
