package jcog.nn.spiking0.learn;

import jcog.nn.spiking0.neuron.IFNeuron;
import jcog.nn.spiking0.neuron.SpikingNeuron;

/** Heuristic implementation of Hebbian learning
 *  from: https://github.com/dieuwkehupkes/Spiking-Network/tree/java/src/izhikevich/spikingnetwork
 * */
public class HebbianSynapseLearning extends SpikeSynapseLearning {

	double activationRate =
		//1;
		//0.1f;
		0.1f;
		//10.0;

	double deactivationRate =
		activationRate/2;

	/**
	 * Update existing connection based on recent activity
	 * of both neurons.
	 */
	@Override
	protected double update(SpikingNeuron a, SpikingNeuron b) {
		double aa = a.getOutput(), bb = b.getOutput();
		double polarity =
		  	  (a instanceof IFNeuron as ? (as.positive ? +1 : -1) : 1 )
			* (b instanceof IFNeuron bs ? (bs.positive ? +1 : -1) : 1 );
//		double polarity = 1;

		boolean firing = aa != 0; //HACK

		if (firing) {
			// check if n2 contributed to firing
			return polarity * activationRate * increment(a.timeFired, b.timeFired);
		} else if (b.timeFired <= a.timeFired) {
			// weight was updated when n1 fired last time
			return 0;
		} else {
			// n1 was not activated through activation of n2
			return polarity * deactivationRate *
					+increment(a.timeFired, b.timeFired)
					//increment(b.lastFireTime, a.lastFireTime)
			;
		}
	}

//	public double updateNonExistingConnection(IzhikevichNeuron n1, IzhikevichNeuron n2) {
//		return 0.0;
//	}

	/**
	 * Compute increment based on how far apart
	 * the two neurons fired.
	 */
	private double increment(double fireTime1, double fireTime2) {
		if (fireTime1 == fireTime2) {   // no causal relationship
			return 0;
		} else {        // if firetime2 is close enough to firetime1, increase weight
			return 1/(fireTime1-fireTime2);
		}
	}

}