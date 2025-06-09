/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package jcog.nn.spiking0;

import jcog.data.list.Lst;

import java.util.List;

/**
 *
 * @author me
 */
public abstract class AbstractBrain /* extends AbstractLocalBrain<SenseNeuron, MotorNeuron, CritterdingNeuron, SimpleSynapse<CritterdingNeuron>>*/ {
    final public List<InputNeuron> sense = new Lst();
    final public List<OutputNeuron> motor = new Lst();

    public double t = 0;
    public double timeStep = 0.1;

    abstract public void forward();

    public InputNeuron getInput(final int i) {
        return sense.get(i);
    }

    public List<InputNeuron> getInputs() {
        return sense;
    }

    //    MotorNeuron motor(int i) {
    //        return motor.get(i);
    //    }
    //
    //    SenseNeuron sense(int i) {
    //        return sense.get(i);
    //    }
    public int getNumInputs() {
        return sense.size();
    }

    public void addInput(InputNeuron s) {
        sense.add(s);
    }

    public void addOutput(OutputNeuron m) {
        motor.add(m);
    }

    public int getNumOutputs() {
        return motor.size();
    }

    public OutputNeuron getOutput(final int i) {
        return motor.get(i);
    }

    public List<OutputNeuron> getOutputs() {
        return motor;
    }

    public InputNeuron newInput() {
        InputNeuron s = new InputNeuron();
        sense.add(s);
        return s;
    }

    public OutputNeuron newOutput() {
        OutputNeuron m = new OutputNeuron();
        motor.add(m);
        return m;
    }
    
}
