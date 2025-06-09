package jcog.learn;

import jcog.activation.LeakyReluActivation;
import jcog.activation.SigLinearActivation;
import jcog.activation.SigmoidActivation;
import jcog.nn.BackpropRecurrentNetwork;
import jcog.nn.NEAT;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static jcog.learn.MLPTest.xor2Test;

class NEATTest {

    @Test void neatXOR2() {
        xor2Test(new NEAT(
        2, 1, 50, 4*2 /* to be safe until deduplication */));
    }

    @Disabled
    @Test void freeformBackpropXOR2_3hidden() {
        freeformBackpropXOR2(3);
    }

    @Test void freeformBackpropXOR2_4hidden() {
        freeformBackpropXOR2(4);
    }
    @Test void freeformBackpropXOR2_5hidden() {
        freeformBackpropXOR2(5);
    }

    private static void freeformBackpropXOR2(int hiddens) {
        BackpropRecurrentNetwork n = new BackpropRecurrentNetwork(2, 1, hiddens, 2);
        xor2Test(n
            .activationFn(
                LeakyReluActivation.the,
                //SigmoidActivation.the,
                SigLinearActivation.the
            )
        );
    }

    @Test void freeformBackpropBackpropIRIS_i2() {
        freeformBackpropIRIS(0, 2);
    }
    @Test void freeformBackpropBackpropIRIS_i3() {
        freeformBackpropIRIS(0, 3);
    }
    @Test void freeformBackpropBackpropIRIS_i4() {
        freeformBackpropIRIS(0, 4);
    }

    private static void freeformBackpropIRIS(int hiddens, int iters) {
        BackpropRecurrentNetwork n = new BackpropRecurrentNetwork(4, 3, hiddens, iters);
        MLPTest.irisTest(n.activationFn(
                        //LeakyReluActivation.the,
                        //ReluActivation.the,
                        SigLinearActivation.the,
                        SigmoidActivation.the
                        //SigLinearActivation.the
                )
        );
    }
}