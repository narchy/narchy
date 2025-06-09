package jcog.pid;

import org.junit.jupiter.api.Test;

import static jcog.Str.n4;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MiniPIDTest {

    @Test
    void test1() {

        MiniPID miniPID = new MiniPID(0.25, 0.01, 0.4);
        //miniPID.outRange(-10,10);

        double actual = 0;
        double target = Double.NaN;
        for (int i = 0; i < 200; i++) {
            target = i == 60 ? 50 : 100;
            double output = miniPID.out(actual, target);
            actual += output;
            System.out.println(n4(output) + ' ' + n4(actual));
        }
        assertTrue(Math.abs(target - actual) < 0.05f);
    }
}