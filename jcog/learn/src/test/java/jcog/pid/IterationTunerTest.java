package jcog.pid;

import jcog.Util;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicLong;

/** TODO finish as complete unit test */
@Disabled
class IterationTunerTest {

    @Test
    void testIterTuner() {
        AtomicLong targetTime = new AtomicLong(1_000_000 * 50);
        AtomicLong loopTime = new AtomicLong(5_000_000);
        IterationTuner t = new IterationTuner(new RandomBits(new XoRoShiRo128PlusRandom())) {

            @Override
            protected void next(int iterations) {
                iterations = Util.clamp(iterations, 1, 20);
                for (int i = 0; i < iterations; i++) {
                    Util.sleepNS(loopTime.get());
                }
            }

            @Override
            public long targetPeriodNS() {
                return targetTime.get();
            }
        };

        t.profileProb = 0.1f;

        for (int i = 0; i < 100; i++) t.run();

        targetTime.set(targetTime.get()/2);

        for (int i = 0; i < 100; i++) t.run();

    }
}