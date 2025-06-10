package nars.memory;

import nars.*;
import org.junit.jupiter.api.Test;

class CaffeineMemoryTest {

    @Test
    void testDynamicWeight() throws Narsese.NarseseException {
        StringBuilder log = new StringBuilder();
        CaffeineMemory index;
        NAR n = new NARS().memory(
            index = new CaffeineMemory(4000, (w) -> {
                int newWeight = 1000 * (w.beliefs().taskCount());
                log.append("weigh ").append(w).append(' ').append(newWeight).append('\n');






                return newWeight;
            }) {
                @Override
                public Concept get(Term x, boolean createIfMissing) {
                    log.append("get ").append(x).append(createIfMissing ? " createIfMissing\n" : "\n");
                    return super.get(x, createIfMissing);
                }
            }).get();


        n.believe("(x-->y).");

        n.believe("(x-->y).");

        System.out.println(log);

    }
}