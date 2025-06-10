package nars.nar;

import com.google.common.primitives.Longs;
import nars.*;
import nars.term.Termed;
import nars.term.util.Testing;
import nars.test.TestNAR;
import nars.util.RuleTest;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static java.lang.System.out;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Created by me on 8/7/15.
 */
class NARTest {


    @Test
    @Disabled
    void testMemoryTransplant() throws Narsese.NarseseException {


        NAR nar = NARS.tmp();


        nar.input("<a-->b>.", "<b-->c>.").run(25);

        nar.input("<a-->b>.", "<b-->c>.");
        nar.stop();

        assertTrue(nar.memory.size() > 5);

        int nc;
        assertTrue((nc = nar.memory.size()) > 0);


        NAR nar2 = NARS.tmp();

        assertTrue(nar.time() > 1);


        assertEquals(nc, nar2.memory.size());


    }

    @Test
    void testFluentBasics() throws Narsese.NarseseException {
        AtomicInteger cycCount = new AtomicInteger(0);

        NAR m = NARS.tmp()
                .input("<a --> b>.", "<b --> c>.");
        m.stopIf(() -> false);
        m.onCycle(nn -> cycCount.incrementAndGet());
        int frames = 32;
        m.run(frames);

        NAR n = NARS.tmp()
                .input("<a --> b>.", "<b --> c>.");
        m.stopIf(() -> false);
        n.onCycle(nn -> cycCount.incrementAndGet());

        assertEquals(frames, cycCount.get());


    }

    @Test
    void testBeforeNextFrameOnlyOnce() {
        AtomicInteger b = new AtomicInteger(0);
        NAR n = NARS.shell();

        n.runLater(b::incrementAndGet);
        n.run(4);
        assertEquals(1, b.get());

    }

    @Test
    void testConceptInstancing() throws Narsese.NarseseException {
        NAR n = NARS.tmp();

        Termed a = $.$("a");
        assertNotNull(a);
        Termed a1 = $.$("a");
        assertEquals(a, a1);

        String statement1 = "<a --> b>.";
        n.input(statement1);
        n.run(4);

        n.input(" <a  --> b>.  ");
        n.run(1);
        n.input(" <a--> b>.  ");
        n.run(1);

        String statement2 = "<a --> c>.";
        n.input(statement2);
        n.run(4);

        Termed a2 = $.$("a");
        assertNotNull(a2);

        Concept ca = n.conceptualize(a2);
        assertNotNull(ca);


    }

    @Test
    void testCycleScheduling() {
        NAR n = NARS.tmp();

        int[] runs = {0};

        long[] events = {2, 4, 4 /* test repeat */};
        for (long w : events) {
            n.runAt(w, () -> {
                assertEquals(w, n.time());
                runs[0]++;
            });
        }

        n.run(1);
        assertEquals(0, runs[0]); /* nothing yet in that 1st cycle */


        n.run((int) Longs.max(events));
        assertEquals(events.length, runs[0]);
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "a:b. b:c.",
            "a:b. b:c. c:d! a@",
            "d(x,c). :|: (x<->c)?",
            "((x &&+1 b) &&+1 c). :|: (c && --b)!"
    })
    void testNARTaskSaveAndReload(String input) throws Narsese.NarseseException, java.io.IOException {

        ByteArrayOutputStream baos = new ByteArrayOutputStream(16384);

        AtomicInteger count = new AtomicInteger();


        NAR a = NARS.tmp()

                .input(new String[]{input});
        a
                .run(16);
        Set<Task> written = new HashSet();
        a
                .synch()
                .outputBinary(baos, (Task t) -> {
                    assertTrue(written.add(t), () -> "duplicate: " + t);
                    count.incrementAndGet();
                    return true;
                })

        ;

        byte[] x = baos.toByteArray();
        out.println(count.get() + " tasks serialized in " + x.length + " bytes");

        NAR b = NARS.shell()
                .inputBinary(new ByteArrayInputStream(x));


        Set<Task> aHas = a.tasks().collect(Collectors.toSet());

        assertEquals(count.get(), aHas.size());

        assertEquals(written, aHas);

        Set<Task> bRead = b.tasks().collect(Collectors.toSet());

        assertEquals(aHas, bRead);


    }


    @Test
    void testA() {
        String somethingIsBird = "bird:$x";
        String somethingIsAnimal = "animal:$x";
        testIntroduction(somethingIsBird, Op.IMPL, somethingIsAnimal, "bird:robin", "animal:robin");
    }


    private static void testIntroduction(String subj, Op relation, String pred, String belief, String concl) {

        NAR n = NARS.shell();

        new TestNAR(n)
                .believe('(' + subj + ' ' + relation + ' ' + pred + ')')
                .believe(belief)
                .mustBelieve(4, concl, 0.81f);
    }

    @Test
    void posNegQuestion() {


        RuleTest.get(new TestNAR(NARS.shell()),
                "a:b?", "(--,a:b).",
                "a:b.",
                0, 0, 0.9f, 0.9f);
    }

    @Test void ImageConceptualize() throws Narsese.NarseseException {
        NAR n = NARS.shell();
        Testing.assertEq("(x,z(y))", n.conceptualize("(x, (y --> (z,/)))").term());

    }
}