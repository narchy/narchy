package nars;

import nars.meta.MetaProgrammingRules;
import nars.term.Term;
import nars.term.atom.Bool;
import nars.term.atom.Int;
import nars.term.obj.QuantityTerm;
import nars.Narsese.NarseseException;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

public class MetaProgrammingTest {

    private NAR nar;
    private static final double DELTA = 1e-6; // For float/double comparisons

    // Store initial NAL values to restore them after each test
    private boolean initialBidiFlag;
    private double initialNovelDurs;
    private int initialCompoundVolumeMax;

    @BeforeEach
    public void setUp() {
        nar = NARS.tmp();

        // Save initial values
        initialBidiFlag = NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI;
        initialNovelDurs = NAL.belief.NOVEL_DURS;
        initialCompoundVolumeMax = NAL.term.COMPOUND_VOLUME_MAX;
    }

    @AfterEach
    public void tearDown() {
        // Restore initial values to ensure test independence for static NAL fields
        NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI = initialBidiFlag;
        NAL.belief.NOVEL_DURS = initialNovelDurs;
        NAL.term.COMPOUND_VOLUME_MAX = initialCompoundVolumeMax;

        if (nar != null) {
            nar.delete(); // Clean up NAR instance
        }
    }

    // Helper to run NAR cycles
    private void runNarCycles(int numCycles) {
        for (int i = 0; i < numCycles; i++) {
            nar.run();
        }
    }

    @Test
    public void testTemporalInductionImplBidi() throws NarseseException {
        // Test Reading (minimal check: ensure query is processed without error)
        nar.input(MetaProgrammingRules.READ_TEMPORAL_INDUCTION_IMPL_BIDI_QUERY);
        runNarCycles(10);
        // No direct assertion on read value via Narsese belief in this simplified test.
        // We are primarily testing the setting mechanism.

        // Test Setting to False
        NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI = true; // Start from a known state (true)
        assertTrue(NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI, "Pre-condition: BIDI flag should be true");

        nar.input("<Self --> set_bidi_false>.");
        nar.input(MetaProgrammingRules.SET_TEMPORAL_INDUCTION_IMPL_BIDI_FALSE_RULE);
        runNarCycles(15); // Allow ample cycles for rule processing
        assertFalse(NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI, "BIDI flag should be false after rule execution");

        // Test Setting to True
        NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI = false; // Start from a known state (false)
        assertFalse(NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI, "Pre-condition: BIDI flag should be false");

        nar.input("<Self --> set_bidi_true>.");
        nar.input(MetaProgrammingRules.SET_TEMPORAL_INDUCTION_IMPL_BIDI_TRUE_RULE);
        runNarCycles(15);
        assertTrue(NAL.temporal.TEMPORAL_INDUCTION_IMPL_BIDI, "BIDI flag should be true after rule execution");
    }

    @Test
    public void testNovelDurs() throws NarseseException {
        // Test Reading (minimal check)
        nar.input(MetaProgrammingRules.READ_NOVEL_DURS_QUERY);
        runNarCycles(10);
        // No direct assertion on read value via Narsese belief.

        // Test Setting (Rule sets it to 2.5)
        double targetNovelDurs = 2.5;
        // Set to a different value first to ensure the rule causes a change
        NAL.belief.NOVEL_DURS = 1.0;
        assertNotEquals(targetNovelDurs, NAL.belief.NOVEL_DURS, DELTA, "Pre-condition: Ensure current NOVEL_DURS is different from target.");

        nar.input("<Self --> set_novel_durs_custom>.");
        nar.input(MetaProgrammingRules.SET_NOVEL_DURS_RULE);
        runNarCycles(15);
        assertEquals(targetNovelDurs, NAL.belief.NOVEL_DURS, DELTA, "NOVEL_DURS should be " + targetNovelDurs + " after rule execution");
    }

    @Test
    public void testCompoundVolumeMax() throws NarseseException {
        // Test Reading (minimal check)
        nar.input(MetaProgrammingRules.READ_COMPOUND_VOLUME_MAX_QUERY);
        runNarCycles(10);
        // No direct assertion on read value via Narsese belief.

        // Test Setting (Rule sets it to 256)
        int targetVolumeMax = 256;
        // Set to a different value first
        NAL.term.COMPOUND_VOLUME_MAX = 100;
        assertNotEquals(targetVolumeMax, NAL.term.COMPOUND_VOLUME_MAX, "Pre-condition: Ensure current COMPOUND_VOLUME_MAX is different from target.");

        nar.input("<Self --> set_volume_max_custom>.");
        nar.input(MetaProgrammingRules.SET_COMPOUND_VOLUME_MAX_RULE);
        runNarCycles(15);
        assertEquals(targetVolumeMax, NAL.term.COMPOUND_VOLUME_MAX, "COMPOUND_VOLUME_MAX should be " + targetVolumeMax + " after rule execution");

        // Test setting to an invalid value (0) - functor should prevent this
        NAL.term.COMPOUND_VOLUME_MAX = targetVolumeMax; // Reset to a valid, known state

        nar.input("<Self --> set_volume_max_invalid_zero>.");
        nar.input("<<Self --> set_volume_max_invalid_zero> =/> <({0}) --> (^setCompoundVolumeMax)>>.");
        runNarCycles(15);
        assertEquals(targetVolumeMax, NAL.term.COMPOUND_VOLUME_MAX, "COMPOUND_VOLUME_MAX should not change for invalid input 0");

        // Test setting to an invalid value (negative) - functor should prevent this
        nar.input("<Self --> set_volume_max_invalid_neg>.");
        nar.input("<<Self --> set_volume_max_invalid_neg> =/> <({-5}) --> (^setCompoundVolumeMax)>>.");
        runNarCycles(15);
        assertEquals(targetVolumeMax, NAL.term.COMPOUND_VOLUME_MAX, "COMPOUND_VOLUME_MAX should not change for invalid input -5");
    }
}
