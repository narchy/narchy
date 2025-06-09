package jcog.math;

import org.junit.jupiter.api.Test;

import static jcog.Str.n2;
import static org.junit.jupiter.api.Assertions.assertEquals;

class DigitizeTest {

    @Test void fuzzyNeedle1() {
        assertEquals("1.0 0.0", n2(Digitize.FuzzyNeedle.digits(0, 2)));
        assertEquals("0.0 1.0", n2(Digitize.FuzzyNeedle.digits(1, 2)));
        assertEquals(".50 .50", n2(Digitize.FuzzyNeedle.digits(0.5f, 2)));
    }
}