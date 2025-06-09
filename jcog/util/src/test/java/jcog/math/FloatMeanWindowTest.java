package jcog.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FloatMeanWindowTest {

    @Test void t1() {
        FloatMeanWindow f = new FloatMeanWindow(3);
        f.accept(1); assertEquals(1, f.mean());
        f.accept(3); assertEquals(2, f.mean());
        f.accept(2); assertEquals(2, f.mean());
        f.accept(4); assertEquals(3, f.mean());

    }
}