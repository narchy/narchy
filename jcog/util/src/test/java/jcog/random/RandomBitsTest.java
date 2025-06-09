package jcog.random;

import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;
import org.hipparchus.stat.Frequency;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RandomBitsTest {

    @Test
    void test1() {
        var r = new RandomBits(new XoRoShiRo128PlusRandom(1));
        //System.out.println(r);
        //r.refresh();
        //System.out.println("  \t" + r);
        var a = r.nextBoolean();
        //System.out.println(a + "\t" + r);
        var b = r.nextBoolean();
        //System.out.println(b + "\t" + r);
        var c = r.nextBoolean();
        //System.out.println(c + "\t" + r);


        var d = r._next(4);
        //System.out.println(d + "  \t" + r);
        var e = r._next(3);
        //System.out.println(e + "  \t" + r);
        var f = r._next(8);
        //System.out.println(f + "  \t" + r);

        assertTrue(a);
        assertFalse(b);
        assertFalse(c);
        assertEquals(46,r.bit);

        r.bit = 3; //force refresh
        var g = r._next(8);
        //System.out.println(g + "  \t" + r);
        assertEquals(56, r.bit);
        assertEquals(13, g);
    }

    @Test void nextInt_4() {
        var r = new RandomBits(new XoRoShiRo128PlusRandom(1));
        var i = r.nextInt(10); //4 bits
        assertEquals(60, r.bit);
        var seen = new IntHashSet();
        for (var j = 0; j < 1000; j++) {
            var z = r.nextInt(10);
            assertTrue(z < 10 && z >= 0, ()->z + " out of range");
            seen.add(z);
        }
        assertEquals(10, seen.size());
    }
    @Test void nextInt_9() {
        var r = new RandomBits(new XoRoShiRo128PlusRandom(1));
        var i = r.nextInt(310); //9 bits
        assertEquals(55, r.bit);
        for (var j = 0; j < 10000; j++) {
            var z = r.nextInt(310);
            assertTrue(z < 310 && z >= 0, ()->z + " out of range");
        }
    }

    @Test void nextBooleanFast() {
        var r = new RandomBits(new XoRoShiRo128PlusRandom(1));
        for (var b : new int[] { 8, 16, 32 }) {
            for (float p = 0; p <= 1; p += 0.05f) {
                assertProb(r, p, b);
            }
        }
    }

    private static void assertProb(RandomBits r, float idealProb, int bits) {
        var samples = 16*1024;
        var tolerance = 0.01;

        var trues = 0;
        for (var i = 0; i < samples; i++) {
            if (r.nextBooleanFast(idealProb, bits))
                trues++;
        }
        var actual = trues/((double)samples);
        assertEquals(idealProb, actual, tolerance);
    }

    @Test void floatFast8() {
        testFloatFast(8);
    }
    @Test void floatFast16() {
        testFloatFast(16);
    }

    private static void testFloatFast(int bitsPerFloat) {
        int bins = 10;
        int samples = 10000;

        var r = new RandomBits(new XoRoShiRo128PlusRandom(1));
        Frequency f = new Frequency<>();
        for (int i = 0; i < samples; i++) {
            float x = r.nextFloatFast(bitsPerFloat);
            int bin = (int)(x * bins);
            f.addValue(bin);
        }
        //System.out.println(f);

        double tolerance = 0.01;
        for (int i = 0; i < bins; i++) {
            double p = f.getPct(i);
            assertTrue(p > 0.1 - tolerance && p < 0.1 + tolerance);
        }
    }
}