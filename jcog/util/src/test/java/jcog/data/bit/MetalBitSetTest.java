package jcog.data.bit;

import jcog.random.XoRoShiRo128PlusRandom;
import org.eclipse.collections.api.block.procedure.primitive.IntProcedure;
import org.eclipse.collections.api.iterator.IntIterator;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class MetalBitSetTest {

    @Test void testIteratorAndToArray() {
        var l = new LongArrayBitSet(
            new long[] {
                0,0,0,
                0b00000010_00000100_10000000_00000000_00000000_00000000_00000000L
            }
        );
        assertEquals(3, l.cardinality());
        short[] s = new short[3];
        l.toArray(0, 64*4, s);
        assertEquals(231, s[0]);
        assertEquals(234, s[1]);
        assertEquals(241, s[2]);
    }

    @Test
    void test1() {
        MetalBitSet b = MetalBitSet.bits(70);
        b.set(64);
        assertEquals(64, b.first(true));
    }
    @Test
    void test2() {
        MetalBitSet b = MetalBitSet.bits(70);
        b.set(5); assertTrue(b.test(5)); assertFalse(b.test(4));
        b.set(65); assertTrue(b.test(65)); assertFalse(b.test(64));
        IntIterator bi = b.iterator(0, 70);
        assertTrue(bi.hasNext()); assertTrue(bi.hasNext()); //extra call shouldnt do anything
        assertEquals(5, bi.next());
        assertTrue(bi.hasNext()); assertTrue(bi.hasNext()); //extra call shouldnt do anything
        assertEquals(65, bi.next());
        assertFalse(bi.hasNext()); assertFalse(bi.hasNext());
    }
    @Test
    void LongArrayBitSet_Iterator() {
        testRandomSetAndIterate(64 * 3 /* LongArrayBitSet */);
    }

    private static void testRandomSetAndIterate(int n) {
        float density = 0.5f;
        int fill = Math.round(n * density);

        Random rng = new XoRoShiRo128PlusRandom(1);
        for (int j = 0; j < 100; j++) {
            MetalBitSet b = MetalBitSet.bits(n);
            int written = 0;
            for (int i = 0; i < fill; i++) {
                int x = rng.nextInt(n);
                if (!b.getAndSet(x, true))
                    written++;

                assertTrue(b.test(x));
                assertEquals(written, b.cardinality());
            }
            IntIterator bb = b.iterator(0, n);
            int read = 0;
            while (bb.hasNext()) {
                bb.next();
                read++;
            }
            assertEquals(written, read);
        }
    }

    @Test
    void testOptimizedForEach_IntBitSet() {
        var bitSet = new IntBitSet();
        bitSet.set(3);
        bitSet.set(12);
        bitSet.set(31);

        var collected = new ArrayList<>();
        bitSet.forEach((IntProcedure) collected::add);

        assertEquals(3, collected.size());
        assertEquals(3, collected.get(0));
        assertEquals(12, collected.get(1));
        assertEquals(31, collected.get(2));
    }

    @Test
    void testOptimizedForEach() {
        LongArrayNBitSet bitSet = new LongArrayNBitSet(128);
        bitSet.set(5);
        bitSet.set(64);
        bitSet.set(127);

        var collected = new ArrayList<>();
        bitSet.forEach((IntProcedure) collected::add);

        assertEquals(3, collected.size());
        assertEquals(5, collected.get(0));
        assertEquals(64, collected.get(1));
        assertEquals(127, collected.get(2));
    }

    @Test
    void testOptimizedMyIntIterator() {
        LongArrayNBitSet bitSet = new LongArrayNBitSet(128);
        bitSet.set(5);
        bitSet.set(64);
        bitSet.set(127);

        var iterator = bitSet.iterator();
        assertTrue(iterator.hasNext());
        assertEquals(5, iterator.next());
        assertTrue(iterator.hasNext());
        assertEquals(64, iterator.next());
        assertTrue(iterator.hasNext());
        assertEquals(127, iterator.next());
        assertFalse(iterator.hasNext());

        //assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Test
    void testOptimizedNext() {
        LongArrayBitSet bitSet = new LongArrayBitSet(256);
        bitSet.set(5);
        bitSet.set(64);
        bitSet.set(127);
        bitSet.set(200);

        assertEquals(5, bitSet.next(true, 0, 256));
        assertEquals(64, bitSet.next(true, 6, 256));
        assertEquals(127, bitSet.next(true, 65, 256));
        assertEquals(200, bitSet.next(true, 128, 256));
        assertEquals(-1, bitSet.next(true, 201, 256));

        assertEquals(6, bitSet.next(false, 5, 256));
        assertEquals(65, bitSet.next(false, 64, 256));
        assertEquals(128, bitSet.next(false, 127, 256));
        assertEquals(201, bitSet.next(false, 200, 256));
    }

//    @Test
//    void testOptimizedResize() {
//        LongArrayBitSet bitSet = new LongArrayBitSet(64);
//        bitSet.set(5);
//        bitSet.set(63);
//
//        // Simulate resize by calling a method that triggers resize
//        bitSet.set(128);
//
//        assertTrue(bitSet.test(5));
//        assertTrue(bitSet.test(63));
//        assertTrue(bitSet.test(128));
//        assertEquals(3, bitSet.cardinality());
//    }

    @Test
    void testOptimizedRandom() {
        MetalBitSet bitSet = MetalBitSet.bits(128);
        bitSet.set(5);
        bitSet.set(64);
        bitSet.set(127);

        Random rng = new XoRoShiRo128PlusRandom(1);
        for (int i = 0; i < 100; i++) {
            int randomBit = bitSet.random(rng);
            assertTrue(randomBit == 5 || randomBit == 64 || randomBit == 127);
        }
    }

    @Test
    void testIntBitSetOptimizations() {
        IntBitSet bitSet = new IntBitSet();

        // Test valid bits
        bitSet.set(0);
        bitSet.set(31);
        assertTrue(bitSet.test(0));
        assertTrue(bitSet.test(31));


        // Test cardinality and clear
        assertEquals(2, bitSet.cardinality());
        bitSet.clear();
        assertEquals(0, bitSet.cardinality());


//        // Test invalid bits
//        assertThrows(ArrayIndexOutOfBoundsException.class, () -> bitSet.set(-1, true));
//        assertThrows(ArrayIndexOutOfBoundsException.class, () -> bitSet.set(32, true));
//        assertThrows(ArrayIndexOutOfBoundsException.class, () -> bitSet.test(-1));
//        assertThrows(ArrayIndexOutOfBoundsException.class, () -> bitSet.test(32));

    }

    @Test void testStupidIterator() {
        IntArrayNBitSet b = new IntArrayNBitSet(3);
        b.set(2);
        var i = b.iterator();
        assertTrue(i.hasNext());
        assertEquals(2, i.next());
        assertFalse(i.hasNext());
    }

    @Test void testStupidIterator2() {
        IntArrayNBitSet b = new IntArrayNBitSet(4);
        b.set(1);
        b.set(3);
        var i = b.iterator();
        assertTrue(i.hasNext());
        assertEquals(1, i.next());
        assertTrue(i.hasNext());
        assertEquals(3, i.next());
        assertFalse(i.hasNext());

        assertEquals(1, b.first());
        assertEquals(3, b.last());
    }
}