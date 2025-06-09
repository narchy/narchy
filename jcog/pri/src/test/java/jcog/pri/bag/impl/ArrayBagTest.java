package jcog.pri.bag.impl;

import com.google.common.base.Joiner;
import com.google.common.collect.Streams;
import jcog.Util;
import jcog.pri.PLink;
import jcog.pri.PriReference;
import jcog.pri.UnitPri;
import jcog.pri.bag.BagTest;
import jcog.pri.op.PriMerge;
import jcog.random.XoRoShiRo128PlusRandom;
import jcog.random.XorShift128PlusRandom;
import org.eclipse.collections.impl.bag.mutable.HashBag;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.*;

import static java.util.stream.Collectors.toList;
import static jcog.pri.bag.BagTest.fillLinear;
import static jcog.pri.bag.BagTest.testBasicInsertionRemoval;
import static jcog.pri.op.PriMerge.*;
import static org.junit.jupiter.api.Assertions.*;


public class ArrayBagTest {


    private static ArrayBag<PLink<String>, PLink<String>> newBag(int n, PriMerge mergeFunction) {
        return new PLinkArrayBag(mergeFunction, n);
    }


    public static void assertSorted(ArrayBag x) {
        assertTrue(x.isSorted(), () -> Joiner.on("\n").join(x));
    }

    @Test
    void BasicInsertionRemovalArray() {
        testBasicInsertionRemoval(new PLinkArrayBag<>(plus, 1));
    }

    @Test
    void BudgetMerge() {
        PriReferenceArrayBag<String, PriReference<String>> a = new PLinkArrayBag<>(plus, 4);
        assertEquals(0, a.size());

        a.put(new PLink("x", 0.1f));
        assertEquals(0.1f, a.get("x").pri());
        assertEquals(1, a.size());

        a.put(new PLink("x", 0.1f));
        assertEquals(0.2f, a.get("x").pri());
        assertEquals(1, a.size());


        PriReference<String> agx = a.get("x");
        UnitPri expect = new UnitPri(0.2f);
        assertTrue(Util.equals(expect.priElseNeg1(), agx.priElseNeg1(), 0.01f),
                () -> agx + "==?==" + expect);

    }

    @Test
    void SortRandom() {
        int j = 0;
        Random rng = new XoRoShiRo128PlusRandom(1);
        for (int c = 3; c < 8; c++) {
            PriReferenceArrayBag<String,PriReference<String>> a = new PLinkArrayBag<>(replace, 32);
            int iters = c * 1000;
            for (int i = 0; i < iters; i++) {
                a.put(new PLink(String.valueOf(j++), rng.nextFloat()));
                assertSorted(a);
            }

            assertEquals(a.capacity(), a.size());
            //a.print();

            //test sorting after scrambling during commit
            a.commit(z -> z.pri(rng.nextFloat()));

            assertSorted(a);
        }


    }
    @Test
    void SortIncrease() {
        PriReferenceArrayBag a = new PLinkArrayBag<>(plus, 4);

        a.put(new PLink("x", 0.1f));
        a.put(new PLink("y", 0.2f));

        a.commit(null);

        Iterator<PriReference<String>> ii = a.iterator();
        assertTrue(ii.hasNext()); assertEquals("y", ii.next().get());
        assertTrue(ii.hasNext()); assertEquals("x", ii.next().get());

        assertEquals("[$0.2 y, $0.1 x]", a.stream().collect(toList()).toString());
        assertEquals(0.1f, a.map.pri(a.map.id("x")));
        assertEquals(0.2f, a.map.pri(a.map.id("y")));

//        System.out.println(a.stream().collect(toList()));

        a.put(new PLink("x", 0.2f));
        assertEquals(0.2f, a.map.pri(a.map.id("y")));
        assertEquals(0.3f, a.map.pri(a.map.id("x")));
        assertTrue(a.isSorted());
        assertEquals(2, a.size());
        assertEquals("$0.3 x", a.get(0).toString());
        assertEquals("$0.2 y", a.get(1).toString());


        a.commit();

        {
            String s = a.stream().collect(toList()).toString(); assertTrue(s.contains("x,")); assertTrue(s.contains("y]"));
        }

        ii = a.iterator();
        assertTrue(ii.hasNext()); assertEquals("x", ii.next().get());
        assertTrue(ii.hasNext()); assertEquals("y", ii.next().get());
        assertTrue(a.isSorted());

    }

    /** tests case where an item shifts downward in the bag */
    @Test void SortDecrease() {
        PriReferenceArrayBag a = new PLinkArrayBag<>(and, 4);

        a.put(new PLink("x", 0.1f));
        a.put(new PLink("y", 0.2f));

        assertTrue(a.isSorted());
        assertEquals("[$0.2 y, $0.1 x]", a.stream().collect(toList()).toString());


        a.put(new PLink("y", 0.3f));
        assertTrue(a.isSorted());
        assertEquals(2, a.size());
        assertEquals("[$0.1 x, $0.06 y]", a.stream().collect(toList()).toString());


    }

    @Test
    void Capacity() {
        PriReferenceArrayBag a = new PLinkArrayBag(plus, 2);

        PLink x = new PLink("x", 0.1f);
        assertSame(x, a.put(x));

        assertEquals(1, a.size());

        PLink y = new PLink("y", 0.2f);
        assertSame(y, a.put(y));

        assertEquals(2, a.size());

        PLink y2 = new PLink("y", 0f);
        assertSame(y, a.put(y2)); //no change, existing link returned

        a.commit(null);
        assertSorted(a);
        assertEquals(0.1f, a.priMin(), 0.01f);

        PLink z = new PLink("z", 0.05f);
        Object zRejected = a.put(z);
        //assertTrue(z.isDeleted());
        assertNull(zRejected);
        assertEquals(0.1f, a.priMin(), 0.01f);
        assertEquals(2, a.size());
        assertTrue(a.contains("x") && a.contains("y"));
        assertFalse(a.contains("z"));

    }

    @Test
    void CapacityChange() {
        PriReferenceArrayBag a = new PLinkArrayBag(plus, 4);
        a.put(new PLink<>("x", 0.5f));
        a.put(new PLink<>("y", 0.4f));
        a.put(new PLink<>("z", 0.3f));
        a.put(new PLink<>("w", 0.2f));

        a.capacity(2);
        assertEquals(2, a.capacity());
        assertEquals(2, a.size());

        a.capacity(0);
        assertEquals(0, a.capacity());
        assertEquals(0, a.size());
        assertTrue(a.isEmpty());
    }

    @Test
    void DeleteBetweenCommits() {
        PLinkArrayBag<String> a = new PLinkArrayBag<>(plus, 4);
        a.put(new PLink<>("x", 0.5f));
        a.put(new PLink<>("y", 0.4f));
        PriReference<String> z = a.put(new PLink<>("z", 0.3f));
        a.put(new PLink<>("w", 0.2f));
        a.commit();
        assertEquals(4, a.size());

        z.delete();

        a.commit();
        assertEquals(3, a.size());
    }

    @Test void PutDeleted() {
        PLinkArrayBag<String> a = new PLinkArrayBag<>(plus, 4);
        assertNull(a.put(new PLink<>("x", Float.NaN)));
        assertTrue(a.isEmpty());
    }
    @Test void ZeroCapacity() {
        PLinkArrayBag<String> a = new PLinkArrayBag<>(plus, 0);
        assertNull(a.put(new PLink<>("x", Float.NaN)));
        assertTrue(a.isEmpty());
    }    
    @Test
    void RemoveByKey() {
        BagTest.testRemoveByKey(new PLinkArrayBag(plus, 2));
    }

    @Disabled
    @Test
    void InsertOrBoostDoesntCauseSort() {
        int[] sorts = {0};
        ArrayBag<PLink<String>, PLink<String>> x = new PLinkArrayBag(plus, 4);
        //{
//            @Override
//            protected void sort() {
//                sorts[0]++;
//                super.sort();
//            }
        //};

        x.put(new PLink("x", 0.2f));
        x.put(new PLink("y", 0.1f));
        x.put(new PLink("z", 0f));

        assertEquals(0, sorts[0]);

        x.commit();

        assertEquals(0, sorts[0]);

        assertSorted(x);

    }

    @ValueSource(ints = {2, 3, 4, 8, 16, 32, 64})  @ParameterizedTest
    void DistributionSmall_Linear(int cap) {
//        for (float batchSizeProp : new float[]{
//            //0.001f, 0.1f, 0.3f
//            0.1f
//        }) {

        BagTest.sampleLinear(newBag(cap, plus), 0.1f);
    }
    @ValueSource(ints = {2, 3, 4, 5, 6, 7, 8})  @ParameterizedTest void BagDistributionSmall_Squashed(int cap) {
        BagTest.sampleSquashed(newBag(cap, plus), 0.1f);

    }
    @ValueSource(ints = {2, 3, 4, 5, 6, 7, 8})  @ParameterizedTest void BagDistributionSmall_Curved(int cap) {
        BagTest.sampleCurved(newBag(cap, plus), 0.1f);

    }

    @Test
    void BagDistribution8_BiggerBatch() {
        for (float batchSizeProp : new float[]{0.5f}) {

            BagTest.sampleLinear(newBag(8, plus), batchSizeProp);

            BagTest.sampleSquashed(newBag(8, plus), batchSizeProp);
        }
    }

    @Test
    void BagDistributions() {
        float err = 0;
        for (int n : new int[]{
                2,3,4,5,6,7,8,
                64,32,16,
                50,100,1000,1001
                })
            for (float sharp : new float[]{ 1/*, 2, 3, 0.75f, 1.5f*/ }) {
                System.out.println(n + "\t" + sharp);
                err += BagTest.sampleLinear(newBag(n, replace), 0, sharp);
            }

        System.out.println("total err=" + err);
    }

    @Test
    void BagDistribution64() {
        for (float batchSizeProp : new float[]{0.05f, 0.1f, 0.2f}) {
            BagTest.sampleLinear(newBag(64, plus), batchSizeProp);
            BagTest.sampleRandom(newBag(64, plus), batchSizeProp);
        }
    }

    @ValueSource(ints = {32, 64})
    @ParameterizedTest
    void BagDistribution32_64__small_batch(int cap) {
        for (float batchSizeProp : new float[]{0.001f}) {
            BagTest.sampleLinear(newBag(cap, plus), batchSizeProp);
        }
    }

    /**
     * should normalize to the utilized range
     */
    @Test
    void BagDistribution_uneven_partial1_a() {
        BagDistribution_uneven_partial1( newBag(32, plus));
    }
//    @Test
//    void BagDistribution_uneven_partial1_b() {
//        testCurveBagDistribution_uneven_partial1(
//            (ArrayBag<?, PLink<String>>)
//                new PriArrayBuffer().merge(plus).capacity(32)
//        );
//    }

    static void BagDistribution_uneven_partial1(ArrayBag<?, PLink<String>> b) {
        /*

        |---\
        |   --\
        |      _____________

        */

        int cap = b.capacity();
        float dynamicRange = 0.25f;
        float dynamicFloor = 0;

        for (int i = 0; i < cap; i++) {
            float r =
                    //i / ((float) cap);
                    Util.cube(i / ((float) cap));
            float p = (r) * dynamicRange + dynamicFloor;
            b.put(new PLink<>(String.valueOf(i), p));
        }
        b.commit();
        b.print();

        Random rng = new XorShift128PlusRandom(1);

//        {
//            Histogram h = new Histogram(1, cap + 1, 3);
//            int samples = cap * 1000;
//            for (int i = 0; i < samples; i++) {
//                int s = b.sampleNext(rng, b.size());
//                h.recordValue(s);
//            }
//            Str.histogramPrint(h, System.out);
//        }
//
//        Tensor d = BagTest.samplingPriDist(b, samples,  60);
//        System.out.println(n4(d.doubleArray()));
//        {
//
//            int scale = 10000;
//            Histogram h =
//                    new Histogram(Math.max(1, Math.round(dynamicFloor * scale) - 1),
//                    Math.round((dynamicFloor + dynamicRange) * scale), 5);
//            h.setAutoResize(true);
//            b.sample(rng, samples, x -> {
//                h.recordValue(Math.round(x.pri() * scale));
//                return true;
//            });
//            Texts.histogramPrint(h, System.out);
//        }

    }


    @Test
    void SampleUnique() {

        for (int c : new int[]{2, 4, 5, 8, 16, 19, 32}) {
            ArrayBag<PLink<String>, PLink<String>> b = newBag(c, replace);
            fillLinear(b, c);

            for (int k = 0; k < c; k++) {
                Set<String> s = new HashSet(c);
                Iterator<PLink<String>> ii = b.sampleUnique(new XoRoShiRo128PlusRandom(k ^ c));
                while (ii.hasNext()) {
                    assertTrue(s.add(ii.next().id), "not unique");
                }
                assertEquals(b.capacity(), s.size());
            }
        }

    }

    /** alternating up/down */
    @Test void SampleUniqueOrder() {
        int c = 10;
        ArrayBag<PLink<String>, PLink<String>> b = newBag(c, replace);
        for (int i = 0; i < c; i++) {
            b.put(new PLink("x" + i, ((float)(i+1))/c));
        }
        b.commit(null);
        Iterator<PLink<String>> s = b.sampleUnique(new Random() {
            @Override
            public float nextFloat() {
                return 0.8f;
            }
        });
        List<PLink<String>> l = Streams.stream(s).limit(4).toList();
        assertEquals(
                "[$0.4 x3, $0.3 x2, $0.5 x4, $0.2 x1]"
                , l.toString());
    }

    @Test void Sample3_HighDynamicRange() {
        int c = 10;
        ArrayBag<PLink<String>, PLink<String>> b = newBag(c, replace);

        b.put(new PLink("a", 1));
        b.put(new PLink("b", 0.5f));
        b.put(new PLink("c", 0.01f));
        b.commit(null);

        Random rng = new XoRoShiRo128PlusRandom();
        HashBag<String> h = new HashBag<>(3);
        for (int i = 0; i < 1000; i++) {
            h.add(b.sample(rng).id);
        }
        var hh = h.toMapOfItemToCount();
        System.out.println(hh);
        assertTrue(hh.get("a") > 1.5f * hh.get("b"));
        assertTrue(hh.get("b") > 1.5f * hh.get("c"));
    }

    @Test void Sample9_HighDynamicRange() {
        int u = 10;
        int c = 10;
        ArrayBag<PLink<String>, PLink<String>> b = newBag(c, replace);

        assertTrue(ArrayBag.DIRECT_SAMPLE_MAX < u); //ensure Sketch for sampling

        b.put(new PLink("a", 1));
        b.put(new PLink("b", 0.9f));
        b.put(new PLink("c", 0.8f));
        b.put(new PLink("d", 0.7f));
        b.put(new PLink("e", 0.6f));
        b.put(new PLink("f", 0.5f));
        b.put(new PLink("g", 0.4f));
        b.put(new PLink("h", 0.3f));
        b.put(new PLink("i", 0.2f));
        b.put(new PLink("j", 0.1f));
        b.commit(null);

        Random rng = new XoRoShiRo128PlusRandom();
        HashBag<String> h = new HashBag<>(3);
        for (int i = 0; i < 1000; i++) {
            h.add(b.sample(rng).id);
        }
        var hh = h.toMapOfItemToCount();
        System.out.println(hh);
        assertTrue(hh.get("a") > 1.1f * hh.get("e"));
        assertTrue(hh.get("f") > 1.1f * hh.get("j"));
    }

}