package jcog.cluster;

import jcog.data.DistanceFunction;
import jcog.data.list.Lst;
import jcog.random.XoRoShiRo128PlusRandom;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class KMeansPlusPlusTest {

    @Test void test1() {
        strTest(List.of("a", "b", "c", "d", "e", "f", "g", "h"),
                "0<1.0 1.0>=[a, b, c] 1<1.0 4>=[d, e, f] 2<1.0 6.5>=[g, h]", 1);
    }
    @Test void test2() {
        strTest(List.of("a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa"),
                "0<1.21 0.0>=[a, aa] 1<1.87 0.0>=[aaa, aaaa] 2<2.34 0.0>=[aaaaa, aaaaaa]", 1);
    }
    @Test void test3() {
        strTest(List.of("a", "z", "a0", "a1", "j", "z0"),
                "0<1.21 25>=[z, z0] 1<1.0 4.5>=[j] 2<1.41 0.0>=[a, a0, a1]", 2);
    }

    @Test void test4() {
        var x = List.of(
                //"a", "aa", "aaa", "aaaa", "b", "bb", "bbb", "bbbb", "c", "cc", "ccc", "cccc"
                "a", "bb", "ccc", "dddd", "eeeee", "ffffff", "aaaaa",
                "f",
                "g", "gggggggg"
        );
        strTest(x, "0<2.83 6>=[gggggggg] 1<1.0 0.0>=[a, bb] 2<2 3>=[ccc, dddd] 3<2.24 0.0>=[aaaaa] 4<1.0 6>=[f, g] 5<2.45 5>=[eeeee, ffffff]", 1, 6);
//        strTest(x, "TODO", 2);
    }

    private static void strTest(List<String> x, String y, int iters) {
        strTest(x, y, iters, 3);
    }
    private static void strTest(List<String> x, String y, int iters, int centroids) {
        var k = new KMeansPlusPlus<String>(centroids, 2,
                DistanceFunction::distanceCartesianSq, new XoRoShiRo128PlusRandom(1)) {
            @Override public void coord(String s, double[] coords) {
                coords[0] = Math.sqrt(s.length());
                coords[1] = s.charAt(0) - 'a';
            }
        };
        k.cluster(new Lst<>(x), iters);
        assertEquals(y, k.toString());
    }
}