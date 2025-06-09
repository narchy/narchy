package jcog.data.set;

import org.junit.jupiter.api.Test;

import java.util.Comparator;

import static org.junit.jupiter.api.Assertions.*;

class SortedArraySetTest {

    @Test
    void test() {
        int c = 10;
        var s = new SortedArraySet<Comparable>(Comparator.comparing(z -> (Comparable)z), new String[c]);
        assertTrue(s.isEmpty());
        s.put("z");
        s.put("a");
        assertNull(s.remove("x"));
        assertEquals(2, s.size());
        assertEquals("[a, z]", s.toString());
        assertEquals("a", s.first());
        assertEquals("z", s.last());
        assertNotEquals(null, s.put("a"));
        s.put("m"); assertEquals("[a, m, z]", s.toString());
        assertTrue(s.contains("a"));
        assertFalse(s.contains("x"));
        s.remove("m"); assertEquals("[a, z]", s.toString());
        s.remove("z"); assertEquals("[a]", s.toString());
        s.remove("a"); assertEquals("[]", s.toString());
        assertNull(s.remove("x"));
        s.put("a"); s.clear(); assertEquals(0, s.size());
    }
}