package jcog.data.list;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LstTest {

    @Test void removeNulls1a() {
        Lst l = new Lst();
        l.add("x");
        l.add(null);
        assertEquals(2, l.size());
        l.removeNulls();
        assertEquals(1, l.size());
        assertEquals("x", l.getFirst());
    }
    @Test void removeNulls1b() {
        Lst l = new Lst();
        l.add(null);
        l.add("x");
        assertEquals(2, l.size());
        l.removeNulls();
        assertEquals(1, l.size());
        assertEquals("x", l.getFirst());
    }
    @Test void removeNulls2_1() {
        Lst l = new Lst();
        l.add("x");
        l.add("y");
        l.add(null);
        assertEquals(3, l.size());
        l.removeNulls();
        assertEquals(2, l.size());
    }
    @Test void removeNulls_many() {
        Lst l = new Lst();
        l.add(null);
        l.add("x");
        l.add(null);
        l.add(null);
        l.add("y");
        l.add(null);
        assertEquals(6, l.size());
        l.removeNulls();
        assertEquals(2, l.size());
    }
}