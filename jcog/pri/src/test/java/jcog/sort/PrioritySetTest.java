package jcog.sort;

import jcog.random.XoRoShiRo128PlusRandom;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashSet;
import java.util.Set;
import java.util.random.RandomGenerator;

import static org.junit.jupiter.api.Assertions.*;

class PrioritySetTest {

    private PrioritySet<String> p;

    private final RandomGenerator rng = new XoRoShiRo128PlusRandom(1);

    @BeforeEach
    void setUp() {
        p = new PrioritySet<>(5);
    }

    @Test
    void testConstructorWithInvalidCapacity() {
        assertThrows(IllegalArgumentException.class, () -> new PrioritySet<>(0));
        assertThrows(IllegalArgumentException.class, () -> new PrioritySet<>(-1));
    }

    @Test
    void testPutAndContains() {
        assertTrue(p.put("A", 1.0));
        assertTrue(p.contains("A"));
        assertFalse(p.contains("B"));
    }

    @Test
    void testPutDuplicate() {
        assertTrue(p.put("A", 1.0));
        assertTrue(p.put("A", 2.0));
        assertEquals(2.0, p.pri("A"));
    }

    @Test
    void testPutWithInvalidPriority() {
        assertThrows(IllegalArgumentException.class, () -> p.put("A", -1));
    }

    @Test
    void testPutNull() {
        assertThrows(NullPointerException.class, () -> p.put(null, 1.0));
    }

    @Test
    void testRemove() {
        p.put("A", 1.0);
        assertTrue(p.remove("A"));
        assertFalse(p.contains("A"));
        assertFalse(p.remove("A"));
    }

    @Test
    void testRemoveNonExistent() {
        assertFalse(p.remove("A"));
    }

    @Test
    void testPriGet() {
        p.put("A", 1.0);
        assertTrue(p.pri("A", 2.0));
        assertEquals(2.0, p.pri("A"));
    }

    @Test
    void testPriNonExistent() {
        assertFalse(p.pri("A", 2.0));
    }
    @Test
    void testPriReject() {
        p.put("A", 1.0);
        p.put("B", 1.0);
        p.put("C", 1.0);
        p.put("D", 1.0);
        p.put("E", 1.0);
        assertFalse(p.put("F", 0.1));
        assertTrue(!p.contains("F"));
    }

    @Test
    void testPriInvalid() {
        p.put("A", 1.0);
        assertThrows(IllegalArgumentException.class, () -> p.pri("A", 0));
        assertThrows(IllegalArgumentException.class, () -> p.pri("A", -1));
    }

    @Test
    void testSample() {
        p.put("A", 1.0);
        p.put("B", 2.0);
        p.put("C", 3.0);

        Set<String> sampledItems = new HashSet<>();
        for (int i = 0; i < 1000; i++) {
            sampledItems.add(p.get(rng));
        }

        assertEquals(3, sampledItems.size());
        assertTrue(sampledItems.contains("A"));
        assertTrue(sampledItems.contains("B"));
        assertTrue(sampledItems.contains("C"));
    }

    @Test
    void testSampleEmpty() {
        assertNull(p.get(rng));
    }

    @Test
    void testCapacityLimit() {
        for (int i = 0; i < 5; i++) {
            assertTrue(p.put("Item" + i, i + 1.0));
        }
        assertEquals(5, p.size());

        assertTrue(p.put("HighPriority", 10.0));
        assertEquals(5, p.size());
        assertTrue(p.contains("HighPriority"));
        assertFalse(p.contains("Item0"));
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 5, 10, 100})
    void testVariousCapacities(int capacity) {
        PrioritySet<Integer> set = new PrioritySet<>(capacity);
        for (int i = 0; i < capacity * 2; i++) {
            set.put(i, i + 1.0);
        }
        assertEquals(capacity, set.size());
        for (int i = capacity; i < capacity * 2; i++) {
            assertTrue(set.contains(i));
        }
    }

    @Test
    void testPri() {
        p.put("A", 1.5);
        assertEquals(1.5, p.pri("A"));
        assertTrue(Double.isNaN(p.pri("B")));
    }

    @Test
    void testIsEmpty() {
        assertTrue(p.isEmpty());
        p.put("A", 1.0);
        assertFalse(p.isEmpty());
        p.remove("A");
        assertTrue(p.isEmpty());
    }
}