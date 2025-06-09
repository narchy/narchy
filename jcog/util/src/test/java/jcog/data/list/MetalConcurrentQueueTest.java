package jcog.data.list;

import com.google.common.base.Joiner;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MetalConcurrentQueueTest {

	@Test void testQueuing() {
		var i = new MetalConcurrentQueue<>(4);

		assertEquals(0, i.size());
		assertEquals(0, i.head());
		assertNull(i.poll());
		assertEquals(0, i.size());
		assertEquals(0, i.head());

		assertNull(i.peek());
		assertEquals(0, i.size());
		assertEquals(0, i.head());

		//assertNull(i.poll(1));

		i.offer(1); i.offer(2); i.offer(3);
		Joiner j = Joiner.on(',');
		assertEquals("1,2,3", j.join(i.stream().iterator()));
		assertEquals(1, i.peek());
		assertEquals(2, i.getHead(1));
		assertEquals(1, i.poll());
		assertEquals("2,3", j.join(i.stream().iterator()));
		assertTrue(i.offer(1));
		assertEquals("2,3,1", j.join(i.stream().iterator()));
		assertEquals(2, i.poll());
		assertEquals(3, i.poll());
		assertTrue(i.offer(2));
		assertTrue(i.offer(3));
		assertEquals("1,2,3", j.join(i.stream().iterator()));
		assertEquals(1, i.poll());
		assertTrue(i.offer(4));
		assertTrue(i.offer(5));
		assertEquals("2,3,4,5", j.join(i.stream().iterator()));
		assertEquals(4, i.size());
		assertFalse(i.offer(6));
		for (int n = 0; n < 4; n++)
			assertEquals(2+n, i.getHead(n));
		i.clear();
		assertEquals("", j.join(i.stream().iterator()));
        assertEquals(0, i.size());
	}
}