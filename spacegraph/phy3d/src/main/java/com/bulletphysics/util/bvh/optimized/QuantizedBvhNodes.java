/*
 * Java port of Bullet (c) 2008 Martin Dvorak <jezek2@advel.cz>
 *
 * Bullet Continuous Collision Detection and Physics Library
 * Copyright (c) 2003-2008 Erwin Coumans  http://www.bulletphysics.com/
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising from
 * the use of this software.
 * 
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

package com.bulletphysics.util.bvh.optimized;

import java.io.Serializable;

/**
 * QuantizedBvhNodes is array of compressed AABB nodes, each of 16 bytes.
 * Node can be used for leaf node or internal node. Leaf nodes can point to 32-bit
 * triangle index (non-negative range).<p>
 * 
 * <i>Implementation note:</i> the nodes are internally stored in int[] array
 * and bit packed. The actual structure is:
 * 
 * <pre>
 * unsigned short  quantizedAabbMin[3]
 * unsigned short  quantizedAabbMax[3]
 * signed   int    escapeIndexOrTriangleIndex
 * </pre>
 * 
 * @author jezek2
 */
public class QuantizedBvhNodes implements Serializable {


	private static final int STRIDE = 4; // 16 bytes
	
	private int[] buf;
	private int size = 0;

	QuantizedBvhNodes() {
		resize(16);
	}
	
	public int add() {
		while (size+1 >= capacity()) {
			resize(capacity()*2);
		}
		return size++;
	}
	
	public int size() {
		return size;
	}
	
	private int capacity() {
		return buf.length / STRIDE;
	}
	
	public void clear() {
		size = 0;
	}
	
	public void resize(int num) {
		int[] oldBuf = buf;
		
		buf = new int[num*STRIDE];
		if (oldBuf != null) {
			System.arraycopy(oldBuf, 0, buf, 0, Math.min(oldBuf.length, buf.length));
		}
	}
	
	static int getNodeSize() {
		return STRIDE*4;
	}
	
	public void set(int destId, QuantizedBvhNodes srcNodes, int srcId) {
		//assert (STRIDE == 4);

		// save field access:
		int[] buf = this.buf;
		int[] srcBuf = srcNodes.buf;

		System.arraycopy(srcBuf, srcId*STRIDE, buf, destId*STRIDE,4);
//		buf[destId*STRIDE+0] = srcBuf[srcId*STRIDE+0];
//		buf[destId*STRIDE+1] = srcBuf[srcId*STRIDE+1];
//		buf[destId*STRIDE+2] = srcBuf[srcId*STRIDE+2];
//		buf[destId*STRIDE+3] = srcBuf[srcId*STRIDE+3];
	}
	
	public void swap(int id1, int id2) {
		//assert (STRIDE == 4);
		
		// save field access:
		int[] buf = this.buf;
		
		int temp0 = buf[id1*STRIDE+0];
		int temp1 = buf[id1*STRIDE+1];
		int temp2 = buf[id1*STRIDE+2];
		int temp3 = buf[id1*STRIDE+3];
		
		buf[id1*STRIDE+0] = buf[id2*STRIDE+0];
		buf[id1*STRIDE+1] = buf[id2*STRIDE+1];
		buf[id1*STRIDE+2] = buf[id2*STRIDE+2];
		buf[id1*STRIDE+3] = buf[id2*STRIDE+3];
		
		buf[id2*STRIDE+0] = temp0;
		buf[id2*STRIDE+1] = temp1;
		buf[id2*STRIDE+2] = temp2;
		buf[id2*STRIDE+3] = temp3;
	}
	
	public int getQuantizedAabbMin(int nodeId, int index) {
		return switch (index) {
			case 0 -> (buf[nodeId * STRIDE + 0]) & 0xFFFF;
			case 1 -> (buf[nodeId * STRIDE + 0] >>> 16) & 0xFFFF;
			case 2 -> (buf[nodeId * STRIDE + 1]) & 0xFFFF;
			default -> throw new UnsupportedOperationException();
		};
	}

	long getQuantizedAabbMin(int nodeId) {
		return (buf[nodeId*STRIDE+0] & 0xFFFFFFFFL) | ((buf[nodeId*STRIDE+1] & 0xFFFFL) << 32);
	}

	void setQuantizedAabbMin(int nodeId, long value) {
		buf[nodeId*STRIDE+0] = (int)value;
		setQuantizedAabbMin(nodeId, 2, (short)((value & 0xFFFF00000000L) >>> 32));
	}

	void setQuantizedAabbMax(int nodeId, long value) {
		setQuantizedAabbMax(nodeId, 0, (short)value);
		buf[nodeId*STRIDE+2] = (int)(value >>> 16);
	}

	void setQuantizedAabbMin(int nodeId, int index, int value) {
		switch (index) {
			case 0 -> buf[nodeId * STRIDE + 0] = (buf[nodeId * STRIDE + 0] & 0xFFFF0000) | (value & 0xFFFF);
			case 1 -> buf[nodeId * STRIDE + 0] = (buf[nodeId * STRIDE + 0] & 0x0000FFFF) | ((value & 0xFFFF) << 16);
			case 2 -> buf[nodeId * STRIDE + 1] = (buf[nodeId * STRIDE + 1] & 0xFFFF0000) | (value & 0xFFFF);
		}
	}

	public int getQuantizedAabbMax(int nodeId, int index) {
		return switch (index) {
			case 0 -> (buf[nodeId * STRIDE + 1] >>> 16) & 0xFFFF;
			case 1 -> (buf[nodeId * STRIDE + 2]) & 0xFFFF;
			case 2 -> (buf[nodeId * STRIDE + 2] >>> 16) & 0xFFFF;
			default -> throw new UnsupportedOperationException();
		};
	}

	long getQuantizedAabbMax(int nodeId) {
		return ((buf[nodeId*STRIDE+1] & 0xFFFF0000L) >>> 16) | ((buf[nodeId*STRIDE+2] & 0xFFFFFFFFL) << 16);
	}

	void setQuantizedAabbMax(int nodeId, int index, int value) {
		switch (index) {
			case 0 -> buf[nodeId * STRIDE + 1] = (buf[nodeId * STRIDE + 1] & 0x0000FFFF) | ((value & 0xFFFF) << 16);
			case 1 -> buf[nodeId * STRIDE + 2] = (buf[nodeId * STRIDE + 2] & 0xFFFF0000) | (value & 0xFFFF);
			case 2 -> buf[nodeId * STRIDE + 2] = (buf[nodeId * STRIDE + 2] & 0x0000FFFF) | ((value & 0xFFFF) << 16);
		}
	}
	
	private int escapeIndexOrTriangleIndex(int nodeId) {
		return buf[nodeId*STRIDE+3];
	}
	
	void escapeIndexOrTriangleIndex(int nodeId, int value) {
		buf[nodeId*STRIDE+3] = value;
	}
	
	boolean isLeafNode(int nodeId) {
		// skipindex is negative (internal node), triangleindex >=0 (leafnode)
		return (escapeIndexOrTriangleIndex(nodeId) >= 0);
	}

	int escapeIndex(int nodeId) {
		//assert (!isLeafNode(nodeId));
		return -escapeIndexOrTriangleIndex(nodeId);
	}

	int triangleIndex(int nodeId) {
		//assert (isLeafNode(nodeId));
		// Get only the lower bits where the triangle index is stored
		return (escapeIndexOrTriangleIndex(nodeId) & ~((~0) << (31 - OptimizedBvh.MAX_NUM_PARTS_IN_BITS)));
	}

	int partId(int nodeId) {
		//assert (isLeafNode(nodeId));
		// Get only the highest bits where the part index is stored
		return (escapeIndexOrTriangleIndex(nodeId) >>> (31 - OptimizedBvh.MAX_NUM_PARTS_IN_BITS));
	}
	
	static int coord(long vec, int index) {
		return (int) switch (index) {
            case 0 -> ((vec & 0x00000000FFFFL));
            case 1 -> ((vec & 0x0000FFFF0000L) >>> 16);
            case 2 -> ((vec & 0xFFFF00000000L) >>> 32);
            default -> throw new UnsupportedOperationException();
        } & 0xFFFF;
	}
	
}