/*
 * This file is part of Beads. See http:
 */
package net.beadsproject.beads.buffers;

import jcog.signal.tensor.ArrayTensor;
import net.beadsproject.beads.data.WaveFactory;

/**
 * Creates a {@link Buffer} consisting of a triangle wave in the range [-1,1].
 *
 * @author ollie
 * @see Buffer BufferFactory
 */
public class TriangleWave extends WaveFactory {

    /* (non-Javadoc)
     * @see net.beadsproject.beads.data.BufferFactory#generateBuffer(int)
     */
    @Override
    public ArrayTensor get(int bufferSize) {
        int size = bufferSize;
        ArrayTensor b = new ArrayTensor(size);
        for (int i = 0; i < bufferSize; i++) {
            b.data[i] = ((i < bufferSize / 2.0f ? i / (bufferSize / 2.0f) : 1.0f - (i - bufferSize / 2.0f) / (bufferSize / 2.0f)) * 2.0f) - 1.0f;
        }
        return b;
    }

    /* (non-Javadoc)
     * @see net.beadsproject.beads.data.BufferFactory#getName()
     */
    @Override
    public String getName() {
        return "Triangle";
    }

}