/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http:
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jcog.signal;

import jcog.Util;
import org.eclipse.collections.api.block.function.primitive.FloatFloatToFloatFunction;

import java.lang.invoke.VarHandle;

public class MutableFloat extends PlainMutableFloat {
    private static final VarHandle V = Util.VAR(MutableFloat.class, "v", float.class);

    /**
     * Constructs a new MutableFloat with the default value of zero.
     */
    public MutableFloat() {
        super();
    }

    public MutableFloat(float v) {
        this();
        setPlain(v);
    }

    @Override
    public final float asFloat() {
        return (float) V.getAcquire(this);
    }


    /**
     * Sets the value.
     *
     * @param value  the value to setAt
     */
    @Override
    public void set(float value) {
        V.setRelease(this, post(value));
    }

    @Override public final float getAndSet(float p) {
        return _getAndSet(post(p));
    }

    private float _getAndSet(float y) {
        return (float)V.getAndSet(this, y);
    }

//    protected final float getAndSetNaN() {
//        return _getAndSet(Float.NaN);
//    }
    protected final boolean wasAndSetNaN() {
        float p = _getAndSet(Float.NaN);
        return p==p;
    }


    public final float update(FloatFloatToFloatFunction f, float y) {
        float prev, next;
        do {
            prev = asFloat();
            next = post(f.valueOf(prev, y));
        } while (
                next!=prev
                &&
                !V.weakCompareAndSetRelease(this, prev, next)
                //!V.compareAndSet(this, prev, next)
        );
        return next;
    }


    public final void setPlain(float value) {
        v = post(value);
        //V.set(this, post(value));
    }

    @Override
    public float add(float y) {
        return update(Float::sum, y);
    }

    @Override
    public final float mul(float y) {
        return update((x,Y)->x * Y, y);
    }


//    /**
//     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance
//     * immediately after the addition operation. This method is not thread safe.
//     *
//     * @param operand the quantity to addAt, not null
//     * @return the value associated with this instance after adding the operand
//     * @since 3.5
//     */
//    public float addAndGet(final float operand) {
//        this.value += operand;
//        return value;
//    }

//    /**
//     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance
//     * immediately after the addition operation. This method is not thread safe.
//     *
//     * @param operand the quantity to addAt, not null
//     * @throws NullPointerException if {@code operand} is null
//     * @return the value associated with this instance after adding the operand
//     * @since 3.5
//     */
//    public float addAndGet(final Number operand) {
//        this.value += operand.floatValue();
//        return value;
//    }
//
//    /**
//     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance
//     * immediately prior to the addition operation. This method is not thread safe.
//     *
//     * @param operand the quantity to addAt, not null
//     * @return the value associated with this instance immediately before the operand was added
//     * @since 3.5
//     */
//    public float getAndAdd(final float operand) {
//        final float last = value;
//        this.value += operand;
//        return last;
//    }
//
//    /**
//     * Increments this instance's value by {@code operand}; this method returns the value associated with the instance
//     * immediately prior to the addition operation. This method is not thread safe.
//     *
//     * @param operand the quantity to addAt, not null
//     * @throws NullPointerException if {@code operand} is null
//     * @return the value associated with this instance immediately before the operand was added
//     * @since 3.5
//     */
//    public float getAndAdd(final Number operand) {
//        final float last = value;
//        this.value += operand.floatValue();
//        return last;
//    }


    //    /** returns the change, between 0 and x */
//    public float subAtMost(float x) {
//        float v = value;
//        if (v > x) {
//            value -= x;
//            return x;
//        } else {
//
//            setAt(0f);
//            return v;
//        }
//    }

}