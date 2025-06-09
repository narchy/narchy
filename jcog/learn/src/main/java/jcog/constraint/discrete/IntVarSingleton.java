/*
 * Copyright 2016, Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http:
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package jcog.constraint.discrete;

import jcog.constraint.discrete.propagation.PropagationQueue;
import jcog.constraint.discrete.propagation.Propagator;
import jcog.constraint.discrete.trail.Trail;

public class IntVarSingleton extends IntVar {

    private final PropagationQueue pQueue;
    private final Trail trail;
    private final int value;

    public IntVarSingleton(PropagationQueue pQueue, Trail trail, int value) {
        this.pQueue = pQueue;
        this.trail = trail;
        this.value = value;
    }

    @Override
    public PropagationQueue propagQueue() {
        return pQueue;
    }

    @Override
    public Trail trail() {
        return trail;
    }

    @Override
    public int min() {
        return value;
    }

    @Override
    public int max() {
        return value;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public boolean isAssigned() {
        return true;
    }

    @Override
    public boolean contains(int value) {
        return this.value == value;
    }

    @Override
    public boolean assign(int value) {
        return this.value == value;
    }

    @Override
    public boolean remove(int value) {
        return this.value != value;
    }

    @Override
    public boolean updateMin(int value) {
        return this.value >= value;
    }

    @Override
    public boolean updateMax(int value) {
        return this.value <= value;
    }

    @Override
    public int copyDomain(int[] array) {
        array[0] = value;
        return 1;
    }

    @Override
    public void watchChange(Propagator propagator) {
    }

    @Override
    public void watchAssign(Propagator propagator) {
    }

    @Override
    public void watchBounds(Propagator propagator) {
    }
}