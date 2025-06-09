/*
 * Copyright (C) 2009-2011 Mathias Doenitz
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.fge.grappa.misc;

import com.google.common.collect.ImmutableList;

import java.util.List;

/**
 * A simple, immutable {@link GraphNode} implementation.
 *
 * @param <T> the actual implementation type of this ImmutableGraphNode
 */
// TODO: rename; this class IS NOT immutable.
public class ImmutableGraphNode<T extends GraphNode<T>> implements GraphNode<T> {
    private final List<T> children;

    public ImmutableGraphNode()
    {
        this(ImmutableList.of());
    }

    public ImmutableGraphNode(final List<T> children) {
        //Objects.requireNonNull(children);
        /*
         * ImmutableLinkedList has no such thing as a "safe copy constructor";
         * ImmutableList (Guava's, that is) does; what is more, if the argument
         * to .copyOf() is _also_ a (Guava...) ImmutableList, it won't even make
         * a copy.
         */
        //this.children = children.isEmpty() ? new Lst<>() : ImmutableList.copyOf(children);

        this.children = children;
    }

    @SuppressWarnings("ReturnOfCollectionOrArrayField")
    @Override
    public final List<T> getChildren()
    {
        return children;
    }
}