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

package com.github.fge.grappa.run.context;

import com.github.fge.grappa.buffers.InputBuffer;
import com.github.fge.grappa.exceptions.GrappaException;
import com.github.fge.grappa.exceptions.InvalidGrammarException;
import com.github.fge.grappa.matchers.ActionMatcher;
import com.github.fge.grappa.matchers.MatcherType;
import com.github.fge.grappa.matchers.base.Matcher;
import com.github.fge.grappa.matchers.wrap.ProxyMatcher;
import com.github.fge.grappa.run.MatchHandler;
import com.github.fge.grappa.stack.ValueStack;
import com.github.fge.grappa.support.IndexRange;
import com.github.fge.grappa.support.Position;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import jcog.data.list.Lst;
import org.jetbrains.annotations.Nullable;


/**
 * <p>The Context implementation orchestrating most of the matching process.</p>
 *
 * <p>The parsing process works as follows:</p>
 *
 * <p>After the rule tree (which is in fact a directed and potentially even
 * cyclic graph of {@link Matcher} instances) has been created a root
 * MatcherContext is instantiated for the root rule (Matcher). A subsequent call
 * to {@link #runMatcher()} starts the parsing process.</p>
 *
 * <p>The MatcherContext delegates to a given {@link MatchHandler} to call
 * {@link Matcher#match(MatcherContext)}, passing itself to the Matcher which
 * executes its logic, potentially calling sub matchers. For each sub matcher
 * the matcher creates/initializes a subcontext with {@link
 * Matcher#getSubContext(MatcherContext)} and then calls {@link #runMatcher()}
 * on it.</p>
 *
 * <p>This basically creates a stack of MatcherContexts, each corresponding to
 * their rule matchers.</p>
 *
 * <p>At each point during the parsing process the matchers and action
 * expressions have access to the current MatcherContext and all "open" parent
 * MatcherContexts through the {@link #getParent()} chain.</p>
 */
@SuppressWarnings("InstanceVariableMayNotBeInitialized")
public final class DefaultMatcherContext<V> implements MatcherContext<V> {

    private static final Joiner JOINER = Joiner.on('/');

    private final InputBuffer inputBuffer;
    private final ValueStack<V> valueStack;
    private final MatchHandler matchHandler;
    private final DefaultMatcherContext<V> parent;
    private final int level;

    private DefaultMatcherContext<V> subContext;
    private int startIndex;
    private int currentIndex;
    private Matcher matcher;
    private String path;
    private boolean hasError;

    /**
     * Initializes a new root MatcherContext.
     *
     * @param inputBuffer  the InputBuffer for the parsing run
     * @param valueStack   the ValueStack instance to use for the parsing run
     * @param matchHandler the MatcherHandler to use for the parsing run
     * @param matcher      the root matcher
     */
    public DefaultMatcherContext(InputBuffer inputBuffer,
                                 ValueStack<V> valueStack,
                                 MatchHandler matchHandler,
                                 Matcher matcher) {
        this(inputBuffer,
                valueStack,
                matchHandler, null, 0);

        // TODO: what the...
        this.matcher = ProxyMatcher.unwrap(matcher);
    }

    private DefaultMatcherContext(InputBuffer inputBuffer,
                                  ValueStack<V> valueStack, MatchHandler matchHandler,
                                  @Nullable DefaultMatcherContext<V> parent,
                                  int level) {
        this.inputBuffer = inputBuffer;
        this.valueStack = valueStack;
        this.matchHandler = matchHandler;
        this.parent = parent;
        this.level = level;
    }

    @Override
    public String toString() {
        return getPath();
    }

    //////////////////////////////// CONTEXT INTERFACE /////////////////////////

    @Override
    public MatcherContext<V> getParent() {
        return parent;
    }

    @Override
    public InputBuffer getInputBuffer() {
        return inputBuffer;
    }

    @Override
    public int getStartIndex() {
        return startIndex;
    }

    @Override
    public void setStartIndex(int startIndex) {
        Preconditions.checkArgument(startIndex >= 0);
        this.startIndex = startIndex;
    }

    @Override
    public Matcher getMatcher() {
        return matcher;
    }

    @Override
    public void setMatcher(Matcher matcher) {
        this.matcher = matcher;
    }

    @Override
    public boolean atEnd() {
        return currentIndex == inputBuffer.length();
    }

    @Override
    public char getCurrentChar() {
        return inputBuffer.charAt(currentIndex);
    }

    @Override
    public int getCurrentCodePoint() {
        return inputBuffer.codePointAt(currentIndex);
    }

    @Override
    public int getCurrentIndex() {
        return currentIndex;
    }

    @Override
    public void setCurrentIndex(int currentIndex) {
        Preconditions.checkArgument(currentIndex >= 0);
        this.currentIndex = currentIndex;
    }

    @Override
    public int getLevel() {
        return level;
    }

    @Override
    public boolean inPredicate() {
        DefaultMatcherContext<V> other = this;
        while (true) {
            //noinspection SimplifiableIfStatement
            if (matcher.getType() == MatcherType.PREDICATE) return true;
            if (parent == null) return false;
            other = other.parent;
        }
    }

    @Override
    public boolean hasError() {
        return hasError;
    }

    @Override
    public CharSequence getMatch() {
        DefaultMatcherContext<V> ctx = subContext;
        return inputBuffer.extract(ctx.startIndex, ctx.currentIndex);
    }

    @Override
    public char getFirstMatchChar() {
        int index = subContext.startIndex;
        if (subContext.currentIndex > index)
            return inputBuffer.charAt(index);

        // TODO: figure out why it says that
        throw new InvalidGrammarException("getFirstMatchChar called "
                + "but previous rule did not match anything");
    }

    @Override
    public int getMatchStartIndex() {
        return subContext.startIndex;
    }

    @Override
    public int getMatchEndIndex() {
        return subContext.currentIndex;
    }

    @Override
    public int getMatchLength() {
        return subContext.currentIndex - subContext.startIndex;
    }

    //////////////////////////////// PUBLIC ////////////////////////////////////

    @Override
    public Position getPosition() {
        return inputBuffer.getPosition(currentIndex);
    }

    @Override
    public IndexRange getMatchRange() {
        return new IndexRange(subContext.startIndex, subContext.currentIndex);
    }

    @Override
    public ValueStack<V> getValueStack() {
        return valueStack;
    }

    @Override
    public void advanceIndex(int delta) {
        currentIndex += delta;
    }

    @Override
    public MatcherContext<V> getBasicSubContext() {
        if (subContext == null) {
            // init new level
            subContext = new DefaultMatcherContext<>(inputBuffer, valueStack,
                    matchHandler, this, level + 1);
        } else {
            // we always need to reset the MatcherPath, even for actions
            subContext.path = null;
        }
        return subContext;
    }

    @Override
    public MatcherContext<V> getSubContext(Matcher matcher) {
        DefaultMatcherContext<V> sc
                = (DefaultMatcherContext<V>) getBasicSubContext();
        sc.matcher = matcher;
        sc.setStartIndex(currentIndex);
        sc.setCurrentIndex(currentIndex);
        sc.hasError = false;
        return sc;
    }

    @Override
    public boolean runMatcher() {
        boolean ret;
        try {
            //ret = matchHandler.match(this);
            ret = matcher.match(this);
        } catch (Throwable e) {
            throw throwGrappa(e);
        }

        // Retire this context
        // TODO: what does the above really mean?
        matcher = null;
        if (ret && parent != null)
            parent.currentIndex = currentIndex;
        return ret;
    }

    private GrappaException throwGrappa(Throwable e)  {
        return e instanceof GrappaException ? ((GrappaException)e) : new GrappaException(String.format(
                "exception thrown when parsing %s '%s' at input position %s",
                matcher instanceof ActionMatcher ? "action" : "rule", getPath(),
                inputBuffer.getPosition(currentIndex)), e);
    }

    private String getPath() {
        if (path != null)
            return path;

        Lst<String> list = new Lst<>();

        MatcherContext<V> ctx;
        Matcher matcher;

        for (ctx = this; ctx != null; ctx = ctx.getParent()) {
            matcher = ctx.getMatcher();
            // TODO: can this really happen?
            if (matcher != null)
                list.add(matcher.toString());
        }

        list.reverseThis();

        path = JOINER.join(list);

        return path;
    }
}