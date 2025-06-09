package com.github.fge.grappa.buffers;

import com.github.fge.grappa.support.IndexRange;
import com.github.fge.grappa.support.Position;

public final class CharSequenceInputBuffer implements InputBuffer {

    static final Position zero = new Position(0, 0);
    //		final Future<LineCounter> lineCounter = null;
    private final char[] charSequence;

    public CharSequenceInputBuffer(CharSequence charSequence) {
        int length = charSequence.length();
        this.charSequence = new char[length];
        for (int i = 0; i < length; i++) {
            this.charSequence[i] = charSequence.charAt(i);
        }
    }

    @Override
    public char charAt(int index) {
        return charSequence[index];
    }

    @SuppressWarnings("ImplicitNumericConversion")
    @Override
    public int codePointAt(int index) {
        int length = charSequence.length;
        if (index >= length)
            return -1;

        char c = charAt(index);
        if (index == length - 1 || !Character.isHighSurrogate(c))
            return c;
        char c2 = charAt(index + 1);
        return Character.isLowSurrogate(c2) ? Character.toCodePoint(c, c2) : c;
    }

    @Override
    public CharSequence subSequence(int start, int end) {
        return end > start ? new CharArraySubSequence(charSequence, start, end) : "";
    }

    @Override
    public CharSequence extract(int start, int end) {
        return subSequence(Math.max(start, 0), Math.min(end, charSequence.length));
    }

    @Override
    public CharSequence extract(IndexRange range) {
        return extract(range.start, range.end);
    }

    @Override
    public Position getPosition(int index) {
        return zero; //HACK
        //return Futures.getUnchecked(lineCounter).toPosition(index);
    }

    @Override
    public String extractLine(int lineNumber) {
        throw new UnsupportedOperationException();
//            Preconditions.checkArgument(lineNumber > 0, "line number is negative");
//            final LineCounter counter = lineCounter.get(); //Futures.getUnchecked(lineCounter);
//            final Range<Integer> range = counter.getLineRange(lineNumber);
//            final int start = range.lowerEndpoint();
//            int end = range.upperEndpoint();
//            if (charAt(end - 1) == '\n')
//                end--;
//            if (charAt(end - 1) == '\r')
//                end--;
//            return extract(start, end);
    }

    @Override
    public IndexRange getLineRange(int lineNumber) {
        throw new UnsupportedOperationException();
//            final Range<Integer> range
//                    = Futures.getUnchecked(lineCounter).getLineRange(lineNumber);
//            return new IndexRange(range.lowerEndpoint(), range.upperEndpoint());
    }

    @Override
    public int getLineCount() {
        throw new UnsupportedOperationException();
//            return Futures.getUnchecked(lineCounter).getNrLines();
    }

    @Override
    public int length() {
        return charSequence.length;
    }

}
