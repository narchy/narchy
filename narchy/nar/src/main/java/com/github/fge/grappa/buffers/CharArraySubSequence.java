package com.github.fge.grappa.buffers;

import jcog.TODO;
import jcog.util.ArrayUtil;

class CharArraySubSequence implements CharSequence {

    final char[] c;
    private final int start, end;

    CharArraySubSequence(char[] seq, int start, int end) {
        this.c = seq;
        this.start = start;
        this.end = end;
    }

    @Override
    public boolean equals(Object x) {
        throw new TODO();
    }

    @Override
    public int hashCode() {
        throw new TODO();
    }

    @Override
    public String toString() {
        return new String(ArrayUtil.subarray(c, start, end));
    }

    @Override
    public int length() {
        return end - start;
    }

    @Override
    public char charAt(int index) {
        return c[start + index];
    }

    @Override
    public CharSequence subSequence(int start, int end) {
        return start == 0 && end == length() ? this : new CharArraySubSequence(c, this.start + start, this.start + end);
    }
}