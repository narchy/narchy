package jcog.memoize;

import jcog.Str;

import java.util.Arrays;

public interface ByteKey  {

    /** of size equal or greater than length() */
    byte[] array();

    int length();

    @Override int hashCode();

    boolean equals(ByteKeyExternal y, int at, int len);

    default boolean equals(ByteKeyExternal y, int len) {
        return equals(y, 0, len);
    }

    static String toString(ByteKey b) {
        return Str.i(b.array(),0, b.length(), 16) + " [" + Integer.toUnsignedString(b.hashCode(),32) + ']';
    }

    static boolean equals(ByteKey x, ByteKey y) {
        return x==y || (x.hashCode() == y.hashCode() && _equals(x, y));
    }

    static boolean _equals(ByteKey x, ByteKey y) {
        int l = x.length();
        if (l != y.length())
            return false;

        if (x instanceof ByteKeyExternal xb) {
            if (y instanceof ByteKeyExternal)
                throw new UnsupportedOperationException();

            return y.equals(xb, l);
        } else {
            return y instanceof ByteKeyExternal yb ?
                x.equals(yb, l) :
                Arrays.equals(x.array(), 0, l, y.array(), 0, l);
        }
    }
}
