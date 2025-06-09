package jcog.data.byt;

import com.google.common.io.ByteArrayDataOutput;
import jcog.TODO;
import jcog.Util;
import jcog.data.IntCoding;
import jcog.util.ArrayUtil;

import java.io.DataOutput;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

import static jcog.data.IntCoding.decodeZigZagInt;
import static jcog.data.IntCoding.decodeZigZagLong;

/**
 * mutable resizable dynamic byte array for fast single-thread use
 *   includes zigzag integer encode/decode operations
 *
 * @see org.objectweb.asm.ByteVector
 */
public sealed class DynBytes implements ByteArrayDataOutput, Appendable, ByteSequence, Comparable<DynBytes> permits RecycledDynBytes {

    /**
     * must remain final for global consistency
     * might as well be 1.0, if it's already compressed to discover what this is, just keep it
     */
    /**
     * must remain final for global consistency
     */
    public int length;
    public byte[] bytes;

    public DynBytes(int bufferSize) {
        this.bytes = alloc(bufferSize);
    }

    public DynBytes(byte[] zeroCopy) {
        this(zeroCopy, zeroCopy.length);
    }

    public DynBytes(byte[] zeroCopy, int length) {
        this.bytes = zeroCopy;
        this.length = length;
    }

    @Override public final int compareTo(DynBytes d) {
        if (d==this) return 0;
        var n = this.length;
        var len = Integer.compare(n, d.length);
        return len != 0 ? len : Arrays.compare(bytes, 0, n, d.bytes, 0, n);
    }

    @Override
    public int hashCode() {
        return Util.hash(bytes, 0, length);
    }

    @Override
    public boolean equals(Object obj) {
        return this == obj || equivalent((DynBytes) obj);
    }

    public final boolean equivalent(DynBytes d) {
        var len = this.length;
        return d.length == len && Arrays.equals(bytes, 0, len, d.bytes, 0, len);
    }

    @Override
    public final int length() {
        return length;
    }

    public final DynBytes length(int newLength) {
        this.length = newLength;
        return this;
    }

    public final DynBytes rewind(int num) {
        return length(length - num);
//        if (/*num > 0 && */(length -= num) < 0)
//            throw new BufferUnderflowException();
//        return this;
    }

    @Override
    public final byte at(int index) {
        return bytes[index];
    }

    @Override
    public final ByteSequence subSequence(int start, int end) {
        var d = end - start;
        if (d == length)
            return this;
        else if (d == 1)
            return new OneByteSeq(at(start));
        else
            return new ArrayBytes(bytes, start, end); //not window since this is mutable
    }

    public final byte[] subBytes(int start, int end) {
        //assert(end!=start);
        return Arrays.copyOfRange(bytes, start, end);
    }

    @Override
    public final void write(int v) {
        writeByte(v);
    }

    @Override
    public final void writeByte(int v) {
        ensureCapacity(1);
        writeByteFast((byte) v);
    }

    public void writeByteFast(byte v) {
        this.bytes[this.length++] = v;
    }

    /**
     * combo: (byte, int)
     */
    public final void writeByteInt(byte b, int i) {
        var len = ensureCapacity(1 + 4);
        var bytes = this.bytes;
        bytes[len++] = b;
        this.length = writeInt(len, i, bytes);
    }

    public final void fillBytes(byte b, int from, int to) {
        Arrays.fill(bytes, from, to, b);
    }

    public final void fillBytes(byte b, int next) {
        Arrays.fill(bytes, this.length, this.length += next, b);
    }


    @Override
    public final void write(byte[] bytes) {
        this.write(bytes, 0, bytes.length);
    }

    @Override
    public final void write(byte[] x, int start, int len) {
        var pos = ensureCapacity(len);
        System.arraycopy(x, start, this.bytes, pos, len);
        this.length += len;
    }

    /** returns current position, which remains constant regardless of reallocation */
    protected int ensureCapacity(int extra) {
        var bytes = this.bytes;
        int capacity = bytes.length, pos = this.length;
        if (capacity < pos + extra)
            grow(extra, bytes, capacity);
        return pos;
    }

    private void grow(int extra, byte[] bytes, int capCurrent) {
        this.bytes = Arrays.copyOf(bytes,
            capCurrent + (capCurrent / 2) + extra
            //(int)Math.ceil((current * GROWTH_RATE) + extra)
            //org.apache.lucene.util.ArrayUtil.oversize(current + extra, 1)
        );
    }

    /** subclasses can implement custom allocation strategies */
    protected static byte[] alloc(int bufferSize) {
        return bufferSize > 0 ? new byte[bufferSize] : null;
    }


    /**
     * isnt this the same as just compact()?
     */
    @Override
    @Deprecated
    public final byte[] arrayCompactDirect() {
        return compact();
    }

//    public final byte[] arrayDeflate() {
//        Deflater d = new Deflater(
//                //BEST_SPEED,
//                BEST_COMPRESSION,
//                false);
//        d.reset();
//        d.setInput(bytes, 0, len);
//        d.finish();
//
//        ensureSized(len);
//        int newLen = d
//                .deflate(bytes, len, len, FULL_FLUSH);
//        d.end();
//        if (newLen < len) {
//
//            //d.end();
//            len = newLen;
//            //compact();
//            this.bytes = Arrays.copyOfRange(bytes, len, len + newLen);
//        }
//        return arrayDirect();
//    }

    public final byte[] arrayDirect() {
        return bytes;
    }


    public final byte[] arrayCopy() {
        return Arrays.copyOf(bytes, length);
    }
    public final byte[] arrayCopy(int offset) {
        return Arrays.copyOfRange(bytes, offset, length);
    }

    public final byte[] compact() {
        return compact(false);
    }

    public final byte[] compact(boolean force) {
        return compact(null, force);
    }

    public byte[] compact(byte[] forceIfSameAs, boolean force) {
        var l = this.length;
        if (l > 0) {
            var b = this.bytes;
            //noinspection ArrayEquality
            if (force || b.length != l || forceIfSameAs == bytes)
                return this.bytes = Arrays.copyOfRange(b, 0, l);
            else
                return bytes;
        } else {
            return this.bytes = ArrayUtil.EMPTY_BYTE_ARRAY;
        }
    }


    @Override
    public final void toArray(byte[] c, int offset) {
        System.arraycopy(bytes, 0, c, offset, length);
    }

    @Override
    public String toString() {
        return Arrays.toString(ArrayUtil.subarray(bytes, 0, length));
    }

    public String toStringFromBytes() {
        return new String(bytes, 0, length);
    }

    @Override
    public final void writeBoolean(boolean v) {
        ensureCapacity(1);
        writeByteFast((byte) (v ? 1 : 0));
    }

    @Override
    public final void writeShort(int v) {
        var s = ensureCapacity(2);
        this.length = writeNextShort(s, v, this.bytes);
    }

    private static int writeNextShort(int s, int v, byte[] e) {
        e[s++] = (byte) (v >> 8);
        e[s++] = (byte) v;
        return s;
    }

    @Override
    public final void writeChar(int v) {
        writeShort(v);
//        int s = ensureSized(2);
//        byte[] e = this.bytes;
//        e[s] = (byte) (v >> 8);
//        e[s + 1] = (byte) v;
//        this.len += 2;
    }

    @Override
    public final void writeInt(int v) {
        var s = ensureCapacity(4);
        this.length = writeInt(s, v, this.bytes);
    }

    private static int writeInt(int s, int v, byte[] e) {
        e[s++] = (byte) (v >> 24);
        e[s++] = (byte) (v >> 16);
        return writeNextShort(s, v, e);
    }

    @Override
    public final void writeLong(long v) {
        var s = ensureCapacity(8);
        var e = this.bytes;
        e[s++] = (byte) ((int) (v >> 56));
        e[s++] = (byte) ((int) (v >> 48));
        e[s++] = (byte) ((int) (v >> 40));
        e[s++] = (byte) ((int) (v >> 32));
        e[s++] = (byte) ((int) (v >> 24));
        e[s++] = (byte) ((int) (v >> 16));
        e[s++] = (byte) ((int) (v >> 8));
        e[s++] = (byte) ((int) v);
        this.length = s;
    }

    @Override
    public final void writeFloat(float v) {
        writeInt(Float.floatToIntBits(v));
    }

    @Override
    public final void writeDouble(double v) {
        throw new TODO();
    }

    @Override
    public void writeBytes(String s) {
        throw new TODO();
    }


    @Override
    @Deprecated
    public byte[] toByteArray() {
        return bytes;
    }

    @Override
    public void writeChars(String s) {
        throw new TODO();
    }

    //final UTF8Writer utf8 = new UTF8Writer();
    @Override
    public void writeUTF(String s) {
        write(s.getBytes());
    }

    @Override
    public final Appendable append(CharSequence csq) {
        return append(csq, 0, csq.length());
    }

    /** TODO optimize with writeNextShort's */
    @Override public final Appendable append(CharSequence csq, int start, int end) {
        for (var i = start; i < end;)
            writeChar(csq.charAt(i++));
        return this;
    }

    @Override
    public final Appendable append(char c) {
        writeChar(c);
        return this;
    }

    public final void appendTo(DataOutput out) throws IOException {
        out.write(bytes, 0, length);
    }

    public final void appendTo(OutputStream o) throws IOException {
        o.write(bytes, 0, length);
    }
    public final void writeUnsignedByte(int i) {
        writeByte(i & 0xff);
    }

    public final DynBytes clear() {
        length = 0;
        return this;
    }

    public HashedArrayBytes rawCopy() {
        return new HashedArrayBytes(compact(true));
    }

    public void writeUnsignedLong(long x) {
        var pos = ensureCapacity(8 + 2 /* max */);
        this.length += IntCoding.encodeUnsignedVariableLong(x, bytes, pos);
    }

    public void writeUnsignedInt(int x) {
        var pos = ensureCapacity(4 + 1 /* max */);
        this.length += IntCoding.encodeUnsignedVariableInt(x, bytes, pos);
    }

    public void writeZigZagLong(long x) {
        var pos = ensureCapacity(8 + 2 /* max */);
        this.length += IntCoding.encodeZigZagVariableLong(x, bytes, pos);
    }

    public void writeZigZagInt(int x) {
        var pos = ensureCapacity(4 + 1 /* max */);
        this.length += IntCoding.encodeZigZagVariableInt(x, bytes, pos);
    }

    private long readVarEncodedUnsignedLong() {
        long unsigned = 0;
        var i = 0;
        long b;
        while (((b = bytes[length++] & 0xff) & 0x80) != 0) {
            unsigned |= (b & 0x7f) << i;
            i += 7;
        }
        return unsigned | (b << i);
    }

    public int readVarEncodedUnsignedInt() {
        var unsigned = 0;
        var i = 0;
        int b;
        while (((b = bytes[length++] & 0xff) & 0x80) != 0) {
            unsigned |= (b & 0x7f) << i;
            i += 7;
        }
        return unsigned | (b << i);
    }

    public long readZigZagLong() {
        return decodeZigZagLong(readVarEncodedUnsignedLong());
    }

    public int readZigZagInt() {
        return decodeZigZagInt(readVarEncodedUnsignedInt());
    }

    public void set(int n, byte b) {
        bytes[n] = b;
    }

    public void write(DynBytes b) {
        write(b.bytes, 0, b.length);
    }

    public final int hashJava() {
        return Util.hashJava(bytes, length);
    }

    public final int hashFNV() {
        return hashCode();
    }

    public static long hashMurmur() {
        //return Murmur3Hash.hash64(bytes, len); //<-- TODO requires any padding at 8 byte boundaries to be zero'd
        throw new TODO();
    }

    public void writeNumber(byte x, int radix) {
        if (radix < 10) throw new TODO();

        if (x < 0) {
            writeByte('-'); //negative sign
            x = (byte) -x;
        }

        if (x < 10)
            writeByte('0' + x);
        else {
            var xx = Integer.toString(x, radix); //TODO without creating String
            var n = xx.length();
            ensureCapacity(n);
            for (var i = 0; i < n; i++)
                writeByteFast((byte)xx.charAt(i));
        }

    }

//    private final static int MIN_COMPRESSION_BYTES = 64;
//    private final static float minCompressionRatio = 1f;
//    public int compress() {
//        return compress(0);
//    }
//
//    /**
//     * return length of the compressed region (not including the from offset).
//     * or -1 if compression was not applied
//     */
//    public int compress(int from) {
//
//
//        int to = length();
//        int len = to - from;
//        if (len < MIN_COMPRESSION_BYTES) {
//            return -1;
//        }
//
//
//        //TODO compress to a temporary suffix the end of the buffer rather than allocating new byte[]
//        int bufferLen = from + Snappy.maxCompressedLength(len);
//
//
//        byte[] compressed = new byte[bufferLen];
//
//        int compressedLength = Snappy.compress(
//                this.bytes, from, len,
//                compressed, from);
//
//
//        if (compressedLength < (len * minCompressionRatio)) {
//
//            if (from > 0)
//                System.arraycopy(this.bytes, 0, compressed, 0, from);
//
//
//            this.bytes = compressed;
//            this.len = from + compressedLength;
//            return compressedLength;
//        } else {
//            return -1;
//
//        }
//    }

}