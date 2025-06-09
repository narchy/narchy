package jcog.memoize;

import jcog.data.byt.DynBytes;
import jcog.pri.PriProxy;
import net.jpountz.lz4.LZ4Compressor;
import net.jpountz.lz4.LZ4Factory;

import java.util.Arrays;

public class ByteKeyExternal implements ByteKey {

    private static final ThreadLocal<DynBytes> buffer = ThreadLocal.withInitial(
            ()->new DynBytes(8 * 1024));

    public final DynBytes key;

    protected int hash;

    final int start;
    private int end = -1;
    byte[] k;

    public ByteKeyExternal() {
		this(buffer.get());
    }

    public ByteKeyExternal(DynBytes key) {
        this.key = key;
        this.start = key.length();
    }

    public void internedNew(PriProxy/*ByteKeyInternal*/ i) {

    }

    protected final void commit() {
        //TODO optional compression
        var l = key.length();
        if (l >= COMPRESSION_THRESHOLD)
            l = compress(l);

        this.hash = key.hashCode(start, this.end = l);
    }


    public void close() {
        key.rewind(end-start);
//        key.close();
//        key = null;
        k = null;
    }


    @Override
    public final int hashCode() {
        return hash;
    }

    @Override
    public final boolean equals(Object obj) {
        return ByteKey.equals(this, (ByteKey) obj);
    }

    @Override
    public final int length() {
        return end-start;
    }

//    @Override
//    public byte at(int i) {
//        return key.at(start + i);
//        //return key.at(i);
//    }

    @Override
    public byte[] array() {
        if (k==null)
            k = Arrays.copyOfRange(key.arrayDirect(), start, end);
        return k;
    }

    @Override
    public boolean equals(ByteKeyExternal y, int at, int len) {
        throw new UnsupportedOperationException();
    }


    /** Integer.MAX_VALUE to disable. untested */
    private static final int COMPRESSION_THRESHOLD =
        //128;
        Integer.MAX_VALUE; //DISABLED

    private static final LZ4Compressor LZ4 = LZ4Factory.fastestJavaInstance()
        .fastCompressor();
        //.highCompressor();

    private static final int COMPRESSION_PADDING = 16;

    private int compress(int l) {
        //var compressed = QuickLZ.compress(key.arrayCopy(start));

        var raw = key.arrayDirect();
        var compressed = new byte[l + COMPRESSION_PADDING];
        int cl = LZ4.compress(raw, start, l, compressed, 0);
        if (cl < l) {
            key.length(start).write(compressed, 0, cl);
            l = cl;
        }
        return l;
    }



}