package jcog.bloom.hash;

import jcog.data.byt.DynBytes;

public abstract class DynBytesHasher<X> implements Hasher<X> {

    protected abstract void write(X x, DynBytes d);

    private final DynBytes d;

//	public DynBytesHasher() {
//		this(RecycledDynBytes.get()); //<- TODO needs closed
//	}

    protected DynBytesHasher(int bytesCap) {
        this(new DynBytes(bytesCap));
    }

    protected DynBytesHasher(DynBytes d) {
        this.d = d;
    }

    @Override
    public int hash1(X t) {
        write(t, d.clear());
        return d.hashJava();
    }

    @Override
    public int hash2(X t) {
        return d.hashFNV();
    }

}
