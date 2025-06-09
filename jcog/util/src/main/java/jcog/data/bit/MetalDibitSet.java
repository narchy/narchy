package jcog.data.bit;

/** 2-bit pairs impl by one contiguous internal BitSet */
public class MetalDibitSet extends LongArrayBitSet {

	public MetalDibitSet(int capacity) {
		super(Math.max(64,capacity * 2));
	}

//	public void set(int i, byte value) {
//		assert(value >= 0 && value <= 3);
//		throw new TODO();
//	}

	/** returns upper */
	public final boolean set(int i, boolean lower, boolean upper) {
		this._set2(i * 2, lower, upper);
		return upper;
	}

	/** gets lower and upper, returned in bits 0 and 1 */
	public byte get(int i) {
		return (byte) ((lower(i) ? 1 : 0) | ((upper(i) ? 1 : 0) << 1));
	}

	public boolean lower(int i) {
		return this.test(i*2);
	}
	public boolean upper(int i) {
		return this.test(i*2+1);
	}

	public int isKnown(int slot) {
		int i = slot*2;
		long[] d = this.data;

		int l = i >>> 6;
		long W = d[l];
		if ((W & (1L << i)) == 0) //first bit?
			return 0;

		//if the index hasnt changed,
		//avoid loading another 64-bits from array
		int h = (++i) >>> 6;

		return ((l == h ? W : d[h]) & (1L << i)) == 0 ?
				1 : 3; //2nd bit?
	}

	public static boolean known(int k)   { return (k & 1)!=0; }
	public static boolean itIs(int k)    { return (k & 2)!=0; }
	private static boolean itIsnt(int k)  { return (k & 2)==0; }

	public static boolean needTest(int pending, int i) {
		return 0 != (pending & (1 << i));
	}

	/** returns -1 if predicate has been previously evaluated false, otherwise the bitset of pending memoizations to compute */
    public int pending(int[] s, MetalBitSet polarity, int n) {
		int pending = 0;
		for (int i = 0; i < n; i++) {
			int k = isKnown(s[i]);
			if (known(k)) {
				if (!is(i, itIs(k), polarity))
					return -1;   //CUT
			} else {
				pending |= (1 << i);
			}
		}
		return pending;
	}

	public static boolean is(int i, boolean k, MetalBitSet polarity) {
		return k == polarity.test(i);
	}

}