package jcog.io.lz;


import org.jetbrains.annotations.NotNull;

public enum QuickLZ {
	;

	public static final int QLZ_POINTERS_1 = 1;
	public static final int QLZ_POINTERS_3 = 16;
	private static final int HASH_VALUES = 4096;
	private static final int MINOFFSET = 2;
	private static final int UNCONDITIONAL_MATCHLEN = 6;
	private static final int UNCOMPRESSED_END = 4;
	private static final int CWORD_LEN = 4;
	private static final int DEFAULT_HEADERLEN = 9;

	static int headerLen(byte[] source, int offset) {
		return (headerLong(source, offset) ? 9 : 3) + offset;
	}

	static boolean headerLong(byte[] source, int offset) {
		return ((source[offset] & 2) == 2);
	}

	public static long sizeDecompressed(byte[] source, int offset) {
		return headerLong(source, offset) ?
			fast_read_long(source, 5 + offset, 4) :
			fast_read_long(source, 2 + offset, 1);
	}

	public static long sizeCompressed(byte[] source) {
		return fast_read_long(source, 1, (headerLong(source, 0)) ? 4 : 1);
	}

	private static void write_header(byte[] dst, int level, boolean compressible, int size_compressed, int size_decompressed) {
        var b = (byte) (2 | (compressible ? 1 : 0));
		b |= (byte) (level << 2);
		b |= (1 << 6);
		//b |= (0 << 4);
		dst[0] = b;
		fast_write(dst, 1, size_decompressed, 4);
		fast_write(dst, 5, size_compressed, 4);
	}

	public static byte[] compress(byte[] in) {
		return compress(in, 3);
	}

	public static byte[] compress(byte[] in, int level) {
		return compress(in, level, 0);
	}

	public static byte[] compress(byte[] in, int level, int prefixReserved) {
        var tmp = new byte[in.length + 400];
        return trim(compress(in, tmp, level), prefixReserved, tmp);
	}

	private static byte @NotNull [] trim(int dst, int prefixReserved, byte[] tmp) {
		if (dst + prefixReserved == tmp.length) return tmp;
		var out = new byte[dst + prefixReserved];
		System.arraycopy(tmp, 0, out, prefixReserved, dst);
		return out;
	}

	/**
	 * level=(1|3)
	 */
	public static int compress(byte[] in, byte[] out, int level) {
        var last_matchstart = in.length - UNCONDITIONAL_MATCHLEN - UNCOMPRESSED_END - 1;

        if (level != 1 && level != 3)
			throw new RuntimeException("Java version only supports level 1 and 3");

        var hashtable = new int[HASH_VALUES][level == 1 ? QLZ_POINTERS_1 : QLZ_POINTERS_3];

		if (in.length == 0)
			return 0;

        var fetch = 0;
        var src = 0;
        if (src <= last_matchstart)
			fetch = fast_read_int(in, src, 3);

        var lits = 0;
        var hash_counter = new byte[HASH_VALUES];
        var cachetable = new int[HASH_VALUES];
        var cword_ptr = DEFAULT_HEADERLEN;
        var cword_val = 0x80000000L;
        var dst = DEFAULT_HEADERLEN + CWORD_LEN;
        while (src <= last_matchstart) {

			if ((cword_val & 1) == 1) {
				if (src > 3 * (in.length >> 2) && dst > src - (src >> 5)) {
					return compressRaw(in, out, level);
				}

				fast_write(out, cword_ptr, (cword_val >>> 1) | 0x80000000L, 4);
				cword_ptr = dst;
				dst += CWORD_LEN;
				cword_val = 0x80000000L;
			}

			if (level == 1) {
                var hash = ((fetch >>> 12) ^ fetch) & (HASH_VALUES - 1);
                var o = hashtable[hash][0];
                var cache = cachetable[hash] ^ fetch;

				cachetable[hash] = fetch;
				hashtable[hash][0] = src;

				if (cache == 0 && hash_counter[hash] != 0 && (src - o > MINOFFSET || (src == o + 1 && lits >= 3 && src > 3 && in[src] == in[src - 3] && in[src] == in[src - 2] && in[src] == in[src - 1] && in[src] == in[src + 1] && in[src] == in[src + 2]))) {
					cword_val = ((cword_val >>> 1) | 0x80000000L);
					if (in[o + 3] != in[src + 3]) {
                        var f = 3 - 2 | (hash << 4);
						out[dst] = (byte) (f);
						out[dst + 1] = (byte) (f >>> 8);
						src += 3;
						dst += 2;
					} else {
                        var old_src = src;
                        var remaining = Math.min(in.length - UNCOMPRESSED_END - src + 1 - 1, 255);

						src += 4;
						if (in[o + src - old_src] == in[src]) {
							src++;
							if (in[o + src - old_src] == in[src]) {
								src++;
								while (in[o + (src - old_src)] == in[src] && (src - old_src) < remaining)
									src++;
							}
						}

                        var matchlen = src - old_src;

						hash <<= 4;
						if (matchlen < 18) {
                            var f = hash | (matchlen - 2);

							out[dst] = (byte) (f);
							out[dst + 1] = (byte) (f >>> 8);
							dst += 2;
						} else {
                            fast_write(out, dst, hash | (matchlen << 16), 3);
							dst += 3;
						}
					}
					lits = 0;
					fetch = fast_read_int(in, src, 3);
				} else {
					lits++;
					hash_counter[hash] = 1;
					out[dst] = in[src];
                    cword_val >>>= 1;
					src++;
					dst++;
					fetch = ((fetch >>> 8) & 0xffff) | ((in[src + 2] & 0xff) << 16);
				}
			} else {
				fetch = fast_read_int(in, src, 3);

				int o;
                var remaining = Math.min(in.length - UNCOMPRESSED_END - src + 1 - 1, 255);
                var hash = ((fetch >>> 12) ^ fetch) & (HASH_VALUES - 1);

                var c = hash_counter[hash];
                var matchlen = 0;
                var offset2 = 0;

                var hth = hashtable[hash];
				for (var k = 0; k < QLZ_POINTERS_3 && (c > k || c < 0); k++) {
					o = hth[k];
					if ((byte) fetch == in[o] && (byte) (fetch >>> 8) == in[o + 1] && (byte) (fetch >>> 16) == in[o + 2] && o < src - MINOFFSET) {
                        /*, best_k = 0*/
                        var m = 3;
                        while (in[o + m] == in[src + m] && m < remaining)
							m++;
						if ((m > matchlen) || (m == matchlen && o > offset2)) {
							offset2 = o;
							matchlen = m;
						}
					}
				}
				o = offset2;
				hth[c & (QLZ_POINTERS_3 - 1)] = src;
				c++;
				hash_counter[hash] = c;

				if (matchlen >= 3 && src - o < 131071) {
                    var offset = src - o;
					for (var u = 1; u < matchlen; u++) {
						fetch = fast_read_int(in, src + u, 3);
						hash = ((fetch >>> 12) ^ fetch) & (HASH_VALUES - 1);
						c = hash_counter[hash]++;
						hashtable[hash][c & (QLZ_POINTERS_3 - 1)] = src + u;
					}

					src += matchlen;
					cword_val = ((cword_val >>> 1) | 0x80000000L);

					if (matchlen == 3 && offset <= 63) {
						fast_write(out, dst, offset << 2, 1);
						dst++;
					} else if (matchlen == 3 && offset <= 16383) {
						fast_write(out, dst, (offset << 2) | 1, 2);
						dst += 2;
					} else if (matchlen <= 18 && offset <= 1023) {
						fast_write(out, dst, ((matchlen - 3) << 2) | (offset << 6) | 2, 2);
						dst += 2;
					} else if (matchlen <= 33) {
						fast_write(out, dst, ((matchlen - 2) << 2) | (offset << 7) | 3, 3);
						dst += 3;
					} else {
						fast_write(out, dst, ((matchlen - 3) << 7) | (offset << 15) | 3, 4);
						dst += 4;
					}
				} else {
					out[dst] = in[src];
                    cword_val >>>= 1;
					src++;
					dst++;
				}
			}
		}

		return compress(in, out, level, src, cword_val, cword_ptr, dst);
	}

	private static int compressRaw(byte[] in, byte[] out, int level) {
		write_header(out, level, false, in.length, in.length + DEFAULT_HEADERLEN);
		System.arraycopy(in, 0, out, DEFAULT_HEADERLEN, in.length);
		return in.length + DEFAULT_HEADERLEN;
	}

	private static int compress(byte[] in, byte[] out, int level, int src, long cword_val, int cword_ptr, int dst) {
		var n = in.length - 1;
		while (src <= n) {
			if ((cword_val & 1) == 1) {
				fast_write(out, cword_ptr, (cword_val >>> 1) | 0x80000000L, 4);
				cword_ptr = dst;
				dst += CWORD_LEN;
				cword_val = 0x80000000L;
			}

			out[dst] = in[src];
			src++;
			dst++;
            cword_val >>>= 1;
		}

		while ((cword_val & 1) != 1) { cword_val >>>= 1; }

		fast_write(out, cword_ptr, (cword_val >>> 1) | 0x80000000L, CWORD_LEN);
		write_header(out, level, true, in.length, dst);

		return dst;
	}

	static long fast_read_long(byte[] a, int i, int numbytes) {
		long l = 0;
		for (var j = 0; j < numbytes; j++)
			l |= ((a[i++] & 0xffL) << j * 8);
		return l;
	}
    static int fast_read_int(byte[] a, int i, int numbytes) {
        var l = 0;
        for (var j = 0; j < numbytes; j++)
            l |= ((a[i++] & 0xffL) << j * 8);
        return l;
    }

    /** TODO separate int (< 4 numbytes) and long versions */
	static void fast_write(byte[] a, int i, long value, int numbytes) {
		for (var j = 0; j < numbytes; j++)
			a[i++] = (byte) (value >>> (j * 8));
	}

	public static byte[] decompress(byte[] in) {
		return decompress(in, 0);
	}

	public static byte[] decompress(byte[] in, int offset) {
        var first = in[offset];
        var level = (first >>> 2) & 0x3;
		if (level != 1 && level != 3)
			throw new RuntimeException("Java version only supports level 1 and 3");

        var initSrc = headerLen(in, offset);
        var size = (int) sizeDecompressed(in, offset);
		if ((first & 1) != 1)
			return decompressRaw(in, size, initSrc);

        var last_hashed = -1;
        var last_matchstart = size - UNCONDITIONAL_MATCHLEN - UNCOMPRESSED_END - 1;
		int[] hashtable = null;
        //new int[hashtable_size];
        var out = new byte[size];
        long cword_val = 1;
        var dst = 0;
        var src = initSrc;
        for (var fetch = 0; ; ) {
			if (cword_val == 1) {
				cword_val = fast_read_long(in, src, 4);
				src += 4;
				if (dst <= last_matchstart)
					fetch = fast_read_int(in, src, level == 1 ? 3 : 4);
			}

			if ((cword_val & 1) == 1) {

                cword_val >>>= 1;

                int offset2;
                int matchlen;
                if (level == 1) {
                    offset2 = hashtable[((fetch >>> 4) & 0xfff)];

					if ((fetch & 0xf) != 0) {
						matchlen = (fetch & 0xf) + 2;
						src += 2;
					} else {
						matchlen = in[src + 2] & 0xff;
						src += 3;
					}
				} else {
					int o;

					if ((fetch & 3) == 0) {
						o = (fetch & 0xff) >>> 2;
						matchlen = 3;
						src++;
					} else if ((fetch & 2) == 0) {
						o = (fetch & 0xffff) >>> 2;
						matchlen = 3;
						src += 2;
					} else if ((fetch & 1) == 0) {
						o = (fetch & 0xffff) >>> 6;
						matchlen = ((fetch >>> 2) & 15) + 3;
						src += 2;
					} else if ((fetch & 127) != 3) {
						o = (fetch >>> 7) & 0x1ffff;
						matchlen = ((fetch >>> 2) & 0x1f) + 2;
						src += 3;
					} else {
						o = (fetch >>> 15);
						matchlen = ((fetch >>> 7) & 255) + 3;
						src += 4;
					}
					offset2 = dst - o;
				}

                //for some reason System.arraycopy doesnt ever work here... i dont know
				//noinspection ManualArrayCopy
				for (var i = 0; i < matchlen; i++)
					out[dst + i] = out[offset2 + i];

				dst += matchlen;

				int toFetch;
				if (level == 1) {
					fetch = fast_read_int(out, last_hashed + 1, 3);
                    if (last_hashed < dst - matchlen) {
                        if (hashtable == null)
                            hashtable = new int[HASH_VALUES]; //lazy alloc
						fetch(last_hashed, hashtable, fetch, out, dst, matchlen);
					}
					toFetch = 3;
				} else {
					toFetch = 4;
				}
                fetch = fast_read_int(in, src, toFetch);
				last_hashed = dst - 1;
			} else {
				if (dst <= last_matchstart) {
					out[dst++] = in[src++];
                    cword_val >>>= 1;


					if (level == 1 && (last_hashed < dst-3)) {
                        if (hashtable == null)
                            hashtable = new int[HASH_VALUES]; //lazy alloc

						last_hashed = fetch2(out, last_hashed, hashtable, dst);
					}

					fetch = fetch >> 8 & 0xffff | (in[src + 2] & 0xff) << 16;

					if (level!=1)
                        fetch |= (in[src + 3] & 0xff) << 24;

				} else
					return decompress(in, dst, size, cword_val, src, out);
			}
		}
	}

	private static byte @NotNull [] decompressRaw(byte[] in, int size, int initSrc) {
        var d2 = new byte[size];
		System.arraycopy(in, initSrc, d2, 0, size);
		return d2;
	}

	private static int fetch2(byte[] out, int last_hashed, int[] hashtable, int dst) {
		do {
            var fetch2 = fast_read_int(out, ++last_hashed, 3);
hashtable[((fetch2 >>> 12) ^ fetch2) & (HASH_VALUES - 1)] = last_hashed;
			//hash_counter[hash] = 1;
		} while (last_hashed < dst - 3);
		return last_hashed;
	}

	private static void fetch(int last_hashed, int[] hashtable, int fetch, byte[] out, int dst, int matchlen) {
		do {
			last_hashed++;
			hashtable[((fetch >>> 12) ^ fetch) & (HASH_VALUES - 1)] = last_hashed;
			//hash_counter[hash] = 1;
			fetch = fetch >>> 8 & 0xffff | (out[last_hashed + 3] & 0xff) << 16;
		} while (last_hashed < dst - matchlen);
	}

	private static byte[] decompress(byte[] in, int dst, int size, long cword_val, int src, byte[] out) {
		while (dst <= size - 1) {
			if (cword_val == 1) {
				src += CWORD_LEN;
				cword_val = 0x80000000L;
			}

			out[dst++] = in[src++];
cword_val >>>= 1;
		}
		return out;
	}
}
