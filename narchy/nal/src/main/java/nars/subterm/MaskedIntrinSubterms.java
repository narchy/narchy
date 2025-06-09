package nars.subterm;

import nars.Term;
import nars.term.anon.Intrin;

import static nars.term.anon.Intrin.ANOMs;

public abstract class MaskedIntrinSubterms extends ProxySubterms<IntrinSubterms> {

    protected MaskedIntrinSubterms(IntrinSubterms ref) {
        super(ref);
    }

    @Override
    public Term sub(int i) {
        short x = ref.subterms[i];
        short ax = x < 0 ? (short) -x : x;
        return (Intrin.group(ax) == ANOMs ? atom((ax & 0xff) - 1) : Intrin._term(ax))
            .negIf(x < 0);
    }

    /** returns the facade atom for the Anom index id minus 1*/
    public abstract Term atom(int index);

    public static class SubtermsMaskedIntrinSubterms extends MaskedIntrinSubterms {
        private final Subterms mask;

        public SubtermsMaskedIntrinSubterms(IntrinSubterms skeleton, Subterms mask) {
            super(skeleton);
            this.mask = mask;
        }

        @Override
        public Term atom(int index) {
            return mask.sub(index);
        }
    }
}