package nars.nal.nal3;


import nars.NAR;
import nars.NARS;
import nars.test.NALTest;

public abstract class NAL3Test extends NALTest {

    static final int cycles = 200;

    @Override
    protected NAR nar() {
        NAR n = NARS.tmp(3, 3);
        n.confMin.set(0.03f);
        n.complexMax.set(8);
        return n;
    }


}

