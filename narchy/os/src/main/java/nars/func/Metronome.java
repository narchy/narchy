package nars.func;

import nars.$;
import nars.NAR;
import nars.Term;
import nars.time.Tense;
import net.beadsproject.beads.core.AudioContext;
import net.beadsproject.beads.core.Auvent;
import net.beadsproject.beads.core.UGen;
import net.beadsproject.beads.data.WaveFactory;
import net.beadsproject.beads.ugens.*;

class Metronome {
    Metronome(Term id, Clock cc, NAR n) {
        cc.on(new Auvent<Clock>() {

            public final Envelope kickEnv;
            public final Envelope snareEnv;
            final AudioContext ac = cc.getContext();

            {
                kickEnv = new Envelope(ac, 0.0f);

                UGen kickGain = new Gain(ac, 1, kickEnv).in(
                        new BiquadFilter(ac, BiquadFilter.BESSEL_LP, 500.0f, 1.0f).in(
                                new WavePlayer(ac, 100.0f, WaveFactory.SINE)));

                ac.out.in(kickGain);

            }

            {
                snareEnv = new Envelope(ac, 0.0f);

                WavePlayer snareNoise = new WavePlayer(ac, 1.0f, WaveFactory.NOISE);
                WavePlayer snareTone = new WavePlayer(ac, 200.0f, WaveFactory.SINE);

                IIRFilter snareFilter = new BiquadFilter(ac, BiquadFilter.BP_SKIRT, 2500.0f, 1.0f);
                snareFilter.in(snareNoise);
                snareFilter.in(snareTone);

                Gain snareGain = new Gain(ac, 1, snareEnv);
                snareGain.in(snareFilter);


                ac.out.in(snareGain);
            }

            @Override
            protected void on(Clock c) {
                if (c.isBeat(16)) {
                    snareEnv.add(0.5f, 2.00f);
                    snareEnv.add(0.2f, 8.0f);
                    snareEnv.add(0.0f, 80.0f);
                    n.believe($.inh("snare", id), Tense.Present);
                }
                if (c.isBeat(4)) {

                    kickEnv.add(0.5f, 2.0f);
                    kickEnv.add(0.2f, 5.0f);
                    kickEnv.add(0.0f, 50.0f);
                    n.believe($.inh("kick", id), Tense.Present);


                }
            }
        });
    }
}