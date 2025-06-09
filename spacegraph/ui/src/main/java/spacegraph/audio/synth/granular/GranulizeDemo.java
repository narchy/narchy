package spacegraph.audio.synth.granular;

import jcog.random.XorShift128PlusRandom;
import spacegraph.audio.Audio;
import spacegraph.audio.SoundSource;
import spacegraph.audio.sample.SampleLoader;

import java.io.IOException;

public enum GranulizeDemo {
    ;

    @SuppressWarnings("HardcodedFileSeparator")
    public static void main(String[] args) throws InterruptedException, IOException {

        var s =
            new Granulize(SampleLoader.load("/home/me/d/r1.wav"),
                    0.01f,
                    1,//1,
                    new XorShift128PlusRandom(1))
                   .setStretchFactor(4.0f)
                   .setPitchFactor(0.1f)
                ;
//            new GrainCloud(List.of(SampleLoader.load("/home/me/d/r1.wav").buf), 44100, 1);

        Audio.the().play(s, SoundSource.center, 0.5f, 1);

        //audio.thread.join();
        System.in.read();
    }


}