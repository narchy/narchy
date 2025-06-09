package nars.gui;

import jcog.Util;
import jcog.data.list.Lst;
import jcog.pri.PLink;
import jcog.pri.PriReference;
import jcog.pri.bag.Bag;
import jcog.pri.bag.impl.PLinkArrayBag;
import jcog.pri.op.PriMerge;
import spacegraph.SpaceGraph;
import spacegraph.space2d.HistogramChart;
import spacegraph.space2d.Surface;
import spacegraph.space2d.container.grid.Containers;
import spacegraph.space2d.widget.slider.FloatSlider;
import spacegraph.space2d.widget.text.Labelling;
import spacegraph.util.math.Color3f;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import static spacegraph.space2d.container.grid.Containers.col;
import static spacegraph.space2d.container.grid.Containers.row;

/**
 * Created by me on 11/29/16.
 */
public class BagLab {

    public static final int BINS = 16;

    int histogramResetPeriod = 64;
    int iteration;


    final Bag<Integer, PriReference<Integer>> bag;
    private final List<FloatSlider> inputSliders;
    private final int uniques;

    float[] selectionHistogram = new float[BINS];

    public BagLab(Bag<Integer, PriReference<Integer>> bag) {
        super();
        this.bag = bag;

        this.uniques = bag.capacity() * 2;

        int inputs = 10;

		inputSliders = (List<FloatSlider>) new Lst(inputs);
        for (int i = 0; i < inputs; i++)
            inputSliders.add(new FloatSlider(0.5f, 0, 1));


    }

    public Surface surface() {
        return row(
                Containers.col(inputSliders),
                
                col(
                        Labelling.the("Bag Selection Distribution (0..1)", new HistogramChart(
                                () -> selectionHistogram, new Color3f(0.5f, 0.25f, 0f), new Color3f(1f, 0.5f, 0.1f))),
                        Labelling.the("Bag Content Distribution (0..1)", new HistogramChart(
                                () -> bag.histogram(new float[BINS]),
                                new Color3f(0f, 0.25f, 0.5f), new Color3f(0.1f, 0.5f, 1f)))
                )
                
        );
    }




























    public static void main(String[] arg) {

        int capacity = 512;
        BagLab bagLab = new BagLab(
                new PLinkArrayBag(
                        //PriMerge.plus,
                        PriMerge.or,
                        capacity
                )

                //new DefaultHijackBag<>(avg,capacity,4)

        );


        SpaceGraph.window(bagLab.surface(), 1200, 800);

        long delayMS = 30;
        
        while (true) {
            bagLab.update();
            Util.sleepMS(delayMS);
        }
        
    }


    private synchronized void update() {


        //inputStochastic();
        inputFlat();

        forget();

        measure();

    }

    private void measure() {
        int bins = selectionHistogram.length;

        if (iteration++ % histogramResetPeriod == 0)
            Arrays.fill(selectionHistogram, 0);

        long seed = System.nanoTime();

        Random rng = //new XorShift128PlusRandom(seed);
                //new XoRoShiRo128PlusRandom(seed);
                ThreadLocalRandom.current();

		List<PriReference<Integer>> sampled = (List<PriReference<Integer>>) new Lst(1024);
        int batchSize = 32;
        float sampleBatches = 1;
        for (int i = 0; i < (int) sampleBatches; i++) {
            sampled.clear();


            bag.sample(rng, batchSize, sampled::add);


            for (PriReference<Integer> sample : sampled) {
                if (sample != null) {
                    float p = sample.priElseZero();
                    selectionHistogram[Util.bin(p, bins - 1)]++;
                } else {
                    break;
                }
            }
        }
    }

    private void forget() {
        bag.commit(bag.forget(1));
    }

    private void inputFlat() {
//        if (!bag.isEmpty())
//            return; //assume done already

        double sum = inputSliders.stream().mapToDouble(FloatSlider::get).sum();
        float totalInputs = (float) sum;
        if (totalInputs < 0.01f)
            return;

        int currentSlider = 0, sliderRemain = -1;
        int cap = bag.capacity();
        int n = inputSliders.size();
        int r = 0;
        for (int i = 0; i < cap && currentSlider < n; i++) {
            if (sliderRemain == -1) {
                r = sliderRemain = Math.round((inputSliders.get(currentSlider++).get()/totalInputs)  * cap);
            }
            bag.put(new PLink<>(i, (((float)currentSlider) / (n-1)) + (((float)sliderRemain)/r) * (1f/n))) ;
            sliderRemain--;
        }
    }

    private void inputStochastic() {
        int n = inputSliders.size();
        int inputRate = n*n;
        for (int j = 0; j < inputRate; j++) {
            for (int i = 0; i < n; i++) {
                if (Math.random() < inputSliders.get(i).get()) {
                    float p = 0.1f;
                            //(i /* + (float) Math.random()*/) / (n - 1);

                    bag.put(new PLink<>((int) Math.floor(Math.random() * uniques), p));
                }
            }
        }
    }

}