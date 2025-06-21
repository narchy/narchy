package jcog.nndepr.layer;

import java.util.Random;

abstract public class StatelessLayer extends AbstractLayer {

    protected StatelessLayer(int i, int o) {
        super(i,o);
    }

    @Override
    public void initialize(Random r) {
    }

}