package jcog.signal.tensor;

import com.google.common.base.Joiner;
import jcog.signal.ITensor;

import java.io.Serializable;

public abstract class AbstractTensor implements ITensor, Serializable {


    @Override
    public abstract int[] stride();

    @Override
    public abstract int[] shape();


    @Override
    public abstract float getAt(int linearCell);

    @Override
    public String toString() {
        return Joiner.on(',').join(iterator(String::valueOf));
    }
}
