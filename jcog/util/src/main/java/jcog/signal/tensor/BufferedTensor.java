package jcog.signal.tensor;

import jcog.signal.ITensor;

public class BufferedTensor extends BatchArrayTensor {

    protected final ITensor from;

    public BufferedTensor(ITensor from) {
        super(from.shape());
        this.from = from;
    }


    @Override public void update() {
        //from.snapshot();
        from.writeTo(data);
    }

}
