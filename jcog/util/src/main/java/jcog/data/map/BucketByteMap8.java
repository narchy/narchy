package jcog.data.map;

public class BucketByteMap8<X> extends BucketByteMap<X> {

    @Override
    protected byte stride() {
        return 8;
    }
}
