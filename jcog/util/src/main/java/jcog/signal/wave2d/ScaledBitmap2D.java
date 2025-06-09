package jcog.signal.wave2d;


import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.function.Supplier;

/**
 * Pan/Zoom filter for a BuferredImage source
 * TODO avoid dependence on Swing Image
 */
public class ScaledBitmap2D extends MonoDBufImgBitmap2D/* TODO extends ArrayBitmap2D directly, bypassing Swing */ /*implements ImageObserver*/ {

    public ScaledBitmap2D(BufferedImage source, int pw, int ph) {
        this(() -> source, pw, ph);
    }

    public ScaledBitmap2D(Supplier<BufferedImage> source) {
        super(source);
    }

    public ScaledBitmap2D(Supplier<BufferedImage> source, int pw, int ph) {
        super(source, pw, ph);
    }

    @Override
    protected void render(BufferedImage in, BufferedImage out) {
        int sw = in.getWidth(), sh = in.getHeight();
        outgfx.drawImage(in,
            0, 0, pw, ph,
            Math.round(sx1 * sw), Math.round(sy1 * sh),
            Math.round(sx2 * sw), Math.round(sy2 * sh),
            Color.BLACK, null);

//        AffineTransform xform = AffineTransform.getScaleInstance(sx1 * sw)
//        outgfx.drawRenderedImage(in, xform);
    }

}