package jcog.signal.wave2d;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.function.Supplier;

import static java.awt.RenderingHints.*;
import static java.awt.image.BufferedImage.TYPE_INT_RGB;

public abstract class MonoDBufImgBitmap2D extends MonoBufImgBitmap2D {

    private static final boolean highQuality = true;

    /**
     * output pixel width / height
     */
    public int pw, ph;
    protected Graphics2D outgfx;

    float sx1 = 0, sy1 = 0;
    float sx2 = 1, sy2 = 1;

    protected MonoDBufImgBitmap2D(int pw, int ph) {
        super();
        this.pw = pw;
        this.ph = ph;
    }

    protected MonoDBufImgBitmap2D(Supplier<BufferedImage> source) {
        this(source, source.get().getWidth(), source.get().getHeight());
    }

    protected MonoDBufImgBitmap2D(Supplier<BufferedImage> source, int pw, int ph) {
        this(pw, ph);
        this.source = source;
    }

    public MonoDBufImgBitmap2D crop(float sx1, float sy1, float sx2, float sy2) {
        this.sx1 = sx1;
        this.sy1 = sy1;
        this.sx2 = sx2;
        this.sy2 = sy2;
        return this;
    }

    @Override
    public int width() {
        return pw;
    }

    @Override
    public int height() {
        return ph;
    }

    @Override
    public void updateBitmap() {

        Supplier<BufferedImage> imgSrc = source;
        if (imgSrc == null)
            return;

        if (imgSrc instanceof Bitmap2D b)
            b.updateBitmap();

        BufferedImage img = imgSrc.get();
        if (img == null)
            return;

        if (outgfx == null || raster.getWidth() != pw || raster.getHeight() != ph)
            initGFX(img);

        render(img, out);
    }

    private void initGFX(BufferedImage in) {
        if (outgfx != null)
            outgfx.dispose();

        out = new BufferedImage(pw, ph, TYPE_INT_RGB /*TYPE_INT_ARGB*/ /* in.getType()*/);
        outgfx = out.createGraphics();

        //if (mode()!=ColorMode.Hue) {
        if (highQuality) {
            outgfx.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            outgfx.setRenderingHint(KEY_ALPHA_INTERPOLATION, VALUE_ALPHA_INTERPOLATION_QUALITY);
            outgfx.setRenderingHint(KEY_INTERPOLATION,
                    VALUE_INTERPOLATION_BICUBIC
                    //VALUE_INTERPOLATION_NEAREST_NEIGHBOR
            );
            outgfx.setRenderingHint(KEY_RENDERING, VALUE_RENDER_QUALITY);
            outgfx.setRenderingHint(KEY_COLOR_RENDERING, VALUE_COLOR_RENDER_QUALITY);
        }
        //}

        img(out);
    }

    protected abstract void render(BufferedImage in, BufferedImage out);
}