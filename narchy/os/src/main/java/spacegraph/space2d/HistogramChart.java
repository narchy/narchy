package spacegraph.space2d;

import com.jogamp.opengl.GL2;
import jcog.Util;
import spacegraph.space2d.container.PaintSurface;
import spacegraph.util.math.Color3f;
import spacegraph.video.Draw;

import java.util.function.Supplier;

/**
 * Created by me on 9/2/16.
 */
public class HistogramChart extends PaintSurface {


    private final Supplier<float[]> data;
    private final Color3f dark;
    private final Color3f light;

    public HistogramChart(Supplier<float[]> source, Color3f dark, Color3f light) {

        this.data = source;
        this.dark = dark;
        this.light = light;

    }


    @Override
    protected void paint(GL2 gl, ReSurface reSurface) {
        Draw.bounds(gl, this, this::paintUnit);
    }

    protected void paintUnit(GL2 gl) {

        gl.glColor4f(0f, 0f, 0f, 0.5f);
        Draw.rect(gl, 0, 0, 1, 1);

        float[] data = this.data.get();

        int N = data.length;
        float max = data[Util.argmax(data)];
        if (max == 0)
            return;

        float x = 0;

        float ra = dark.x;
        float ga = dark.y;
        float ba = dark.z;
        float rb = light.x;
        float gb = light.y;
        float bb = light.z;

        float dx = 1f / N;
        for (int i = 0; i < N; i++) {

            float v = data[i] / max;

            gl.glColor3f(Util.lerpSafe(v, ra, rb), Util.lerpSafe(v, ga, gb), Util.lerpSafe(v, ba, bb));

            Draw.rect(x, 0, dx, v, gl);

            x += dx;
        }

    }

}

