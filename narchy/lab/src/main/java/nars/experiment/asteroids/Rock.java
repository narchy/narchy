package nars.experiment.asteroids;

import java.awt.*;

public class Rock extends VectorSprite {
    int c;
    int d;
    int e;
    int f;
    int g;
    int h;
    int i;
    int j;
    int k;
    int l;
    int m;
    int n;
    double a;
    double b;

    public Rock() {
        size = 4;
        initializeAsteroid();
    }

    public Rock(double x, double y, double size, double x2, double y2) {
        this.size = size;
        initializeAsteroid();
        xposition = x;
        yposition = y;
        xspeed = x2 + ((Math.random() - 0.5) * 2);
        yspeed = y2 + ((Math.random() - 0.5) * 2);
    }

    public void initializeAsteroid() {
        c = (int) (Math.random() * 12 * size);
        d = (int) (Math.random() * 12 * size);
        e = (int) (Math.random() * 12 * size);
        f = (int) (Math.random() * 12 * size);
        g = (int) (Math.random() * 12 * size);
        h = (int) (Math.random() * 12 * size);
        i = (int) (Math.random() * 12 * size);
        j = (int) (Math.random() * 12 * size);
        k = (int) (Math.random() * 12 * size);
        l = (int) (Math.random() * 12 * size);

        shape = new Polygon();
        shape.addPoint(c, i);
        shape.addPoint(d, -j);
        shape.addPoint(-e, -k);
        shape.addPoint(-h, n);
        shape.addPoint(0, m);

        drawShape = new Polygon();
        drawShape.addPoint(c, i);
        drawShape.addPoint(d, -j);
        drawShape.addPoint(-e, -k);
        drawShape.addPoint(-h, n);
        drawShape.addPoint(0, m);

        health = 3 * (size - 2);

        xposition = 450;
        yposition = 300;

        a = Math.random() * 2;
        b = Math.random() * 2 * Math.PI;

        xspeed = Math.cos(b) * a;
        yspeed = Math.sin(b) * a;

        a = Math.random() * 400 + 100;
        b = Math.random() * 2 * Math.PI;

        xposition = Math.cos(b) * a + 450;
        yposition = Math.sin(b) * a + 300;

        ROTATION = (Math.random() - 0.5) / 5;
    }

    @Override
    public void updatePosition(int w, int h) {
        angle += ROTATION;
        super.updatePosition(w, h);
    }
}
