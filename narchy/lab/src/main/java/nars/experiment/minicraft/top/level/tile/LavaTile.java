package nars.experiment.minicraft.top.level.tile;


import nars.experiment.minicraft.top.entity.Entity;
import nars.experiment.minicraft.top.gfx.Color;
import nars.experiment.minicraft.top.gfx.Screen;
import nars.experiment.minicraft.top.level.Level;

import java.util.Random;

public class LavaTile extends Tile {
    public LavaTile(int id) {
        super(id);
        connectsToSand = true;
        connectsToLava = true;
    }

    private final Random wRandom = new Random();

    @Override
    public void render(Screen screen, Level level, int x, int y) {
        wRandom.setSeed((tickCount + (x / 2 - y) * 4311) / 10 * 54687121L + x * 3271612L + y * 3412987161L);
        int col = Color.get(500, 500, 520, 550);
        int transitionColor1 = Color.get(3, 500, level.dirtColor - 111, level.dirtColor);
        int transitionColor2 = Color.get(3, 500, level.sandColor - 110, level.sandColor);

        boolean u = !level.getTile(x, y - 1).connectsToLava;
        boolean d = !level.getTile(x, y + 1).connectsToLava;
        boolean l = !level.getTile(x - 1, y).connectsToLava;
        boolean r = !level.getTile(x + 1, y).connectsToLava;

        boolean su = u && level.getTile(x, y - 1).connectsToSand;
        boolean sd = d && level.getTile(x, y + 1).connectsToSand;
        boolean sl = l && level.getTile(x - 1, y).connectsToSand;
        boolean sr = r && level.getTile(x + 1, y).connectsToSand;

        if (!u && !l) {
            screen.render(x * 16 + 0, y * 16 + 0, wRandom.nextInt(4), col, wRandom.nextInt(4));
        } else
            screen.render(x * 16 + 0, y * 16 + 0, (l ? 14 : 15) + (u ? 0 : 1) * 32, (su || sl) ? transitionColor2 : transitionColor1, 0);

        if (!u && !r) {
            screen.render(x * 16 + 8, y * 16 + 0, wRandom.nextInt(4), col, wRandom.nextInt(4));
        } else
            screen.render(x * 16 + 8, y * 16 + 0, (r ? 16 : 15) + (u ? 0 : 1) * 32, (su || sr) ? transitionColor2 : transitionColor1, 0);

        if (!d && !l) {
            screen.render(x * 16 + 0, y * 16 + 8, wRandom.nextInt(4), col, wRandom.nextInt(4));
        } else
            screen.render(x * 16 + 0, y * 16 + 8, (l ? 14 : 15) + (d ? 2 : 1) * 32, (sd || sl) ? transitionColor2 : transitionColor1, 0);
        if (!d && !r) {
            screen.render(x * 16 + 8, y * 16 + 8, wRandom.nextInt(4), col, wRandom.nextInt(4));
        } else
            screen.render(x * 16 + 8, y * 16 + 8, (r ? 16 : 15) + (d ? 2 : 1) * 32, (sd || sr) ? transitionColor2 : transitionColor1, 0);
    }

    @Override
    public boolean mayPass(Level level, int x, int y, Entity e) {
        return e.canSwim();
    }

    @Override
    public void tick(Level level, int xt, int yt) {
        int xn = xt;
        int yn = yt;

        if (random.nextBoolean())
            xn += random.nextInt(2) * 2 - 1;
        else
            yn += random.nextInt(2) * 2 - 1;

        if (level.getTile(xn, yn) == Tile.hole) {
            level.setTile(xn, yn, this, 0);
        }
    }

    @Override
    public int getLightRadius(Level level, int x, int y) {
        return 6;
    }
}