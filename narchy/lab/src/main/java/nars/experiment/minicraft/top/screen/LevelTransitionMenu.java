package nars.experiment.minicraft.top.screen;

import nars.experiment.minicraft.top.gfx.Screen;

public class LevelTransitionMenu extends Menu {
    private final int dir;
    private int time;

    public LevelTransitionMenu(int dir) {
        this.dir = dir;
    }

    @Override
    public void tick() {
        time += 2;
        if (time == 30) game.changeLevel(dir);
        if (time == 60) game.setMenu(null);
    }

    @Override
    public void render(Screen screen) {
        for (int x = 0; x < 20; x++) {
            for (int y = 0; y < 15; y++) {
                int dd = (y + x % 2 * 2 + x / 3) - time;
                if (dd < 0 && dd > -30) {
                    if (dir > 0)
                        screen.render(x * 8, y * 8, 0, 0, 0);
                    else
                        screen.render(x * 8, screen.h - y * 8 - 8, 0, 0, 0);
                }
            }
        }
    }
}
