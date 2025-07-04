package nars.experiment.mario.sprites;

import nars.experiment.mario.level.SpriteTemplate;
import spacegraph.audio.SoundSource;

import java.awt.*;

public class Sprite implements SoundSource {
    public static SpriteContext spriteContext;

    public float xOld;
    public float yOld;
    public float x;
    public float y;
    public float xa;
    public float ya;

    public int xPic;
    public int yPic;
    public int wPic = 32;
    public int hPic = 32;
    public int xPicO;
    public int yPicO;
    public boolean xFlipPic;
    public boolean yFlipPic;
    public Image[][] sheet;
    public boolean visible = true;

    public int layer = 1;

    public SpriteTemplate spriteTemplate;

    public void move() {
        x += xa;
        y += ya;
    }

    public void render(Graphics og, float alpha) {
        if (!visible) return;

        int xPixel = (int) (xOld + (x - xOld) * alpha) - xPicO;
        int yPixel = (int) (yOld + (y - yOld) * alpha) - yPicO;

        og.drawImage(sheet[xPic][yPic], xPixel + (xFlipPic ? wPic : 0), yPixel + (yFlipPic ? hPic : 0), xFlipPic ? -wPic : wPic, yFlipPic ? -hPic : hPic, null);
    }
    
/*  private void blit(Graphics og, Image bitmap, int x0, int y0, int x1, int y1, int w, int h)
    {
        if (!xFlipPic)
        {
            if (!yFlipPic)
            {
                og.drawImage(bitmap, x0, y0, x0+w, y0+h, x1, y1, x1+w, y1+h, null);
            }
            else
            {
                og.drawImage(bitmap, x0, y0, x0+w, y0+h, x1, y1+h, x1+w, y1, null);
            }
        }
        else
        {
            if (!yFlipPic)
            {
                og.drawImage(bitmap, x0, y0, x0+w, y0+h, x1+w, y1, x1, y1+h, null);
            }
            else
            {
                og.drawImage(bitmap, x0, y0, x0+w, y0+h, x1+w, y1+h, x1, y1, null);
            }
        }
    }*/

    public final void tick() {
        xOld = x;
        yOld = y;
        move();
    }

    public final void tickNoMove() {
        xOld = x;
        yOld = y;
    }

    @Override
    public float getX(float alpha) {
        return (xOld + (x - xOld) * alpha) - xPicO;
    }

    @Override
    public float getY(float alpha) {
        return (yOld + (y - yOld) * alpha) - yPicO;
    }

    public void collideCheck() {
    }

    public void bumpCheck(int xTile, int yTile) {
    }

    public boolean shellCollideCheck(Shell shell) {
        return false;
    }

    public void release(Mario mario) {
    }

    public boolean fireballCollideCheck(Fireball fireball) {
        return false;
    }
}