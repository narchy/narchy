/*
 * Copyright 2012 Jonathan Leahey
 *
 * This file is part of Minicraft
 *
 * Minicraft is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * Minicraft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Minicraft. If not, see http:
 */

package nars.experiment.minicraft.side;

public class Player extends LivingEntity {

    public final Int2 handBreakPos = new Int2(0, 0);
    public final Int2 handBuildPos = new Int2(0, 0);
    public float handStartX;
    public float handStartY;
    public float handEndX;
    public float handEndY;

    private final Sprite leftWalkSprite;
    private final Sprite rightWalkSprite;

    public Player(boolean gravityApplies, float x, float y, int width, int height) {
        super(gravityApplies, x, y, width, height);

        leftWalkSprite = SpriteStore.get().getSprite("sprites/entities/left_man.png");
        rightWalkSprite = SpriteStore.get().getSprite("sprites/entities/right_man.png");
        sprite = SpriteStore.get().getSprite("sprites/entities/player.gif");
    }

    public void setHotbarItem(int hotbarIdx) {
        inventory.hotbarIdx = hotbarIdx;
    }

    public void updateHand(GraphicsHandler g, float cameraX, float cameraY, float mouseX,
                           float mouseY, World world, int tileSize) {


        float x = .5f + (int) this.getCenterX(tileSize);
        float y = .5f + (int) this.getCenterY(tileSize);

        handStartX = x;
        handStartY = y;

        float tMax = (float) Math.ceil(armLength);
        handEndX = -1;
        handEndY = -1;

        handBuildPos.x = -1;
        handBuildPos.y = -1;

        float m;
        if (x - mouseX == 0) {
            m = Float.MAX_VALUE;
        } else {
            m = (y - mouseY) / (x - mouseX);
        }

        int hitY = -1;
        int hitX = -1;
        for (float i = 0; i <= Math.ceil(armLength) * 2; i++) {
            for (float j = 0; j <= Math.ceil(armLength) * 2; j++) {
                float px = (float) (x - Math.ceil(armLength) + i) - .5f;
                float py = (float) (y - Math.ceil(armLength) + j) - .5f;
                if (!world.isBreakable((int) px, (int) py)) {
                    continue;
                }

                float down = -1;
                float left = -1;
                float up = -1;
                float right = -1;

                if ((x >= px && x >= mouseX) &&
                        (y >= py && y >= mouseY)) {
                    right = findIntersection(x, y, m, px + 1, py, px + 1, py + 1);
                    down = findIntersection(x, y, m, px, py + 1, px + 1, py + 1);
                } else if ((x - .5f <= px && x <= mouseX) &&
                        (y - .5f >= py && y >= mouseY)) {
                    left = findIntersection(x, y, m, px, py, px, py + 1);
                    down = findIntersection(x, y, m, px, py + 1, px + 1, py + 1);
                } else if ((x >= px && x >= mouseX) &&
                        (y - 1 < py && y <= mouseY)) {
                    right = findIntersection(x, y, m, px + 1, py, px + 1, py + 1);
                    up = findIntersection(x, y, m, px, py, px + 1, py);
                } else if ((x - .5f <= px && x <= mouseX) &&
                        (y - .5f <= py && y <= mouseY)) {
                    left = findIntersection(x, y, m, px, py, px, py + 1);
                    up = findIntersection(x, y, m, px, py, px + 1, py);
                } else {
                    continue;
                }

                if (down != -1 || left != -1 || up != -1 || right != -1) {


                    float newTMax = (float) Math.sqrt(Math.pow(Math.abs(x) - Math.abs(px), 2)
                            + Math.pow(Math.abs(y) - Math.abs(py), 2));
                    if (newTMax >= tMax) {
                        continue;
                    }

                    if (up != -1) {
                        handEndX = up;
                        float upY = py - 1;
                        handEndY = upY;
                        handBuildPos.x = (int) px;
                        handBuildPos.y = (int) py - 1;
                    }
                    if (down != -1) {
                        handEndX = down;
                        float downY = py + 1;
                        handEndY = downY;
                        handBuildPos.x = (int) px;
                        handBuildPos.y = (int) py + 1;
                    }
                    if (left != -1) {
                        float leftX = px;
                        handEndX = leftX;
                        handEndY = left;
                        handBuildPos.x = (int) px - 1;
                        handBuildPos.y = (int) py;
                    }
                    if (right != -1) {
                        float rightX = px + 1;
                        handEndX = rightX;
                        handEndY = right;
                        handBuildPos.x = (int) px + 1;
                        handBuildPos.y = (int) py;
                    }

                    hitX = (int) px;
                    hitY = (int) py;

                    tMax = newTMax;
                }
            }
        }
        handBreakPos.x = hitX;
        handBreakPos.y = hitY;
    }

    @Override
    public void draw(GraphicsHandler g, float cameraX, float cameraY, int screenWidth,
                     int screenHeight, int tileSize) {
        Int2 pos = StockMethods.computeDrawLocationInPlace(cameraX, cameraY, screenWidth,
                screenHeight, tileSize, x, y);
        if (StockMethods.onScreen) {
            int frame = (int) x % 4;
            if (facingRight) {
                if (frame == 0 || frame == 2 || dx <= 0) {
                    sprite.draw(g, pos.x, pos.y, widthPX, heightPX);
                } else if (frame == 1) {
                    rightWalkSprite.draw(g, pos.x, pos.y, widthPX, heightPX);
                } else {
                    leftWalkSprite.draw(g, pos.x, pos.y, widthPX, heightPX);
                }
            } else {
                if (frame == 0 || frame == 2 || dx >= 0) {
                    sprite.draw(g, pos.x + widthPX, pos.y, -widthPX, heightPX);
                } else if (frame == 1) {
                    rightWalkSprite.draw(g, pos.x + widthPX, pos.y, -widthPX, heightPX);
                } else {
                    leftWalkSprite.draw(g, pos.x + widthPX, pos.y, -widthPX, heightPX);
                }
            }
        }
    }
}