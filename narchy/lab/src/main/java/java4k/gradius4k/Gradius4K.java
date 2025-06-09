package java4k.gradius4k;

/*
 * Gradius 4K
 * Copyright (C) 2011 meatfighter.com
 *
 * This file is part of Gradius 4K.
 *
 * Gradius 4K is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gradius 4K is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http:
 *
 */

import java4k.GamePanel;
import jcog.Util;
import jcog.data.list.Lst;
import jcog.random.XoRoShiRo128PlusRandom;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.util.Random;

public class Gradius4K extends GamePanel {

    private boolean easy = true;

    public int Afterlife_Time =
        60 * 1;
        //60 * 8;

    public long updateMS = 20;
    public float SPEED = 2;

    int playerGun = 2;

    public int level = 1;

    public boolean DIE_IF_HIT_ENEMY = true,
            DIE_IF_HIT_ENEMY_BULLET = true,
            DIE_IF_HIT_WALL = true;

    public static final int VK_LEFT = 0x25;
    public static final int VK_RIGHT = 0x27;
    public static final int VK_UP = 0x26;
    public static final int VK_DOWN = 0x28;
    public static final int VK_SHOOT = 0x42;
    public static final int OBJ_X = 0;
    public static final int OBJ_Y = 1;
    public static final int OBJ_TYPE = 2;
    public static final int OBJ_SPRITE = 3;
    public static final int OBJ_REMOVE = 4;
    public static final int OBJ_VX = 5;
    public static final int OBJ_VY = 6;
    public static final int OBJ_SCALE_X = 7;
    public static final int OBJ_SCALE_Y = 8;
    public static final int OBJ_TIMER = 9;
    public static final int OBJ_NONRENDERABLE = 10;
    public static final int OBJ_ANGLE = 11;
    public static final int OBJ_X2 = 12;
    public static final int OBJ_Y2 = 13;
    public static final int OBJ_ENEMY_BULLET = 14;
    public static final int OBJ_RADIUS = 15;
    public static final int OBJ_MAKE_POWER_UP = 16;
    public static final int OBJ_WALL_EXPLOSION = 17;
    public static final int OBJ_HITS = 18;
    public static final int OBJ_SHOOT_DELAY = 19;
    public static final int OBJ_BOSS = 20;
    public static final int OBJ_BUBBLE = 21;
    public static final int OBJ_FORCE_FIELD = 22;
    public static final int OBJ_DISABLED = 23;
    public static final int OBJ_SHRINKER = 24;
    public static final int OBJ_ATTRACTOR = 25;
    public static final int OBJ_FAN_BLADE = 26;
    public static final int TYPE_EXPLOSION = 0;
    public static final int TYPE_EXPLOSION_SEED = 1;
    public static final int TYPE_PLAYER = 2;
    public static final int TYPE_BULLET = 3;
    public static final int TYPE_ENEMY = 4;
    public static final int TYPE_POWER_UP = 5;
    static final int VIDEO_WIDTH = 256;
    static final int VIDEO_HEIGHT = 256;
    private final static String S = "\uff00\u0000\u0100\u0000\u0000\u0100\u0000\u0000\u0100\u0000\u0100\u0000\u0000\u0100\u0000\u0000\u0100\u01ff\uf000\u0c00\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u00f0\u001c\u00f0\uf000\ufc00\u0000\uf000\ue000\u0000\u0000\u0000\u0000\u0000\u0000\u0300\u00ff\u003c\u0fc0\u00ff\u003c\u3ff0\u0100\u8000\u8000\u0100\u8000\u8000\u0100\u8000\u0080\u8001\u0080\u0080\u8001\u0080\u8080\u8001\u0080\u8080\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u00fe\u0000\u0000\u008c\u0000\u00e0\u0000\u0000\u00f8\u0000\u003f\u00fe\u00fc\u803f\u00ff\u00fe\ue03f\u00ff\u000f\u00ff\uf000\u0003\u003c\uc000\u0000\u0000\u00ff\u0000\u0000\u00ff\u0003\u003c\uc000\u000f\u00ff\uf000\uc3c3\u00c3\u7c00\u8181\u0081\u3000\u0000\u0000\u0000\u0000\u0000\u001f\u8181\u0081\u301f\uc3c3\u00c3\u7c1f\u0d01\u0205\uba1b\ub6b7\u1776\u0a1a\u0202\u0101\u130e\u2322\u9062\u8689\u4464\u4952\u087e\u0304\u3f08\u8050\u888f\u22f7\uf722\u8f88\u5585\u0a3f\u807f\u4443\u142b\u2b28\u2c2b\u2b17\u4344\u7f80\u150e\u6412\u8c84\uc9fa\u8889\u244f\u1514\u0f0c\u0a07\u3912\u603c\u9292\uc2a2\u1122\u2010\u3f40\u1b06\u5b2a\uc7a2\u54be\ube54\ua2c7\u2a5b\u061b\u1807\u403e\u807e\ufe81\u7142\ua2cb\u4e92\u1e2e\u0407\u4538\u5458\u5762\ucacc\u9ce9\u93ba\u2b55\u990f\uf191\u322e\u4120\u4442\u526d\ud52d\ua4a6\u1c07\u6330\ucf47\u9c9e\u9e9c\u47cf\u3063\u0718\u0505\u790f\u8949\u9191\u84e5\ud584\ubfb5\u2191\u0101\u8202\uee92\uaae9\u8a8c\u8acb\ua1a5\u0820";
    private final Graphics2D g;
    public boolean[] keys = new boolean[32768];
    public BufferedImage image;
    public int score;
    public int playerDead = 1;
    public float cameraX;
    public float[] player = new float[256];
    public boolean paused = false;
    public float playerX;
    public float playerY;
    int i = 0;
    int j = 0;
    int k = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    int s = 0;
    int v = 0;
    float dx = 0;
    float dy = 0;
    Random random = new XoRoShiRo128PlusRandom(7);
    float halfPipeOffset = 0;
    Lst<float[]> queue = new Lst<>();
    int[][] levelMap = null;
    Graphics2D g2 = null;
    boolean bossMode = false;
    float cameraVx = 0;
    float mag = 0;
    int counter = 0;


    int fireworks = 0;
    int playerShootDelay = 0;
    BufferedImage[] sprites = new BufferedImage[14 * 4096];
    int[] background = new int[65536];
    private BufferedImage tileImage;
    private int[] backgroundImage;
    private int[][][] halfPipe;
    private AffineTransform defaultTransform;


    public static void main(String[] args) {
        var g = new Gradius4K();
        new Thread(g).start();
    }

    public Gradius4K() {
        enableEvents(8);

        JFrame f = new JFrame();
        f.setContentPane(this);
        f.setSize(500, 500);
        f.setVisible(true);
        f.setIgnoreRepaint(true);
        f.setResizable(false);

        image = new BufferedImage(VIDEO_WIDTH, VIDEO_HEIGHT, 1);
        g = (Graphics2D) image.getGraphics();
        defaultTransform = g.getTransform();

        int[][][][][][] orbs = new int[3][3][16][16][16][16];
        for (s = 0; s < 14; s++) {
            for (i = 0; i < 3; i++) {
                for (j = 0; j < 3; j++) {
                    for (k = 0; k < 16; k++) {
                        for (z = 0; z < 16; z++) {


                            dx = 0.393f * k;
                            dy = 0.393f * z;

                            int red1 = 127 + (int) (127 * Math.sin(dx));
                            int red2 = 127 + (int) (127 * Math.sin(dy));

                            int green1 = 127 + (int) (127 * Math.sin(dx + 1.05f));
                            int green2 = 127 + (int) (127 * Math.sin(dy + 1.05f));

                            int blue1 = 127 + (int) (127 * Math.sin(dx + 2.09f));
                            int blue2 = 127 + (int) (127 * Math.sin(dy + 2.09f));


                            if (i == 1) {
                                red1 >>= 1;
                                green1 >>= 1;
                                blue1 >>= 1;
                            }
                            if (j == 1) {
                                red2 >>= 1;
                                green2 >>= 1;
                                blue2 >>= 1;
                            }


                            if (i == 2) {
                                red1 = (red1 + 255) >> 1;
                                green1 = (green1 + 255) >> 1;
                                blue1 = (blue1 + 255) >> 1;
                            }
                            if (j == 2) {
                                red2 = (red2 + 255) >> 1;
                                green2 = (green2 + 255) >> 1;
                                blue2 = (blue2 + 255) >> 1;
                            }

                            sprites[(s << 12) | (i << 10) | (j << 8) | (k << 4) | z] = new BufferedImage(16, 16, 2);
                            if (s == 0) {


                                for (y = 0; y < 16; y++) {
                                    for (x = 0; x < 16; x++) {
                                        dx = 7.5f - x;
                                        dy = 7.5f - y;
                                        float radius = (float) Math.sqrt(dx * dx + dy * dy) / 8;
                                        if (radius <= 1) {
                                            int red = red1 + (int) ((red2 - red1) * radius);
                                            int green = green1 + (int) ((green2 - green1) * radius);
                                            int blue = blue1 + (int) ((blue2 - blue1) * radius);

                                            orbs[i][j][k][z][y][x] = 0xFF000000 | (red << 16) | (green << 8) | blue;
                                        }
                                    }
                                    sprites[(i << 10) | (j << 8) | (k << 4) | z].setRGB(0, y, 16, 1, orbs[i][j][k][z][y], 0, 16);
                                }
                            } else {


                                for (y = 0; y < 16; y++) {
                                    v = S.charAt(144 + (y >> 1) + ((s - 1) << 3));
                                    if ((y & 1) == 0) {
                                        v &= 0xFF;
                                    } else {
                                        v >>= 8;
                                    }
                                    for (x = 0; x < 8; x++) {
                                        background[((7 - x) << 4) + y] = background[((8 + x) << 4) + y] = v == 0 ? 0 : (0xFF000000 | (((v & 1) == 0) ? ((red1 << 16) | (green1 << 8) | blue1)
                                                : ((red2 << 16) | (green2 << 8) | blue2)));
                                        v >>= 1;
                                    }
                                }
                                sprites[(s << 12) | (i << 10) | (j << 8) | (k << 4) | z].setRGB(0, 0, 16, 16, background, 0, 16);
                            }
                        }
                    }
                }
            }
        }

        halfPipe = new int[256][256][2];
        for (y = 0; y < 256; y++) {
            for (x = 0; x < 256; x++) {
                float d = x - 128;
                float e = y - 128;
                float A = e * e + 4096;
                float t = (8192 + (float) Math.sqrt(67108864 + 49152 * A)) / (2 * A);
                float hx = d * t;
                halfPipe[y][x][0] = ((int) (512 + hx)) & 0xFF;
                float hz = 64 - 64 * t;
                float hy = e * t;
                halfPipe[y][x][1] = ((int) (512 + Math.atan2(hy, hz) * 128)) & 0xFF;
            }
        }


        backgroundImage = new int[65536];
        for (k = 0; k < 4096; k++) {
            i = random.nextInt(512) & 0xFF;
            j = random.nextInt(512) & 0xFF;
            for (y = 0; y < 16; y++) {
                for (x = 0; x < 16; x++) {
                    if (orbs[2][1][9][9][y][x] != 0) {
                        backgroundImage[(((i + y) & 0xFF) << 8) + ((j + x) & 0xFF)] = orbs[2][1][9][9][y][x];
                        background[(((i + y) & 31) << 5) + ((j + x) & 31)] = orbs[2][0][8][8][y][x];
                    }
                }
            }
        }
        tileImage = new BufferedImage(32, 32, 1);
        tileImage.setRGB(0, 0, 32, 32, background, 0, 32);
    }

    @Override
    public void run() {
        while (true) {
            if (!paused)
                next();

            Util.sleepMS(updateMS);
        }
    }

    public void next() {
        if (playerDead > 0 && --playerDead == 0) {

            queue.clear();
            cameraX = 0;
            cameraVx = level == 3 ? 8 : 0.5f;
            bossMode = false;
            halfPipeOffset = 0;

            if (fireworks < 0) {

                fireworks = 0;
                queue.add(player);
                player[OBJ_X] -= 2304;
            } else {

                player = new float[256];
                queue.add(player);
                player[OBJ_TYPE] = TYPE_PLAYER;
                player[OBJ_SPRITE] = 4816;
                player[OBJ_SCALE_X] = 1.5f;
                player[OBJ_SCALE_Y] = 1;
                player[OBJ_X] = 64;
                player[OBJ_Y] = 128;
                playerGun = 1;
            }


            levelMap = new int[8][256];
            for (x = 24; x < 72; x++) {
                levelMap[0][x] = 1;
                levelMap[7][x] = 1;
            }
            for (y = 0, z = 18 * level; y < 6; y++) {
                for (x = 0; x < 3; x++) {
                    k = S.charAt(z++);
                    for (i = 0; i < 16; i++) {
                        levelMap[y + 1][24 + (x << 4) + i] = k & 1;
                        k >>= 1;
                    }
                }
            }

            if (level == 2) {

                for (x = 0; x < 6; x++) {
                    float[] enemy = new float[256];
                    queue.add(enemy);
                    enemy[OBJ_TYPE] = TYPE_ENEMY;
                    enemy[OBJ_X] = 785 + (x << 8) + (random.nextInt(8) << 5);
                    enemy[OBJ_Y] = enemy[OBJ_Y2] = 128;
                    enemy[OBJ_SPRITE] = 2302;
                    enemy[OBJ_SCALE_X] = 2;
                    enemy[OBJ_SCALE_Y] = 24;
                    enemy[OBJ_HITS] = 4096;
                    enemy[OBJ_SHOOT_DELAY] = 64;
                    enemy[OBJ_FORCE_FIELD] = 1;
                }
            }

            if (level == 3) {
                for (x = 0; x < 5; x++) {

                    float[] powerUp = new float[256];
                    queue.add(powerUp);
                    powerUp[OBJ_TYPE] = TYPE_POWER_UP;
                    powerUp[OBJ_X] = 264 + (x << 5);
                    powerUp[OBJ_Y] = 128;
                    powerUp[OBJ_SCALE_X] = 1;
                    powerUp[OBJ_SCALE_Y] = 1;
                    powerUp[OBJ_SPRITE] = 18463;
                }
            }

            if (level == 4) {
                for (i = 0; i < 32; i++) {

                    float[] shrinker = new float[256];
                    queue.add(shrinker);
                    shrinker[OBJ_TYPE] = TYPE_ENEMY;
                    shrinker[OBJ_X] = 800 + (random.nextInt(22) << 6);
                    shrinker[OBJ_Y] = shrinker[OBJ_Y2] = 64 + (random.nextInt(3) << 6);
                    shrinker[OBJ_SCALE_X] = 4;
                    shrinker[OBJ_SCALE_Y] = 4;
                    shrinker[OBJ_SPRITE] = 2338;
                    shrinker[OBJ_HITS] = 4096;
                    shrinker[OBJ_TIMER] = 4096;
                    shrinker[OBJ_SHRINKER] = 1;
                }
            }

            if (level == 7) {
                for (k = 0; k < 3; k++) {
                    for (j = 0; j < 3; j++) {
                        mag = 2.09f * j;
                        dy = (float) Math.sin(mag);
                        dx = (float) Math.cos(mag);
                        for (i = 0; i < 7; i++) {

                            float[] blade = new float[256];
                            queue.add(blade);
                            blade[OBJ_TYPE] = TYPE_ENEMY;
                            blade[OBJ_X2] = (k << 8) + 896;
                            blade[OBJ_Y2] = switch (k) {
                                case 0 -> 0.01f;
                                case 1 -> -0.01f;
                                default -> 0.012f;
                            };
                            blade[OBJ_SCALE_X] = 1;
                            blade[OBJ_SCALE_Y] = 1;
                            blade[OBJ_SPRITE] = 2301;
                            blade[OBJ_HITS] = 4096;
                            blade[OBJ_TIMER] = 4096;
                            blade[OBJ_ANGLE] = mag;
                            blade[OBJ_RADIUS] = i << 4;
                            blade[OBJ_FAN_BLADE] = 1;
                        }
                    }
                }
            }
        }

        if (cameraX >= 2304) {
            cameraVx = 0;
            if (!bossMode) {
                bossMode = true;


                float[] boss = new float[256];
                queue.add(boss);
                boss[OBJ_TYPE] = TYPE_ENEMY;
                boss[OBJ_X] = 2624;
                boss[OBJ_Y] = boss[OBJ_Y2] = 128;
                boss[OBJ_VX] = -1;
                boss[OBJ_SPRITE] = switch (level) {
                    case 0 -> 21128;
                    case 1 -> 26679;
                    case 2 -> 29320;
                    case 3 -> 35123;
                    case 4 -> 39148;
                    case 5 -> 43096;
                    case 6 -> 54784;
                    default -> 51319;
                };
                boss[OBJ_SCALE_X] = 4 + (level >> 1);
                boss[OBJ_SCALE_Y] = 4 + (level >> 1);
                boss[OBJ_RADIUS] = 8 + 1.5f * level;
                boss[OBJ_HITS] = 512;
                boss[OBJ_SHOOT_DELAY] = 32 - level;
                boss[OBJ_BOSS] = 1;
            }
        }
        cameraX += cameraVx;

        if (fireworks > 0) {
            if (--fireworks == 0) {
                if (level == 7) {
                    fireworks = 64;
                } else {
                    fireworks = -1;
                    playerDead = 1;
                    level++;
                }
            } else if ((fireworks & 3) == 0) {
                float[] explosionSeed = new float[256];
                queue.add(explosionSeed);
                explosionSeed[OBJ_TYPE] = TYPE_EXPLOSION_SEED;
                explosionSeed[OBJ_X] = random.nextInt(256) + cameraX;
                explosionSeed[OBJ_Y] = random.nextInt(256);
                explosionSeed[OBJ_NONRENDERABLE] = 1;
            }
        }

        if (bossMode) {
            if (fireworks == 0) {
                halfPipeOffset += 0.5f;
            }
        } else if (((counter++) & 0xFF) == 0) {
            float Y = 0;
            for (i = 0; i < 16; i++) {
                Y = 48 + (random.nextInt(6) << 5);
                if (levelMap[(int) Y >> 5][((int) cameraX >> 5) + 8] == 0 && levelMap[((int) Y >> 5) + 1][((int) cameraX >> 5) + 8] == 0) {
                    break;
                }
            }

            if (level != 3) {
                if (!(level == 1 || level == 7) || cameraX < 384 || (level == 7 && cameraX > 1408)) {
                    if (random.nextBoolean()) {
                        for (i = 0; i < 5; i++) {

                            float[] enemy = new float[256];
                            queue.add(enemy);
                            enemy[OBJ_TYPE] = TYPE_ENEMY;
                            enemy[OBJ_X] = cameraX + 256 + i * 24;
                            enemy[OBJ_Y] = enemy[OBJ_Y2] = Y;
                            enemy[OBJ_VX] = -1;
                            enemy[OBJ_SPRITE] = 8518;
                            enemy[OBJ_SCALE_X] = 1;
                            enemy[OBJ_SCALE_Y] = 1;
                            enemy[OBJ_ANGLE] = -1.2f * i;
                            enemy[OBJ_RADIUS] = 8;
                            enemy[OBJ_HITS] = 1;
                            enemy[OBJ_SHOOT_DELAY] = 240;
                        }
                    } else if (level == 6) {
                        for (i = 0; i < 3; i++) {

                            dx = 6.28f * random.nextFloat();
                            float[] enemy = new float[256];
                            queue.add(enemy);
                            enemy[OBJ_TYPE] = TYPE_ENEMY;
                            enemy[OBJ_X] = cameraX + 256 + 20 * i;
                            enemy[OBJ_Y] = Y + 20 * i;
                            enemy[OBJ_SPRITE] = 47180;
                            enemy[OBJ_SCALE_X] = 1;
                            enemy[OBJ_SCALE_Y] = 1;
                            enemy[OBJ_HITS] = 1;
                            enemy[OBJ_SHOOT_DELAY] = 240;
                            enemy[OBJ_VX] = (float) Math.cos(dx);
                            enemy[OBJ_VY] = (float) Math.sin(dy);
                            enemy[OBJ_ATTRACTOR] = 1;
                        }
                    } else {

                        for (i = 0; i < 3; i++) {
                            float[] enemy = new float[256];
                            queue.add(enemy);
                            enemy[OBJ_TYPE] = TYPE_ENEMY;
                            enemy[OBJ_X] = cameraX + 256;
                            enemy[OBJ_Y] = enemy[OBJ_Y2] = Y + 20 * i;
                            enemy[OBJ_VX] = -1;
                            enemy[OBJ_SPRITE] = 12375;
                            enemy[OBJ_SCALE_X] = 1;
                            enemy[OBJ_SCALE_Y] = 1;
                            enemy[OBJ_HITS] = 1;
                            enemy[OBJ_SHOOT_DELAY] = 240;
                        }
                    }
                } else if (level == 1 && cameraX > 512) {

                    float[] enemy = new float[256];
                    queue.add(enemy);
                    enemy[OBJ_TYPE] = TYPE_ENEMY;
                    enemy[OBJ_X] = cameraX + 320;
                    enemy[OBJ_Y] = enemy[OBJ_Y2] = Y + 20 * i;
                    enemy[OBJ_VX] = -0.5f;
                    enemy[OBJ_SPRITE] = 2301;
                    enemy[OBJ_SCALE_X] = 4;
                    enemy[OBJ_SCALE_Y] = 4;
                    enemy[OBJ_HITS] = 2;
                    enemy[OBJ_TIMER] = 4096;
                    enemy[OBJ_SHOOT_DELAY] = 4096;
                    enemy[OBJ_BUBBLE] = 1;
                    enemy[OBJ_RADIUS] = 8;
                }
            }
        }

        boolean playerExploded = false;

        for (i = queue.size() - 1; i >= 0; i--) {
            float[] object = queue.get(i);

            if (object[OBJ_REMOVE] == 1) {
                queue.remove(i);
                continue;
            }

            switch ((int) object[OBJ_TYPE]) {
                case TYPE_PLAYER -> {
                    object[OBJ_X] += cameraVx;
                    if (keys[VK_UP]) {
                        if (object[OBJ_Y] > 11) {
                            object[OBJ_Y] -= SPEED;
                        }
                    } else if (keys[VK_DOWN]) {
                        if (object[OBJ_Y] < 220) {
                            object[OBJ_Y] += SPEED;
                        }
                    }
                    if (keys[VK_LEFT]) {
                        if (object[OBJ_X] > cameraX + 13) {
                            object[OBJ_X] -= SPEED;
                        }
                    } else if (keys[VK_RIGHT]) {
                        if (object[OBJ_X] < cameraX + 243) {
                            object[OBJ_X] += SPEED;
                        }
                    }
                    playerX = object[OBJ_X] - cameraX;
                    playerY = object[OBJ_Y];
                    if (playerShootDelay > 0) {
                        playerShootDelay--;
                    } else if (keys[VK_SHOOT]) {
                        playerShootDelay = 5;


                        for (k = 0; k < 2; k++) {
                            for (j = 0; j < playerGun; j++) {

                                float[] bullet = new float[256];
                                queue.add(bullet);
                                bullet[OBJ_TYPE] = TYPE_BULLET;
                                bullet[OBJ_X] = object[OBJ_X] + 24 * (float) Math.cos(0.25f * j);
                                bullet[OBJ_Y] = object[OBJ_Y] + (k == 0 ? 24 : -24) * (float) Math.sin(0.25f * j);
                                bullet[OBJ_SPRITE] = 733;
                                bullet[OBJ_VX] = 7.5f + cameraVx;
                                bullet[OBJ_SCALE_X] = 1.5f;
                                bullet[OBJ_SCALE_Y] = 0.25f;
                            }
                        }
                    }
                    for (j = queue.size() - 1; j >= 0; j--) {
                        float[] enemy = queue.get(j);
                        if ((enemy[OBJ_TYPE] == TYPE_ENEMY || enemy[OBJ_TYPE] == TYPE_POWER_UP) && object[OBJ_X] - 12 < enemy[OBJ_X] + 8 * enemy[OBJ_SCALE_X]
                                && object[OBJ_X] + 12 > enemy[OBJ_X] - 8 * enemy[OBJ_SCALE_X] && object[OBJ_Y] - 8 < enemy[OBJ_Y] + 8 * enemy[OBJ_SCALE_Y]
                                && object[OBJ_Y] + 8 > enemy[OBJ_Y] - 8 * enemy[OBJ_SCALE_Y]) {
                            if (enemy[OBJ_TYPE] == TYPE_POWER_UP) {

                                enemy[OBJ_REMOVE] = 1;
                                if (playerGun < 5)
                                    playerGun++;

                            } else if (enemy[OBJ_DISABLED] == 0) {

                                playerExploded = DIE_IF_HIT_ENEMY;
                            }
                        }
                    }
                    if (levelMap[((((int) object[OBJ_Y]) - 5) >> 5) & 7][((((int) object[OBJ_X])) >> 5) & 0xFF] == 1
                            || levelMap[((((int) object[OBJ_Y]) + 5) >> 5) & 7][((((int) object[OBJ_X])) >> 5) & 0xFF] == 1
                            || levelMap[((((int) object[OBJ_Y])) >> 5) & 7][((((int) object[OBJ_X]) - 7) >> 5) & 0xFF] == 1
                            || levelMap[((((int) object[OBJ_Y])) >> 5) & 7][((((int) object[OBJ_X]) + 7) >> 5) & 0xFF] == 1) {
                        playerExploded = DIE_IF_HIT_WALL;
                    }
                }
                case TYPE_EXPLOSION -> {
                    object[OBJ_X] += object[OBJ_VX];
                    object[OBJ_Y] += object[OBJ_VY];
                    object[OBJ_SCALE_X] *= 0.8f;
                    object[OBJ_SCALE_Y] *= 0.8f;
                    if (--object[OBJ_TIMER] == 0) {
                        object[OBJ_REMOVE] = 1;
                    }
                }
                case TYPE_EXPLOSION_SEED -> {
                    object[OBJ_REMOVE] = 1;
                    for (j = (object[OBJ_WALL_EXPLOSION] == 1) ? 16 : 128; j >= 0; j--) {

                        float[] explosion = new float[256];
                        queue.add(explosion);
                        explosion[OBJ_TYPE] = TYPE_EXPLOSION;
                        explosion[OBJ_X] = object[OBJ_X];
                        explosion[OBJ_Y] = object[OBJ_Y];
                        dx = 6.28f * random.nextFloat();
                        dy = ((object[OBJ_WALL_EXPLOSION] == 1) ? 1 : 3) * random.nextFloat();
                        explosion[OBJ_VX] = dy * (float) Math.sin(dx);
                        explosion[OBJ_VY] = dy * (float) Math.cos(dx);
                        explosion[OBJ_SPRITE] = 56;
                        explosion[OBJ_TIMER] = 15;
                        explosion[OBJ_SCALE_X] = 4;
                        explosion[OBJ_SCALE_Y] = 4;
                    }
                    if (object[OBJ_MAKE_POWER_UP] == 1) {

                        float[] powerUp = new float[256];
                        queue.add(powerUp);
                        powerUp[OBJ_TYPE] = TYPE_POWER_UP;
                        powerUp[OBJ_X] = object[OBJ_X];
                        powerUp[OBJ_Y] = object[OBJ_Y];
                        powerUp[OBJ_SCALE_X] = 1;
                        powerUp[OBJ_SCALE_Y] = 1;
                        powerUp[OBJ_SPRITE] = 18463;
                    }
                }
                case TYPE_BULLET -> {
                    object[OBJ_X] += object[OBJ_VX];
                    object[OBJ_Y] += object[OBJ_VY];
                    if (object[OBJ_X] < cameraX - 32 || object[OBJ_X] > cameraX + 287 || object[OBJ_Y] < -32 || object[OBJ_Y] > 287) {

                        object[OBJ_REMOVE] = 1;
                    } else if (levelMap[((int) object[OBJ_Y] >> 5) & 7][((int) object[OBJ_X] >> 5) & 0xFF] == 1) {

                        object[OBJ_REMOVE] = 1;
                        if (object[OBJ_ENEMY_BULLET] == 0) {
                            float[] explosionSeed = new float[256];
                            queue.add(explosionSeed);
                            explosionSeed[OBJ_TYPE] = TYPE_EXPLOSION_SEED;
                            explosionSeed[OBJ_X] = object[OBJ_X];
                            explosionSeed[OBJ_Y] = object[OBJ_Y];
                            explosionSeed[OBJ_NONRENDERABLE] = 1;
                            explosionSeed[OBJ_WALL_EXPLOSION] = 1;
                        }
                    } else if (object[OBJ_ENEMY_BULLET] == 1) {
                        if (fireworks > 0) {

                            object[OBJ_REMOVE] = 1;
                        } else if (playerDead == 0
                                && object[OBJ_X] >= player[OBJ_X] - 8 && object[OBJ_Y] >= player[OBJ_Y] - 8 && object[OBJ_X] <= player[OBJ_X] + 8 && object[OBJ_Y] <= player[OBJ_Y] + 8) {
                            playerExploded = DIE_IF_HIT_ENEMY_BULLET;
                        }
                    } else {

                        for (j = queue.size() - 1; j >= 0; j--) {
                            float[] enemy = queue.get(j);
                            if (enemy[OBJ_TYPE] == TYPE_ENEMY && enemy[OBJ_DISABLED] == 0 && object[OBJ_X] >= enemy[OBJ_X] - 8 * enemy[OBJ_SCALE_X]
                                    && object[OBJ_Y] >= enemy[OBJ_Y] - 8 * enemy[OBJ_SCALE_Y] && object[OBJ_X] <= enemy[OBJ_X] + 8 * enemy[OBJ_SCALE_X]
                                    && object[OBJ_Y] <= enemy[OBJ_Y] + 8 * enemy[OBJ_SCALE_Y]) {

                                object[OBJ_REMOVE] = 1;
                                score++;

                                float[] explosionSeed = new float[256];
                                queue.add(explosionSeed);
                                explosionSeed[OBJ_TYPE] = TYPE_EXPLOSION_SEED;
                                explosionSeed[OBJ_NONRENDERABLE] = 1;
                                if (--enemy[OBJ_HITS] == 0) {

                                    if (enemy[OBJ_BOSS] == 1) {

                                        fireworks = 512;
                                    }


                                    enemy[OBJ_REMOVE] = 1;
                                    if (enemy[OBJ_BUBBLE] == 1 && enemy[OBJ_SCALE_X] > 1) {
                                        for (k = 0; k < 2; k++) {

                                            float[] bubble = new float[256];
                                            queue.add(bubble);
                                            bubble[OBJ_TYPE] = TYPE_ENEMY;
                                            bubble[OBJ_VX] = random.nextFloat() - 0.5f;
                                            bubble[OBJ_SPRITE] = 2301;
                                            bubble[OBJ_SCALE_X] = enemy[OBJ_SCALE_X] * 0.75f;
                                            bubble[OBJ_SCALE_Y] = enemy[OBJ_SCALE_Y] * 0.75f;
                                            bubble[OBJ_X] = enemy[OBJ_X];
                                            bubble[OBJ_Y] = bubble[OBJ_Y2] = enemy[OBJ_Y] + (k - 0.5f) * bubble[OBJ_SCALE_Y] * 16;
                                            bubble[OBJ_HITS] = 1;
                                            bubble[OBJ_TIMER] = 4096;
                                            bubble[OBJ_SHOOT_DELAY] = 4096;
                                            bubble[OBJ_BUBBLE] = 1;
                                            bubble[OBJ_RADIUS] = 8;
                                            bubble[OBJ_ANGLE] = 6.28f * random.nextFloat();
                                        }
                                    } else {
                                        explosionSeed[OBJ_X] = enemy[OBJ_X];
                                        explosionSeed[OBJ_Y] = enemy[OBJ_Y];
                                        explosionSeed[OBJ_MAKE_POWER_UP] = random.nextInt(13) == 0 ? 1 : 0;
                                    }
                                } else {

                                    explosionSeed[OBJ_X] = object[OBJ_X];
                                    explosionSeed[OBJ_Y] = object[OBJ_Y];
                                    explosionSeed[OBJ_WALL_EXPLOSION] = 1;

                                    if (enemy[OBJ_SHRINKER] == 1) {
                                        enemy[OBJ_SCALE_X] -= 0.25f;
                                        enemy[OBJ_SCALE_Y] -= 0.25f;
                                        if (enemy[OBJ_SCALE_X] <= 0.5f) {
                                            enemy[OBJ_HITS] = 1;
                                        }
                                    }
                                }

                            }
                        }
                    }
                }
                case TYPE_ENEMY -> {
                    if (fireworks > 0) {

                        object[OBJ_REMOVE] = 1;
                    }
                    if (object[OBJ_FAN_BLADE] == 1) {
                        object[OBJ_ANGLE] += object[OBJ_Y2];
                        object[OBJ_X] = object[OBJ_X2] + object[OBJ_RADIUS] * (float) Math.cos(object[OBJ_ANGLE]);
                        object[OBJ_Y] = 128 + object[OBJ_RADIUS] * (float) Math.sin(object[OBJ_ANGLE]);
                    } else if (object[OBJ_ATTRACTOR] == 1) {
                        if ((object[OBJ_VX] < 0 && (levelMap[(((int) object[OBJ_Y]) >> 5) & 7][(((int) (object[OBJ_X] - 12 * object[OBJ_SCALE_X])) >> 5) & 0xFF] == 1))
                                || (object[OBJ_VY] > 0 && (levelMap[(((int) object[OBJ_Y]) >> 5) & 7][(((int) (object[OBJ_X] + 12 * object[OBJ_SCALE_X])) >> 5) & 0xFF] == 1))) {
                            object[OBJ_VX] = -object[OBJ_VX];
                        }
                        if ((object[OBJ_VY] < 0 && (levelMap[((int) (object[OBJ_Y] - 12 * object[OBJ_SCALE_Y]) >> 5) & 7][(((int) (object[OBJ_X])) >> 5) & 0xFF] == 1))
                                || (object[OBJ_VY] > 0 && (levelMap[((int) (object[OBJ_Y] + 12 * object[OBJ_SCALE_Y]) >> 5) & 7][(((int) (object[OBJ_X])) >> 5) & 0xFF] == 1))) {
                            object[OBJ_VY] = -object[OBJ_VY];
                        }
                        dx = player[OBJ_X] - object[OBJ_X];
                        dy = player[OBJ_Y] - object[OBJ_Y];
                        mag = 64 * (float) Math.sqrt(dx * dx + dy * dy);
                        object[OBJ_VX] += dx / mag;
                        object[OBJ_VY] += dy / mag;
                        object[OBJ_VX] *= 0.995f;
                        object[OBJ_VY] *= 0.995f;
                        object[OBJ_X] += object[OBJ_VX];
                        object[OBJ_Y] += object[OBJ_VY];
                    } else {
                        if ((object[OBJ_VX] < 0 && (levelMap[(((int) object[OBJ_Y]) >> 5) & 7][(((int) (object[OBJ_X] - 12 * object[OBJ_SCALE_X])) >> 5) & 0xFF] == 1 || (object[OBJ_BUBBLE] == 0 && object[OBJ_X] < cameraX
                                + 12 * object[OBJ_SCALE_X])))
                                || (object[OBJ_VX] > 0 && (levelMap[(((int) object[OBJ_Y]) >> 5) & 7][(((int) (object[OBJ_X] + 12 * object[OBJ_SCALE_X])) >> 5) & 0xFF] == 1 || (object[OBJ_BOSS] == 1 && object[OBJ_X]
                                + 8 * object[OBJ_SCALE_X] >= 2560)))) {
                            object[OBJ_VX] = -object[OBJ_VX] + cameraVx;
                        }

                        if (levelMap[(((int) (object[OBJ_Y] - 8 * object[OBJ_SCALE_Y])) >> 5) & 7][((((int) object[OBJ_X])) >> 5) & 0xFF] == 1) {
                            object[OBJ_Y2]++;
                        }
                        if (levelMap[(((int) (object[OBJ_Y] + 8 * object[OBJ_SCALE_Y])) >> 5) & 7][((((int) object[OBJ_X])) >> 5) & 0xFF] == 1) {
                            object[OBJ_Y2]--;
                        }
                        object[OBJ_X] += object[OBJ_VX];
                        object[OBJ_Y] = object[OBJ_Y2] + object[OBJ_RADIUS] * (float) Math.sin(object[OBJ_ANGLE]);
                        object[OBJ_ANGLE] += 0.075f;
                    }
                    if (object[OBJ_FORCE_FIELD] == 1) {
                        object[OBJ_SPRITE] = object[OBJ_SPRITE] == 766 ? 2302 : 766;
                    }
                    if (object[OBJ_TIMER] > 0) {
                        object[OBJ_TIMER]--;
                    } else {
                        object[OBJ_TIMER] = object[OBJ_SHOOT_DELAY];

                        if (object[OBJ_FORCE_FIELD] == 1) {

                            object[OBJ_DISABLED] = object[OBJ_DISABLED] == 1 ? 0 : 1;
                        } else {

                            if (!easy) {
                                float[] bullet = new float[256];
                                queue.add(bullet);
                                bullet[OBJ_X] = object[OBJ_X];
                                bullet[OBJ_Y] = object[OBJ_Y];
                                bullet[OBJ_TYPE] = TYPE_BULLET;
                                bullet[OBJ_ENEMY_BULLET] = 1;
                                bullet[OBJ_SPRITE] = 2338;
                                bullet[OBJ_SCALE_X] = 0.5f;
                                bullet[OBJ_SCALE_Y] = 0.5f;
                                bullet[OBJ_NONRENDERABLE] = 0;
                                dx = player[OBJ_X] - bullet[OBJ_X];
                                dy = player[OBJ_Y] - bullet[OBJ_Y];
                                mag = 0.5f * (float) Math.sqrt(dx * dx + dy * dy);
                                bullet[OBJ_VX] = dx / mag + cameraVx;
                                bullet[OBJ_VY] = dy / mag;
                            }
                        }
                    }
                    if (object[OBJ_X] + 128 < cameraX) {
                        object[OBJ_REMOVE] = 1;
                    }
                }
                case TYPE_POWER_UP -> {
                    if (object[OBJ_X] + 8 < cameraX) {
                        object[OBJ_REMOVE] = 1;
                    }
                    if (object[OBJ_TIMER] > 0) {
                        object[OBJ_TIMER]--;
                    } else {
                        object[OBJ_TIMER] = 4;
                        object[OBJ_SPRITE] = (object[OBJ_SPRITE] == 17137) ? 18463 : 17137;
                    }
                }
            }
        }

        if (playerExploded) {
            player[OBJ_REMOVE] = 1;
            float[] explosionSeed = new float[256];
            explosionSeed[OBJ_TYPE] = TYPE_EXPLOSION_SEED;
            explosionSeed[OBJ_X] = player[OBJ_X];
            explosionSeed[OBJ_Y] = player[OBJ_Y];
            explosionSeed[OBJ_NONRENDERABLE] = 1;
            queue.add(explosionSeed);

            this.playerDead = Afterlife_Time;
        }


        image.setRGB(0, 0, 256, 256, background, 0, 256);


        for (i = queue.size() - 1; i >= 0; i--) {
            float[] object = queue.get(i);
            if (object[OBJ_NONRENDERABLE] == 0 && object[OBJ_DISABLED] == 0) {
                g.translate(object[OBJ_X] - 8 * object[OBJ_SCALE_X] - cameraX, object[OBJ_Y] - 8 * object[OBJ_SCALE_Y]);
                g.scale(object[OBJ_SCALE_X], object[OBJ_SCALE_Y]);
                g.drawImage(sprites[(int) object[OBJ_SPRITE]], 0, 0, null);
                g.setTransform(defaultTransform);
            }
        }


        z = ((int) cameraX) & 31;
        for (x = 0; x < 9; x++) {
            int ix = x + (((int) cameraX) >> 5);
            for (y = 0; y < 8; y++) {
                if (levelMap[y][ix] == 1) {
                    g.drawImage(tileImage, (x << 5) - z, y << 5, null);
                }
            }
        }

        if (g2 == null) {
            g2 = (Graphics2D) getGraphics();
            requestFocus();
        }

        g2.drawImage(image, 0, 0, 512, 512, null);
    }

    @Override
    public void processKeyEvent(KeyEvent keyEvent) {

        int k = keyEvent.getKeyCode();
        if (k > 0) {
            final int VK_D = 0x44;
            final int VK_A = 0x41;
            final int VK_S = 0x53;
            final int VK_W = 0x57;
            final int VK_DOWN = 0x28;
            final int VK_LEFT = 0x25;
            switch (k) {
                case VK_W -> {
                    final int VK_UP = 0x26;
                    k = VK_UP;
                }
                case VK_D -> {
                    final int VK_RIGHT = 0x27;
                    k = VK_RIGHT;
                }
                case VK_A -> k = VK_LEFT;
                case VK_S -> k = VK_DOWN;
            }
            final int VK_SHOOT = 0x42;
            keys[(k >= VK_LEFT && k <= VK_DOWN) ? k : VK_SHOOT] = keyEvent.getID() != 402;
        }

    }

    
    /*public static void main(String[] args) throws Throwable {
      javax.swing.JFrame frame = new javax.swing.JFrame("Gradius 4K");
	  frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
	  a applet = new a();
	  applet.setPreferredSize(new java.awt.Dimension(512, 512));
	  frame.addAt(applet, java.awt.BorderLayout.CENTER);
	  frame.setResizable(false);
	  frame.pack();
	  frame.setLocationRelativeTo(null);
	  frame.setVisible(true);
	  Thread.sleep(250);
	  applet.start();
	}*/
}