package java4k.spacedevastation;

/*
Copyright (C) 2013 Mikael GUILLEMOT

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

import java.applet.Applet;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.util.Random;

/*
 * Space Devastation feb 2013
 * @author	Mikael GUILLEMOT
 * @version 	1.1
 * 
 */

public class A extends Applet implements Runnable {

	
	private static final int GAMESTART = -1;
	private static final int GAMEPAUSE = 0;
	private static final int GAMEINIT = 1;
	private static final int GAMERUN = 2;
    private static final int GAMEWIN = 4;
    private static final int _X = 0;
	private static final int _Y = 1;


    private final String[] text = {"", "Thrusters Upgrade Unlocked!", "", "Double Damage Unlocked!", "",
            "Regeneration Boost Unlocked!", "", "Energy Boost Unlocked!", "", ""};
    private final String[] textlvl = {"Please! Don't hurt me captain galaxy!",
            "I love the smell of laser in the morning.", "Do you know Chuck norris? it's my follower!"};

    private static final int MAP1T = 3;

    private static final int EXPLODE_FOE = 10;
	private static final int EXPLODE_S = 110;
	private static final int SHIP_HIT = 120;

    private static final int SHOOT2 = 150;

    private static final int SHIP_S = 152;
	private static final int SHIP_E = 159;
	private static final int SHOOT1 = 159;
	private static final int E_SHOOT = 160;
    private static final int CUR = 173;
	private static final int FOE = 174;

    private final float[] BLUR = { .1f, .1f, .1f, .1f, .21f, .1f, .1f, .1f, .1f };
	private final Color[][] color = new Color[5][60];
	private final Color BLACK = new Color(0x0);
	private final Color WHITE = new Color(0xFFFFFFFF);

	
	private int movespeed;
	private int regeneration_div;
	private int shootdmg;
	private int wavefrequency = 200;
	private int wave_multiplier = 2;
	private int foe_firerate_div = 50;
	private int bossmove = 4;
	private int chalenge = 170;

	
	private final int[][] firetable = new int[2][128];
	
	private final double[][] enemy_fire = new double[6][128];
	
	private final int[][] explode = new int[5][128];
	
	private final int[][] enemy = new int[8][128];
	private boolean boss;
	private int emax = 100;
	private int gamestatus = GAMESTART;
	private int wave;
	private int firestatus1;
	private int firestatus2;
	private int sens = 1;
	private int lastwave;
	private int level = 1;
	private int score;
	private int energy = 100;
    private Random rand;
	private boolean r = true;
	
	private int x;
    private int y;
    private int i;
    private int j;
    private int k;
    private int l;
	private int dx = 350;
    private int dy = 400;
	private double d1;
    private double d2;
    private Graphics2D g;
    private Graphics2D[] bg;
    private BufferedImage[] buf;
	private int frame;

	@Override
    public void init() {
		rand = new Random();
		
		for (x = 1; x < 6; x++)
			for (y = 10; y < 70; y++) {
				j = 15 - ((y % 50 - 15) < 0 ? -(y % 50 - 15) : (y % 50 - 15));
				k = 20 - ((y - 30) < 0 ? -(y - 30) : (y - 30));
				l = 20 - ((y - 50) < 0 ? -(y - 50) : (y - 50));
				j = j > 0 ? (Math.min(j, 10)) : 0;
				k = k > 0 ? (Math.min(k, 10)) : 0;
				l = l > 0 ? (Math.min(l, 10)) : 0;
				color[x - 1][y - 10] = new Color(25 * j, 25 * k, 25 * l, 51 * x);
			}
		enableEvents(48); 
							
		g = (Graphics2D) getGraphics();

        int _SMLSPRIT = 184;
        buf = new BufferedImage[_SMLSPRIT];
		bg = new Graphics2D[_SMLSPRIT];
		for (i = 0; i < _SMLSPRIT; i++) {
            int _SPECIAL = 151;
            int _MIDSPRIT = 150;
            int _BIGSPRIT = 10;
            if (i < _BIGSPRIT)
				buf[i] = new BufferedImage(800, 600, 2);
			else if (i < _MIDSPRIT)
				buf[i] = new BufferedImage(64, 64, 2);
			else if (i < _SPECIAL)
				buf[i] = new BufferedImage(32, 600, 2);
			else
				buf[i] = new BufferedImage(32, 48, 2);
			bg[i] = (Graphics2D) buf[i].getGraphics();
			clear(i);
		}
		
		clear(MAP1T);
		for (i = 0; i < 1200; i++) {
			buf[MAP1T].setRGB(rand.nextInt(800), rand.nextInt(600), 0x00FFFFFF | (rand.nextInt(255) << 24));
		}

        int TMP = 151;
        bg[TMP].setColor(WHITE);
		bg[TMP].drawOval(12, 2, 8, 20);
		bg[TMP].setColor(BLACK);
		bg[TMP].fillOval(0, 14, 31, 14);
		bg[TMP].setColor(WHITE);
		bg[TMP].drawArc(0, 14, 31, 14, 140, 260);
		bg[TMP].fillRect(6, 12, 2, 12);
		bg[TMP].fillRect(25, 12, 2, 12);
		
		bg[SHIP_S].setColor(color[3][36]);
		bg[SHIP_S].fillOval(2, 18, 28, 28);
		bg[SHIP_S].setColor(WHITE);
		bg[SHIP_S].fillOval(10, 26, 12, 12);
		blur(SHIP_S, 3);
		
		for (i = SHIP_S + 1; i < SHIP_E; i++) {
			bg[i].drawImage(buf[i - 1], null, this);
			blur(i, 3);
			bg[i - 1].drawImage(buf[TMP], null, this);
		}
		bg[SHIP_E - 1].drawImage(buf[TMP], null, this);
		
		bg[SHOOT1].setColor(color[3][42]);
		bg[SHOOT1].fillOval(0, 0, 15, 24);
		bg[SHOOT1].fillOval(17, 0, 15, 24);
		bg[SHOOT1].setColor(WHITE);
		bg[SHOOT1].fillRect(6, 4, 4, 16);
		bg[SHOOT1].fillRect(24, 4, 4, 16);
		
		bg[SHOOT2].setColor(color[3][45]);
		bg[SHOOT2].fillRect(4, 0, 24, 590);
		bg[SHOOT2].fillOval(0, 568, 32, 32);
		bg[SHOOT2].setColor(WHITE);
		bg[SHOOT2].fillRect(12, 0, 8, 590);
		bg[SHOOT2].fillOval(8, 576, 16, 16);
		
		bg[CUR].setColor(WHITE);
		bg[CUR].fillRect(8, 14, 16, 4);
		bg[CUR].fillRect(14, 8, 4, 16);
		
		blur(SHOOT1, 3);
		blur(SHOOT2, 3);
		blur(CUR, 1);
		
		for (i = 0; i < 5; i++) {
			bg[E_SHOOT + 2 * i].setColor(color[i][34]);
			bg[E_SHOOT + 2 * i].fillOval(8, 4, 18, 24);
            int e_SHOOT2 = 161;
            bg[e_SHOOT2 + 2 * i].setColor(color[i][50]);
			bg[e_SHOOT2 + 2 * i].fillOval(6, 6, 20, 20);
			bg[e_SHOOT2 + 2 * i].setColor(WHITE);
			bg[E_SHOOT + 2 * i].setColor(WHITE);
			bg[E_SHOOT + 2 * i].fillRect(14, 6, 6, 20);
			bg[e_SHOOT2 + 2 * i].fillOval(10, 10, 12, 12);
			blur(E_SHOOT + 2 * i, 8 - i);
			blur(e_SHOOT2 + 2 * i, 8 - i);
		}
		
		for (i = 0; i < 10; i++) {
			bg[FOE + i].setColor(color[3][i * 4 + 8]);
			bg[FOE + i].fillOval(0, 0, 32, 32);
			blur(FOE + i, 4);
			bg[FOE + i].setColor(color[4][i * 4]);
			bg[FOE + i].fillOval(5, 5, 22, 22);
			bg[FOE + i].setColor(BLACK);
			bg[FOE + i].fillOval(11, 11, 10, 10);
			bg[FOE + i].setColor(WHITE);
			if (i % 2 == 0)
				bg[FOE + i].fillRect(14, 24, 4, 10);
			blur(FOE + i, 1);
			bg[FOE + i].drawOval(5, 5, 22, 22);
		}
		
		makeexp(EXPLODE_S, 40, 2);
		makeexp(SHIP_HIT, 34, 3);
		for (k = 0; k < 10; k++)
			makeexp(EXPLODE_FOE + k * 10, 1 + k * 5, 4);
	}

	@Override
    public void start() {
		new Thread(this).start();
	}

	@Override
    public void stop() {
		
		r = false;
	}

	@Override
    protected void processMouseMotionEvent(MouseEvent e) {
		if (gamestatus == GAMERUN) {
			dx = e.getX();
			dy = e.getY();
		}
		if (gamestatus == GAMESTART) {
			
			wavefrequency = 220;
			wave_multiplier = 2;
			foe_firerate_div = 52;
			bossmove = 4;
			chalenge = 170;
			if (e.getY() > 270) {
				
				wavefrequency = 160;
				wave_multiplier = 4;
				foe_firerate_div = 44;
				bossmove = 5;
				chalenge = 270;
			}
			if (e.getY() > 370) {
				
				wavefrequency = 140;
				wave_multiplier = 6;
				foe_firerate_div = 42;
				bossmove = 6;
				chalenge = 370;
			}
		}
	}

	@Override
    protected void processMouseEvent(MouseEvent e) {
		if (e.getID() == 504 || e.getID() == 505)
			return;
		
		if (e.getID() == 501)
			if (e.getButton() == 1)
				firestatus1 = SHOOT1;
			else
				firestatus2 = SHOOT2;
		if (e.getID() == 502) 
		{
            int NOFIRE = 0;
            if (e.getButton() == 1)
				firestatus1 = NOFIRE;
			firestatus2 = NOFIRE;
			if (gamestatus == GAMEPAUSE)
				gamestatus = GAMEINIT;
			if (gamestatus == GAMEWIN)
				gamestatus = GAMEPAUSE;
			if (gamestatus == GAMESTART)
				gamestatus = GAMEPAUSE;
		}
		
	}

	@Override
    public void run() { 
		while (r) {

            int BKBUFFER = 0;
            bg[BKBUFFER].setColor(BLACK);
			bg[BKBUFFER].fillRect(0, 0, 800, 600);
            long timer = System.nanoTime();
			frame++;
			
			if (dx > x + movespeed)
				x += movespeed;
			else if (dx < x - movespeed)
				x -= movespeed;
			else
				x = dx;
			if (dy > y + movespeed)
				y += movespeed;
			else if (dy < y - movespeed)
				y -= movespeed;
			else
				y = dy;
			if (y > 600)
				y = 600;
			if (y < 0)
				y = 0;
			if (x > 800)
				x = 800;
			if (x < 0)
				x = 0;

            int MAP1 = 2;
            clear(MAP1);
			bg[MAP1].drawImage(buf[MAP1T], 0, 6, this);
			for (i = 0; i < 12; i++)
				buf[MAP1].setRGB(rand.nextInt(800), rand.nextInt(1), 0x00FFFFFF | (rand.nextInt(255) << 24));
			clear(MAP1T);
			bg[MAP1T].drawImage(buf[MAP1], null, this);
			bg[BKBUFFER].drawImage(buf[MAP1], null, this);

            int baselife = 18;
			int addlife = 4;
			int HOLDFIRE = 1;
            final int GAMELOOSE = 3;
            switch (gamestatus) {
                case GAMESTART -> {
                    setCursor(null);
                    bg[BKBUFFER].setColor(color[1][40]);
                    bg[BKBUFFER].fillRect(100, chalenge, 600, 50);
                    bg[BKBUFFER].setColor(WHITE);
                    bg[BKBUFFER].drawRect(100, 170, 600, 50);
                    bg[BKBUFFER].drawRect(100, 270, 600, 50);
                    bg[BKBUFFER].drawRect(100, 370, 600, 50);
                    bg[BKBUFFER].drawString(textlvl[0], 300, 200);
                    bg[BKBUFFER].drawString(textlvl[1], 300, 300);
                    bg[BKBUFFER].drawString(textlvl[2], 300, 400);
                }
                case GAMEPAUSE -> {
                    if (level > 10) {
                        level = 1;
                        gamestatus = GAMESTART;
                    }
                    x = 350;
                    y = 400;
                    wave = baselife + addlife * level;
                    setCursor(null);
                    bg[BKBUFFER].setColor(WHITE);
                    bg[BKBUFFER].drawString("LEVEL       - GET READY !", 400, 400);
                    bg[BKBUFFER].drawString(String.valueOf(level), 440, 400);
                    bg[BKBUFFER].drawString("(Clik to continue!)", 400, 420);
                    bg[BKBUFFER].drawString("Expected threats: ", 200, 200);
                    if (text[level - 1].length() > 1) {
                        bg[BKBUFFER].setColor(color[4][10]);
                        bg[BKBUFFER].drawString(text[level - 1], 300, 500);
                        bg[BKBUFFER].drawRect(100, 470, 600, 50);
                    }
                    for (i = 0; i < level; i++) {
                        bg[BKBUFFER].drawImage(buf[FOE + i], 200 + 40 * i, 220, this);
                        bg[BKBUFFER].drawImage(buf[E_SHOOT + i], 200 + 40 * i, 260, this);

                    }
                }
                case GAMEINIT -> {
                    rand.setSeed(10);
                    if (level == 1)
                        score = 0;
                    int energyboost = 16;
                    emax = 80 + energyboost * level;
                    movespeed = 8;
                    shootdmg = 8;
                    regeneration_div = 8;
                    if (level > 1)
                        movespeed = 12;
                    if (level > 3)
                        shootdmg = 16;
                    if (level > 5)
                        regeneration_div = 5;
                    if (level > 7)
                        emax += 80;
                    energy = emax;
                    frame = 0;
                    wave = 0;
                    lastwave = wave_multiplier * level + 2;
                    for (i = 0; i < 128; i++) {
                        enemy[4][i] = 0;
                        enemy_fire[4][i] = 0;
                        explode[2][i] = 0;
                        firetable[_Y][i] = 0;
                    }
                    gamestatus = GAMERUN;
                    setCursor(getToolkit().createCustomCursor(buf[CUR], new Point(16, 8), ""));
                }
                case GAMERUN -> {
                    if (frame % wavefrequency == 0)
                        if (wave < lastwave) {
                            wave++;
                            for (i = 2; i < 14; i++)
                                if (wave != lastwave || i % 6 == 0)
                                    for (j = 0; j < 128; j++)
                                        if (enemy[4][j] == 0) {
                                            boss = (wave == lastwave);
                                            int nextship = (boss ? (level - (i < 7 && level != 1 ? 2 : 1)) : rand
                                                    .nextInt(level));

                                            enemy[0][j] = 50 * i;
                                            enemy[1][j] = 0;
                                            enemy[2][j] = (rand.nextBoolean() ? 1 : -1) * (1 + rand.nextInt(3));
                                            enemy[3][j] = rand.nextInt(3) + 1;
                                            enemy[4][j] = FOE + nextship;
                                            enemy[6][j] = (boss ? 1 : 0);
                                            int bossmul = 64;
                                            enemy[5][j] = (baselife + addlife * nextship) + (enemy[6][j] * bossmul * level);
                                            enemy[7][j] = enemy[5][j];
                                            break;
                                        }
                        } else {

                            for (i = 0; i < 128; i++)
                                if (enemy[4][i] != 0 || enemy_fire[4][i] != 0)
                                    break;
                            if (i == 128) {
                                gamestatus = GAMEWIN;
                                level++;
                            }
                        }
                    if (frame % regeneration_div == 0)
                        if (energy < emax)
                            energy++;
                    int firerate_div = 6;
                    if (firestatus1 == HOLDFIRE && frame % firerate_div == 0)
                        firestatus1 = SHOOT1;
                    for (i = 0; i < 128; i++)
                        if (enemy[4][i] != 0) {
                            boss = (enemy[6][i] == 1);
                            bg[BKBUFFER].drawImage(buf[enemy[4][i]], enemy[0][i] - (boss ? 64 : 16), enemy[1][i]
                                    - (boss ? 64 : 16), boss ? 128 : 32, boss ? 192 : 48, this);
                            if (enemy[1][i] > 150) {
                                if (boss) {
                                    if (enemy[0][i] > 750 && sens == 1)
                                        sens = -1;
                                    if (enemy[0][i] < 50 && sens == -1)
                                        sens = 1;
                                    enemy[2][i] = sens * bossmove;

                                } else if (enemy[4][i] % 2 == 1)
                                    enemy[1][i] += enemy[3][i];
                                enemy[0][i] += enemy[2][i];
                            } else
                                enemy[1][i]++;

                            if (enemy[1][i] > 600 || enemy[0][i] > 800 || enemy[0][i] < 0)
                                enemy[4][i] = 0;

                            if (boss) {

                                bg[BKBUFFER].setColor(color[2][55]);
                                bg[BKBUFFER].fillRect(enemy[0][i] - 64, enemy[1][i] - 64,
                                        (int) (enemy[5][i] * 128.0 / enemy[7][i]), 4);
                                bg[BKBUFFER].setColor(color[4][55]);
                                bg[BKBUFFER].drawRect(enemy[0][i] - 64, enemy[1][i] - 64, 128, 4);
                            }
                        }
                    if (firestatus2 == SHOOT2 && energy > emax / 4) {
                        bg[BKBUFFER].drawImage(buf[SHOOT2], x - 16, y - 616, this);
                        energy -= 1;
                        for (j = 0; j < 128; j++)
                            if (enemy[4][j] != 0) {
                                l = x - enemy[0][j];
                                i = x - (l / 2);
                                l *= l;
                                if (((l < 4096 && enemy[6][j] == 1) || l < 1024) && enemy[1][j] < y) {

                                    enemy[5][j] -= shootdmg / 4;
                                    hit();
                                    setexp(i, enemy[1][j], EXPLODE_S);
                                }
                            }

                    }
                    for (i = 0; i < 128; i++) {
                        if (firetable[_Y][i] > 0) {
                            bg[BKBUFFER].drawImage(buf[SHOOT1], firetable[_X][i] - 16, firetable[_Y][i] - 16, this);
                            for (j = 0; j < 128; j++)
                                if (enemy[4][j] != 0) {
                                    k = firetable[_Y][i] - enemy[1][j];
                                    l = firetable[_X][i] - enemy[0][j];
                                    l = l * l + k * k;
                                    if ((l < 4096 && enemy[6][j] == 1) || l < 1024) {
                                        enemy[5][j] -= shootdmg;
                                        hit();
                                        setexp(firetable[_X][i], firetable[_Y][i], EXPLODE_S);
                                        firetable[_Y][i] = 0;
                                        break;
                                    }
                                }
                            int shootspeed = 10;
                            firetable[_Y][i] -= shootspeed;

                        } else if (firestatus1 == SHOOT1) {
                            firetable[_X][i] = x;
                            firetable[_Y][i] = y;
                            firestatus1 = HOLDFIRE;
                        }
                    }
                    for (i = 0; i < 128; i++) {
                        boss = enemy[6][i] == 1;
                        l = foe_firerate_div - level / 2;
                        if (frame % l == 0 || frame % (l / 3) == 0 && boss)
                            if (enemy[4][i] != 0 && enemy[1][i] > 10 && (rand.nextInt(level + 5) > 4 || boss)) {
                                l = 2 + level / 4 + enemy[6][i];
                                if (chalenge > 200)
                                    l++;
                                if (chalenge > 300)
                                    l++;
                                int nextshoot = enemy[4][i] % 2;
                                for (j = 0; j < 128; j++)
                                    if (enemy_fire[4][j] == 0) {
                                        enemy_fire[0][j] = enemy[0][i];
                                        enemy_fire[1][j] = enemy[1][i];
                                        enemy_fire[2][j] = 0;
                                        enemy_fire[3][j] = l;
                                        if (nextshoot == 1) {

                                            d1 = x - enemy[0][i];
                                            d2 = y - enemy[1][i];
                                            k = (int) d1;
                                            d1 = Math.atan(d2 / d1);
                                            d2 = Math.sin(d1) * l;
                                            d1 = Math.cos(d1) * l;
                                            l = (k < 0 ? -1 : 1);


                                            enemy_fire[2][j] = l * d1;
                                            enemy_fire[3][j] = l * d2;
                                        }
                                        enemy_fire[4][j] = E_SHOOT + enemy[4][i] - FOE;
                                        int boss_dmg_div = 8;
                                        enemy_fire[5][j] = enemy[5][i] / (1 + enemy[6][i] * boss_dmg_div);
                                        break;
                                    }
                            }
                    }
                    for (i = 0; i < 128; i++)
                        if (enemy_fire[4][i] > 0) {
                            bg[BKBUFFER].drawImage(buf[(int) enemy_fire[4][i]], (int) enemy_fire[0][i] - 16,
                                    (int) enemy_fire[1][i] - 16, this);
                            enemy_fire[0][i] += enemy_fire[2][i];
                            enemy_fire[1][i] += enemy_fire[3][i];
                            k = (int) enemy_fire[_Y][i] - y;
                            l = (int) enemy_fire[_X][i] - x;
                            l = l * l + k * k;
                            if (enemy_fire[1][i] > 600 || enemy_fire[1][i] < 0 || enemy_fire[0][i] > 800
                                    || enemy_fire[0][i] < 0)
                                enemy_fire[4][i] = 0;
                            else if (l < 800) {
                                energy -= enemy_fire[5][i];
                                enemy_fire[4][i] = 0;
                                boss = false;
                                setexp(x, y, SHIP_HIT);
                            }
                        }
                    if (energy < 0) {
                        energy = 0;
                        gamestatus = GAMELOOSE;
                        boss = true;
                        setexp(x, y, EXPLODE_FOE);
                    }
                    bg[BKBUFFER].setColor(color[(energy < emax / 10 ? 1 : 0)][1]);
                    if (energy < emax / 4)
                        bg[BKBUFFER].fillRect(0, 0, 800, 600);
                }
                case GAMELOOSE -> {
                    for (i = 0; i < 128; i++) {
                        if (explode[3][i] != 0)
                            break;
                    }
                    if (i == 128) {
                        score /= 2;
                        gamestatus = GAMEPAUSE;
                    }
                }
                case GAMEWIN -> {
                    dx = 350;
                    dy = 400;
                    if (x == 350 && y == 400)
                        if (level > 10)
                            setexp(rand.nextInt(800), rand.nextInt(600), EXPLODE_FOE + rand.nextInt(11) * 10);
                        else
                            gamestatus = GAMEPAUSE;
                }
            }
			
			if (gamestatus != GAMELOOSE && gamestatus != GAMESTART)
				bg[BKBUFFER].drawImage(buf[SHIP_S + frame % (SHIP_E - SHIP_S)], x - 16, y - 16, this);
			
			for (i = 0; i < 128; i++) {
				if (explode[3][i] != 0) {
					boss = explode[4][i] == 1;
					bg[BKBUFFER].drawImage(buf[explode[3][i] + explode[2][i]], explode[_X][i] - (boss ? 128 : 32),
							explode[_Y][i] - (boss ? 128 : 32), boss ? 256 : 64, boss ? 256 : 64, this);
					if (frame % (boss ? 6 : 3) == 0 || explode[3][i] == EXPLODE_S || explode[3][i] == SHIP_HIT)
						explode[2][i]++;
					if (explode[2][i] > 9)
						explode[3][i] = 0;
				}
			}
			
			if (level > 10) {
				bg[BKBUFFER].setColor(color[2][40]);
				bg[BKBUFFER].fillRect(390, 380, 250, 48);
				bg[BKBUFFER].setColor(WHITE);
				bg[BKBUFFER].drawString("Congratulation! You Win!", 400, 400);
			}
			
			bg[BKBUFFER].setColor(color[1][40]);
			bg[BKBUFFER].fillRect(0, 0, 800, 16);
			bg[BKBUFFER].setColor(color[4][37 - chalenge / 10]);
			bg[BKBUFFER].drawString("Score :", 10, 13);
			bg[BKBUFFER].drawString(String.valueOf(score), 50, 13);
			bg[BKBUFFER].drawString("Level :", 730, 13);
			bg[BKBUFFER].drawString(String.valueOf(level), 770, 13);
			i = (energy * 400) / emax;
			bg[BKBUFFER].setColor(color[1][i / 18]);
			bg[BKBUFFER].fillRect(200, 580, i, 20);
			bg[BKBUFFER].drawRect(200, 580, 400, 20);
			
			g.drawImage(buf[BKBUFFER], 0, 0, this);
			if (chalenge > 300)
				timer += 12000000; 
			else
				timer += 16000000; 

			while (System.nanoTime() < timer)
				Thread.yield();
		}
	}

	
	private void hit() {
		if (enemy[5][j] < 0) {
			score += enemy[7][j] * chalenge / 100;
			boss = (enemy[6][j] == 1);
			setexp(enemy[0][j], enemy[1][j], EXPLODE_FOE + (enemy[4][j] - FOE) * 10);
			enemy[4][j] = 0;
		}
		boss = false;
	}

	
	private void blur(int spriteIndex, int number) {

		Kernel kernel = new Kernel(3, 3, BLUR);
		ConvolveOp cop = new ConvolveOp(kernel);
		for (j = 0; j < number; j++)
			buf[spriteIndex] = cop.filter(buf[spriteIndex], null);
		bg[spriteIndex] = (Graphics2D) buf[spriteIndex].getGraphics();
	}

	
	private void clear(int index) {
		bg[index].setComposite(AlphaComposite.Clear);
		bg[index].fillRect(0, 0, buf[index].getWidth(), buf[index].getHeight());
		bg[index].setComposite(AlphaComposite.SrcOver);
	}

	
	private void makeexp(int picindex, int colorindex, int size) {
		float[] lng = new float[300];
		for (i = picindex; i < 10 + picindex; i++) {
			for (j = 0; j < 300; j++) {
				if (lng[j] == 0)
					lng[j] = rand.nextFloat() * size;
				d1 = Math.cos(.063 * j) * lng[j] * (i - picindex);
				d2 = Math.sin(.063 * j) * lng[j] * (i - picindex);
				bg[i].setColor(color[2][rand.nextInt(10) + colorindex]);
				if (lng[j] < 1.1 && lng[j] > .6)
					bg[i].setColor(WHITE);
				bg[i].fillOval(31 + (int) d1, 31 + (int) d2, 5, 5);
			}
			blur(i, 2);
		}
	}

	
	private void setexp(int x, int y, int sprite) {
		for (k = 0; k < 128; k++)
			if (explode[3][k] == 0) {
				explode[3][k] = sprite;
				explode[_X][k] = x;
				explode[_Y][k] = y;
				explode[2][k] = 0;
				explode[4][k] = (boss ? 1 : 0);
				break;
			}
	}
}