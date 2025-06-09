package java4k.pitfall4k;

/*
 * Pitfall 4K
 * Copyright (C) 2012 meatfighter.com
 *
 * This file is part of Pitfall 4K.
 *
 * Pitfall 4K is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Pitfall 4K is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http:
 *
 */

import java4k.GamePanel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class a extends GamePanel {

	
	private final boolean[] a = new boolean[32768];

	@Override
	public void start() {
		enableEvents(8);
		new Thread(this).start();
	}

	@Override
    public void run() {

        final int BACKGROUND_0 = 0;
		final int BACKGROUND_1 = 1;
		final int BACKGROUND_2 = 2;

        final int SPRITE_DIGIT_1 = 1;
		final int SPRITE_DIGIT_2 = 2;
		final int SPRITE_DIGIT_3 = 3;
		final int SPRITE_DIGIT_4 = 4;
		final int SPRITE_DIGIT_5 = 5;
		final int SPRITE_DIGIT_6 = 6;
		final int SPRITE_DIGIT_7 = 7;
		final int SPRITE_DIGIT_8 = 8;
		final int SPRITE_DIGIT_9 = 9;
        final int SPRITE_LEAVES_0 = 17;
		final int SPRITE_LEAVES_1 = 18;
		final int SPRITE_LEAVES_2 = 19;
        final int SPRITE_GOLD_BAR_1 = 23;
        final int SPRITE_SILVER_BAR_1 = 25;
        final int SPRITE_RATTLESNAKE_1 = 33;
        final int SPRITE_COPYRIGHT_1 = 37;
		final int SPRITE_COPYRIGHT_2 = 38;
		final int SPRITE_COPYRIGHT_3 = 39;
		final int SPRITE_COPYRIGHT_4 = 40;
		final int SPRITE_COPYRIGHT_5 = 41;
        final int SPRITE_HARRY_RUNNING_1 = 46;
		final int SPRITE_HARRY_RUNNING_2 = 47;
		final int SPRITE_HARRY_RUNNING_3 = 48;

        final int SCENE_SHIFTING_QUICKSAND = 7;

		final int ITEM_1_ROLLING_LOG = 0;
        final int ITEM_1_STATIONARY_LOG = 4;
        final int ITEM_FIRE = 6;
		final int ITEM_RATTLESNAKE = 7;

		final int OBJECT_ARRAY_SIZE = 8;

		ArrayList<int[]> queue = new ArrayList<>();

        int[] object = new int[OBJECT_ARRAY_SIZE];
		queue.add(object);
        final int OBJECT_X = 0;
        object[OBJECT_X] = 116;
        final int OBJECT_Y = 1;
        object[OBJECT_Y] = 111;
        final int OBJECT_SPRITE_INDEX = 2;
        final int SPRITE_LOG = 35;
        object[OBJECT_SPRITE_INDEX] = SPRITE_LOG;
        final int OBJECT_SPRITE_DIRECTION = 3;
        final int LEFT = 1;
        object[OBJECT_SPRITE_DIRECTION] = LEFT;

        BufferedImage[][] sprites = new BufferedImage[50][2];
		BufferedImage image = new BufferedImage(152, 192, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = (Graphics2D) image.getGraphics();

        int i;
		int j;
		int k = 0;
		int x;
		int y;


        final String S = "  nnnn   nn  nn  nn  nn  nn  nn  nn  nn  nn  nn  nn  nn  " + " nnnn     nn     nnn      nn      nn      nn      nn      nn     nnn"
                + "n    nnnn   n   nn      nn      nn   nnnn   nn      nn      nnnnnn  " + " nnnn   n   nn      nn     nn      nn       nn  n   nn   nnnn      n"
                + "n     nnn    n nn   n  nn   nnnnnn     nn      nn      nn   nnnnnn  " + "nn      nn      nnnnn       nn      nn  n   nn  nnnnn    nnnn   nn  "
                + " n  nn      nnnnn   nn  nn  nn  nn  nn  nn   nnnn   nnnnnn  n    n  " + "    nn     nn     nn      nn      nn      nn     nnnn   nn  nn  nn  "
                + "nn   nnnn    nnnn   nn  nn  nn  nn   nnnn    nnnn   nn  nn  nn  nn  " + "nn  nn   nnnnn      nn  n   nn   nnnn             nn      nn        "
                + "              nn      nn              nn       nn       nn nnnnnnnnn" + "nnnnnnn     nn     nn     nn      nn      nn      nn   n  nn  nnn nn"
                + " nn nnnnnn   nnnn     nn      nn     nnnn   nnnnnn nn nn nnn  nn  n " + "  nn      nn      nn   b  bb  bb  bb  bb  bb  bb  bb  bb  bb  bb  bb"
                + "  bbb bb bb bbbbbb                         j       jjjj    jjjjjj  j" + "jjjjjj jjjjjjjj                        k       kkkk    kkkkkk  kkkkk"
                + "kk kkkkkkkkooooooo oooooo   oooo     oo    ooooooooooooooo o oooo   " + "  oo    ooooooo  oooo o   oo           oooooooooo  ooooo     oo     "
                + "  o                          mmm    mmmmm    mmm      c      ccc    " + "cc cc   c   c   c   c   c   c   cc cc    ccc                        "
                + "              m             m m m            m m                    " + " ccccc  cccccc ccccccc ccccccc cccccc  ccccc                      m "
                + "           m  m  m          m m m             m              ccccc  " + "cccccc ccccccc ccccccc cccccc  ccccc                              m "
                + "            m m m            m m                     lllll  llllll l" + "llllll lllllll llllll  lllll                      m            m  m "
                + " m          m m m             m              lllll  llllll lllllll l" + "llllll llllll  lllll           fff fff fff fff fff fff lllllll f fff"
                + " f f fff f f fff f lllllll fff fff fff fff fff fff lllllll f fff f f" + " fff f f fff f lllllll                                              "
                + "    mmm    mm mm   m   m   m  m  m mm   m  mmmmm m  mmmm     mmmm   " + " mm  mm m  m  m                                         mmm    mm mm"
                + "   m   m   m  m    m     m mm   mm mmmmm    mmmm     mmmm m  mm  m m" + "    m m          qq qq    qqq      a      qqq    qq qq   qq qq  qq  "
                + " qq qq qqqq qq   qq qqqq qq qq   qq qqq qqq qqq qqq  qqqqq          " + "                                            r       rr rrrrrrrr r r "
                + "r rr r r rrrrrrrrrr                                                 " + "       r       rrr     r rrr r   r rrr     r rr      rr      rrr r r"
                + " rrrrrrrrrr                                                f        " + " kk      kkk       k       kk      kk      k      k     kk     kkkkk"
                + " k qqqqq q kkkkk  kqqqqq  qkkkkkkk                  f        kk     " + " kkk       k       kk      kk      k      k     kk     kkkkk  kqqqqq"
                + "  qkkkkk  kqqqqq  qkkkkkkk             d       d       dd      dd   " + "  ddd     ddd     dddd   ddddd   eeeee   eeee     ee     bbbb   bbbb"
                + "bb bbb  bbbbb    bb                   aa     aaaa   aa aaa  aaaa a  " + "aaaaaa  aa aaa  aaaa a  aaaaaa  aa  aa  a aa a  a aa a  a aa a   a  "
                + "a     aa           nnnn    n  n    n       n    nnnn  n n nnnnn nnn " + "                                                                    "
                + "                                    nnn n n n n n n nnn nnn n     n " + "n   nnn            nnnn n     nnnn nn nn n n  nnnn n  nn n n  nn n n"
                + "n n            n            nn n nn n  n n  n  n nn              nn " + "       nnnn       n   n n n  nn n n nn  n nnn   n nn    n n         "
                + " n   n  nnn  n   n n nnn n n n n n n n n n n       n               n" + "nnnnnn n         nnn n n n   n n nnn n n   n n n nnn n            n "
                + "nnn   n n n   n nnn   n   n   n   n                                 " + "       nnn n  nn n nn nn n nnnnn n n nnnnn n  n         nnn nnn n n "
                + "  n nnn nnn n n n   nnn nnn                                         " + "                                          aa      hh      hh    h   "
                + "    s ss    ssss     ssss     sss     sss     ss     ooo    oooo    " + "oo o    o  o    oo o     o o     o o    oo o       o       o       o"
                + "       oo          aa      hh h    hh h    h  h    ss s    ssss    s" + "s      ss      ss      ss      ss      oo ooo  oooo oo oooo  o  oo  "
                + " oo                                                           aa    " + "  hh      hh      h       ss      ss      ss s    ssss    sss     ss"
                + "      ss      oo      oo      oo      oo      oo      oo      oo    " + "  ooo     o       oo              aa      hh      hh      h       ss"
                + "      ss     sss     sss s   sssss   ssss     ss      oo      oo    " + "  ooo     ooo     o o   ooo o   o   o   o    oo      o              "
                + "      aa      hh      hh      h       ss      ss      ss      ss    " + "  ss      sss     sss     oo      oo      ooo      ooo     o o   ooo"
                + "oo   o o     o o       o       oo             aa      hh      hh    " + "  h       ss      ss      ss      ss      sss     sss     ss      oo"
                + "      ooo     oooo    o oo   oo  o   oo o    o  o    o   o   o      " + "  o               aa      hh      hh      h       ss      ss     sss"
                + "     sss s   sssss   ssss     ss      oo      ooo    ooooo   oo oo  " + "oo   o  oo   o oo    ooo       o                          aa      hh"
                + "      hh      h       ss      ss s   sssss  sssss   s ss    s ss    " + "  ss      ooo     oooo oo oo o  ooo  o   oo  oo                     " + "                   ";
		final int LIGHT_GREEN = 0x5CBA5C;
		final int DARKEST_GREEN = 0x143C00;
		final int DARKEST_GRAY = 0x6F6F6F;
		final int DARK_YELLOW = 0xBBBB35;
		final int DARK_GREEN = 0x355F18;
		final int GRAY = 0xD6D6D6;
		final int WHITE = 0xECECEC;
		final int DARK_GRAY = 0x8E8E8E;
		final int BLACK = 0x000000;
		final int BLUE = 0x2D6D98;
		final int GREEN = 0x6E9C42;
		final int PINK = 0xE46F6F;
		final int YELLOW_GREEN = 0x86861D;
		final int RED = 0xA71A1A;
		final int ORANGE = 0xFCBC74;
		final int LIGHT_ORANGE = 0xECC860;
		final int YELLOW = 0xFCFC54;
		final int DARK_BROWN = 0x484800;
		final int BROWN = 0x69690F;
		int[] COLORS = {BROWN,
                DARK_BROWN,
                YELLOW,
                LIGHT_ORANGE,
                ORANGE,
                RED,
                YELLOW_GREEN,
                PINK,
                GREEN,
                BLUE,
                BLACK,
                DARK_GRAY,
                WHITE,
                GRAY,
                DARK_GREEN,
                DARK_YELLOW,
                DARKEST_GRAY,
                DARKEST_GREEN,
                LIGHT_GREEN,
        };
        for (i = 0; i < 50; i++) {
			j = (i < 17) ? 8 : (i < 21) ? 4 : (i < 42) ? 16 : 22;
			sprites[i][0] = new BufferedImage(8, j, BufferedImage.TYPE_INT_ARGB_PRE);
			sprites[i][1] = new BufferedImage(8, j, BufferedImage.TYPE_INT_ARGB_PRE);
			for (y = 0; y < j; y++) {
				for (x = 0; x < 8; x++, k++) {
                    int z = (S.charAt(k) == ' ') ? 0 : (0xFF000000 | COLORS[S.charAt(k) - 'a']);
                    sprites[i][0].setRGB(x, y, z);
					sprites[i][1].setRGB(7 - x, y, z);
				}
			}
		}

		long nextFrameStartTime = System.nanoTime();
        int vineY = 0;
        int vineX = 0;
        float vy = 0;
        float vx = 0;
        float vineAngle = 0;
        float harryVineRadius = 0;
        float harryVy = 0;
        final int UPPER_FLOOR_Y = 97;
        float harryY = UPPER_FLOOR_Y;
        boolean paused = false;
        boolean hintsEnabled = false;
        boolean harryLanded = true;
        boolean vine = false;
        boolean shiftingPit = false;
        boolean crocodiles = false;
        boolean pit = false;
        boolean resetScreen = false;
        boolean wallOnLeft = false;
        boolean harrySwinging = false;
        boolean harrySinking = false;
        boolean harryClimbedDown = false;
        boolean harryClimbing = false;
        boolean harryJumping = false;
        boolean pauseKeyReleased = true;
        boolean hintsKeyReleased = true;
        boolean jumpReleased = true;
        boolean attractMode = true;
        final int SPRITE_CROCODILE_0 = 30;
        int crocodileSprite = SPRITE_CROCODILE_0;
        int crocodileTimer = 0;
        int pitTimer = 0;
        int pitSprite = 0;
        int pitOffset = 0;
        int extraLives = 2;
        int vineTimer = 0;
        int timer = 0;
        int harryCrushed = 0;
        int harryRunTimer = 0;
        final int SPRITE_HARRY_STANDING = 44;
        int harrySprite = SPRITE_HARRY_STANDING;
        int harryOffsetY = 0;
        int harryDirection = 0;
        final int BACKGROUND_3 = 3;
        int background = BACKGROUND_3;
        final int SCENE_LADDER = 0;
        int scene = SCENE_LADDER;
        int screenIndex = 0;
        int screen = 196;
        int restartDelay = 0;
        int copyrightTimer = 0;
        int copyrightOffset = 0;
        int harryX = 8;
        int clockTicks = 0;
        int clockSeconds = 0;
        int clockMinutes = 20;
        int score = 2000;
        int treasures = 32;
        Graphics2D g2 = null;
        boolean[] collectedTreasures = new boolean[256];
        Color COLOR_DARK_YELLOW = new Color(DARK_YELLOW);
        Color COLOR_GRAY = new Color(GRAY);
        Color COLOR_GREEN = new Color(GREEN);
        Color COLOR_YELLOW_GREEN = new Color(YELLOW_GREEN);
        Color COLOR_DARK_GREEN = new Color(DARK_GREEN);
        Color COLOR_BLACK = new Color(0);
        Color COLOR_DARK_BROWN = new Color(DARK_BROWN);
        final int OBJECT_SPRITE_INDEX_2 = 6;
        final int OBJECT_TYPE = 5;
        final int OBJECT_ROLLING = 4;
        final int OBJECT_TYPE_SCORPION = 7;
        final int OBJECT_TYPE_DIAMOND_RING = 6;
        final int OBJECT_TYPE_GOLD_BRICK = 5;
        final int OBJECT_TYPE_SILVER_BRICK = 4;
        final int OBJECT_TYPE_MONEY_BAG = 3;
        final int OBJECT_TYPE_RATTLESNAKE = 2;
        final int OBJECT_TYPE_FIRE = 1;
        final int OBJECT_TYPE_LOG = 0;
        final int ITEM_3_STATIONARY_LOGS = 5;
        final int ITEM_3_ROLLING_LOGS = 3;
        final int ITEM_2_FAR_ROLLING_LOGS = 2;
        final int ITEM_2_CLOSE_ROLLING_LOGS = 1;
        final int SCENE_SHIFTING_TAR_WITH_VINE = 6;
        final int SCENE_SHIFTING_TAR_WITH_TREASURE = 5;
        final int SCENE_CROCODILES = 4;
        final int SCENE_QUICKSAND_WITH_VINE = 3;
        final int SCENE_TAR_WITH_VINE = 2;
        final int SCENE_LADDER_AND_HOLES = 1;
        final int SPRITE_HARRY_RUNNING_4 = 49;
        final int SPRITE_HARRY_RUNNING_0 = 45;
        final int SPRITE_HARRY_SWINGING = 43;
        final int SPRITE_HARRY_CLIMBING = 42;
        final int SPRITE_COPYRIGHT_0 = 36;
        final int SPRITE_FIRE = 34;
        final int SPRITE_RATTLESNAKE_0 = 32;
        final int SPRITE_CROCODILE_1 = 31;
        final int SPRITE_MONEY_BAG = 29;
        final int SPRITE_SCORPION_1 = 28;
        final int SPRITE_SCORPION_0 = 27;
        final int SPRITE_BRICK_WALL = 26;
        final int SPRITE_SILVER_BAR_0 = 24;
        final int SPRITE_GOLD_BAR_0 = 22;
        final int SPRITE_DIAMOND_RING = 21;
        final int SPRITE_LEAVES_3 = 20;
        final int SPRITE_TAR = 16;
        final int SPRITE_WATER = 15;
        final int SPRITE_TREE_BRANCHES = 14;
        final int SPRITE_ARROW_2 = 13;
        final int SPRITE_ARROW_1 = 12;
        final int SPRITE_ARROW_0 = 11;
        final int SPRITE_COLON = 10;
        final int SPRITE_DIGIT_0 = 0;
        final int VK_HINTS = 0x38;
        final int VK_PAUSE = 0x50;
        final int VK_JUMP = 0x42;
        final int VK_DOWN = 0x28;
        final int VK_UP = 0x26;
        final int VK_RIGHT = 0x27;
        final int VK_LEFT = 0x25;
        final int RIGHT = 0;
        final int LADDER_BOTTOM_Y = 152;
        final int LADDER_TOP_Y = 112;
        final int UPPER_FLOOR_LOWER_Y = 142;
        final int LOWER_FLOOR_Y = 152;
        final int JUMP_SPEED = -1;
        final float GRAVITY = 0.05f;
        while (true) {

			do {
				nextFrameStartTime += 16666667;

				

				if (!a[VK_JUMP]) {
					jumpReleased = true;
					if (!a[VK_LEFT] && !a[VK_RIGHT]) {
						harryClimbedDown = true;
					}
				}
				if (!a[VK_HINTS]) {
					hintsKeyReleased = true;
				}
				if (a[VK_HINTS] && hintsKeyReleased) {
					hintsKeyReleased = false;
					hintsEnabled = !hintsEnabled;
				}
				if (!a[VK_PAUSE]) {
					pauseKeyReleased = true;
				}
				if (a[VK_PAUSE] && pauseKeyReleased) {
					pauseKeyReleased = false;
					paused = !paused;
				}
				if (paused) {
					continue;
				}

				if (attractMode) {

					
					copyrightTimer++;
					if (copyrightTimer >= 120 && copyrightTimer <= 184) {
						copyrightOffset = (copyrightTimer - 120) >> 3;
					} else if (copyrightTimer > 274) {
						copyrightTimer = 0;
						copyrightOffset = 0;
					}

					if (restartDelay > 0) {
						
						
						

						restartDelay--;
					} else {
						boolean b = IntStream.of(VK_JUMP, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT).anyMatch(v -> a[v]);
						if (b) {


							attractMode = false;
							harryJumping = false;
							harryClimbing = false;
							harryClimbedDown = false;
							harrySinking = false;
							harrySwinging = false;
							wallOnLeft = false;
							resetScreen = false;
							pit = false;
							crocodiles = false;
							vine = false;
							harryLanded = true;
							treasures = 32;
							score = 2000;
							clockMinutes = 20;
							clockSeconds = 0;
							clockTicks = 0;
							harryX = 8;
							harryY = UPPER_FLOOR_Y;
							harryVy = 0;
							screen = 196;
							screenIndex = 0;
							scene = SCENE_LADDER;
							background = BACKGROUND_3;
							harryDirection = 0;
							harryOffsetY = 0;
							harryCrushed = 0;
							timer = 0;
							vineTimer = 0;
							pitTimer = 0;
							crocodileTimer = 0;
							extraLives = 2;
							harrySprite = SPRITE_HARRY_STANDING;
							collectedTreasures = new boolean[256];

							queue.clear();
							object = new int[OBJECT_ARRAY_SIZE];
							queue.add(object);
							object[OBJECT_X] = 116;
							object[OBJECT_Y] = 111;
							object[OBJECT_SPRITE_INDEX] = SPRITE_LOG;
							object[OBJECT_SPRITE_DIRECTION] = LEFT;
						}
					}

				} else if (restartDelay > 0) {

					

					if (--restartDelay == 0) {
						if (extraLives > 0) {
							extraLives--;
							harryX = 12;
							harryY = harryY > UPPER_FLOOR_Y ? 127 : 26;
							harryVy = 0;
							harryDirection = RIGHT;
							resetScreen = true;
							harryJumping = true;
							harryClimbing = false;
							harryCrushed = 0;
						} else {

							
							attractMode = true;
							restartDelay = 255;
						}
					}

				} else if (harrySinking) {

					harrySprite = SPRITE_HARRY_STANDING;

					
					harryY++;
					if (harryY > 115) {
						
						harrySinking = false;
						restartDelay = 60;
						harryY = 26;
					}

				} else {

					
					timer++;

					
					copyrightOffset = 8;

					
					if (--clockTicks < 0) {
						clockTicks = 59;
						if (--clockSeconds < 0) {
							clockSeconds = 59;
							if (--clockMinutes < 0) {

								
								clockMinutes = 0;
								clockSeconds = 0;
								attractMode = true;
								restartDelay = 255;
							}
						}
					}

					
					if (++crocodileTimer == 210) {
						crocodileTimer = 0;
						crocodileSprite = crocodileSprite != SPRITE_CROCODILE_0 ? SPRITE_CROCODILE_0 : SPRITE_CROCODILE_1;
					}

					
					if (vine) {
						vineAngle = 0.393f * (float) Math.sin(++vineTimer / 46f);
						vx = (float) Math.sin(vineAngle);
						vy = (float) Math.cos(vineAngle);
						if (!harrySwinging && harryJumping && harryLanded) {

							
							x = harryX + 4;
							y = (int) (harryY + 4);
							float B = 2 * (72 * vx + 24 * vy - vx * x - vy * y);
							float C = 5744 - 144 * x + x * x - 48 * y + y * y;
							float D = B * B - 4 * C;
							if (D >= 0) {
								harryVineRadius = (-B - (float) Math.sqrt(D)) / 2;
								harrySwinging = harryVineRadius <= 76;
							}
						}

						vineX = (int) (72 + 76 * vx);
						vineY = (int) (24 + 76 * vy);
					}

					
					if (++pitTimer == 320) {
						pitTimer = 0;
					}
					pitOffset = 0;
					if (shiftingPit) {
                        pitOffset = switch (pitTimer) {
                            case int i1 when i1 < 20 -> pitTimer >> 2;
                            case int i1 when i1 < 160 -> 5;
                            case int i1 when i1 < 180 -> 4 - ((pitTimer - 160) >> 2);
                            default -> pitOffset;
                        };
					}

					
					if (harrySwinging) {
						harrySprite = SPRITE_HARRY_SWINGING;
						harryX = (int) (69 + harryVineRadius * vx) - harryDirection;
						harryY = 21 + harryVineRadius * vy;

						if (jumpReleased && a[VK_JUMP]) {
							jumpReleased = false;
							harrySwinging = false;
							harryJumping = true;
							harryVy = JUMP_SPEED;
							harryLanded = false;
						}
					} else if (harryClimbing) {
						if (a[VK_DOWN]) {
							if (++harryRunTimer == 8) {
								harryRunTimer = 0;
								harryDirection ^= 1;
								harryY += 4;
								if (harryY >= LADDER_BOTTOM_Y) {
									harryClimbing = false;
									harryY = LOWER_FLOOR_Y;
									harryRunTimer = 0;
								}
								harryClimbedDown = true;
							}
						} else if (a[VK_UP]) {
							if (harryY > LADDER_TOP_Y && ++harryRunTimer == 8) {
								harryRunTimer = 0;
								harryDirection ^= 1;
								harryY -= 4;
							}
						}

						if (((jumpReleased && a[VK_JUMP]) || (harryClimbedDown && (a[VK_LEFT] || a[VK_RIGHT]))) && harryY == LADDER_TOP_Y) {
							jumpReleased = false;
							harryJumping = true;
							harryVy = JUMP_SPEED;
							harryClimbing = false;
							harryY = UPPER_FLOOR_Y;
						}
					} else {
						if (harryCrushed == 0) {
							if (a[VK_LEFT]) {
								harryDirection = LEFT;
								if ((timer & 1) == 0) {
									if ((harryX != 37 && harryX != 89) || harryY <= UPPER_FLOOR_Y || harryY >= UPPER_FLOOR_LOWER_Y) {
										harryX--;
									}
								}
								if (++harryRunTimer == 4) {
									harryRunTimer = 0;
									if (++harrySprite > SPRITE_HARRY_RUNNING_4) {
										harrySprite = SPRITE_HARRY_RUNNING_0;
									}
								}
							} else if (a[VK_RIGHT]) {
								harryDirection = RIGHT;
								if ((timer & 1) == 0) {
									if ((harryX != 47 && harryX != 99) || harryY <= UPPER_FLOOR_Y || harryY >= UPPER_FLOOR_LOWER_Y) {
										harryX++;
									}
								}
								if (++harryRunTimer == 4) {
									harryRunTimer = 0;
									if (++harrySprite > SPRITE_HARRY_RUNNING_4) {
										harrySprite = SPRITE_HARRY_RUNNING_0;
									}
								}
							} else {
								harrySprite = SPRITE_HARRY_STANDING;
							}
						}

						if (scene <= SCENE_LADDER_AND_HOLES && harryY > UPPER_FLOOR_Y) {
							if (wallOnLeft) {
                                switch (harryX) {
                                    case 4 -> {
                                        harrySprite = SPRITE_HARRY_STANDING;
                                        harryRunTimer = 0;
                                        harryX = 3;
                                    }
                                    case 15 -> {
                                        harrySprite = SPRITE_HARRY_STANDING;
                                        harryRunTimer = 0;
                                        harryX = 16;
                                    }
                                }
							} else {
                                switch (harryX) {
                                    case 122 -> {
                                        harrySprite = SPRITE_HARRY_STANDING;
                                        harryRunTimer = 0;
                                        harryX = 121;
                                    }
                                    case 133 -> {
                                        harrySprite = SPRITE_HARRY_STANDING;
                                        harryRunTimer = 0;
                                        harryX = 134;
                                    }
                                }
							}
						}

						if (harryJumping) {
							harrySprite = SPRITE_HARRY_RUNNING_4;
							i = (int) harryY;
							harryY += harryVy;
							harryVy += GRAVITY;
							if (i <= UPPER_FLOOR_Y && (int) harryY >= UPPER_FLOOR_Y) {
								harryJumping = false;
								harryLanded = true;
								harryY = UPPER_FLOOR_Y;
							} else if (i <= LOWER_FLOOR_Y && (int) harryY >= LOWER_FLOOR_Y) {
								harryJumping = false;
								harryLanded = true;
								harryY = LOWER_FLOOR_Y;
							}
						} else if (jumpReleased && a[VK_JUMP] && harryCrushed == 0) {
							jumpReleased = false;
							harryJumping = true;
							harryVy = JUMP_SPEED;
						}

						if (!harryJumping && scene <= SCENE_LADDER_AND_HOLES) {
							if (harryX >= 65 && harryX <= 71 && (harryY == UPPER_FLOOR_Y || a[VK_UP])) {
								
								harryClimbing = true;
								harryX = 68;
								harrySprite = SPRITE_HARRY_CLIMBING;
								harryRunTimer = 0;
								if (harryY == UPPER_FLOOR_Y) {
									harryY = LADDER_TOP_Y;
									harryClimbedDown = false;
								} else {
									harryY = LADDER_BOTTOM_Y;
									harryClimbedDown = true;
								}
							} else if (harryY == UPPER_FLOOR_Y && scene == SCENE_LADDER_AND_HOLES) {
								if ((harryX >= 37 && harryX <= 47) || (harryX >= 89 && harryX <= 99)) {
									harryJumping = true;
									harryY++;
									score -= 100; 
									if (score < 0) {
										score = 0;
									}
								}
							}
						}
					}

					
					harrySinking = harryY == UPPER_FLOOR_Y && pit && pitOffset < 5 && harryX >= 36 && harryX <= 100 && (image.getRGB(harryX, 119) & 0xFFFFFF) != DARK_YELLOW
							&& (image.getRGB(harryX + 7, 119) & 0xFFFFFF) != DARK_YELLOW;
					if (crocodiles) {
						for (i = 0; i < 3; i++) {
							if (harryX <= 56 + (i << 4) && harryX >= 48 + 6 * (crocodileSprite - SPRITE_CROCODILE_0) + (i << 4)) {
								harrySinking = false;
							}
						}
					}

					
					harryOffsetY = 0;
					if (harryCrushed > 0) {
						harryCrushed--;
					}
					for (i = queue.size() - 1; i >= 0; i--) {
						object = queue.get(i);
						j = object[OBJECT_TYPE];
						x = object[OBJECT_X];
						y = object[OBJECT_Y];

                        switch (j) {
                            case OBJECT_TYPE_LOG:


                                if (object[OBJECT_ROLLING] == 1) {
                                    object[OBJECT_Y] = (timer & 15) < 2 ? 112 : 111;
                                    if ((timer & 3) == 0) {
                                        object[OBJECT_SPRITE_DIRECTION] ^= 1;
                                    }
                                    if ((timer & 1) == 0 && --object[OBJECT_X] < -8) {
                                        object[OBJECT_X] = 152;
                                    }
                                }
                                break;
                            case OBJECT_TYPE_SCORPION:


                                if ((timer & 7) == 0) {
                                    if (object[OBJECT_SPRITE_DIRECTION] == LEFT) {
                                        if (--object[OBJECT_X] < harryX - 16) {
                                            object[OBJECT_SPRITE_DIRECTION] = RIGHT;
                                        }
                                    } else {
                                        if (++object[OBJECT_X] > harryX + 16) {
                                            object[OBJECT_SPRITE_DIRECTION] = LEFT;
                                        }
                                    }
                                    object[OBJECT_SPRITE_INDEX] = object[OBJECT_SPRITE_INDEX] != SPRITE_SCORPION_0 ? SPRITE_SCORPION_0 : SPRITE_SCORPION_1;
                                }
                                break;
                        }

						if ((timer & 1) == 0 && Math.random() < 0.5f) {


                            switch (j) {
                                case OBJECT_TYPE_FIRE -> object[OBJECT_SPRITE_DIRECTION] ^= 1;
                                case OBJECT_TYPE_RATTLESNAKE, OBJECT_TYPE_SILVER_BRICK, OBJECT_TYPE_GOLD_BRICK -> object[OBJECT_SPRITE_INDEX] = object[OBJECT_SPRITE_INDEX] != object[OBJECT_SPRITE_INDEX_2] ? object[OBJECT_SPRITE_INDEX_2] : object[OBJECT_SPRITE_INDEX_2] + 1;
                            }
						}

						if (!harrySwinging && harryX >= x - 4 && harryX <= x + 4 && harryY >= y - 16 + (j == OBJECT_TYPE_SCORPION ? 6 : 0) && harryY <= y + 1) {


                            switch (j) {
                                case OBJECT_TYPE_FIRE, OBJECT_TYPE_RATTLESNAKE, OBJECT_TYPE_SCORPION ->
                                        restartDelay = 60;
                                case int i1 when i1 >= OBJECT_TYPE_MONEY_BAG && i1 <= OBJECT_TYPE_DIAMOND_RING -> {

                                    queue.remove(i);
                                    collectedTreasures[screenIndex] = true;
                                    score += (j - OBJECT_TYPE_MONEY_BAG + 2) * 1000;
                                    if (--treasures == 0) {

                                        attractMode = true;
                                        restartDelay = 255;
                                    }
                                }
                                case OBJECT_TYPE_LOG -> {

                                    if (!harryClimbing) {
                                        harrySprite = SPRITE_HARRY_RUNNING_4;
                                        harryOffsetY = 5;
                                    }
                                    if (object[OBJECT_ROLLING] == 1) {
                                        harryCrushed = 2;
                                    }
                                    score--;
                                    if (score < 0) {
                                        score = 0;
                                    }
                                }
								default -> { }
                            }
						}
					}

					i = screen;
					if (harryX > 150) {
						
						for (i = harryY > UPPER_FLOOR_Y ? 3 : 1; i > 0; i--) {
							screen = 0xFF & ((screen << 1) | ((1 & (screen >> 3)) ^ (1 & (screen >> 4)) ^ (1 & (screen >> 5)) ^ (1 & (screen >> 7))));
							if (++screenIndex > 254) {
								screenIndex = 0;
							}
						}
						harryX = 0;
					} else if (harryX < -6) {
						
						for (i = harryY > UPPER_FLOOR_Y ? 3 : 1; i > 0; i--) {
							screen = 0xFF & ((screen >> 1) | ((1 & (screen >> 4)) ^ (1 & (screen >> 5)) ^ (1 & (screen >> 6)) ^ (1 & screen)) << 7);
							if (--screenIndex < 0) {
								screenIndex = 254;
							}
						}
						harryX = 140;
					}
					if (i != screen || resetScreen == true) {

						

						resetScreen = false;
						background = screen >> 6;
						scene = (screen >> 3) & 7;
						wallOnLeft = (screen & 128) == 0;
						queue.clear();

						
						crocodiles = scene == SCENE_CROCODILES;

						
						vine = scene == SCENE_TAR_WITH_VINE || scene == SCENE_QUICKSAND_WITH_VINE || scene == SCENE_SHIFTING_TAR_WITH_VINE || (scene == SCENE_CROCODILES && (screen & 2) == 2);

						
						pitSprite = scene == SCENE_TAR_WITH_VINE || scene == SCENE_SHIFTING_TAR_WITH_TREASURE || scene == SCENE_SHIFTING_TAR_WITH_VINE ? SPRITE_TAR : SPRITE_WATER;
						shiftingPit = scene > SCENE_CROCODILES;
						pitOffset = 0;
						if (shiftingPit) {
                            pitOffset = switch (pitTimer) {
                                case int i1 when i1 < 20 -> pitTimer >> 2;
                                case int i1 when i1 < 160 -> 5;
                                case int i1 when i1 < 180 -> 4 - ((pitTimer - 160) >> 2);
                                default -> pitOffset;
                            };
						}
						if (pit = (scene > SCENE_LADDER_AND_HOLES)) {

							

							object = new int[OBJECT_ARRAY_SIZE];
							queue.add(object);
							object[OBJECT_X] = 68;
							object[OBJECT_Y] = 158;
							object[OBJECT_SPRITE_INDEX] = SPRITE_SCORPION_0;
							object[OBJECT_TYPE] = OBJECT_TYPE_SCORPION;
							object[OBJECT_SPRITE_DIRECTION] = harryX == 0 ? LEFT : RIGHT;
						}

						if (scene == SCENE_SHIFTING_TAR_WITH_TREASURE) {

							if (!collectedTreasures[screenIndex]) {
								

								j = screen & 3;

								object = new int[OBJECT_ARRAY_SIZE];
								queue.add(object);
								object[OBJECT_X] = 116;
								object[OBJECT_Y] = 111;
								object[OBJECT_SPRITE_INDEX_2] = object[OBJECT_SPRITE_INDEX] = j == 0 ? SPRITE_MONEY_BAG : j == 1 ? SPRITE_SILVER_BAR_0 : j == 2 ? SPRITE_GOLD_BAR_0
										: SPRITE_DIAMOND_RING;
								object[OBJECT_TYPE] = j + OBJECT_TYPE_MONEY_BAG;
							}
						} else if (scene != SCENE_CROCODILES) {

							

							j = screen & 7;

							for (i = 0; i < ((j == ITEM_2_CLOSE_ROLLING_LOGS || j == ITEM_2_FAR_ROLLING_LOGS) ? 2 : ((j == ITEM_3_ROLLING_LOGS || j == ITEM_3_STATIONARY_LOGS) ? 3 : 1)); i++) {
								object = new int[OBJECT_ARRAY_SIZE];
								queue.add(object);
								object[OBJECT_X] = ((i << (j == ITEM_2_CLOSE_ROLLING_LOGS ? 4 : 5)) + 116) % 160;
								object[OBJECT_Y] = 111;
								object[OBJECT_SPRITE_INDEX] = object[OBJECT_SPRITE_INDEX_2] = j == 6 ? SPRITE_FIRE : j == 7 ? SPRITE_RATTLESNAKE_0 : SPRITE_LOG;
								object[OBJECT_TYPE] = j == 6 ? OBJECT_TYPE_FIRE : j == 7 ? OBJECT_TYPE_RATTLESNAKE : OBJECT_TYPE_LOG;
								object[OBJECT_ROLLING] = (j >> 2) ^ 1;
								object[OBJECT_SPRITE_DIRECTION] = j < 6 ? LEFT : RIGHT;
							}
						}
					}
				}

				

			} while (nextFrameStartTime < System.nanoTime());

			

			
			g.setColor(COLOR_BLACK);
			g.fillRect(0, 0, 152, 210);

			
			g.setColor(COLOR_GREEN);
			g.fillRect(0, 46, 152, 65);

			
			g.setColor(COLOR_DARK_BROWN);
			for (i = 0; i < 2; i++) {
				j = ((background + 1) << 3) + (i << 5);
				if (background == 3) {
					j -= 4;
				}
				g.fillRect(j, 59, 4, 52);
				g.fillRect(140 - j, 59, 4, 52);
				g.drawImage(sprites[SPRITE_TREE_BRANCHES][0], j - 2, 51, null);
				g.drawImage(sprites[SPRITE_TREE_BRANCHES][0], 138 - j, 51, null);
			}

			
			if (vine) {
				g.drawLine(72, 24, vineX, vineY);
			}

			
			g.setColor(COLOR_YELLOW_GREEN);
			g.fillRect(0, 174, 152, 6);
			g.fillRect(0, 127, 152, 15);

			
			g.setColor(COLOR_DARK_YELLOW);
			g.fillRect(0, 111, 152, 11);

			if (scene <= SCENE_LADDER_AND_HOLES) {
				
				g.setColor(COLOR_BLACK);
				if (scene == SCENE_LADDER_AND_HOLES) {
					g.fillRect(40, 116, 12, 26); 
					g.fillRect(92, 116, 12, 26); 
				}
				g.fillRect(68, 116, 8, 26); 

				
				g.setColor(COLOR_YELLOW_GREEN);
				for (i = 0; i < 11; i++) {
					g.fillRect(70, 130 + (i << 2), 4, 2);
				}

				
				g.drawImage(sprites[SPRITE_BRICK_WALL][0], wallOnLeft ? 10 : 128, 142, null);
				g.drawImage(sprites[SPRITE_BRICK_WALL][0], wallOnLeft ? 10 : 128, 158, null);
			}

			if (pit) {
				
				g.setClip(40, 111, 64, 8);
				g.drawImage(sprites[pitSprite][1], 40, 111 + pitOffset, 32, 8, null);
				g.drawImage(sprites[pitSprite][0], 72, 111 + pitOffset, 32, 8, null);

				if (crocodiles) {

					

					for (i = 0; i < 3; i++) {
						g.drawImage(sprites[crocodileSprite][0], 52 + (i << 4), 111, null);
					}
				}

				g.setClip(null);
			}

			
			g.drawImage(sprites[harrySprite][harryDirection], harryX, harryOffsetY + (int) harryY, null);

			if (scene > SCENE_LADDER_AND_HOLES) {
				
				g.setColor(COLOR_YELLOW_GREEN);
				g.fillRect(0, 127, 152, 15);
			}

			
			g.setColor(COLOR_DARK_YELLOW);
			g.fillRect(0, 122, 152, 5);

			if (pit) {
				
				g.fillRect(40, 119, 64, 8);

				g.setClip(40, 119, 64, 8);
				g.drawImage(sprites[pitSprite][1], 40, 127 - pitOffset, 32, -8, null);
				g.drawImage(sprites[pitSprite][0], 72, 127 - pitOffset, 32, -8, null);

				if (crocodiles) {

					

					for (i = 0; i < 3; i++) {
						g.drawImage(sprites[crocodileSprite][0], 52 + (i << 4), 111, null);
					}
				}

				g.setClip(null);
			}

			
			g.setColor(COLOR_DARK_GREEN);
			g.fillRect(0, 0, 152, 51);

			
			for (i = 0; i < 6; i++) {
				g.drawImage(sprites[SPRITE_LEAVES_3 - background][1 - (i & 1)], (i << 5) - 24, 51, 32, 8, null);
			}

			
			for (i = queue.size() - 1; i >= 0; i--) {
				object = queue.get(i);
				g.drawImage(sprites[object[OBJECT_SPRITE_INDEX]][object[OBJECT_SPRITE_DIRECTION]], object[OBJECT_X], object[OBJECT_Y], null);
			}

			
			j = score;
			x = 53;
			do {
				g.drawImage(sprites[SPRITE_DIGIT_0 + (j % 10)][0], x, 3, null);
				x -= 8;
				j /= 10;
			} while (j > 0);

			
			j = clockSeconds;
			x = 53;
			do {
				g.drawImage(sprites[SPRITE_DIGIT_0 + (j % 10)][0], x, 16, null);
				x -= 8;
				j /= 10;
			} while (x > 37);

			
			j = clockMinutes;
			x = 29;
			do {
				g.drawImage(sprites[SPRITE_DIGIT_0 + (j % 10)][0], x, 16, null);
				x -= 8;
				j /= 10;
			} while (j > 0);

			
			g.drawImage(sprites[SPRITE_COLON][0], 37, 16, null);

			
			g.setColor(COLOR_GRAY);
			if (extraLives > 0) {
				g.fillRect(13, 16, 1, 8);
			}
			if (extraLives > 1) {
				g.fillRect(15, 16, 1, 8);
			}

			
			g.setClip(13, 183, 48, 8);
			for (i = 0; i < 6; i++) {
				g.drawImage(sprites[SPRITE_COPYRIGHT_0 + i][0], 13 + (i << 3), 183 - copyrightOffset, null);
			}
			g.setClip(null);

			if (hintsEnabled) {

				
				

				i = (harryY < 124 && (screenIndex == 11 || screenIndex == 37 || screenIndex == 116 || screenIndex == 185 || screenIndex == 224 || screenIndex == 235)) ? SPRITE_ARROW_1
						: (harryY >= 124 && (screenIndex == 25 || screenIndex == 92 || screenIndex == 179 || screenIndex == 209 || screenIndex == 226 || screenIndex == 245)) ? SPRITE_ARROW_2
								: SPRITE_ARROW_0;
				j = harryY < 124
						&& ((screenIndex >= 0 && screenIndex < 11) || (treasures == 30 && screenIndex >= 226 && screenIndex <= 228) || (treasures == 13 && screenIndex >= 92 && screenIndex <= 95) || (treasures == 1
								&& screenIndex >= 25 && screenIndex <= 28)) ? RIGHT : LEFT;
				g.drawImage(sprites[i][j], 136, 11, null);
			}

			

			
			if (g2 == null) {
				g2 = (Graphics2D) getGraphics();
				requestFocus();
			} else {
				g2.drawImage(image, 0, 0, 608, 384, null);
			}

			
			while (nextFrameStartTime - System.nanoTime() > 0) {
				Thread.yield();
			}
		}
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
            final int VK_UP = 0x26;
            final int VK_RIGHT = 0x27;
            final int VK_LEFT = 0x25;
            k = k == VK_W ? VK_UP : k == VK_D ? VK_RIGHT : k == VK_A ? VK_LEFT : k == VK_S ? VK_DOWN : k;
            final int VK_HINTS = 0x38;
            final int VK_PAUSE = 0x50;
            final int VK_JUMP = 0x42;
            a[(k >= VK_LEFT && k <= VK_DOWN) || k == VK_HINTS || k == VK_PAUSE ? k : VK_JUMP] = keyEvent.getID() != 402;
		}
	}

	
	public static void main(String[] args) throws Throwable {
	  JFrame frame = new JFrame("Pitfall 4K");
	  frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
	  a applet = new a();
	  applet.setPreferredSize(new Dimension(608, 384));
	  frame.add(applet, BorderLayout.CENTER);
	  frame.setResizable(false);
	  frame.pack();
	  frame.setLocationRelativeTo(null);
	  frame.setVisible(true);
	  Thread.sleep(250);
	  applet.start();
	}
}
