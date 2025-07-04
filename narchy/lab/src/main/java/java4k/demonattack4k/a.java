package java4k.demonattack4k;

/*
 * Demon Attack 4K
 * Copyright (C) 2011 meatfighter.com
 *
 * This file is part of Demon Attack 4K.
 *
 * Demon Attack 4K is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Demon Attack 4K is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http:
 *
 */

import java4k.GamePanel;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;

public class a extends GamePanel {

	private static final int WIDTH = 640;
	private static final int HEIGHT = 420;

	static final int VK_LEFT = 0x25;
	static final int VK_RIGHT = 0x27;
	static final int VK_SHOOT = 0x42;
	static final int VK_PAUSE = 0x50;

	static final float DEMON_REFORM_SPEED = 0.035f;
	static final float BULLET_DELTA_ANGLE = 0.5f;

	static final int BULLET_X = 0;
	static final int BULLET_Y = 1;
	static final int BULLET_ANGLE = 2;
	static final int BULLET_DEMON_X = 3;
	static final int BULLET_OFFSET_X = 4;
	static final int BULLET_RADIUS = 5;

	static final int DEMON_X = 0;
	static final int DEMON_Y = 1;
	static final int DEMON_SPRITE = 2;
	static final int DEMON_ROW = 3;
	static final int DEMON_START_Y = 4;
	static final int DEMON_DELTA_Y = 5;
	static final int DEMON_PERCENT_Y = 6;
	static final int DEMON_DELTA_PERCENT_Y = 7;
	static final int DEMON_START_X = 8;
	static final int DEMON_DELTA_X = 9;
	static final int DEMON_PERCENT_X = 10;
	static final int DEMON_DELTA_PERCENT_X = 11;
	static final int DEMON_FORMING = 12;
	static final int DEMON_X2 = 13;
	static final int DEMON_DELTA_X2 = 14;
	static final int DEMON_HIDDEN = 15;
	static final int DEMON_EXPLODING = 16;
	static final int DEMON_EXPLODING_INDEX = 17;
	static final int DEMON_DELAY = 18;
	static final int DEMON_BIRD = 19;
	static final int DEMON_SHOOTER = 20;
	static final int DEMON_DIVING = 21;
	static final int DEMON_DIVING_Y = 22;

	static final int scoreColor = 0xFFDFB755;
	static final int playerColor = 0xFFB846A2;
	final Color playerBulletColor = new Color(0xD48CFC);
	final Color enemyBulletColor = new Color(0xFC9090);
	final Color[] extraLivesColors = new Color[100];
	final Color[] explodingFlashColors = new Color[16];

	int i;
	int j;
	int k;
	int x;
	int y;
	int z;

	int level;
	int score;
	int playerX = 76;
	int playerBulletX = 79;
	float playerBulletY = 175;
	float playerBulletSpeed = 4;
	boolean fireReleased = true;
	int advanceSprite;
	int extraLives = 2;
	int extraDemons = 5;
	int nextLevelDelay = 1;
	float shootTimer = 1;
	int shootCounter = 1;
	int playerExploding;
	int gameOver;
	boolean shooting;
	boolean shoot;
	boolean lasers;
	boolean tinyEnemies;
	float demonSpeed;
	float shootDelay;
	float bulletSpeed;
	boolean demoMode = true;
	int demoTargetX = 76;
	int demoShootDelay;
	int gameOverDelay;
	int noShootDelay;
	boolean paused;
	boolean pauseReleased = true;

	BufferedImage[] sprites;
	BufferedImage[][] enemySprites;
	BufferedImage offscreenImage;
	Graphics2D offscreenGraphics;

	int[] palette = new int[8];
	int[] pixels = new int[16];
	Random random = new Random();
	Color[][] floorColors = new Color[120][7];
	ArrayList<float[]> demons = new ArrayList<>();
	ArrayList<float[]> bullets = new ArrayList<>();
	ArrayList<float[]>[] demonsInRows = new ArrayList[4];

	
	private final boolean[] a = new boolean[32768];

	long nextFrameStartTime;

	@Override
	public Dimension getPreferredSize() {
		return new Dimension(WIDTH, HEIGHT);
	}

	public a() {
		super(true);

		sprites = new BufferedImage[32];
		enemySprites = new BufferedImage[256][12];
		offscreenImage = new BufferedImage(160, 210, 1);
		offscreenGraphics = (Graphics2D) offscreenImage.getGraphics();

		for (i = 0; i < 4; i++) {
			demonsInRows[i] = new ArrayList<>();
		}

		String S = "\u3e00\u2626\u2626\u2626\u3e26\u1c00\u1818\u1818\u1818\u1818\u3e00\u3232\u3c30\u3202\u3e32\u3e00\u3232\u1c30\u3230\u3e32\u3200\u3232\u3232\u7e32\u3030\u3e00\u0232\u303e\u3230\u3e32\u3e00\u3232\u3e02\u3232\u3e32\u3e00\u3232\u1830\u0c18\u0c0c\u3e00\u2626\u3e26\u3232\u3e32\u3e00\u3232\u3e32\u3230\u3e32\u02fc\u9112\u9191\u9291\ufc02\u00ff\u2ac5\uea2a\u2a2a\uff00\u00ff\ua598\ub585\u99a5\uff00\u403f\u924c\u8282\u4c92\u3f40\u1414\u1414\u3636\u7777\u6363\u6363\u8010\uf840\u62c4\u0032\u2000\u8e00\uc073\u0060\u4000\u980f\ue070\u00c0\u0000\uf0c0\u2428\"\u6000\u26f8\u4121\u0002\uf030\u2428\u8444\b\u0824\u2712\uf94d\u0040\u5000\u2410\u784c\u0090\u2000\u2050\ub840\u006c\u124c\u31e1\u3219\u00e4\u0e20\ue111\u1931\u00f2\u2600\u1109\uf1e1\u00f9\ue2fc\u2242\u0412\b\ue418\u24c4\u2414\u0004\u2810\uc4c4\u2224\u0020\u6000\u9090\u6090\u0000\u0000\ua0c0\u00c0\u0000\u0000\u8080\u0000\u0000\u9000\u2004\u0002\"\u1240\u2000\u1002\u0002\u0208\u0040\u0410\u0011\u2804\u1610\u0428\u0000\u0221\u0528\u2048\u0002\u4209\u0128\u2088\u0004\u3e08\u4163\u2241\u0000\u0000\u6b36\u4141\u0000\u6341\u1436\b\u0000\u0021\u0000\u0001\u0024\u0110\u0002\u4010\u0209\u0000\u2200\u0841\u0000\u0208\u4020\b\u0820\u0040\u0050\u4000\u8020\u0060\u0000\uc080\u0060";

		
		for (i = 0; i < 15; i++) {
			k = i < 14 ? 10 : 12;
			sprites[i] = new BufferedImage(8, k, 2);
			for (y = 0; y < k; y++) {
				j = S.charAt(i * 5 + (y >> 1));
				if ((y & 1) == 1) {
					j >>= 8;
				}
				for (x = 0; x < 8; x++) {
					pixels[15 - x] = pixels[x] = ((j & 1) == 0) ? 0 : i < 14 ? scoreColor : playerColor;
					j >>= 1;
				}
				sprites[i].setRGB(0, y, 8, 1, pixels, 0, 8);
			}
		}

		
		
		for (k = 0; k < 256; k++) {

			
			random.setSeed(18 + (k << 3));
			for (i = 0; i < 8; i++) {
				palette[i] = 0xFF;
			}
			for (i = 0; i < 3; i++) {
				float offset = 6.28f * random.nextFloat();
				float scale = 0.3f * (1 + random.nextFloat());
				for (j = 0; j < 8; j++) {
					palette[j] <<= 8;
					palette[j] |= (int) (180 + 75 * Math.sin(scale * j + offset));
				}
			}

			if (k == 2) {
				
				for (i = 0; i < 8; i++) {
					sprites[i + 15] = new BufferedImage(16, 48, 2);
					for (y = 0; y < 6; y++) {
						j = S.charAt(184 + i * 3 + (y >> 1));
						if ((y & 1) == 1) {
							j >>= 8;
						}
						for (x = 0; x < 8; x++) {
							pixels[15 - x] = pixels[x] = ((j & 1) == 0) ? 0 : palette[7 - y];
							j >>= 1;
						}
						for (j = 0; j < 2; j++) {
							sprites[i + 15].setRGB(0, (y << 3) + j, 16, 1, pixels, 0, 16);
						}
					}
				}
			}

			
			for (i = 0; i < 12; i++) {
				z = i < 9 ? 16 : 8;
				enemySprites[k][i] = new BufferedImage(z, 8, 2);
				for (y = 0; y < 8; y++) {
					j = S.charAt(i < 3 ? 76 + 12 * ((k >> 1) % 6) + (i << 2) + (y >> 1) : 148 + ((i - 3) << 2) + (y >> 1));
					if ((y & 1) == 1) {
						j >>= 8;
					}
					for (x = 0; x < 8; x++) {
						pixels[15 - x] = pixels[x] = ((j & 1) == 0) ? 0 : palette[7 - y];
						j >>= 1;
					}
					enemySprites[k][i].setRGB(0, y, z, 1, pixels, 0, z);
				}
			}
		}

		
		for (i = 0; i < 120; i++) {
			for (j = 0; j < 7; j++) {
				float w = (float) Math.sin(6 * Math.PI * i / 119);
				floorColors[i][j] = new Color(Color.HSBtoRGB((234 + j - (i << 1)) / 360f, 0.51f + 0.08f * j, 0.94f - 0.06f * j - 0.25f * w * w));
			}
		}

		
		for (i = 0; i < 100; i++) {
			extraLivesColors[i] = new Color(Color.HSBtoRGB(i / 100f, 0.46f, 0.65f + 0.35f * (float) Math.cos(0.594245f + 0.35f * i)));
		}

		
		for (i = 0; i < 16; i++) {
			float intensity = (float) Math.sin(0.196f * i);
			explodingFlashColors[i] = new Color(intensity, intensity, intensity);
		}

		nextFrameStartTime = System.nanoTime();

	}

	@Override
	public void paintComponent(Graphics g) {

		
		while (nextFrameStartTime - System.nanoTime() > 0) {
			Thread.yield();
		}

		do {
			nextFrameStartTime += 16666667;

			

			if (!a[VK_SHOOT]) {
				fireReleased = true;
			}

			if (!a[VK_PAUSE]) {
				pauseReleased = true;
			}

			if (pauseReleased && a[VK_PAUSE]) {
				pauseReleased = false;
				paused = !paused;
			}

			if (paused || level == 84) {
				continue;
			}

			if (demoMode || gameOver == 119) {
				
				if (fireReleased && a[VK_SHOOT]) {
					
					fireReleased = false;
					demoMode = false;
					level = -1;
					score = 0;
					playerX = 76;
					playerBulletX = 79;
					playerBulletY = 175;
					playerBulletSpeed = 4;
					extraLives = 2;
					extraDemons = 5;
					nextLevelDelay = 1;
					shootTimer = 1;
					shootCounter = 1;
					playerExploding = 0;
					gameOver = 0;
					shooting = false;
					shoot = false;
					shootDelay = 0;
					noShootDelay = 0;
					demons.clear();
					bullets.clear();
					for (i = 0; i < 4; i++) {
						demonsInRows[i].clear();
					}
				}
			}

			if (gameOver > 0) {
				if (gameOver < 119) {
					gameOver++;
					gameOverDelay = 0;
				} else if (gameOverDelay < 600) {
					gameOverDelay++;
				} else {
					
					demoMode = true;
					level = 0;
					playerX = 76;
					playerBulletX = 79;
					playerBulletY = 175;
					playerBulletSpeed = 4;
					extraLives = 2;
					extraDemons = 5;
					nextLevelDelay = 1;
					shootTimer = 1;
					shootCounter = 1;
					playerExploding = 0;
					gameOver = 0;
					shooting = false;
					shoot = false;
					shootDelay = 0;
					noShootDelay = 0;
					demons.clear();
					bullets.clear();
					for (i = 0; i < 4; i++) {
						demonsInRows[i].clear();
					}
				}
			} else if (playerExploding > 0) {
				if (--playerExploding == 0) {
					playerX = 76;
					playerBulletX = 79;
					playerBulletY = 175;

					
					
					noShootDelay = 30;

					if (!demoMode) {
						if (extraLives == 0) {
							gameOver = 1;
						} else {
							extraLives--;
						}
					}
				}
			} else {
				if (demoMode) {

					
					if (playerX < demoTargetX) {
						playerX++;
					} else if (playerX > demoTargetX) {
						playerX--;
					} else {
						demoTargetX = 25 + random.nextInt(110);
					}

					
					if (playerBulletY == 175) {
						playerBulletX = playerX + 3;
						if (--demoShootDelay < 0) {
							demoShootDelay = 15 + random.nextInt(30);
							playerBulletY--;
						}
					} else if (playerBulletY < 12) {
						playerBulletY = 175;
						playerBulletX = playerX + 3;
					} else {
						playerBulletY -= playerBulletSpeed;
					}
				} else {
					
					if (a[VK_LEFT] && playerX > 25) {
						playerX--;
					} else if (a[VK_RIGHT] && playerX < 135) {
						playerX++;
					}

					
					if (playerBulletY == 175) {
						playerBulletX = playerX + 3;
						if (fireReleased && a[VK_SHOOT]) {
							fireReleased = false;
							playerBulletY--;
						}
					} else if (playerBulletY < 12) {
						playerBulletY = 175;
						playerBulletX = playerX + 3;
					} else {
						playerBulletY -= playerBulletSpeed;
					}
				}
			}

			
			for (i = bullets.size() - 1; i >= 0; i--) {
				float[] bullet = bullets.get(i);
				bullet[BULLET_Y] += bulletSpeed;
				bullet[BULLET_ANGLE] += BULLET_DELTA_ANGLE;
				bullet[BULLET_X] = bullet[BULLET_DEMON_X] + bullet[BULLET_OFFSET_X] + ((Math.sin(bullet[BULLET_ANGLE]) > 0.5) ? bullet[BULLET_RADIUS] : 0);
				if (bullet[BULLET_Y] > 190) {
					bullets.remove(i);
				}

				
				if (playerExploding == 0 && bullet[BULLET_Y] <= 185 && bullet[BULLET_Y] >= 175 && bullet[BULLET_X] >= playerX && bullet[BULLET_X] <= playerX + 7) {
					bullets.remove(i);
					playerExploding = 63;
				}
			}

			
			shoot = false;
			if (noShootDelay > 0) {
				noShootDelay--;
			} else if (playerExploding == 0 && --shootTimer <= 0) {
				if (shooting) {
					shoot = true;
					if (--shootCounter == 0) {
						shooting = false;
						shootTimer = shootDelay;
					} else {
						shootTimer += (lasers ? 2 : 8) / bulletSpeed;
					}
				} else {
					shooting = true;
					shootCounter = (lasers ? 6 : 3) + (level < 10 ? (level >> 1) : 5) + random.nextInt(lasers ? 6 : 3);
					shootTimer = 1;
				}
			}

			if (nextLevelDelay > 0) {
				if (gameOver == 0 && --nextLevelDelay == 0) {
					if (extraLives < 6) {
						
						extraLives++;
					}
					level++;
					if (level == 84) {
						gameOver = 1;
					}
					demonSpeed = (level <= 11) ? 0.4f + 0.2f * level / 11 : 0.6f + 0.2f * (level - 11) / 72;
					shootDelay = level > 30 ? 30 : 60 - level;
					bulletSpeed = level > 11 ? 3f : 1.5f + 1.5f * level / 11;

					lasers = ((level >> 1) & 1) == 1;
					playerBulletSpeed = 4 + 4f * level / 11;
					if (playerBulletSpeed > 8) {
						playerBulletSpeed = 8;
					}

					extraDemons = 5;
					tinyEnemies = (level % 12) > 9;

					
					for (i = 0; i < 3; i++) {

						j = 16 + random.nextInt(96);

						
						float[] demon = new float[32];
						demons.add(demon);
						demon[DEMON_X] = -32;
						demon[DEMON_X2] = 160;
						demon[DEMON_Y] = 40 + (i << 5);
						demon[DEMON_ROW] = i;
						demon[DEMON_SPRITE] = i;
						demon[DEMON_PERCENT_Y] = 1;
						demon[DEMON_PERCENT_X] = 0;
						demon[DEMON_FORMING] = 1;
						demon[DEMON_DELTA_X] = (j + 32) * DEMON_REFORM_SPEED;
						demon[DEMON_DELTA_X2] = (j - 160) * DEMON_REFORM_SPEED;
						demon[DEMON_HIDDEN] = 30 + 50 * (2 - i);

						demonsInRows[i].add(demon);
					}
				}
			} else if (extraDemons == 0 && demons.size() == 0 && playerExploding == 0 && bullets.size() == 0) {
				nextLevelDelay = 99;
			} else {

				if (level > 3) {

					
					if (demonsInRows[2].size() == 1) {
						float[] demon = demonsInRows[2].get(0);
						if (demon[DEMON_BIRD] == 1 && demon[DEMON_DIVING] == 0 && demonsInRows[3].size() == 0) {
							demonsInRows[2].remove(0);
							demonsInRows[3].add(demon);
							demon[DEMON_DIVING_Y] = demon[DEMON_Y];
							demon[DEMON_DIVING] = 1;
							demon[DEMON_PERCENT_X] = 0;
							demon[DEMON_DELTA_X] = demon[DEMON_X] < playerX ? -demonSpeed : demonSpeed;
							demon[DEMON_ROW] = 3;
						}
					}

					
					for (i = 1; i >= 0; i--) {
						if (demonsInRows[i + 1].size() == 0) {
							for (j = demonsInRows[i].size() - 1; j >= 0; j--) {
								float[] demon = demonsInRows[i].remove(j);
								demon[DEMON_ROW]++;
								demonsInRows[i + 1].add(demon);
							}
						}
					}
				}

				
				if (extraDemons > 0) {
					for (i = 2; i >= 0; i--) {
						if (demonsInRows[i].size() == 0) {
							if (!demoMode) {
								extraDemons--;
							}

							
							j = 16 + random.nextInt(96);
							float[] demon = new float[32];
							demons.add(demon);
							demon[DEMON_X] = -32;
							demon[DEMON_X2] = 160;
							demon[DEMON_Y] = 40 + (i << 5);
							demon[DEMON_PERCENT_Y] = 1;
							demon[DEMON_PERCENT_X] = 0;
							demon[DEMON_FORMING] = 1;
							demon[DEMON_DELTA_X] = (j + 32) * DEMON_REFORM_SPEED;
							demon[DEMON_DELTA_X2] = (j - 160) * DEMON_REFORM_SPEED;
							demon[DEMON_HIDDEN] = 30;
							demon[DEMON_EXPLODING] = 0;
							demon[DEMON_ROW] = i;

							demonsInRows[i].add(demon);
						}
					}
				}

				
				for (i = demons.size() - 1; i >= 0; i--) {
					float[] demon = demons.get(i);

					
					if (advanceSprite == 0) {
						if (++demon[DEMON_SPRITE] == 4) {
							demon[DEMON_SPRITE] = 0;
						}
					}

					if (demon[DEMON_HIDDEN] > 0) {
						demon[DEMON_HIDDEN]--;
					} else if (demon[DEMON_FORMING] == 1) {
						
						demon[DEMON_PERCENT_X] += DEMON_REFORM_SPEED;
						demon[DEMON_X] += demon[DEMON_DELTA_X];
						demon[DEMON_X2] += demon[DEMON_DELTA_X2];
						if (demon[DEMON_PERCENT_X] >= 1) {
							demon[DEMON_FORMING] = 0;
							demon[DEMON_X] += 8;
						}
					} else if (demon[DEMON_EXPLODING] == 1) {
						if (++demon[DEMON_DELAY] == 5) {
							demon[DEMON_DELAY] = 0;
							if (++demon[DEMON_EXPLODING_INDEX] == 3) {
								if (level > 3 && demon[DEMON_BIRD] == 0) {

									
									demon[DEMON_BIRD] = 1;
									demon[DEMON_EXPLODING] = 0;
									demon[DEMON_SHOOTER] = 1;
									demon[DEMON_PERCENT_X] = 1;

									float[] bird = new float[32];
									demons.add(bird);
									bird[DEMON_X] = demon[DEMON_X] + 8;
									bird[DEMON_Y] = demon[DEMON_Y];
									bird[DEMON_ROW] = demon[DEMON_ROW];
									bird[DEMON_PERCENT_Y] = 1;
									bird[DEMON_PERCENT_X] = 1;
									bird[DEMON_BIRD] = 1;

									demonsInRows[(int) demon[DEMON_ROW]].add(bird);

								} else {
									demons.remove(i);
									demonsInRows[(int) demon[DEMON_ROW]].remove(demon);
								}
							}
						}
					} else {

						if (demon[DEMON_DIVING] == 1) {
							
							demon[DEMON_Y] = demon[DEMON_DIVING_Y] + 16 * (float) Math.sin(3.14f * demon[DEMON_PERCENT_X]);
							demon[DEMON_DIVING_Y] += 0.25f;
							demon[DEMON_PERCENT_X] += demonSpeed * 0.07f;
							demon[DEMON_X] += demon[DEMON_DELTA_X];
							if (demon[DEMON_PERCENT_X] >= 1) {
								demon[DEMON_PERCENT_X] = 0;
								if (demon[DEMON_DELTA_X] < 0) {
									if (demon[DEMON_X] < playerX - 8) {
										demon[DEMON_DELTA_X] = demonSpeed;
									}
								} else {
									if (demon[DEMON_X] > playerX + 8) {
										demon[DEMON_DELTA_X] = -demonSpeed;
									}
								}
							}

							
							if (playerExploding == 0 && demon[DEMON_Y] >= 171 && demon[DEMON_Y] <= 185 && demon[DEMON_X] + 7 >= playerX && demon[DEMON_X] <= playerX + 7) {
								demons.remove(i);
								demonsInRows[3].remove(0);
								playerExploding = 63;
							} else if (demon[DEMON_Y] > 182) {
								
								demons.remove(i);
								demonsInRows[3].remove(0);
							}

						} else {

							boolean shooter = demon[DEMON_ROW] == 2 && (demon[DEMON_BIRD] == 0 || demon[DEMON_SHOOTER] == 1) && demonsInRows[3].size() == 0;

							
							if (shooting && shooter) {

								if (shoot) {
									
									
									if (lasers || random.nextBoolean()) {
										float[] bullet = new float[32];
										bullets.add(bullet);
										bullet[BULLET_DEMON_X] = demon[DEMON_X];
										bullet[BULLET_X] = demon[DEMON_X] + (bullet[BULLET_OFFSET_X] = (lasers || demon[DEMON_BIRD] == 0 ? 4 : 0) + (lasers ? 0 : random.nextInt(3)));
										bullet[BULLET_Y] = demon[DEMON_Y] + 8;
										bullet[BULLET_RADIUS] = demon[DEMON_BIRD] == 1 ? 1 : lasers ? 0 : 1;

										if (demon[DEMON_BIRD] == 0 || !lasers) {
											float[] bullet2 = new float[32];
											bullets.add(bullet2);
											bullet2[BULLET_DEMON_X] = demon[DEMON_X];
											bullet2[BULLET_X] = demon[DEMON_X] + (bullet2[BULLET_OFFSET_X] = (demon[DEMON_BIRD] == 1 ? 0 : 4) + (lasers ? 3 : random.nextInt(3)) + 4);
											bullet2[BULLET_Y] = demon[DEMON_Y] + 8;
											bullet2[BULLET_RADIUS] = lasers ? 0 : 1;
										}
									} else {
										float[] bullet = new float[32];
										bullets.add(bullet);
										bullet[BULLET_DEMON_X] = demon[DEMON_X];
										bullet[BULLET_X] = demon[DEMON_X] + (bullet[BULLET_OFFSET_X] = (demon[DEMON_BIRD] == 1 ? 0 : 4) + random.nextInt(8));
										bullet[BULLET_Y] = demon[DEMON_Y] + 8;
										bullet[BULLET_RADIUS] = lasers ? 0 : 1;
									}
								}

							} else {

								
								if (demon[DEMON_PERCENT_Y] >= 1) {
									demon[DEMON_PERCENT_Y] = 0;
									demon[DEMON_DELTA_PERCENT_Y] = 1.57f / (30 + random.nextInt(15));
									demon[DEMON_START_Y] = demon[DEMON_Y];
									demon[DEMON_DELTA_Y] = 40 + ((int) demon[DEMON_ROW] << 5) + random.nextInt(demon[DEMON_ROW] == 2 ? 14 : 30) - demon[DEMON_Y];
									float dy = demon[DEMON_DELTA_Y] < 0 ? -demon[DEMON_DELTA_Y] : demon[DEMON_DELTA_Y];
									if (dy < 1) {
										dy = 1;
									}
									demon[DEMON_DELTA_PERCENT_Y] = (0.9f + 0.1f * random.nextFloat()) * demonSpeed * 1.57f / dy;
								} else {
									demon[DEMON_PERCENT_Y] += demon[DEMON_DELTA_PERCENT_Y];
									demon[DEMON_Y] = demon[DEMON_START_Y] + demon[DEMON_DELTA_Y] * (float) Math.sin(demon[DEMON_PERCENT_Y]);
								}

								
								if (demon[DEMON_PERCENT_X] >= 1) {
									demon[DEMON_PERCENT_X] = 0;
									demon[DEMON_START_X] = demon[DEMON_X];

									if (random.nextInt(11) == 1) {
										x = random.nextInt(144);
									} else {
										if (demon[DEMON_ROW] < 2 || (demon[DEMON_BIRD] == 1 && demon[DEMON_SHOOTER] == 0)) {
											
											y = playerX + 8 + random.nextInt(136 - playerX);
											z = random.nextInt(playerX - 8);
											x = (demon[DEMON_X] + 4 < playerX) ? (playerX < 57) ? y : z : (playerX > 103) ? z : y;
										} else {
											
											x = playerX - 28 + random.nextInt(48);
										}
									}

									demon[DEMON_DELTA_X] = x - demon[DEMON_X];
									float dx = demon[DEMON_DELTA_X] < 0 ? -demon[DEMON_DELTA_X] : demon[DEMON_DELTA_X];
									if (dx < 1) {
										dx = 1;
									}
									demon[DEMON_DELTA_PERCENT_X] = (0.9f + 0.1f * random.nextFloat()) * demonSpeed * 1.57f / dx;
								} else {
									demon[DEMON_PERCENT_X] += demon[DEMON_DELTA_PERCENT_X];
									demon[DEMON_X] = demon[DEMON_START_X] + demon[DEMON_DELTA_X] * (float) Math.sin(demon[DEMON_PERCENT_X]);
									if (shooter && level > 7) {
										
										for (j = bullets.size() - 1; j >= 0; j--) {
											bullets.get(j)[BULLET_DEMON_X] = demon[DEMON_X];
										}
									}
								}
							}
						}

						
						if (playerBulletY < 175 && playerBulletY <= demon[DEMON_Y] + 7 && playerBulletY + 7 >= demon[DEMON_Y]
								&& playerBulletX >= demon[DEMON_X] + (demon[DEMON_BIRD] == 0 && tinyEnemies ? 4 : 0)
								&& playerBulletX <= demon[DEMON_X] + (demon[DEMON_BIRD] == 1 ? 7 : (tinyEnemies ? 11 : 15))) {
							demon[DEMON_EXPLODING] = 1;
							demon[DEMON_EXPLODING_INDEX] = 0;
							demon[DEMON_DELAY] = 0;
							playerBulletY = 175;
							playerBulletX = playerX + 3;

							if (!demoMode) {
								
								j = 10 + 5 * (level >> 1);
								j = (j > 35) ? 35 : j;
								j <<= (int) (demon[DEMON_BIRD] + demon[DEMON_DIVING]);
								score += j;
							}
						}
					}
				}

				
				if (advanceSprite == 0) {
					advanceSprite = 8;
				} else {
					advanceSprite--;
				}
			}

			

		} while (nextFrameStartTime < System.nanoTime());

		

		
		Color backgroundColor = playerExploding > 47 ? explodingFlashColors[63 - playerExploding] : Color.BLACK;
		offscreenGraphics.setColor(backgroundColor);
		offscreenGraphics.fillRect(0, 20, 160, 168);

		if (gameOver == 0) {

			
			offscreenGraphics.setColor(enemyBulletColor);
			for (i = bullets.size() - 1; i >= 0; i--) {
				float[] bullet = bullets.get(i);
				offscreenGraphics.fillRect((int) bullet[BULLET_X], (int) bullet[BULLET_Y], 1, 4);
			}

			
			for (i = demons.size() - 1; i >= 0; i--) {
				float[] demon = demons.get(i);
				j = (demon[DEMON_SPRITE] == 3) ? 1 : (int) demon[DEMON_SPRITE];
				if (demon[DEMON_EXPLODING] == 1) {
					if (demon[DEMON_BIRD] == 1) {
						offscreenGraphics.drawImage(enemySprites[level][6 + (int) demon[DEMON_EXPLODING_INDEX]], (int) demon[DEMON_X], (int) demon[DEMON_Y], (int) demon[DEMON_X] + 8,
								(int) demon[DEMON_Y] + 8, 0, 0, 8, 8, null);
					} else {
						offscreenGraphics.drawImage(enemySprites[level][(level > 3 ? 6 : 3) + (int) demon[DEMON_EXPLODING_INDEX]], (int) demon[DEMON_X], (int) demon[DEMON_Y], null);
					}
				} else if (demon[DEMON_FORMING] == 1) {
					offscreenGraphics.drawImage(enemySprites[level][j + 3], (int) demon[DEMON_X], (int) demon[DEMON_Y], (int) demon[DEMON_X] + 32, (int) demon[DEMON_Y] + 8, 0, 0, 8, 8, null);
					offscreenGraphics.drawImage(enemySprites[level][j + 3], (int) demon[DEMON_X2], (int) demon[DEMON_Y], (int) demon[DEMON_X2] + 32, (int) demon[DEMON_Y] + 8, 7, 0, 15, 8, null);
				} else {
					offscreenGraphics.drawImage(enemySprites[level][demon[DEMON_BIRD] == 1 ? 9 + j : j], (int) demon[DEMON_X], (int) demon[DEMON_Y], null);
				}
			}

			if (playerExploding > 0) {
				
				offscreenGraphics.drawImage(sprites[(playerExploding >> 3) + 15], playerX - 4, 153, null);
			} else {
				
				offscreenGraphics.setColor(playerBulletColor);
				offscreenGraphics.fillRect(playerBulletX, (int) playerBulletY, 1, 8);

				
				offscreenGraphics.drawImage(sprites[14], playerX, 174, null);
			}
		}

		
		offscreenGraphics.setColor(backgroundColor);
		offscreenGraphics.fillRect(0, 0, 160, 20);
		if (demoMode && score == 0) {
			
			for (i = 0; i < 4; i++) {
				offscreenGraphics.drawImage(sprites[10 + i], 63 + (i << 3), 6, null);
			}
		} else {
			
			j = score;
			x = 95;
			do {
				offscreenGraphics.drawImage(sprites[j % 10], x, 6, null);
				x -= 8;
				j /= 10;
			} while (j > 0);
		}

		if (level < 84) {
			
			for (i = 0; i < 7; i++) {
				offscreenGraphics.setColor(floorColors[gameOver][i]);
				offscreenGraphics.fillRect(0, 188 + i, 160, 1);
			}
			offscreenGraphics.fillRect(0, 195, 160, 15);

			if (!demoMode) {
				
				offscreenGraphics.setColor(extraLivesColors[nextLevelDelay]);
				for (i = 0; i < extraLives; i++) {
					offscreenGraphics.fillRect(17 + (i << 3), 190, 1, 3);
					offscreenGraphics.fillRect(18 + (i << 3), 188, 1, 3);
					offscreenGraphics.fillRect(19 + (i << 3), 190, 1, 3);
				}
			}
		} else {
			
			offscreenGraphics.setColor(backgroundColor);
			offscreenGraphics.fillRect(0, 20, 160, 190);
		}

		

		
		g.drawImage(offscreenImage, 0, 0, 640, 420, null);

	}

	@Override
	public void processAWTEvent(AWTEvent e) {
		if (e instanceof KeyEvent keyEvent) {

            int k = keyEvent.getKeyCode();
			if (k > 0) {
                final int VK_D = 0x44;
                final int VK_A = 0x41;
                final int VK_RIGHT = 0x27;
                final int VK_LEFT = 0x25;
                switch (k) {
                    case VK_D -> k = VK_RIGHT;
                    case VK_A -> k = VK_LEFT;
                }
                final int VK_P = 0x50;
                final int VK_SHOOT = 0x42;
                a[(k == VK_LEFT || k == VK_RIGHT || k == VK_P) ? k : VK_SHOOT] = keyEvent.getID() != 402;
			}
		}
	}

	
	/*public static void main(String[] args) throws Throwable {
	  javax.swing.JFrame frame = new javax.swing.JFrame("Demon Attack 4K");
	  frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
	  a applet = new a();
	  applet.setPreferredSize(new java.awt.Dimension(640, 420));
	  frame.addAt(applet, java.awt.BorderLayout.CENTER);
	  frame.setResizable(false);
	  frame.pack();
	  frame.setLocationRelativeTo(null);
	  frame.setVisible(true);
	  Thread.sleep(250);
	  applet.start();
	}*/
}