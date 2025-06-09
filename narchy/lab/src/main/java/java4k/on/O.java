package java4k.on;

import java.applet.Applet;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * "On!" - 4k shooter, elements from Raiden II.
 *
 * Functionality remaining:
 * - shoot logic for each unit
 * - damage done by each weapon
 * - explosions for hit collisions
 * - scoring based on events.
 * - pick-ups / power ups, and how they are spawned.
 * - player weapon upgrades + shooting logic
 * - Level / enemy spawn pattern / enemy flight pattern coding.
 *      * part of flight pattern coded with ship AI value.
 *
 * Optional functionality that may not be in the final game:
 * - unit animation
 * - music
 * - actual interactive background
 *
 *
 *
 *
 * @author Matt Albrecht
 * @version Jan 14, 2009
 * @since Sep 20, 2012
 */
public class O extends Applet implements Runnable {

	private final boolean[] keys = new boolean[32767];

	@Override
    public void start() {
		new Thread(this).start();
	}

	@Override
    public void run() {
		
		while (!isActive()) {
			Thread.yield();
		}

		int[] STATIC = new int[STATIC_DATA_LENGTH];

		{

			for (int i = 0; i < 726; i++) {
				int j = STATIC_DATA.charAt(i);
				STATIC[i] = (0xfffffffe * (j & 0x8000)) | j;
			}
		}

		
		BufferedImage image = new BufferedImage(RENDER_WIDTH, RENDER_HEIGHT, BufferedImage.TYPE_INT_RGB);
		Graphics ogr = image.getGraphics();

		Random random = new Random();
		int[] display = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();

		long nextFrame = System.nanoTime();

		int[] units = new int[UNIT_SIZE];
		for (int pp = STAR_INDEX; pp < STAR_INDEX_FINAL; pp += UNIT_LEN) {
			
			units[pp + UNIT_IDX_X] = random.nextInt(UNIT_MAX_X);
			units[pp + UNIT_IDX_Y] = random.nextInt(UNIT_MAX_Y);
			units[pp + UNIT_IDX_VX] = 0;
			units[pp + UNIT_IDX_VY] = random.nextInt(STAR_VELOCITY_RANDOM) + STAR_VELOCITY_MIN;
			if (units[pp + UNIT_IDX_VY] == 0) {
				units[pp + UNIT_IDX_VY] = 1;
			}
			
			
			units[pp + UNIT_IDX_HP] = 1;
			units[pp + UNIT_IDX_SCALE_Y] = units[pp + UNIT_IDX_VY] / (STAR_VELOCITY_RANDOM / 32);
			units[pp + UNIT_IDX_SHIP_TEMPLATE] = STAR_SHIP_TEMPLATE;
		}

		int lives = 0;
		int highScore = 0;
		startGame: while (true) {
			
			
			int score = 0;

			int powerupLaser = LASER_START;
			int powerupPlasma = PLASMA_START;
			boolean firingLaser = true;

			int viewStartX = 0;

			
			
			int levelSwarmPos = NORMAL_SWARM_START_POS;
			int swarmShipsRemaining = 10; 
			int nextShipFrameCount = FIRST_SWARM_DELAY;
			int swarmShipIndex = 0;
			int levelDifficulty = INITIAL_DIFFICULTY;
			int wavesUntilBoss = BOSS_WAVE_FREQUENCY;

			
			
			
			int bossUnitIndex = 0;
			int nextBossSwarmIndex = SWARM_BOSS1_POS;
			int nextLevelSwarmIndex = 0;

			startLevel: while (true) {
				
				units[PLAYER_UNIT_IDX_HP] = PLAYER_START_HP;
				units[PLAYER_UNIT_IDX_X] = PLAYER_START_X;
				units[PLAYER_UNIT_IDX_Y] = PLAYER_START_Y;

				int[] plasmaHittingUnitIndex = new int[PLASMA_MAX_HITS_COUNT];

				
				

				Graphics sg = getGraphics();
				loopGame: while (true) {


					for (int i = 0; i < RENDER_PIXEL_COUNT; i++) {
						display[i] = 0;
					}

					
					
					
					viewStartX = (units[PLAYER_UNIT_IDX_X] >> UNIT_POSITION_SHIFT) / 2;
					if (viewStartX < 0) {
						
						
						
						viewStartX = 0;
					}
					int viewEndX = viewStartX + RENDER_WIDTH;

					if (lives > 0) {

						if (keys[KEY_CHANGE_WEAPON]) {
							firingLaser = !firingLaser;
							keys[KEY_CHANGE_WEAPON] = false;

							
							for (int i = 1; i < PLASMA_MAX_HITS_COUNT; i++) {
								plasmaHittingUnitIndex[i] = 0;
							}
						}

						
						

						
						
						
						STATIC[HEALTHBAR_SCALE_IDX] = units[PLAYER_UNIT_IDX_HP];
						units[PLAYER_UNIT_IDX_SHIP_TEMPLATE] = PLAYER_SHIP_TEMPLATE;

						
						
						if (this.keys[KEY_LEFT]) {
							if (units[PLAYER_UNIT_IDX_VX] > PLAYER_MIN_VX) {
								units[PLAYER_UNIT_IDX_VX] -= PLAYER_VX_ADJUST;
							}
						} else if (this.keys[KEY_RIGHT]) {
							if (units[PLAYER_UNIT_IDX_VX] < PLAYER_MAX_VX) {
								units[PLAYER_UNIT_IDX_VX] += PLAYER_VX_ADJUST;
							}
						} else {
							/*
							    
							    units[PLAYER_UNIT_IDX_VX] = 0;
							*/

							
							if (units[PLAYER_UNIT_IDX_VX] < 0) {
								units[PLAYER_UNIT_IDX_VX] += PLAYER_KEYRELEASE_VX_ADJUST;
							} else if (units[PLAYER_UNIT_IDX_VX] > 0) {
								units[PLAYER_UNIT_IDX_VX] -= PLAYER_KEYRELEASE_VX_ADJUST;
							}
							if (units[PLAYER_UNIT_IDX_VX] < PLAYER_KEYRELEASE_VX_ADJUST && units[PLAYER_UNIT_IDX_VX] > 0 - PLAYER_KEYRELEASE_VX_ADJUST) {
								units[PLAYER_UNIT_IDX_VX] = 0;
							}
						}
						if (units[PLAYER_UNIT_IDX_X] < PLAYER_MIN_X && units[PLAYER_UNIT_IDX_VX] < 0) {
							units[PLAYER_UNIT_IDX_X] = PLAYER_MIN_X_ADJ;
							units[PLAYER_UNIT_IDX_VX] = 0;
							this.keys[KEY_LEFT] = false;
						}
						if (units[PLAYER_UNIT_IDX_X] >= PLAYER_MAX_X && units[PLAYER_UNIT_IDX_VX] > 0) {
							units[PLAYER_UNIT_IDX_X] = PLAYER_MAX_X_ADJ;
							units[PLAYER_UNIT_IDX_VX] = 0;
							this.keys[KEY_RIGHT] = false;
						}

						if (this.keys[KEY_UP]) {
							if (units[PLAYER_UNIT_IDX_VY] > PLAYER_MIN_VY) {
								units[PLAYER_UNIT_IDX_VY] -= PLAYER_VY_ADJUST;
							}
						} else if (this.keys[KEY_DOWN]) {
							if (units[PLAYER_UNIT_IDX_VY] < PLAYER_MAX_VY) {
								units[PLAYER_UNIT_IDX_VY] += PLAYER_VY_ADJUST;
							}
						} else {
							/*
							    
							    units[PLAYER_UNIT_IDX_VY] = 0;
							*/

							
							if (units[PLAYER_UNIT_IDX_VY] < 0) {
								units[PLAYER_UNIT_IDX_VY] += PLAYER_KEYRELEASE_VY_ADJUST;
							} else if (units[PLAYER_UNIT_IDX_VY] > 0) {
								units[PLAYER_UNIT_IDX_VY] -= PLAYER_KEYRELEASE_VY_ADJUST;
							}
							if (units[PLAYER_UNIT_IDX_VY] < PLAYER_KEYRELEASE_VY_ADJUST && units[PLAYER_UNIT_IDX_VY] > 0 - PLAYER_KEYRELEASE_VY_ADJUST) {
								units[PLAYER_UNIT_IDX_VY] = 0;
							}
						}
						if (units[PLAYER_UNIT_IDX_Y] < PLAYER_MIN_Y && units[PLAYER_UNIT_IDX_VY] < 0) {
							units[PLAYER_UNIT_IDX_Y] = PLAYER_MIN_Y_ADJ;
							units[PLAYER_UNIT_IDX_VY] = 0;
							this.keys[KEY_UP] = false;
						}
						if (units[PLAYER_UNIT_IDX_Y] >= PLAYER_MAX_Y && units[PLAYER_UNIT_IDX_VY] > 0) {
							units[PLAYER_UNIT_IDX_Y] = PLAYER_MAX_Y_ADJ;
							units[PLAYER_UNIT_IDX_VY] = 0;
							this.keys[KEY_DOWN] = false;
						}

						
						

						
						
						
						
						
						while (--nextShipFrameCount <= 0) {
							
							int swarmShipPos = levelSwarmPos + SWARM_BASE_LEN + (swarmShipIndex * SWARM_IDX_SHIP_LEN);
							nextShipFrameCount = (STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_FRAMES_TO_NEXT] & SWARM_FRAMES_TO_NEXT_SPAWN_MASK) + levelDifficulty;
							if (--swarmShipsRemaining < 0) {
								
								nextShipFrameCount = ((STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_FRAMES_TO_NEXT] >> SWARM_FRAMES_TO_NEXT_SHIP_SHIFT) & SWARM_FRAMES_TO_NEXT_SHIP_MASK) + levelDifficulty;

								swarmShipPos += SWARM_IDX_SHIP_LEN;

								
								
								if (++swarmShipIndex >= STATIC[levelSwarmPos + SWARM_IDX_SHIP_COUNT]) {

									
									

									
									

									
									if (bossUnitIndex > 0) {
										
										swarmShipIndex--;
										break;
									}

									swarmShipIndex = 0;

									levelSwarmPos = STATIC[levelSwarmPos + SWARM_IDX_NEXT_SWARM_POS];
									if (--wavesUntilBoss <= 0) {
										
										
										nextLevelSwarmIndex = levelSwarmPos;
										levelSwarmPos = STATIC[nextBossSwarmIndex + SWARM_IDX_NEXT_SWARM_POS];
									}

									swarmShipPos = levelSwarmPos + SWARM_BASE_LEN;
								}
								swarmShipsRemaining = STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_SHIP_COUNT];
								
								
								
								
								
								
								break;
							}

							

							for (int pp = OTHER_START_INDEX; pp < UNIT_SIZE; pp += UNIT_LEN) {
								if (units[pp + UNIT_IDX_SHIP_TEMPLATE] == 0) {
									
									if (nextLevelSwarmIndex > 0 && bossUnitIndex == 0) {
										bossUnitIndex = pp;
									}

									int templatePos = STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_TEMPLATE_POS];
									units[pp + UNIT_IDX_SHIP_TEMPLATE] = templatePos;
									units[pp + UNIT_IDX_X] = (STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_X] + (STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_DX] * (STATIC[swarmShipPos
											+ SWARM_IDX_SHIP_IDX_SHIP_COUNT] - swarmShipsRemaining))) << UNIT_POSITION_SHIFT;
									units[pp + UNIT_IDX_Y] = (STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_Y] + (STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_DY] * (STATIC[swarmShipPos
											+ SWARM_IDX_SHIP_IDX_SHIP_COUNT] - swarmShipsRemaining))) << UNIT_POSITION_SHIFT;
									

									units[pp + UNIT_IDX_VX] = STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_VX];
									units[pp + UNIT_IDX_VY] = STATIC[swarmShipPos + SWARM_IDX_SHIP_IDX_VY];
									units[pp + UNIT_IDX_AX] = 0;
									units[pp + UNIT_IDX_AY] = 0;
									units[pp + UNIT_IDX_SCALE_X] = units[pp + UNIT_IDX_SCALE_Y] = random.nextInt(0x30) - 0x18;
									units[pp + UNIT_IDX_HP] = STATIC[templatePos + SHIP_IDX_HP] << 3;
									units[pp + UNIT_IDX_TIMER] = 0;
									break;
								}
							}

							
							
							break;
						}
					} else {
						
						
						units[PLAYER_UNIT_IDX_SHIP_TEMPLATE] = 0;
						units[HEALTHBAR_INDEX + UNIT_IDX_SHIP_TEMPLATE] = 0;

						

						ogr.drawString("Click To Start", STARTTEXT_X, STARTTEXT_Y);

						if (this.keys[0]) {
							
							lives = PLAYER_START_LIVES;
							

							
							units[PLAYER_UNIT_IDX_VX] = 0;
							units[PLAYER_UNIT_IDX_VY] = 0;
							units[PLAYER_UNIT_IDX_SCALE_X] = 0;
							units[PLAYER_UNIT_IDX_SCALE_Y] = 0;
							
							
							
							
							

							
							
							units[HEALTHBAR_UNIT_IDX_Y] = HEALTHBAR_Y;
							units[HEALTHBAR_UNIT_IDX_AX] = HEALTHBAR_X; 
							units[HEALTHBAR_UNIT_IDX_HP] = 1; 
							units[HEALTHBAR_UNIT_IDX_SHIP_TEMPLATE] = HEALTHBAR_SHIP_TEMPLATE;


							for (int pp = OTHER_START_INDEX; pp < UNIT_SIZE; pp++) {
								units[pp] = 0;
							}

							continue startGame;
						}

					}


					int[] hitDetect = new int[SCENE_PIXEL_COUNT];
					unitDrawLoop: for (int upos = 0; upos < UNIT_SIZE; upos += UNIT_LEN) {
						int templatePos = units[upos + UNIT_IDX_SHIP_TEMPLATE];

						if (templatePos > 0) {
							int ai = STATIC[templatePos + SHIP_IDX_AI_HIT];

							if ((ai & SHIP_LIFE_COUNTDOWN) != 0) {
								units[upos + UNIT_IDX_HP]--;
							}

							
							
							if (units[upos + UNIT_IDX_HP] <= 0) {
								
								if ((ai & SHIP_HIT_PLAYER_WEAPON_DAMAGES_UNIT) != 0) {
									score += STATIC[templatePos + SHIP_IDX_HP];
								}

								
								
								
								for (int i = 0; i < PLASMA_MAX_HITS_COUNT; i++) {
									if (plasmaHittingUnitIndex[i] == upos) {
										while (i < PLASMA_MAX_HITS_COUNT - 1) {
											plasmaHittingUnitIndex[i] = plasmaHittingUnitIndex[i + 1];
											i++;
										}
										plasmaHittingUnitIndex[i] = 0;
										break;
									}
								}

								units[upos + UNIT_IDX_SHIP_TEMPLATE] = 0;

								if (upos == PLAYER_INDEX) {
									
									
									lives = 0;
									continue startLevel;
								}
								if (upos == bossUnitIndex && bossUnitIndex > 0) {
									
									

									
									for (int spos = OTHER_START_INDEX; spos < UNIT_SIZE; spos += UNIT_LEN) {
										int tpos = units[spos + UNIT_IDX_SHIP_TEMPLATE];
										if (tpos > 0 && (STATIC[tpos + SHIP_IDX_AI_HIT] & SHIP_IS_BOSS) != 0) {
											units[spos + UNIT_IDX_HP] = 0;
										}
									}

									
									levelSwarmPos = nextLevelSwarmIndex;
									nextLevelSwarmIndex = 0;
									nextBossSwarmIndex = STATIC[nextBossSwarmIndex + SWARM_IDX_NEXT_SWARM_POS];
									bossUnitIndex = 0;
									wavesUntilBoss = BOSS_WAVE_FREQUENCY;

									
									levelDifficulty -= LEVEL_DIFFICULTY;
									if (levelDifficulty < MAX_DIFFICULTY) {
										levelDifficulty = MAX_DIFFICULTY;
									}
								}
								int deathMask = (ai & SHIP_ONDEATH_MASK);
								if (deathMask == SHIP_ONDEATH_SPAWN_BULLETS_DOWN) {
									
								}
								if (deathMask == SHIP_ONDEATH_SPAWN_BULLETS_CIRCLE) {
									
								}
								if (deathMask == SHIP_ONDEATH_SPAWN_BOOM) {
									
									
									units[upos + UNIT_IDX_SHIP_TEMPLATE] = EXPLOSION_SHIP_TEMPLATE;
									units[upos + UNIT_IDX_HP] = STATIC[EXPLOSION_SHIP_TEMPLATE + SHIP_IDX_HP];
									units[upos + UNIT_IDX_VX] = 0;
									units[upos + UNIT_IDX_VY] = 0;
									units[upos + UNIT_IDX_SCALE_X] = units[upos + UNIT_IDX_SCALE_Y] = random.nextInt(0x100) - 0x80;
								}
								if (deathMask == SHIP_ONDEATH_SCENERY) {
									
									units[upos + UNIT_IDX_SHIP_TEMPLATE] = templatePos;
									units[upos + UNIT_IDX_HP] = STATIC[templatePos + SHIP_IDX_HP];
									units[upos + UNIT_IDX_X] = random.nextInt(UNIT_MAX_X);
									units[upos + UNIT_IDX_Y] = UNIT_MIN_Y;
								}

								continue;
							}

							units[upos + UNIT_IDX_TIMER]++;

							
							
							int aiMask = ai & SHIP_AI_MASK;
							if (aiMask == SHIP_AI_BOUNCE) {
								if (units[upos + UNIT_IDX_X] <= UNIT_MIN_X) {
									units[upos + UNIT_IDX_VX] = (random.nextInt(3) + 1) << UNIT_POSITION_SHIFT;
								}
								if (units[upos + UNIT_IDX_X] >= UNIT_MAX_X) {
									units[upos + UNIT_IDX_VX] = (random.nextInt(3) - 4) << UNIT_POSITION_SHIFT;
								}
								if (units[upos + UNIT_IDX_Y] <= UNIT_MIN_Y) {
									units[upos + UNIT_IDX_VY] = (random.nextInt(3) + 1) << UNIT_POSITION_SHIFT;
								}
								if (units[upos + UNIT_IDX_Y] >= UNIT_MAX_Y) {
									units[upos + UNIT_IDX_VY] = (random.nextInt(3) - 4) << UNIT_POSITION_SHIFT;
								}
							}
							if (aiMask == SHIP_AI_DECAL) {
								
								
								
								units[upos + UNIT_IDX_X] = (viewStartX << UNIT_POSITION_SHIFT) + units[upos + UNIT_IDX_AX];
							}

							if (aiMask == SHIP_AI_SHORT_ZIGZAG) {
								if (units[upos + UNIT_IDX_TIMER] % SHORT_ZIGZAG_MOVE == 0) {
									units[upos + UNIT_IDX_VX] *= -1;
								}
							}
							if (aiMask == SHIP_AI_ZIGZAG) {
								if (units[upos + UNIT_IDX_VX] > 0) {
									if (units[upos + UNIT_IDX_X] > (3 * SCENE_WIDTH / 4) << UNIT_POSITION_SHIFT) {
										units[upos + UNIT_IDX_VX] *= -1;
									}
								} else if (units[upos + UNIT_IDX_X] < (SCENE_WIDTH / 4) << UNIT_POSITION_SHIFT) {
									units[upos + UNIT_IDX_VX] *= -1;
								}
							}
							if (aiMask == SHIP_AI_HOOK_X) {
								
								if (units[upos + UNIT_IDX_AX] == 0) {
									units[upos + UNIT_IDX_AX] = 2;
									if (units[upos + UNIT_IDX_X] > (SCENE_CENTER_X << UNIT_POSITION_SHIFT)) {
										units[upos + UNIT_IDX_AX] *= -1;
									}
								}

								if ((units[upos + UNIT_IDX_AX] > 0 && units[upos + UNIT_IDX_X] > (SCENE_CENTER_X << UNIT_POSITION_SHIFT))
										|| (units[upos + UNIT_IDX_AX] < 0 && units[upos + UNIT_IDX_X] < (SCENE_CENTER_X << UNIT_POSITION_SHIFT))) {
									units[upos + UNIT_IDX_AX] *= -1;
								}
							}
							if (aiMask == SHIP_AI_HOOK_Y) {
								
								if (units[upos + UNIT_IDX_AY] == 0) {
									units[upos + UNIT_IDX_AY] = 2;
									if (units[upos + UNIT_IDX_Y] > (SCENE_CENTER_Y << UNIT_POSITION_SHIFT)) {
										units[upos + UNIT_IDX_AY] *= -1;
									}
								}

								if ((units[upos + UNIT_IDX_AY] > 0 && units[upos + UNIT_IDX_Y] > ((SCENE_CENTER_Y / 2) << UNIT_POSITION_SHIFT))
										|| (units[upos + UNIT_IDX_AY] < 0 && units[upos + UNIT_IDX_Y] < ((SCENE_CENTER_Y / 2) << UNIT_POSITION_SHIFT))) {
									units[upos + UNIT_IDX_AY] *= -1;
								}
							}
							if (aiMask == SHIP_AI_VERTICAL_BOUNCE) {
								if (units[upos + UNIT_IDX_Y] >= UNIT_MAX_Y || units[upos + UNIT_IDX_Y] <= UNIT_MIN_Y) {
									units[upos + UNIT_IDX_VY] *= -1;
								}
							}
							if ((ai & SHIP_REMOVE_AT_MAXY) != 0 && units[upos + UNIT_IDX_Y] > UNIT_MAX_Y) {
								
								
								units[upos + UNIT_IDX_HP] = 0;
							}
							if ((ai & SHIP_REMOVE_AT_MINY) != 0 && units[upos + UNIT_IDX_Y] < UNIT_MIN_Y) {
								
								
								units[upos + UNIT_IDX_HP] = 0;
							}
							
							if (units[upos + UNIT_IDX_X] < UNIT_ABSOLUTE_MIN_X || units[upos + UNIT_IDX_X] > UNIT_ABSOLUTE_MAX_X) {
								
								
								units[upos + UNIT_IDX_HP] = 0;
							}

							
							

							int basex = ((units[upos + UNIT_IDX_X]) >> UNIT_POSITION_SHIFT);
							int basey = (units[upos + UNIT_IDX_Y]) >> UNIT_POSITION_SHIFT;
							int pcount = STATIC[templatePos + SHIP_IDX_DAMAGE_PART_COUNT] & SHIP_PART_COUNT_MASK;

							int sppos = templatePos + SHIP_BASE_LEN;
							int playerDamage = 0;
							for (int j = 0; j < pcount; j++, sppos += SHIP_IDX_PART_LEN) {
								int scaleX = STATIC[sppos + SHIP_IDX_PART_IDX_SCALE_X] + units[upos + UNIT_IDX_SCALE_X];
								int scaleY = STATIC[sppos + SHIP_IDX_PART_IDX_SCALE_Y] + units[upos + UNIT_IDX_SCALE_Y];
								int ppos = STATIC[sppos + SHIP_IDX_PART_IDX_IDX];
								int partx = basex + STATIC[sppos + SHIP_IDX_PART_IDX_X];
								int party = basey + STATIC[sppos + SHIP_IDX_PART_IDX_Y];
								int baseColor = STATIC[sppos + SHIP_IDX_PART_IDX_BASECOLOR];


								int ccount = STATIC[ppos + PART_IDX_COMPONENT_COUNT];


								int spawnVY = 0;
								int spawnTemplate = 0;

								int partAi = STATIC[ppos + PART_IDX_AI];
								if (partAi == PART_AI_LIGHT_FIRE) {
									

									if ((units[upos + UNIT_IDX_TIMER] + upos) % MEDIUM_FIRE_RATE == 0) {
										spawnVY = BULLET_SPEED_MEDIUM;
										spawnTemplate = LONG_BULLET_SHIP_TEMPLATE;
									}
								}
								int spawnVX = 0;
								if (partAi == PART_AI_RADIAL_FIRE) {
									int time = units[upos + UNIT_IDX_TIMER] % (RADIAL_FIRE_RATE * 8);
									if (time == 0) {
										
										spawnVY = BULLET_SPEED_SLOW;
									}
									if (time == RADIAL_FIRE_RATE) {
										
										spawnVX = spawnVY = BULLET_SPEED_SLOW / 2;
									}
									if (time == RADIAL_FIRE_RATE * 2) {
										
										spawnVX = BULLET_SPEED_SLOW;
									}
									if (time == RADIAL_FIRE_RATE * 3) {
										
										spawnVX = BULLET_SPEED_SLOW / 2;
										spawnVY = -BULLET_SPEED_SLOW / 2;
									}
									if (time == RADIAL_FIRE_RATE * 4) {
										
										spawnVY = -BULLET_SPEED_SLOW;
									}
									if (time == RADIAL_FIRE_RATE * 5) {
										
										spawnVY = spawnVX = -BULLET_SPEED_SLOW / 2;
									}
									if (time == RADIAL_FIRE_RATE * 6) {
										
										spawnVX = -BULLET_SPEED_SLOW;
									}
									if (time == RADIAL_FIRE_RATE * 7) {
										
										spawnVX = -BULLET_SPEED_SLOW / 2;
										spawnVY = BULLET_SPEED_SLOW / 2;
									}
									if (spawnVX != 0 || spawnVY != 0) {
										
										spawnTemplate = SHORT_BULLET_SHIP_TEMPLATE;
									}
								}
								if (partAi == PART_AI_SQUARE_FIRE) {
									int time = units[upos + UNIT_IDX_TIMER] % SLOW_FIRE_RATE;
									if (time < 4) {
										spawnTemplate = SHORT_BULLET_SHIP_TEMPLATE;
									}
									if (time == 0) {
										
										spawnVY = BULLET_SPEED_FAST;
									}
									if (time == 1) {
										
										spawnVX = BULLET_SPEED_FAST;
									}
									if (time == 2) {
										
										spawnVY = -BULLET_SPEED_FAST;
									}
									if (time == 3) {
										
										spawnVX = -BULLET_SPEED_FAST;
									}
								}
								if (spawnTemplate != 0) {
									
									for (int k = OTHER_START_INDEX; k < UNIT_SIZE; k += UNIT_LEN) {
										if (units[k + UNIT_IDX_SHIP_TEMPLATE] == 0) {
											
											units[k + UNIT_IDX_X] = partx << UNIT_POSITION_SHIFT;
											units[k + UNIT_IDX_Y] = party << UNIT_POSITION_SHIFT;
											units[k + UNIT_IDX_VX] = spawnVX;
											units[k + UNIT_IDX_VY] = spawnVY;
											
											
											units[k + UNIT_IDX_HP] = STATIC[spawnTemplate + SHIP_IDX_HP];
											units[k + UNIT_IDX_SHIP_TEMPLATE] = spawnTemplate;
											break;
										}
									}
								}

								int cpos = ppos + PART_BASE_LEN;
								int colorB = ((baseColor & 0x00f) << 8) | ((baseColor & 0x00f) << 12);
								int colorG = ((baseColor & 0x0f0) << 4) | ((baseColor & 0x0f0) << 8);
								int colorR = (baseColor & 0xf00) | ((baseColor & 0xf00) << 4);
								for (int k = 0; k < ccount; k++, cpos += PART_IDX_COMPONENT_LEN) {
									
									
									int alpha = STATIC[cpos + PART_IDX_COMPONENT_IDX_GRADIENT];


									int startX = partx + ((scaleX * STATIC[cpos + PART_IDX_COMPONENT_IDX_X]) >> SHIP_SCALE_X_SHR);
									int endX = startX + ((scaleX * STATIC[cpos + PART_IDX_COMPONENT_IDX_WIDTH]) >> SHIP_SCALE_X_SHR);
									if (endX < startX) {
										
										int temp = endX;
										endX = startX;
										startX = temp;
									}

									
									
									
									
									
									
									
									
									
									
									if (startX < 0) {
										startX = 0;
									}
									if (endX > SCENE_WIDTH) {
										endX = SCENE_WIDTH;
									}

									
									
									
									if (endX > startX) {
										int y = party + ((scaleY * STATIC[cpos + PART_IDX_COMPONENT_IDX_Y]) >> SHIP_SCALE_Y_SHR);
										int endY = y + ((scaleY * STATIC[cpos + PART_IDX_COMPONENT_IDX_HEIGHT]) >> SHIP_SCALE_Y_SHR);

										
										if (y > endY) {
											int temp = y;
											y = endY;
											endY = temp;
										}

										if (y < 0) {
											y = 0;
										}
										if (endY > RENDER_HEIGHT) {
											endY = RENDER_HEIGHT;
										}
										int gradient = (alpha & 0xff) + 1;
										for (int color = (((gradient * colorR)) & 0xff0000) | (((gradient * colorG) >>> 8) & 0xff00) | (((gradient * colorB) >>> 16) & 0xff); y < endY; y++) {
											for (int x = startX; x < endX; x++) {
												
												int hitPos = x + (y * SCENE_WIDTH);

												
												
												
												
												int hitDamage = (STATIC[templatePos + SHIP_IDX_DAMAGE_PART_COUNT] >> SHIP_DAMAGE_SHIFT) & SHIP_DAMAGE_MASK;
												if (hitDetect[hitPos] == PLAYER_INDEX && upos != PLAYER_INDEX) {
													if (hitDamage > playerDamage) {
														playerDamage = hitDamage;
													}
												} else {
													hitDetect[hitPos] = upos;

													if (x >= viewStartX && x < viewEndX) {
														display[x - viewStartX + (y * RENDER_WIDTH)] = color;
													}
													if (x < viewStartX && hitDamage > 0) {
														
														display[y * RENDER_WIDTH] = color;
														display[y * RENDER_WIDTH + 1] = color;
													}
													if (x >= viewEndX && hitDamage > 0) {
														
														display[(y + 1) * RENDER_WIDTH - 1] = color;
														display[(y + 1) * RENDER_WIDTH - 2] = color;
													}
												}
											}
										}
									}
								}
							}
							if (playerDamage > 0) {
								if (templatePos == PICKUP_LASER_SHIP_TEMPLATE) {
									powerupLaser += LASER_INC;
									if (powerupLaser > LASER_MAX) {
										powerupLaser = LASER_MAX;
									}
								}
								if (templatePos == PICKUP_PLASMA_SHIP_TEMPLATE) {
									powerupPlasma += PLASMA_INC;
									if (powerupPlasma > PLASMA_MAX) {
										powerupPlasma = PLASMA_MAX;
									}
								}
								int aiHit = STATIC[templatePos + SHIP_IDX_AI_HIT];
								if ((aiHit & SHIP_HIT_PLAYER_COLLIDE_DESTROYS_UNIT) != 0) {
									units[upos + UNIT_IDX_HP] = 0;
								}
								if ((aiHit & SHIP_HIT_PLAYER_COLLIDE_DAMAGES_UNIT) != 0) {
									units[upos + UNIT_IDX_HP] -= PLAYER_COLLISION_DAMAGE;
								}
								if ((aiHit & SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER) != 0) {
									units[PLAYER_UNIT_IDX_HP] -= playerDamage;
									powerupPlasma--;
									if (powerupPlasma < PLASMA_START) {
										powerupPlasma = PLASMA_START;
										powerupLaser--;
										if (powerupLaser < LASER_START) {
											powerupLaser = LASER_START;
										}
									}
								}
							}

							
							if ((ai & SHIP_USES_ACCELERATION) != 0) {
								units[upos + UNIT_IDX_VX] += units[upos + UNIT_IDX_AX];
								units[upos + UNIT_IDX_VY] += units[upos + UNIT_IDX_AY];
							}
							units[upos + UNIT_IDX_X] += units[upos + UNIT_IDX_VX];
							units[upos + UNIT_IDX_Y] += units[upos + UNIT_IDX_VY];
							if ((ai & SHIP_IS_BOSS) != 0 && units[upos + UNIT_IDX_Y] > MAX_BOSS_Y) {
								units[upos + UNIT_IDX_Y] = MAX_BOSS_Y;
							}
						}
					}

					
					if (lives > 0) {
						if (firingLaser) {
							
							

							
							
							int p2x = (units[PLAYER_UNIT_IDX_X] - units[PLAYER_UNIT_IDX_VX]) >> UNIT_POSITION_SHIFT;
							for (int z = -powerupLaser; z < powerupLaser; z++) {
								if (p2x + z >= viewStartX && p2x + z < viewEndX) {
									
									for (int y = ((units[PLAYER_UNIT_IDX_Y] - units[PLAYER_UNIT_IDX_VY]) >> UNIT_POSITION_SHIFT); y >= 0; y--) {
										
										
										int hitUnitIndex = hitDetect[p2x + z + (y * SCENE_WIDTH)];
										if (hitUnitIndex > PLAYER_INDEX) {
											int hitTemplatePos = units[hitUnitIndex + UNIT_IDX_SHIP_TEMPLATE];
											if ((STATIC[hitTemplatePos + SHIP_IDX_AI_HIT] & SHIP_HIT_PLAYER_WEAPON_DAMAGES_UNIT) > 0) {
												
												
												units[hitUnitIndex + UNIT_IDX_HP] -= 8;
												break;
											}
										}
										int intensity = z * 3 + 80;
										if (z < 0) {
											intensity = 80 - z * 3;
										}
										display[p2x - viewStartX + z + (y * RENDER_WIDTH)] = 0xff | (intensity << 8) | (intensity << 16);
									}
								}
							}
						} else {
							

							
							
							
							
							

							
							
							
							
							
							
							
							int p2x = units[PLAYER_UNIT_IDX_X] - units[PLAYER_UNIT_IDX_VX];
							int p2y = units[PLAYER_UNIT_IDX_Y] - BEND_SHOT_Y;
							int pp0 = PLAYER_INDEX;
							plasmaUnitTraceList: for (int i = 0, loopCount = 0; i < PLASMA_UNIT_LOOP && loopCount < MAX_PLASMA_UNIT_LOOP; i++, loopCount++) {
								int pp3 = 1;
								if (i < PLASMA_MAX_HITS_COUNT) {
									pp3 = plasmaHittingUnitIndex[i];
                                }
								if (pp3 > 0) {
									int p0x = units[pp0 + UNIT_IDX_X];
									int p0y = units[pp0 + UNIT_IDX_Y];
									if (pp0 == PLAYER_INDEX) {
										p0x -= units[PLAYER_UNIT_IDX_VX];
										p0y -= units[PLAYER_UNIT_IDX_VY];
									}
									int p3x = p0x;
									int p3y = 0;
									if (pp3 > 1) {
										p3x = units[pp3 + UNIT_IDX_X];
										p3y = units[pp3 + UNIT_IDX_Y];
									}

									
									
									
									int p1x = 2 * p0x - p2x;
									int p1y = 2 * p0y - p2y;
									int pp2 = i + 1;
									for (; pp2 < PLASMA_MAX_HITS_COUNT && plasmaHittingUnitIndex[pp2] == 0; pp2++)
										;
									if (pp2 >= PLASMA_MAX_HITS_COUNT) {
										
										p2x = p3x;
										p2y = p3y;
									} else {
										pp2 = plasmaHittingUnitIndex[pp2];
										p2x = 2 * p3x - units[pp2 + UNIT_IDX_X];
										p2y = 2 * p3y - units[pp2 + UNIT_IDX_Y];
									}

									
									
									
									int steps = (((p0x > p3x ? (p0x - p3x) : (p3x - p0x)) + (p0y > p3y ? (p0y - p3y) : (p3y - p0y))) >> UNIT_POSITION_SHIFT) / powerupPlasma / 2;
									
									if (steps > MAX_PLASMA_STEPS) {
										steps = MAX_PLASMA_STEPS;
									}
									float du = 0.25f / (float) steps;

									for (float t1 = 0.f; t1 <= 1.f; t1 += du) {
										float it1 = (1.f - t1);
										float it2 = it1 * it1;
										float it3 = it2 * it1;
										float t2 = t1 * t1;
										float t3 = t2 * t1;
										float it2_t1_3 = 3 * it2 * t1;
										float it1_t2_3 = 3 * it1 * t2;
										int x0 = ((int) ((p0x * it3) + (it2_t1_3 * p1x) + (it1_t2_3 * p2x) + (t3 * p3x))) >> UNIT_POSITION_SHIFT;
										int y0 = ((int) ((p0y * it3) + (it2_t1_3 * p1y) + (it1_t2_3 * p2y) + (t3 * p3y))) >> UNIT_POSITION_SHIFT;

										for (int z = -powerupPlasma; z < powerupPlasma; z++) {
											
											
											if (x0 + z >= 0 && x0 + z < SCENE_WIDTH) {
												

												for (int q = -powerupPlasma; q < powerupPlasma; q++) {
													if (y0 + q >= 0 && y0 + q < RENDER_HEIGHT) {
														
														
														int hitPos = x0 + z + ((y0 + q) * SCENE_WIDTH);
														int hitUnitIndex = hitDetect[hitPos];
														if (hitUnitIndex == PLAYER_INDEX) {
															
															continue;
														}
														if (hitUnitIndex > PLAYER_INDEX) {
															int hitTemplatePos = units[hitUnitIndex + UNIT_IDX_SHIP_TEMPLATE];
															if ((STATIC[hitTemplatePos + SHIP_IDX_AI_HIT] & SHIP_HIT_PLAYER_WEAPON_DAMAGES_UNIT) > 0) {

																
																
																
																boolean found = IntStream.range(0, PLASMA_MAX_HITS_COUNT).anyMatch(j -> plasmaHittingUnitIndex[j] == hitUnitIndex);
																if (!found) {
																	for (int j = PLASMA_MAX_HITS_COUNT; --j > i;) {
																		plasmaHittingUnitIndex[j] = plasmaHittingUnitIndex[j - 1];
																	}
																	
																	
																	for (; i == PLASMA_MAX_HITS_COUNT || (i > 0 && plasmaHittingUnitIndex[i - 1] == 0); i--)
																		;

																	plasmaHittingUnitIndex[i] = hitUnitIndex;
																	i--;
																	pp0 = hitUnitIndex;

																	continue plasmaUnitTraceList;
																}
															}
														}

														if (x0 + z >= viewStartX && x0 + z < viewEndX) {

															int j = (20 + z + q) << 2;
															if (j < 80) {
																j = 80 - j;
															}
															display[x0 - viewStartX + z + ((y0 + q) * RENDER_WIDTH)] = 0x8f008f | (j << 8);
														}

														
														

													}
												}
											}
										}
									}

									if (pp3 > 1 && (STATIC[units[pp3 + UNIT_IDX_SHIP_TEMPLATE] + SHIP_IDX_AI_HIT] & SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER) > 0) {
										
										if (units[PLAYER_UNIT_IDX_HP] < PLAYER_START_HP) {
											units[PLAYER_UNIT_IDX_HP] += powerupPlasma;
										}
									}

									pp0 = pp3;

								}
							}
						}

					} 

					
					
					if (score > highScore) {
						highScore = score;
					}
					
					ogr.drawString("Score:", SCORE_TEXT1_X, SCORE_TEXT_Y);
					ogr.drawString(Integer.toString(score), SCORE_TEXT2_X, SCORE_TEXT_Y);
					ogr.drawString("High Score:", SCORE_TEXT3_X, SCORE_TEXT_Y);
					ogr.drawString(Integer.toString(highScore), SCORE_TEXT4_X, SCORE_TEXT_Y);

					
					
					sg.drawImage(image, 0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, 
							0, 0, RENDER_WIDTH, RENDER_HEIGHT, 
							null);

					long time;
					do {
						time = System.nanoTime();
						Thread.yield();
					} while (time - nextFrame < 0);
					nextFrame = time + (1000000000L / FRAMES_PER_SECOND);
					if (!isActive()) {
						return;
					}
				}
			}
		}
	}

	@Override
    public boolean handleEvent(Event e) {
		
		
		
		
		
		boolean down = IntStream.of(KEY_PRESS, KEY_ACTION, MOUSE_DOWN).anyMatch(i -> e.id == i);


		if (e.id >= KEY_PRESS && e.id <= KEY_ACTION_RELEASE) {
			
			
			
			this.keys[e.key] = down;
		}
		if (e.id == MOUSE_DOWN || e.id == MOUSE_UP) {
			this.keys[0] = down;
		}
		return false;
	}

	
	
	private static final int DISPLAY_WIDTH = 400;
	private static final int DISPLAY_HEIGHT = 600;
	private static final int RENDER_WIDTH = DISPLAY_WIDTH;
	private static final int RENDER_HEIGHT = DISPLAY_HEIGHT;
	private static final int RENDER_PIXEL_COUNT = RENDER_WIDTH * RENDER_HEIGHT;
	private static final int RENDER_CENTER_X = RENDER_WIDTH / 2;
	private static final int RENDER_CENTER_Y = RENDER_HEIGHT / 2;
	private static final int FRAMES_PER_SECOND = 60; 

	private static final int UNIT_POSITION_SHIFT = 4; 

	
	private static final int SCENE_WIDTH = RENDER_WIDTH * 2;
	private static final int SCENE_HEIGHT = RENDER_HEIGHT;
	private static final int SCENE_CENTER_X = SCENE_WIDTH / 2;
	private static final int SCENE_CENTER_Y = SCENE_HEIGHT / 2;
	private static final int SCENE_PIXEL_COUNT = SCENE_WIDTH * SCENE_HEIGHT;

	private static final int SCENE_RIGHT_SCROLL = 5 * RENDER_WIDTH / 8;
	private static final int SCENE_LEFT_SCROLL = 3 * RENDER_WIDTH / 8;
	private static final int UNIT_RIGHT_SCROLL = SCENE_RIGHT_SCROLL << UNIT_POSITION_SHIFT;
	private static final int UNIT_LEFT_SCROLL = SCENE_LEFT_SCROLL << UNIT_POSITION_SHIFT;
	private static final int SCENE_MAX_SCROLL = SCENE_WIDTH - RENDER_WIDTH;

	private static final int STARTTEXT_X = RENDER_CENTER_X - 30;
	private static final int STARTTEXT_Y = RENDER_CENTER_Y;

	private static final int MAX_PLASMA_STEPS = RENDER_HEIGHT + RENDER_WIDTH;

	
	
	private static final int KEY_UP = 1004;
	private static final int KEY_DOWN = 1005;
	private static final int KEY_LEFT = 1006;
	private static final int KEY_RIGHT = 1007;
	private static final int KEY_CHANGE_WEAPON = 32;
	
	
	

	private static final int KEY_PRESS = Event.KEY_PRESS;
	private static final int KEY_ACTION = Event.KEY_ACTION;
	private static final int MOUSE_DOWN = Event.MOUSE_DOWN;
	private static final int KEY_ACTION_RELEASE = Event.KEY_ACTION_RELEASE;
	private static final int MOUSE_UP = Event.MOUSE_UP;

	
	
	private static final int MAX_UNITS = 2000;

	private static final int UNIT_MIN_X = 0;
	private static final int UNIT_MAX_X = SCENE_WIDTH << UNIT_POSITION_SHIFT;
	private static final int UNIT_ABSOLUTE_MIN_X = UNIT_MIN_X - (60 << UNIT_POSITION_SHIFT);
	private static final int UNIT_ABSOLUTE_MAX_X = UNIT_MAX_X + (60 << UNIT_POSITION_SHIFT);
	private static final int UNIT_MIN_Y = 0;
	private static final int UNIT_MAX_Y = SCENE_HEIGHT << UNIT_POSITION_SHIFT;

	private static final int UNIT_IDX_SHIP_TEMPLATE = 0;
	private static final int UNIT_IDX_X = 1; 
	private static final int UNIT_IDX_Y = 2; 
	private static final int UNIT_IDX_VX = 3; 
	private static final int UNIT_IDX_VY = 4; 
	private static final int UNIT_IDX_AX = 5; 
	private static final int UNIT_IDX_AY = 6; 
	private static final int UNIT_IDX_HP = 7;
	private static final int UNIT_IDX_SCALE_X = 8;
	private static final int UNIT_IDX_SCALE_Y = 9;
	private static final int UNIT_IDX_TIMER = 10;
	private static final int UNIT_LEN = 11;

	private static final int UNIT_SIZE = MAX_UNITS * UNIT_LEN;

	private static final int STAR_VELOCITY_MIN = 4 << UNIT_POSITION_SHIFT;
	private static final int STAR_VELOCITY_MAX = 20 << UNIT_POSITION_SHIFT;
	private static final int STAR_VELOCITY_RANDOM = STAR_VELOCITY_MAX - STAR_VELOCITY_MIN;

	private static final int STAR_INDEX = 0;
	private static final int STAR_COUNT = 90;
	private static final int HEALTHBAR_INDEX = STAR_COUNT * UNIT_LEN;
	private static final int STAR_INDEX_FINAL = HEALTHBAR_INDEX;
	private static final int PLAYER_INDEX = HEALTHBAR_INDEX + UNIT_LEN;
	private static final int OTHER_START_INDEX = PLAYER_INDEX + UNIT_LEN;

	
	
	
	

	
	private static final int SHIP_PART_COUNT_MASK = 0x3f; 
	private static final int SHIP_DAMAGE_SHIFT = 6; 
	private static final int SHIP_DAMAGE_MASK = 0x3ff;

	
	
	
	

	private static final int SHIP_SCALE_X_SHR = 8;
	private static final int SHIP_SCALE_Y_SHR = 8;

	
	

	
	
	
	
	
	
	
	
	
	
	
	

	private static final int SHIP_AI_MASK = 0x0f;
	private static final int SHIP_AI_BOUNCE = 0;
	private static final int SHIP_AI_DECAL = 1;
	private static final int SHIP_AI_ZIGZAG = 2;
	private static final int SHIP_AI_SHORT_ZIGZAG = 3;
	
	private static final int SHIP_AI_HOOK_X = 6;
	private static final int SHIP_AI_HOOK_Y = 7;
	private static final int SHIP_AI_VERTICAL_BOUNCE = 8;
	private static final int SHIP_AI_NONE = 0x0f; 

	private static final int SHIP_REMOVE_AT_MAXY = 1 << 4;
	private static final int SHIP_REMOVE_AT_MINY = 1 << 5;
	private static final int SHIP_LIFE_COUNTDOWN = 1 << 6;

	private static final int SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER = 1 << 7;
	private static final int SHIP_HIT_PLAYER_COLLIDE_DAMAGES_UNIT = 1 << 8;
	private static final int SHIP_HIT_PLAYER_COLLIDE_DESTROYS_UNIT = 1 << 9;

	private static final int SHIP_HIT_PLAYER_WEAPON_DAMAGES_UNIT = 1 << 10;

	private static final int SHIP_IS_BOSS = 1 << 11;
	private static final int SHIP_USES_ACCELERATION = 1 << 12;

	private static final int SHIP_ONDEATH_SHL = 13;
	private static final int SHIP_ONDEATH_MASK = 0x07 << SHIP_ONDEATH_SHL;
	private static final int SHIP_ONDEATH_NONE = 0;
	private static final int SHIP_ONDEATH_SCENERY_V = 1;
	private static final int SHIP_ONDEATH_SCENERY = SHIP_ONDEATH_SCENERY_V << SHIP_ONDEATH_SHL;
	private static final int SHIP_ONDEATH_SPAWN_BOOM_V = 2;
	private static final int SHIP_ONDEATH_SPAWN_BOOM = SHIP_ONDEATH_SPAWN_BOOM_V << SHIP_ONDEATH_SHL;
	private static final int SHIP_ONDEATH_SPAWN_BULLETS_DOWN_V = 3;
	private static final int SHIP_ONDEATH_SPAWN_BULLETS_DOWN = SHIP_ONDEATH_SPAWN_BULLETS_DOWN_V << SHIP_ONDEATH_SHL;
	private static final int SHIP_ONDEATH_SPAWN_BULLETS_CIRCLE_V = 4;
	private static final int SHIP_ONDEATH_SPAWN_BULLETS_CIRCLE = SHIP_ONDEATH_SPAWN_BULLETS_CIRCLE_V << SHIP_ONDEATH_SHL;

	private static final int SHIP_AI_HIT_PLAYER = SHIP_AI_NONE;
	private static final int SHIP_AI_HIT_PICKUP = SHIP_AI_BOUNCE | SHIP_HIT_PLAYER_COLLIDE_DESTROYS_UNIT | SHIP_REMOVE_AT_MAXY | SHIP_LIFE_COUNTDOWN;
	private static final int SHIP_AI_HIT_STAR = SHIP_AI_NONE | SHIP_REMOVE_AT_MAXY | SHIP_ONDEATH_SCENERY;
	private static final int SHIP_AI_HIT_DECAL = SHIP_AI_DECAL;
	private static final int SHIP_AI_HIT_LONG_BULLET = SHIP_AI_NONE | SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER | SHIP_HIT_PLAYER_COLLIDE_DESTROYS_UNIT | SHIP_REMOVE_AT_MINY | SHIP_REMOVE_AT_MAXY;
	private static final int SHIP_AI_HIT_SHORT_BULLET = SHIP_AI_NONE | SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER | SHIP_HIT_PLAYER_COLLIDE_DESTROYS_UNIT | SHIP_REMOVE_AT_MINY | SHIP_REMOVE_AT_MAXY
			| SHIP_LIFE_COUNTDOWN;
	private static final int SHIP_AI_HIT_ZIGZAG = SHIP_AI_ZIGZAG | SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER | SHIP_HIT_PLAYER_COLLIDE_DAMAGES_UNIT | SHIP_HIT_PLAYER_WEAPON_DAMAGES_UNIT
			| SHIP_REMOVE_AT_MAXY | SHIP_ONDEATH_SPAWN_BOOM;
	private static final int SHIP_AI_HIT_EXPLOSION = SHIP_AI_NONE | SHIP_LIFE_COUNTDOWN;
	private static final int SHIP_AI_HIT_BOSS1 = SHIP_AI_SHORT_ZIGZAG | SHIP_IS_BOSS | SHIP_HIT_PLAYER_COLLIDE_DAMAGES_PLAYER | SHIP_HIT_PLAYER_COLLIDE_DAMAGES_UNIT
			| SHIP_HIT_PLAYER_WEAPON_DAMAGES_UNIT | SHIP_REMOVE_AT_MAXY | SHIP_ONDEATH_SPAWN_BOOM;

	
	
	
	
	

	private static final int PART_AI_NOOP = 0;
	private static final int PART_AI_LIGHT_FIRE = 1;
	private static final int PART_AI_RADIAL_FIRE = 2;
	private static final int PART_AI_SQUARE_FIRE = 3;

	
	
	
	
	

	

	private static final int SWARM_FRAMES_TO_NEXT_SPAWN_MASK = 0xff;
	private static final int SWARM_FRAMES_TO_NEXT_SHIP_MASK = 0xff;
	private static final int SWARM_FRAMES_TO_NEXT_SHIP_SHIFT = 8;

	
	

	private static final int SWARM_IDX_NEXT_SWARM_POS = 1;
	private static final int PART_IDX_COMPONENT_IDX_HEIGHT = 3;
	private static final int SWARM_MINIBOSS1_POS = 617;
	private static final int PART_STATIC_5_POS = 46;
	private static final int ENEMY1_SHIP_TEMPLATE = 194;
	private static final int SHIP_TEMPLATE_START_POS = 155;
	private static final int SHIP_IDX_DAMAGE_PART_COUNT = 2;
	private static final int SHIP_STATIC_6_POS = 215;
	private static final int SHIP_IDX_PART_IDX_BASECOLOR = 5;
	private static final int PART_STATIC_1_POS = 0;
	private static final int SHIP_STATIC_3_POS = 170;
	private static final int PART_START_POS = 0;
	private static final int SHIP_STATIC_5_POS = 194;
	private static final int SHIP_STATIC_10_POS = 260;
	private static final int SWARM_SIDEWAYS1_POS = 450;
	private static final int BOSS_SWARM_STATIC_2_POS = 715;
	private static final int EXPLOSION_SHIP_TEMPLATE = 239;
	private static final int BOSS_SWARM_STATIC_0_POS = 675;
	private static final int HEALTHBAR_SHIP_TEMPLATE = 179;
	private static final int SHIP_IDX_PART_IDX_IDX = 0;
	private static final int PART_STATIC_9_POS = 121;
	private static final int SWARM_SMALL4_POS = 628;
	private static final int SHIP_STATIC_9_POS = 239;
	private static final int SWARM_VERTICAL_SPRAY0_POS = 461;
	private static final int SHIP_STATIC_4_POS = 179;
	private static final int SWARM_IDX_SHIP_IDX_VX = 6;
	private static final int SHIP_IDX_HP = 1;
	private static final int PICKUP_LASER_SHIP_TEMPLATE = 260;
	private static final int SWARM_IDX_SHIP_IDX_VY = 7;
	private static final int BOSS_SWARM_START_POS = 675;
	private static final int SWARM_IDX_SHIP_COUNT = 0;
	private static final int PART_IDX_COMPONENT_IDX_WIDTH = 2;
	private static final int SWARM_IDX_SHIP_IDX_DX = 4;
	private static final int SWARM_IDX_SHIP_IDX_DY = 5;
	private static final int PART_STATIC_2_POS = 12;
	private static final int RADIAL_SHIP_TEMPLATE = 341;
	private static final int SMALL_ENEMY_TEMPLATE = 311;
	private static final int STATIC_DATA_LENGTH = 726;
	private static final int SIDEWAYS_ENEMY_TEMPLATE = 320;
	private static final int VERTICAL_SPRAY_TEMPLATE = 290;
	private static final int PART_IDX_COMPONENT_LEN = 5;
	private static final int SHIP_STATIC_1_POS = 155;
	private static final int PART_STATIC_6_POS = 53;
	private static final int LONG_BULLET_SHIP_TEMPLATE = 215;
	private static final int PART_STATIC_8_POS = 104;
	private static final int SWARM_IDX_SHIP_IDX_SHIP_COUNT = 1;
	private static final int SWARM_STATIC_0_POS = 410;
	private static final int SWARM_SMALL0_POS = 430;
	private static final int PART_STATIC_4_POS = 29;
	private static final int SHIP_IDX_PART_IDX_SCALE_Y = 4;
	private static final int SWARM_IDX_SHIP_IDX_TEMPLATE_POS = 0;
	private static final int SHIP_IDX_PART_IDX_SCALE_X = 3;
	private static final int SWARM_IDX_SHIP_IDX_FRAMES_TO_NEXT = 8;
	private static final int SWARM_IDX_SHIP_LEN = 9;
	private static final int PART_IDX_COMPONENT_IDX_GRADIENT = 4;
	private static final int PART_STATIC_10_POS = 133;
	private static final int PART_STATIC_3_POS = 60;
	private static final int BOSS1B_SHIP_TEMPLATE = 389;
	private static final int SWARM_BASE_LEN = 2;
	private static final int SWARM_IDX_SHIP_IDX_Y = 3;
	private static final int NORMAL_SWARM_START_POS = 410;
	private static final int SWARM_IDX_SHIP_IDX_X = 2;
	private static final int PART_IDX_COMPONENT_IDX_X = 0;
	private static final int BOSS_SWARM_STATIC_1_POS = 704;
	private static final int SHIP_IDX_PART_LEN = 6;
	private static final int PART_IDX_COMPONENT_IDX_Y = 1;
	private static final int SWARM_SMALL_CROSS1_POS = 492;
	private static final int SHIP_IDX_PART_IDX_X = 1;
	private static final int PART_IDX_AI = 0;
	private static final int SHIP_STATIC_12_POS = 350;
	private static final int SWARM_VERTICAL_SPRAY_POS = 577;
	private static final int SHIP_STATIC_11_POS = 275;
	private static final int SWARM_STATIC_3_POS = 597;
	private static final int SHIP_BASE_LEN = 3;
	private static final int SWARM_BOSS1_POS = 675;
	private static final int PART_BASE_LEN = 2;
	private static final int SHIP_IDX_AI_HIT = 0;
	private static final int PICKUP_PLASMA_SHIP_TEMPLATE = 275;
	private static final int PART_IDX_COMPONENT_COUNT = 1;
	private static final int SWARM_STATIC_1_POS = 472;
	private static final int BOSS1A_SHIP_TEMPLATE = 350;
	private static final int SHORT_BULLET_SHIP_TEMPLATE = 224;
	private static final int PART_STATIC_7_POS = 87;
	private static final int PLAYER_SHIP_TEMPLATE = 155;
	private static final int SHIP_STATIC_7_POS = 224;
	private static final int STAR_SHIP_TEMPLATE = 170;
	private static final int SHIP_IDX_PART_IDX_Y = 2;
	private static final int SHIP_STATIC_13_POS = 389;
	private static final int SWARM_STATIC_2_POS = 548;

	private static final String STATIC_DATA = "u0000u0002u0000u00002u0007u00ffu000fufffbu0014u0007u00e0u0000u0003u0000u0004fu001eu00e0u0004u0000u0004(u00ffu0003u0005u0006u000eu0000u0000u0003uffffu0000u0003u0004u00ffu0000ufffcu0001u0006u00fdu0000ufff6u0001u0006u00fbu0000u0001u0000u0000u00ccu000eu00ffu0000u0001u0000u0000dnu00ffu0001u0005ufffduffd6b$u00f0u0002uffd4b&u00e0u0007uffd2b u00d0fuffcdbu001cu00c0u0000uffcdu00063u0090u0000u0003u0000tu0018u0006u00fftu0000u0006u0018u00ffu0004u0004u0010u0010u00ffu0000u0003ufffeufff8u0004u0010u00ffufffcufffcbbu00ffufff8ufffeu0012u0004u00ffu0003u0002ufff6ufffeu0014u0004u00ffufffeufff6u0004u0014u00ffu0002u0004ufff8uffecu0010(u00ffuffe0ufff4@bu00c0ufff0ufff8 u0017u00d8ufffcuffe8bu001eu0080u000fu0000u0002u0000ufff0u001eu00a8u0100u063ffufffbu0000u00d0u0100u09cfu201fu0001u0001u001du0000u0000u0100u0100u0444u0001u0001u0002.u0000u0000u0100u0100u04445u0002u0002u0200u0100u0f00u4592u001eu0283<uffe23u0100u0100u0099<u001e3uff00u0100u0099u0000uffe7u0007u0100u0100u00c6u02bfu0001u1401u001du0000u0000u0280u0180u0f00u02ffu00b4u2302Wu0000u0000u0100u0100u0f22Wu0004u0003u0090u0090u0f96Ou001eu0003Wuffeeuffeeu0200u0200u0800Wufffaufffau0100u0100u0a63Wu0000u0000u0080u0080u0aaau0270u012cBWu0000u0000u0100u0100u0888hffu0100u0100u088fu0270u012cBWu0000u0000u0100u0100u0888hffu0100u0100u0f8fu4598Pu03c3u001du0000u0000u0800u0180u00c9u001du0000nu0800ufe80u0ff6yu0004u0005u0180u0130u0090u5597u0002u03c1<u00003u0090pu00f8u5596<u08c3Wuffe7ufffbu0100pu0cf6Wu0004ufffbu0100pu0cf6u0000ufff5ufffeu0080uff80u0f99u5596u00b4u08c1u0085u0000u0000u00d0uff30u06c0u4d93u0640u1906Wuffc4uffecu0510u0140u08d4u0000uffbfu0004u0100uff00u0fb0u0000u000eu0004u0100uff00u0fb0huffc4ufffbufee0u0150u03a6h<ufffbu0120u0150u03a6yu0000ufff8u0100u0100u00f9u4d93u0320u1903u0085u0000u0000u0100u0100u0ff6Wuffceuffecu0100u0100u09f6Wu0018uffecu0100u0100u09f6u0002u01aeu00c2u000fu0014u0000u0000u0000:fu1428u0104u0001Pu0000u0000u0000f$u1414u0002u01c2u0137u0014u0320u0000u0000u0000uffd82u140cu0137u0014u0000u0000u0000n(2u500cu0001u01cdu0140bu0320u001eu0000u0000uffd8u0014u5012u0001u01d8u0122u0002<nu0118u0000u0000nuc83cu0002u01ecu00c2u000fu02e4u0000uffecu0000uffccu000eu1e28u0113u0001u00b4u0000u0000u0000Pnuc804u0006u0224u0137u0005(u0000u0000u000022u140eu0122u0001Pu0256u0000u0000u0000ufff6u1414u0140bu0320u001eu0000u0000uffd8u0014u1412u0104u0001Pu0000u0000u0000f$(u0137u0005u0320u0000u0000u0000uffce2u140eu0122u0001u0154u0256u0000u0000u0000ufff6u5000u0003u0241u00c2u000fu0014u0000u0000u0000:fu143cu0155u0002u0000du0000u0000u0010u0000u5a46u0104u0001Pu0000u0000u0000f$u0003u0002u0255u0122u0002u0014u0256u0154u0000u0000ufff6u1eb4u0122u0002du0256u00c8u0000u0000ufff6uc8b4u0002u0269u00c2u000fu0014u0000u0000u0000:fu3c18u0104u0001Pu0000u0000u0000f$u0003u0001u0274u0155u0004u00002u0000u001eu0010u0000u5a5au0005u019au0137n(u0000ufffbufffa(2u140au0140u0004u0320u001eu0000u0000uffd8u0014u5012u0137nu0320u0256ufffbufffauffd8uffceu140au0140u0004u0000u001eu0000u0000(u0014u5012u0113u0001u00b4u0000u0000u0000Pnuc804u0003u02c0u015eu0001u0190u0000u0000u0000bu0003u0000u0185u0001u0154ru0000u0000bu0003u0000u0185u0001u01ccu0010u0000u0000bu0003u0300u0001u02cbu0185u0001u0190u0000u0000u0000u0001u0003u0300u0001u02a3u015eu0001u0190u0000u0000u0000u0001u0003u0300";

	
	
	private static final int PLAYER_MAX_VX = 256;
	private static final int PLAYER_MIN_VX = -PLAYER_MAX_VX;
	private static final int PLAYER_MAX_VY = 256;
	private static final int PLAYER_MIN_VY = -PLAYER_MAX_VY;

	private static final int PLAYER_VX_ADJUST = 20;
	private static final int PLAYER_VY_ADJUST = 10;
	private static final int PLAYER_KEYRELEASE_VX_ADJUST = 32;
	private static final int PLAYER_KEYRELEASE_VY_ADJUST = 20;

	private static final int PLAYER_MIN_X = 0 << UNIT_POSITION_SHIFT;
	private static final int PLAYER_MAX_X = SCENE_WIDTH << UNIT_POSITION_SHIFT;
	private static final int PLAYER_MIN_Y = 2 << UNIT_POSITION_SHIFT;
	private static final int PLAYER_MAX_Y = (SCENE_HEIGHT - 40) << UNIT_POSITION_SHIFT;

	private static final int PLAYER_MIN_X_ADJ = PLAYER_MIN_X + (1 << UNIT_POSITION_SHIFT);
	private static final int PLAYER_MAX_X_ADJ = PLAYER_MAX_X - (1 << UNIT_POSITION_SHIFT);
	private static final int PLAYER_MIN_Y_ADJ = PLAYER_MIN_Y + (1 << UNIT_POSITION_SHIFT);
	private static final int PLAYER_MAX_Y_ADJ = PLAYER_MAX_Y - (1 << UNIT_POSITION_SHIFT);

	private static final int PLAYER_START_X = SCENE_CENTER_X << UNIT_POSITION_SHIFT;
	private static final int PLAYER_START_Y = PLAYER_MAX_Y - (30 << UNIT_POSITION_SHIFT);

	
	

	private static final int PLAYER_UNIT_IDX_X = PLAYER_INDEX + UNIT_IDX_X;
	private static final int PLAYER_UNIT_IDX_Y = PLAYER_INDEX + UNIT_IDX_Y;
	private static final int PLAYER_UNIT_IDX_VX = PLAYER_INDEX + UNIT_IDX_VX;
	private static final int PLAYER_UNIT_IDX_VY = PLAYER_INDEX + UNIT_IDX_VY;
	private static final int PLAYER_UNIT_IDX_AX = PLAYER_INDEX + UNIT_IDX_AX;
	private static final int PLAYER_UNIT_IDX_AY = PLAYER_INDEX + UNIT_IDX_AY;
	private static final int PLAYER_UNIT_IDX_HP = PLAYER_INDEX + UNIT_IDX_HP;
	private static final int PLAYER_UNIT_IDX_SHIP_TEMPLATE = PLAYER_INDEX + UNIT_IDX_SHIP_TEMPLATE;
	private static final int PLAYER_UNIT_IDX_SCALE_X = PLAYER_INDEX + UNIT_IDX_SCALE_X;
	private static final int PLAYER_UNIT_IDX_SCALE_Y = PLAYER_INDEX + UNIT_IDX_SCALE_Y;

	private static final int HEALTHBAR_UNIT_IDX_X = HEALTHBAR_INDEX + UNIT_IDX_X;
	private static final int HEALTHBAR_UNIT_IDX_Y = HEALTHBAR_INDEX + UNIT_IDX_Y;
	private static final int HEALTHBAR_UNIT_IDX_VX = HEALTHBAR_INDEX + UNIT_IDX_VX;
	private static final int HEALTHBAR_UNIT_IDX_VY = HEALTHBAR_INDEX + UNIT_IDX_VY;
	private static final int HEALTHBAR_UNIT_IDX_AX = HEALTHBAR_INDEX + UNIT_IDX_AX;
	private static final int HEALTHBAR_UNIT_IDX_AY = HEALTHBAR_INDEX + UNIT_IDX_AY;
	private static final int HEALTHBAR_UNIT_IDX_HP = HEALTHBAR_INDEX + UNIT_IDX_HP;
	private static final int HEALTHBAR_UNIT_IDX_SHIP_TEMPLATE = HEALTHBAR_INDEX + UNIT_IDX_SHIP_TEMPLATE;

	private static final int SHADOW_VX_ADJUST = PLAYER_VX_ADJUST / 2;
	private static final int SHADOW_MAX_VX = PLAYER_MAX_VX / 2;

	
	private static final int HEALTHBAR_SCALE_IDX = HEALTHBAR_SHIP_TEMPLATE + SHIP_BASE_LEN + SHIP_IDX_PART_LEN + SHIP_IDX_PART_IDX_SCALE_X;

	
	

	private static final int BEND_SHOT_Y = 20 << UNIT_POSITION_SHIFT;

	private static final int SCORE_TEXT_Y = 15;
	private static final int SCORE_TEXT1_X = 10;
	private static final int SCORE_TEXT2_X = 60;
	private static final int SCORE_TEXT3_X = RENDER_CENTER_X;
	private static final int SCORE_TEXT4_X = SCORE_TEXT3_X + 80;

	private static final int HEALTHBAR_X = (RENDER_CENTER_X - 100) << UNIT_POSITION_SHIFT;
	private static final int HEALTHBAR_Y = UNIT_MAX_Y - (15 << UNIT_POSITION_SHIFT);

	
	

	private static final int PLAYER_START_LIVES = 1;
	private static final int PLAYER_START_HP = 0x200; 

	private static final int PLASMA_MAX_HITS_COUNT = 4;
	private static final int PLASMA_UNIT_LOOP = PLASMA_MAX_HITS_COUNT + 1;

	private static final int PLAYER_COLLISION_DAMAGE = 30;

	private static final int PLASMA_INC = 2;
	private static final int LASER_INC = 2;
	private static final int PLASMA_START = 1;
	private static final int LASER_START = 2;
	private static final int PLASMA_MAX = 6;
	private static final int LASER_MAX = 12;

	private static final int MAX_PLASMA_UNIT_LOOP = PLASMA_MAX_HITS_COUNT * 2;

	private static final int FIRST_SWARM_DELAY = FRAMES_PER_SECOND * 2;

	private static final int LEVEL_DIFFICULTY = 2;
	private static final int MAX_DIFFICULTY = 2;
	private static final int INITIAL_DIFFICULTY = FRAMES_PER_SECOND / 4;
	private static final int BOSS_WAVE_FREQUENCY = 6;

	private static final int SHORT_ZIGZAG_MOVE = FRAMES_PER_SECOND * 7;

	private static final int SLOW_FIRE_RATE = FRAMES_PER_SECOND * 3 / 2;
	private static final int MEDIUM_FIRE_RATE = FRAMES_PER_SECOND * 8 / 7;
	private static final int FAST_FIRE_RATE = FRAMES_PER_SECOND * 3 / 4;
	private static final int RADIAL_FIRE_RATE = FRAMES_PER_SECOND / 3;

	private static final int BULLET_SPEED_SLOW = PLAYER_MAX_VY / 5;
	private static final int BULLET_SPEED_MEDIUM = PLAYER_MAX_VY / 3;
	private static final int BULLET_SPEED_FAST = PLAYER_MAX_VY / 2;

	private static final int MAX_BOSS_Y = SCENE_CENTER_Y << UNIT_POSITION_SHIFT;
}
