package java4k.porta4k;

import java4k.GamePanel;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.SourceDataLine;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.stream.IntStream;

public class P extends GamePanel {
	
	private static final int NUM_TRACKS = 4;
	private static final int SEQ_LENGTH = 16;
	private static final int TRACK_LENGTH = 32;
	private static final int INSTRUMENT_SIZE = 32;
	private static final int SEQ_WRAP = SEQ_LENGTH * TRACK_LENGTH - 1;
	
	private static final int SEQUENCE_OFFSET = 0;
	private static final int INSTRUMENT_OFFSET = 2;
	private static final int PATTERN_OFFSET = INSTRUMENT_OFFSET + NUM_TRACKS;
	
	private static final int SAMPLE_RATE = 44100 / 2;
	private static final int TICKS_PER_BEAT = 4;
	private static final int TICKS_PER_MINUTE = 125 / 2;
	
	private static final int SAMPLES_PER_TICK = 60 * SAMPLE_RATE / (TICKS_PER_BEAT * TICKS_PER_MINUTE);
	private static final int BUFFER_SIZE = 2 * SAMPLES_PER_TICK;
	private static final int WAVE_BUFFER = 65536;
	private static final int WAVE_BUFFER_MASK = 65535;
	private static final float WAVE_FREQ = WAVE_BUFFER / SAMPLE_RATE * 440f;
	private static final int DELAY_SIZE = 1 << 18;
	private static final int DELAY_MASK = DELAY_SIZE - 1;
	
	private static final int FP = 16;
	private static final int FP_S1 = (1 << (FP - 1)) - 5120;
	private static final int FP_U1 = (1 << FP) - 5120;
	
	public static final int PITCH = 0;
	public static final int DELAY = 1;
	public static final int DELAY_FEEDBACK = 2;
	public static final int DELAY_MIX = 3;
	public static final int ENV_RATE = 4; 
	public static final int ENV_LEVEL = 6; 
	public static final int ENV_STAGE = 10;
	public static final int DELAY_POS = 11;
	public static final int OSC1_RATE = 12;
	public static final int OSC1_PHASE = 13;
	public static final int ARP_STAGE = 14;
	public static final int ARP_SPEED = 15;
	public static final int ARP_STEP_1 = 16; 
	public static final int ARP_STEP_2 = 17; 
	public static final int VOLUME = 18;
	private static final int OSC_TYPE = 19;

	
	private static final int ENV_ATTACK = ENV_RATE + 0;
	private static final int ENV_DECAY = ENV_RATE + 1;

	
	private Thread thread;

	
	
	

	
	private static final int WIDTH = 800;
	private static final int HEIGHT = 600;
	private static final float NANOTIME = 1000000000;

	
	private static final int MOUSE_X = 255;
	private static final int MOUSE_Y = 254;
	private static final int MOUSE_BUTTON = 253;

	private static final int COMPONENTS = 10;
	private static final int PORTAL1 = 0 * COMPONENTS; 
	private static final int PORTAL2 = 1 * COMPONENTS; 
	private static final int EXIT = 2 * COMPONENTS;
	private static final int LEVEL_DATA = 3 * COMPONENTS;
	private static final int NUM_PLATES = 10;

	private static final int X = 0;
	private static final int Y = 1;
	private static final int X2 = 2;
	private static final int Y2 = 3;
	private static final int LENGTH = 4;
	private static final int DX = 5; 
	private static final int DY = 6; 
	private static final int DEAD = 7; 
	private static final int INDEX = 8;
	private static final int DIE = 9;
	private static final int RAY_T = 2; 
	private static final int LINE_T = 3; 
	private static final int LINE_INDEX = 4;

	
	private static final int AIMX = 2;
	private static final int AIMY = 3;
	private static final int VX = 4; 
	private static final int VY = 5;
	private static final int ROTATION = 6; 
											
	private static final int BLOCKTIME = 7; 
	private static final int ANIMATIONTIME = 8;

	
	private static final int COLOR_BACKGROUND = 0;
	private static final int COLOR_FACE = 1;
	private static final int COLOR_WALL = 2;
	private static final int COLOR_WALL_DIE = 4;
	private static final int COLOR_P1 = 5;
	
	private static final int COLOR_PLATE_BORDER = 8;
	private static final int COLOR_WALL_BORDER = 9;
	private static final int COLOR_FACE_BORDER = 10;
	private static final int COLOR_WHITE = 11;

	
	private static final char UP = 'w';
	private static final char LEFT = 'a';
	private static final char RIGHT = 'd';
	private static final char SPACE = ' ';

	
	private static final int PLAYER_H = 10;
	private static final float HALF_PORTAL_WIDTH = 25;
	private static final float PORTAL_WIDTH_2 = (2 * HALF_PORTAL_WIDTH) * (2 * HALF_PORTAL_WIDTH);

	private static final float zNear = 0.95f;
	private static final float zFar = 1.05f;

	
	private static final float PORTAL_BLOCK_TIME = .25f;
	private static final int ENTRY_X = 30;
	private static final int BORDER = 30;
	private static final int DOOR_WIDTH = 60;

	private static final int STATE_WAIT_FOR_INPUT = 16;
	private static final int STATE_NO_UPDATE = 8;
	private static final int STATE_GAME_COMPLETE = 4;
	private static final int STATE_DIED = 2;
	private static final int STATE_LEVEL_START = 1;

	
	private static final float AIR_SPEED = 8;
	private static final float GROUND_SPEED = 60;
	private static final float JUMP_SPEED = -250f;
	private static final float FRICTION = 0.8f;
	private static final float FRICTION_AIR = 0.97f;
	private static final float GRAVITY = 700;

	private static final float PLAYER_WALL_BOUNDING = 10;
	private static final float PLAYER_WALL_BOUNDING_2 = PLAYER_WALL_BOUNDING * PLAYER_WALL_BOUNDING;

	private static final float ANIMATION_TIME = 0.1f;
	private static final float PI = 3.1415f;

	
	private static final int[] key = new int[256];



	@Override
    public void run() {
		try {
			 {
				
				
				
				AudioFormat format = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, SAMPLE_RATE, 16, 1, 2, SAMPLE_RATE, false);
				SourceDataLine line = AudioSystem.getSourceDataLine(format);
				line.open(format, BUFFER_SIZE);
				line.start();

                 int index;
				int sequence = 0;

				
				
				
				int[][] song = new int[16][32];
				for (index = 0; index < song_data.length(); index++) {
					int j = (2 * index) & 0x1f;
					song[index >> 4][j + 0] = (byte) (song_data.charAt(index) >> 8);
					song[index >> 4][j + 1] = (byte) (song_data.charAt(index) & 0xff);
				}



				
				int[][] wave = new int[3][WAVE_BUFFER];
				for (index = 0; index < WAVE_BUFFER; index++) {
					
					wave[0][index] = (int) (FP_S1 * (float) Math.sin(index * (2f * PI / WAVE_BUFFER)));
					wave[1][index] = index < (WAVE_BUFFER >> 1) ? FP_S1 : -FP_S1;
					wave[2][index] = (int) (FP_U1 * (float) Math.random()) - FP_S1;
					
				}
                 int[][] delay = new int[NUM_TRACKS][DELAY_SIZE];
                 byte[] out = new byte[BUFFER_SIZE];
                 new Thread(new Runnable() {
					int sequence;

					@Override
					public void run() {

						while(true) {
							for (int track = 0; track < NUM_TRACKS; track++) {
								int[] ins = song[INSTRUMENT_OFFSET + track];
								ins[ENV_LEVEL + 1] = FP_U1;

								
								int pattern = song[SEQUENCE_OFFSET + (track >> 1)][((track & 1) * SEQ_LENGTH) + (sequence >> 5)];
								int note = song[PATTERN_OFFSET + pattern][sequence & 0x1f];

								if (note == 1 && ins[ENV_STAGE] < (2 << FP)) {
									ins[ENV_STAGE] = (2 << FP);
								}
								if (note > 1) {
									ins[OSC1_RATE] = frequencies.charAt((note - 2 + ins[PITCH]) & 0xf) << ((note - 2 + ins[PITCH]) >> 4);
									ins[ENV_STAGE] = 0;
								}
								for (int index = 0, offset = 0; index < SAMPLES_PER_TICK; index++, offset += 2) {
									if (track == 0) {
										
										out[offset] = 0;
										out[offset + 1] = 0;
									}
									
									int stage = ins[ENV_STAGE] >> FP;
									ins[ENV_STAGE] += ins[ENV_RATE + stage];

									
									ins[OSC1_PHASE] += ins[OSC1_RATE];

									int value = wave[ins[OSC_TYPE]][ins[OSC1_PHASE] & WAVE_BUFFER_MASK];
									int env = ins[ENV_LEVEL + stage] + (int) ((long) (ins[ENV_LEVEL + stage + 1] - ins[ENV_LEVEL + stage]) * (ins[ENV_STAGE] & FP_U1) >> FP);
									value = (((value * env) >> (FP + 2)) * ins[VOLUME]) >> 6;

									
									value += ((delay[track][(ins[DELAY_POS] - ((ins[DELAY] * SAMPLES_PER_TICK >> 1))) & DELAY_MASK]) * ins[DELAY_MIX]) / 128;
									delay[track][ins[DELAY_POS]] = (value * ins[DELAY_FEEDBACK]) / 128;

									out[offset + 0] += (byte) value;
									out[offset + 1] += (byte) (value >> 8);

									ins[DELAY_POS] = (ins[DELAY_POS] + 1) & DELAY_MASK;
								}
							}

							int index = 0;
							while (index < BUFFER_SIZE) {
								index += line.write(out, index, BUFFER_SIZE - index);
							}
							sequence = (sequence + 1) & SEQ_WRAP;
						}
					}
				}).start();
			}

			
			BufferedImage screen = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
			Graphics g = screen.getGraphics();
			Graphics appletGraphics = getGraphics();

			Font smallFont = g.getFont();


            int bound = str_colors.length() >> 1;
			Color[] color = IntStream.range(0, bound).mapToObj(i1 -> new Color((str_colors.charAt(2 * i1 + 0) << 16) + str_colors.charAt(2 * i1 + 1))).toArray(Color[]::new);

            int i;
            int j = data.length();
			int[] level_data = new int[j * 2];
			for (i = 0; i < j; i++) {
				level_data[2 * i + 0] = (data.charAt(i) >> 8);
				level_data[2 * i + 1] = (data.charAt(i)) & 0xff;
			}
			int num_levels = level_data[1] >> 1;

			
			float lastTime = System.nanoTime() / NANOTIME;


            int animation = 0;
            int gameState = 0;
            int levelCount = 0;
            int portals = 0;
            float[] level = null;
            float[][] solids = null;
            int[] faceY = new int[32];
            int[] faceX = new int[32];
            float[] intersect = new float[32];
            float[] player = new float[9];
            Font bigFont = smallFont.deriveFont(0, 50);
            mainLoop: while (true) {
				float time = System.nanoTime() / NANOTIME;
                float dt = time - lastTime;
                dt = Math.min(dt, 0.05f);
				dt = Math.max(dt, 0.01f);
				lastTime = time;

				/***********************
				 * State initialization
				 ***********************/
				if ((int) player[Y] > (HEIGHT - BORDER - (DOOR_WIDTH / 2)) && (int) player[X] > (WIDTH - BORDER - DOOR_WIDTH)) {
					if (levelCount + 1 == num_levels) {
						gameState = STATE_WAIT_FOR_INPUT + STATE_NO_UPDATE + STATE_GAME_COMPLETE;
					} else {
						levelCount++;
						level = null;
					}
				}

				if (player[Y] > HEIGHT) {
					gameState = STATE_WAIT_FOR_INPUT + STATE_NO_UPDATE + STATE_DIED;
					level = null;
				}


                int k;
                if (level == null) {
					player[X] = ENTRY_X + DOOR_WIDTH / 2;
					player[Y] = HEIGHT - BORDER - PLAYER_WALL_BOUNDING;
					player[AIMX] = player[X] + 10;
					player[AIMY] = player[Y];
					player[VX] = 0;
					player[VY] = 0;
					portals = 0;
					if (gameState == 0) {
						gameState = STATE_WAIT_FOR_INPUT + STATE_NO_UPDATE + STATE_LEVEL_START;
					}

					
					int offset = (level_data[2 * levelCount + 0] << 8) + (level_data[2 * levelCount + 1]);
					int length = level_data[offset++];
					int shapes = level_data[offset++];

					int index = LEVEL_DATA;
					level = new float[length * COMPONENTS + index];
					solids = new float[shapes][];

					level[EXIT + X] = WIDTH - BORDER;
					level[EXIT + Y] = HEIGHT - BORDER;
					level[EXIT + X2] = WIDTH - BORDER - DOOR_WIDTH;
					level[EXIT + Y2] = HEIGHT - BORDER;

					k = 12;
					int point = offset + shapes; 
					for (i = 0; i < shapes; i++, k = 0) {
						int linesInShape = level_data[offset + i];
						int last = point + 2 * linesInShape - 2;
						solids[i] = new float[2 * linesInShape + k];

						for (j = 0; j < linesInShape; j++, last = point, point += 2, index += COMPONENTS) {
							level[index + X] = 10 * (level_data[last + 0] & 0x7f);
							level[index + Y] = 10 * (level_data[last + 1] & 0x7f);
							level[index + X2] = 10 * (level_data[point + 0] & 0x7f);
							level[index + Y2] = 10 * (level_data[point + 1] & 0x7f);
							level[index + DEAD] = (level_data[point + 0] >> 7);
							level[index + DIE] = (level_data[point + 1] >> 7);

							solids[i][2 * j + X] = level[index + X];
							solids[i][2 * j + Y] = level[index + Y];

							float dx = level[index + X2] - level[index + X];
							float dy = level[index + Y2] - level[index + Y];
							float len = (float) Math.sqrt(dx * dx + dy * dy);
							level[index + LENGTH] = len;
							level[index + DX] = dx / len;
							level[index + DY] = dy / len;
						}
						if (i == 0) {
							j = 2 * linesInShape;
							solids[0][j + X] = solids[0][X];
							solids[0][j + Y] = solids[i][Y];
							
							
							j += 4;
							solids[i][j + X] = WIDTH;
							
							j += 2;
							solids[i][j + X] = WIDTH;
							solids[i][j + Y] = HEIGHT;
							j += 2;
							
							solids[i][j + Y] = HEIGHT;
						}
					}

					

					
					
					
					
					
					
					
					
					
					
					
					
					
					
					
				}

				/*********************
				 * UPDATE
				 *********************/
				
				int onGround = 0;

				
				for (i = 0; i < level.length; i += COMPONENTS) {
					
					
					
					
					float x = level[i + X];
					float y = level[i + Y];
					float length = level[i + LENGTH];
					float lx = level[i + DX];
					float ly = level[i + DY];
					float position = ((player[X] - x) * lx + (player[Y] - y) * ly);

                    if (0 < position && position <= length) {
                        float ax = x + position * lx;
                        float dx = ax - player[X];
                        float ay = y + position * ly;
                        float dy = ay - player[Y];

						if (dx * dx + dy * dy <= PLAYER_WALL_BOUNDING_2) {
							if (i <= PORTAL2) {
								if (player[BLOCKTIME] > time || portals < 3) {
									
									continue;
								}
								
								int direction = i / COMPONENTS;

                                player[BLOCKTIME] = time + PORTAL_BLOCK_TIME;

                                int dstPortal = (1 - direction) * COMPONENTS;
                                float ddx = level[dstPortal + DX];
								float ddy = level[dstPortal + DY];
								player[X] = level[dstPortal + X] + ddx * HALF_PORTAL_WIDTH - ddy * (PLAYER_WALL_BOUNDING + 2);
								player[Y] = level[dstPortal + Y] + ddy * HALF_PORTAL_WIDTH + ddx * (PLAYER_WALL_BOUNDING + 2);

								float cosR = (float) Math.cos(2 * PI * direction - player[ROTATION]);
								float sinR = (float) Math.sin(2 * PI * direction - player[ROTATION]);

								float vx = player[VX] * cosR - player[VY] * sinR;
								player[VY] = player[VX] * sinR + player[VY] * cosR;
								player[VX] = vx;
								break;
							} else {
								if ((int) level[i + DIE] != 0) {
									level = null;
									gameState = STATE_WAIT_FOR_INPUT + STATE_NO_UPDATE + STATE_DIED;
									continue mainLoop;
								}

								float nx = -ly;
								float ny = lx;

								
								if (ny > 0) {
									
									player[VY] = 0;
								} else if (ny < 0) {
									
									player[VY] = 0;
									player[Y] = ay - PLAYER_WALL_BOUNDING;
									onGround = 1;
									
								} else {
									
									player[VX] = (player[VX] * nx < 0) ? 0 : player[VX];
									player[X] = x + nx * PLAYER_WALL_BOUNDING;
								}
							}
						}
					}
				}

				/************
				 * INPUT
				 ************/
				
				if ((gameState & STATE_WAIT_FOR_INPUT) != 0 && key[SPACE] != 0) {
					if ((gameState & STATE_GAME_COMPLETE) != 0) {
						levelCount = -1;
						gameState = STATE_WAIT_FOR_INPUT + STATE_NO_UPDATE + STATE_LEVEL_START;
					} else {
						gameState = 0;
					}
					key[SPACE] = 0;
				}

				if (key[RIGHT] + key[LEFT] != 0 && player[ANIMATIONTIME] - time <= 0) {
					player[ANIMATIONTIME] = time + ANIMATION_TIME;
					animation = (animation + 1) & 1;
				}

				if (onGround == 1) {
					
					player[VX] = (player[VX] + (key[RIGHT] - key[LEFT]) * GROUND_SPEED) * FRICTION;
					player[VY] = JUMP_SPEED * (key[UP] | key[SPACE]);
				} else {
					
					player[VY] += GRAVITY * dt;
					player[VX] += (key[RIGHT] - key[LEFT]) * AIR_SPEED;
					if (key[RIGHT] + key[LEFT] != 0) {
						player[VX] *= FRICTION_AIR;
					}
				}

				if ((gameState & STATE_NO_UPDATE) == 0) {
					player[X] += player[VX] * dt;
					player[Y] += player[VY] * dt;

					
					player[AIMX] = key[MOUSE_X];
					player[AIMY] = key[MOUSE_Y];

					float x3 = player[X];
					float y3 = player[Y];
					float x4 = player[X2];
					float y4 = player[Y2];

					intersect[X] = 0;
					intersect[Y] = 0;

					float closest = 0x1000;

					for (i = LEVEL_DATA; i < level.length; i += COMPONENTS) {
						float x1 = level[i + X];
						float y1 = level[i + Y];
						float x2 = level[i + X2];
						float y2 = level[i + Y2];

						float denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
						if (denom != 0) {
							float t0 = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
							if (0 <= t0 && t0 <= 1) {
								float t1 = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;
								if (t1 >= 0 && t1 < closest) {
									closest = t1;
									intersect[X] = x1 + t0 * (x2 - x1);
									intersect[Y] = y1 + t0 * (y2 - y1);
									intersect[RAY_T] = t1;
									intersect[LINE_T] = t0;
									intersect[LINE_INDEX] = i;
								}
							}
						}
					}

					if (key[MOUSE_BUTTON] != 0) {
						
						
						
						

						int line = (int) intersect[LINE_INDEX];
						if ((int) level[line + DEAD] + (int) level[line + DIE] == 0) {
                            float dx = level[line + DX] * HALF_PORTAL_WIDTH;
							float dy = level[line + DY] * HALF_PORTAL_WIDTH;

                            float x1 = intersect[X] - dx;
                            float y1 = intersect[Y] - dy;
                            float x2 = intersect[X] + dx;
                            float y2 = intersect[Y] + dy;

							if (intersect[LINE_T] * level[line + LENGTH] < HALF_PORTAL_WIDTH) {
								x1 = level[line + X];
								y1 = level[line + Y];
								x2 = x1 + dx * 2;
								y2 = y1 + dy * 2;
							}
							if ((1 - intersect[LINE_T]) * level[line + LENGTH] < HALF_PORTAL_WIDTH) {
								x2 = level[line + X2];
								y2 = level[line + Y2];
								x1 = x2 - dx * 2;
								y1 = y2 - dy * 2;
							}

							int index = COMPONENTS * (key[MOUSE_BUTTON] - 1);

                            portals |= 1 << (key[MOUSE_BUTTON] - 1);
                            boolean apply = true;
                            int other = PORTAL2 - index;
                            if (portals == 3 && level[other + INDEX] == line) {
								float Dx = x1 - level[other + X];
								float Dy = y1 - level[other + Y];
								apply = (Dx * Dx + Dy * Dy >= PORTAL_WIDTH_2);
							}
							if (apply) {
								level[index + X] = x1;
								level[index + Y] = y1;
								level[index + X2] = x2;
								level[index + Y2] = y2;
								level[index + LENGTH] = 2 * HALF_PORTAL_WIDTH;
								level[index + DX] = level[line + DX];
								level[index + DY] = level[line + DY];
								level[index + INDEX] = line;

								
								player[ROTATION] = (float) Math.acos(level[PORTAL1 + DX] * level[PORTAL2 + DX] + level[PORTAL1 + DY] * level[PORTAL2 + DY]) + PI;
								if (level[PORTAL1 + DX] * level[PORTAL2 + DY] + level[PORTAL1 + DY] * level[PORTAL2 + DX] > 0) {
									player[ROTATION] *= -1;
								}
							}
						}
						key[MOUSE_BUTTON] = 0;
					}
				}

				/*********************
				 * RENDER
				 *********************/
				g.setColor(color[COLOR_BACKGROUND]);
				g.fillRect(0, 0, WIDTH, HEIGHT);

				float px = player[X] / 2 + WIDTH / 4;
				float py = player[Y] / 2 + HEIGHT / 4 - 30;

				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				

				
				for (i = LEVEL_DATA; i < level.length; i += COMPONENTS) {
					project(faceX, faceY, level, i, px, py);
					
					
					g.setColor(color[COLOR_WALL + (int) (level[i + DEAD] + 2 * level[i + DIE])]);

					float dot = level[i + DX] * (faceY[3] - faceY[0]) - level[i + DY] * (faceX[3] - faceX[0]);
					if (dot >= 0) {
						g.fillPolygon(faceX, faceY, 4);
						g.setColor(color[COLOR_WALL_BORDER]);
						g.drawPolygon(faceX, faceY, 4);
					}
				}

				
				
				
				
				
				
				
				
				
				
				
				

				
				k = 6;
				for (i = 0; i < solids.length; i++, k = 0) {
					int length = solids[i].length >> 1;
					for (j = 0; j < length; j++) {
						faceX[j] = (int) ((px * zNear - px + solids[i][2 * j + X]) / zNear);
						faceY[j] = (int) ((py * zNear - py + solids[i][2 * j + Y]) / zNear);
					}
					g.setColor(color[COLOR_FACE]);
					g.fillPolygon(faceX, faceY, length);
					g.setColor(color[COLOR_FACE_BORDER]);
					g.drawPolygon(faceX, faceY, length - k);
				}

				
				j = 0;
				for (i = 0; i < 3; i++, j += COMPONENTS) {
					if ((int) level[j + X] != 0) {
						project(faceX, faceY, level, j, px, py);
						g.setColor(color[COLOR_P1 + i]);
						g.fillPolygon(faceX, faceY, 4);
						g.setColor(color[COLOR_FACE_BORDER]);
						g.drawPolygon(faceX, faceY, 4);
					}
				}

				
				g.setColor(color[COLOR_WHITE]);
				g.drawOval((int) player[X] - 2, (int) player[Y] - PLAYER_H, 4, 4);
				g.drawLine((int) player[X], (int) player[Y] - PLAYER_H + 4, (int) player[X], (int) player[Y]);
				g.drawLine((int) player[X], (int) player[Y], (int) player[X] + animation * 4, (int) player[Y] + PLAYER_H);
				g.drawLine((int) player[X], (int) player[Y], (int) player[X] - animation * 4, (int) player[Y] + PLAYER_H);

				
				float aimx = player[AIMX] - player[X];
				float aimy = player[AIMY] - player[Y];
				float aimLength = (float) Math.sqrt(aimx * aimx + aimy * aimy);
				
				
				
				g.drawLine((int) player[X], (int) player[Y] - 2, (int) (player[X] + 10 * aimx / aimLength), (int) (player[Y] - 2 + 10 * aimy / aimLength));

				if (gameState != 0) {
					g.setFont(bigFont);
					if ((gameState & STATE_GAME_COMPLETE) != 0) {
						g.drawString("TESTS COMPLETE", 160, 400);
					}
					if ((gameState & STATE_LEVEL_START) != 0) {
						g.drawString("TEST " + (levelCount + 1), 320, 400);
					}
					if ((gameState & STATE_DIED) != 0) {
						g.drawString("REGENERATING", 220, 400);
					}
					g.setFont(smallFont);
					g.drawString("Press <SPACE> to continue", 320, 420);
				}

				
				appletGraphics.drawImage(screen, 0, 0, null);

				Thread.sleep(10);
			}
		} catch (Exception exc) {
			exc.printStackTrace();
		}
	}

	private static void project(int[] faceX, int[] faceY, float[] level, int index, float px, float py) {
		faceX[0] = (int) ((px * zNear - px + level[index + X]) / zNear);
		faceY[0] = (int) ((py * zNear - py + level[index + Y]) / zNear);
		faceX[1] = (int) ((px * zNear - px + level[index + X2]) / zNear);
		faceY[1] = (int) ((py * zNear - py + level[index + Y2]) / zNear);

		faceX[2] = (int) ((px * zFar - px + level[index + X2]) / zFar);
		faceY[2] = (int) ((py * zFar - py + level[index + Y2]) / zFar);
		faceX[3] = (int) ((px * zFar - px + level[index + X]) / zFar);
		faceY[3] = (int) ((py * zFar - py + level[index + Y]) / zFar);
	}

	@Override
	public boolean handleEvent(Event e) {
        switch (e.id) {
            case Event.KEY_PRESS -> key[e.key & 0xff] = 1;
            case Event.KEY_RELEASE -> key[e.key & 0xff] = 0;
            case Event.MOUSE_DOWN -> key[MOUSE_BUTTON] = 1 + (e.modifiers == 0 ? 0 : 1);
            case Event.MOUSE_UP -> key[MOUSE_BUTTON] = 0;
            case Event.MOUSE_DRAG, Event.MOUSE_MOVE -> {
                key[MOUSE_X] = e.x;
                key[MOUSE_Y] = e.y;
            }
        }
		return false;
	}

	private static final String str_colors = "u0080u8080" 
			+ "u00d0ud0d0" 
			+ "u00c0uc0c0" 
			+ "u0070u7070" 
			+ "u00ffu7070" 
			+ "u00ffub030" 
			+ "u0000ub0ff" 
			+ "u0080uc070" 
			+ "u0040u4040" 
			+ "u0020u2020" 
			+ "u0000u0000" 
			+ "u00ffuffff";

	private static final String frequencies = "u00c2u00cdu00dau00e7u00f4u0103u0112u0123u0134u0146u015au016e";

	private static final String data = "u0010'u004cu0063u0088u00a5u00c4u00f7u0a01u0a03u03cdu034du393eu3935u36b5u391bub99bu3612u3903u3911u0111u0303u4d03u4d39u3e39u3e1eu279eu111eu9120u3920u3939u0339u032eu302eub02cu282cu0bacu032cu0a01u0a03u03cdu03cdu39beu39b5u36b5u391bub99bu3612u3903u3910u020cu0483u03cdu03cdu3928u3928u0fa6u0fa6u3903u3903u2388u2388u1e03u19a4u37a4u321eu3219u3700u0c02u0804u8303ucd03ucd39u3b39u3b1bu341bub439u0339uae35u2e25ua825u1535u000eu010eu8303ucd03ucd39u2839u280fu1e0fu9e11u2611ua639u0339u8323u8c1fu8c1du0319u1801u1883u03cdu034du1e37u9eb7u204du204du392bu392bu1240u12c0u0f17u0f97u12a8u12a8u2223u27a3u2a28u2f28u3903u3903u2288u1d88u1a03u1517u0213u0484u1403u0e83u089au081au03cdu034dub93eu393eu1938u9931ub927u3927u9e07u9e87u2124u2124u3903u3983u1587u2b07u3514u350au2b00";

	
	
	private static final String song_data = "u0101u0101u0404u0404u0101u0101u0404u0404u0202u0202u0505u0505u0202u0202u0505u0505u0000u0303u0000u0404u0707u0303u0606u0000bu0808u0808u0202u0202u0808u0803u0808u0000u0000u0802u0000u0000u0000u0000u0000u0000u2000u0000u0000u0000u0000u0000u0000u0002u2828u6414u0000u0000u0000u0000u0000u0000u2800u0000u0000u0000u0000u0000u0000u2002u2828u6420u0000u0000u0000u0000u0000u0000u1401u0000u0000u0000u0000u0000u0000u0001u2828u7f7fu0000u0000u0000u0000u0000u0000u0a02u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0900u0000u0000u0000u0800u0000u0000u0000u0200u0000u0000u0000u0000u0000u0500u0000u0203u0203u1200u0002u0302u0002u0300u0000u0200u0300u1200u0002u0302u0002u0300u0000u1200u0000u0012ftu0000u0000u0007u0000u1200u0012u0000u0000u1500u0019u0000u0200u0200u0000u0000u0000u0003u0300u0000u0000u0500u0500u0000u0000u0003u0300u0000u0200u0000u1212u0000u0200u0013u0013u0000u0200u0000u1515u0000u0200u0000u1313u0000u0200u0012u2219u0000u0300u0013u0000u0000u0000u1500u1500u0000u0000u1213u0000u0000u0000u0900u0900u0000u0000u0800u0007u0000u0200u0012u0000u0000u0000u0000u0000u0000u1200u1200u0000u0012u1200u0000u0000u0000u1200u1200u0000u0012u1200u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000u0000";

	public P() {
		super();
	}

	
	public static void main(String[] args) {
		JFrame frame = new JFrame("Porta 4K");
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		P applet = new P();
		applet.setPreferredSize(new Dimension(608, 384));
		frame.add(applet, BorderLayout.CENTER);
		frame.setResizable(false);
		frame.pack();
		frame.setLocationRelativeTo(null);
		frame.setVisible(true);
		applet.start();
		applet.run();
	}
}