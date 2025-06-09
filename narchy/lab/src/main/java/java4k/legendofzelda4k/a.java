package java4k.legendofzelda4k;

/*
 * Legend of Zelda 4K
 * Copyright (C) 2011 meatfighter.com
 *
 * This file is part of Legend of Zelda 4K.
 *
 * Legend of Zelda 4K is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Legend of Zelda 4K is distributed in the hope that it will be useful,
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
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Random;

public class a extends GamePanel {

  
  private final boolean[] a = new boolean[32768];

  @Override
  public void start() {
    enableEvents(8);
    new Thread(this).start();
  }

  @Override
  public void run() {

    int i;
    int j;

    Random random = new Random();
    AffineTransform affineTransform = new AffineTransform();


    BufferedImage candleLight = new BufferedImage(112, 112, 2);
    for(i = 0; i < 112; i++) {
      for(j = 0; j < 112; j++) {
        float X = 55.5f - j;
        float Y = 55.5f - i;
        float R2 = X * X + Y * Y;
        float F = R2 / 1600;        
        candleLight.setRGB(i, j, (R2 < 1600 ? (int)(255 * F * F) : 255) << 24);
      }
    }

    
    int[] pixels = new int[16];
    int[] pixels2 = new int[16];
    BufferedImage[] sprites = new BufferedImage[64];
    int y;
    int x;
    final String S = "\u002a\ua800\u00aa\uaa00\u0c95\u5630\u0c55\u5530\u0f7b\uedf0\u0f77\uddf0\u07ff\uffc0\u06bd\u7e80\u15af\uf554\u17aa\u9575\u1fe9\u75fd\u3f55\ub575\u0ea9\u7575\u00aa\ub555\u0054\u1ffc\u0000\u1500\u002a\ua800\u00aa\uaa00\u0c95\u5630\u0c55\u5530\u0f7b\uedf0\u0f77\uddf0\u07ff\uffc0\u05bd\u7e00\u0eaf\ud550\u0eaa\u55d4\u01a5\ud7f4\u0256\ud5d4\u02a5\ud5d4\u006a\ud554\u0054\u3ff0\u0054\u0000\u0000\u0000\u002a\ua140\u00aa\ua9d0\u02a5\u5ad4\u3295\u56f5\u3e5b\uedf5\u0e77\uddd7\u03fd\u7e5c\u017d\u7a70\u055f\ueac0\u055a\ua900\u0955\u6600\u0a56\u5a00\u1a7d\u6940\u56bf\ua150\u553f\u0000\n\ua000\u00aa\u9540\u0aba\u5550\u2abd\u5540\u22bf\u7ec1\u025f\u7dfd\u0057\uffc1\u002a\ubfc1\u01aa\ua95d\u056f\uea5d\u055f\uea51\u055f\ua901\u025a\u5501\u0aaa\uaa00\u0015\u4000\u0015\u5000\u0000\u0000\n\ua000\u00aa\u9540\u0aba\u5550\u2abd\u5540\u22bf\u7ec0\u025f\u7dfc\u0057\uffc4\u002a\ubfc4\u025a\ufd74\u0155\ufe74\u0955\ufa44\u0a55\ua904\u16aa\u5584\u15aa\uaa50\u0540\u0540\u0000\u0000\u0002\ua800\u002a\ua550\u00ae\u9554\u00af\u5550\u02af\udfb0\u0a97\udf7f\u0815\ufff0\u002a\uaffc\u00a5\u55fc\u0095\u55f0\u0295\u5580\u0aa5\u6a50\u16aa\u9550\u15aa\uaa94\u0540\u0055\u002a\ua800\u00aa\uaa00\u0caa\uaa30\u0eaa\uaab0\u0daa\uaa70\u0f5a\ua5f0\u0356\u95c0\u0195\u5640\u016a\uaa50\u0d6a\uaa50\u0d6a\ua940\u0f95\u5680\u02aa\uaa80\u006a\ua540\u0014\u1540\u0000\u0500\u0002\uaa40\n\uaa8c\u002a\uaa9c\u032a\uaabc\u03da\uaa74\uc0d5\ua974\u70e5\u6594\u5d59\u5694\u575a\uaa90\u55da\uaa90\u157a\uaa60\u05d5\u55a4\u03aa\uaaa5\u016a\u8015\u0550\u0000\u0550\u0000\u0002\ua000\u0003\uf000\u0002\ua000\u0023\uf200\u002a\uaa00\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0001\u5000\u0000\u4000\uffff\ufdaf\uffff\uf6ab\uff5a\u96bb\ufdaa\ub6ae\uf5aa\ub6ae\uf6aa\uadae\ud6a9\uadaa\ud569\uadaa\ud669\uadab\ud665\uadab\ud665\uadab\ud566\ua9aa\ud96a\ua9aa\ufa56\uaaaa\u555a\u9aab\uf555\u5555\uffff\ufdaf\uffff\uf6ab\uff59\u96bb\ufda6\ub6a6\uf5a9\ub69e\uf6aa\u6d6e\ud6a9\u9d9a\ud569\u6da6\ud669\u9d9b\ud665\ua56b\ud665\u9dab\ud566\u69aa\ud969\ua9aa\ufa56\u6aaa\u555a\u9aab\uf555\u5555\u0000\u0410\u0005\u1450\u0001\u5550\u0055\uf554\u405f\ufe54\u401d\u5d59\u5575\u7595\u557f\uf556\u557f\uf556\u5575\u7595\u401d\u5d59\u405f\ufe54\u0055\uf554\u0001\u5550\u0005\u1450\u0000\u0410\u0001\u0104\u0001\u4514\u0041\u5550\u0055\uf554\u041f\ufe54\u051d\u5d59\u0575\u7595\u057f\uf556\u057f\uf556\u0575\u7595\u051d\u5d59\u041f\ufe54\u0055\uf554\u0041\u5550\u0001\u4514\u0001\u0104\u0001\u0000\u0005\u5000\u0006\u5400\u0015\u9400\u0006\u9000\u0000\u0000\u000f\uf000\u000f\uf000\u000f\uf000\u008f\uf000\u008f\uf000\u008f\uf000\u008f\uf000\u008f\uf000\u00aa\uaa00\u002a\ua800\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0014\u5000\u0055\u5400\u0055\u5400\u0055\u5400\u0055\u5400\u0015\u5000\u0005\u4000\u0001\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0006\ua000\u001a\uac00\u0068\u1b00\u0060\u0700\u0060\u0700\u006a\uaa00\u0056\uab00\u0001\u8000\u0001\u8000\u0001\u8000\u0001\u8000\u0001\uac00\u0001\ua800\u0001\ua000\u0001\ua800\u0001\u8000\u0000\u0a00\u0000\u0a00\u0000\u2a80\u0000\u2a80\u0000\uaaa0\u0000\uaaa0\u0002\uaaa8\u0002\uaaa8\n\uaaaa\n\uaaaa\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u7fff\ufffe\u5fff\ufffb\u57ff\uffef\u56aa\uaabf\u56aa\uaabf\u56a9\u6abf\u56a5\u5abf\u56a5\u5abf\u56a9\u6abf\u56a9\u6abf\u56a9\u6abf\u56a9\u6abf\u56aa\uaabf\u5955\u557f\u6555\u555f\u9555\u5557\u5555\u5555\u5555\u5555\u5565\u5565\u5555\u5555\u5555\u5555\u5555\u5555\u5556\u5556\u5555\u5555\u5555\u5555\u5655\u5655\u5555\u5555\u5555\u5555\u5555\u5555\u5595\u5595\u5555\u5555\u5555\u5555\u6aaa\u6aaa\ueeae\ueeae\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueaaa\ueeae\ueeae\ueaaa\ueaaa\ufffd\ufffd\u003f\ufc00\u00be\ube00\u00bf\ufe00\u03f2\u8fc0\u03e2\u8bc0\u03ea\uabc0\u03fb\uefc0\u03d6\u97c0\u0f95\u56f0\u0fa5\u5af0\u0f9e\ub6f0\u0a55\u55a0\u0f55\u55f0\u0d55\u5570\u0d55\u5570\u1555\u5554\u4fff\uffd4\u4aaa\uaaa4\u4aaa\uaaa4\u4aaa\uaaa4\u4fff\uffd4\u4aaa\uaaa4\u4aaa\uaaa4\u4aaa\uaaa4\u4fff\uffd4\u4aaa\uaaa4\u4aaa\uaaa4\u4aaa\uaaa4\u4fff\uffd4\u4aaa\uaaa4\u4aaa\uaaa4\u4aaa\uaaa4\uffff\u8003\u8003\u924b\u0003\u124b\u0003\u924b\u8003\u8003\ubfff\ubfff\ubfff\ubfff\ubfff\ubfff\u8003\u8003\u8003\u8003\uc003\uc003\uc003\u8003\u8003\u8103\u8403\u1153\u8403\u8103\uc003\ufff7\ufff7\ufff7\ufff7\u0003\u0403\u1003\u0403\u1003\u0403\u0003\uffef\uffef\uffef\uffef\u0003\u1101\u0401\u1109\u0401\u1101\u0003\uffff\uffff\uffff\uffff\uffff\uf3f3\ue1e0\u0000\ue000\ue003\ue007\uffff\uffff\uffff\uffff\u0003\ufe01\ufe01\ufe01\ufe01\ufe01\u0003\uffff\uffff\uffff\ufe7f\u0003\ufff1\ufff1\ufff0\ufff1\ufff1\u0003\uc19f\uffff\uffff\uffff\uc000\u8078\u0700\u0078\u0700\u8078\uc000\ucfff\u4fff\u4fff\u4fff\uc000\uc100\u8444\u0100\u0444\u8100\uc000\uffff\uffff\uffff\ud001\u17fd\uf501\u0577\ufd10\uc1d7\u5451\ud7fd\u1001\uf7ff\uf7ff\uf7ff\uf000\uffff\uffff\uc3ff\uc3ff\uc3ff\uc000\uffff\uffff\uffff\ufe7f\uc000\u8003\u0003\u0003\u0003\u8003\uc000\uc00f\uffff\uffff\uffff\uc603\u9029\u0000\u1028\u0000\u9029\uc603\ufe7f\ufe70\ufe77\ufe37\ufe07\ufe03\ufe01\u0000\u0000\uc003\uc003\uc003\uffff\uffff\u8507\ub574\ua415\ubff4\u8101\ubd7f\ua540\uad5f\u2040\uffff\uffff\uffff\ue7c3\ue7c3\u67c1\u0001\u6001\ue003\ue003\uffff\uffff\uffff\uffff\uc003\uc001\ud008\uc000\ud008\uc001\uc003\uffff\uffff\uffff\u8007\u8003\ua211\u8004\uaa50\u8004\ua211\u8003\uaa57\uaa57\uaa57\uaa57\u0003\u2a41\u0015\u2a40\u0014\u2a41\u0003\uffff\uffff\uffff\u87e1\u8db1\u8991\u8991\u8e71\u83c1\u83c1\u8001\u83c0\uffff\uffff\u8001\u8001\u8841\ua214\u8840\ua214\u8841\u8001\uff7f\uff7f\uff7f\u8a05\uaafd\uaa01\uabf5\ub855\uaadd\uaa45\uaf75\ua021\ubdf7\ubdf7\u8101\ubffd\u8005\ubff5\ua411\uaddf\ua101\uaffd\u8801\ufdff\ufdff\ufdff\uf03e\ue03c\ue238\ue530\ue200\ue000\uf030\uffff\uffff";
    final int SPRITE_LADDER = 21;
    final int SPRITE_PRINCESS = 20;
    final int SPRITE_BRIDGE = 19;
    final int SPRITE_WATER = 18;
    final int SPRITE_CRACKED_ROCK = 10;
    final int SPRITE_ROCK = 9;
    final int SPRITE_FLIPPED = 32;
    final int SPRITE_COUNT = 22;
    for(i = 0; i < SPRITE_COUNT; i++) {
      sprites[i] = new BufferedImage(16, 16, 2);
      sprites[i + SPRITE_FLIPPED] = new BufferedImage(16, 16, 2);
      for(y = 0; y < 16; y++) {
        for(x = 0; x < 16; x++) {
          j = (S.charAt((x < 8 ? 1 : 0) + (y << 1) + (i << 5))
              >> ((x & 7) << 1)) & 3;
          pixels2[15 - x] = pixels[x] = j == 0 ? 0 : 0xFF000000
              | (i < SPRITE_ROCK ? (j == 1 ? 0x994E00 : j == 2
                      ? 0x88D800 : 0xEA9E22)
                  : i <= SPRITE_CRACKED_ROCK
                      ? (j == 1 ? 0 : j == 2 ? 0x994E00 : 0xF7D8A5)
                  : i < SPRITE_WATER
                      ? (j == 1 ? 0xB53120 : j == 2 ? 0xEA9E22 : 0xFFFFFF)
                  : i < SPRITE_BRIDGE
                      ? (j == 1 ? 0x4240FF : 0x0D9300)
                  : i < SPRITE_PRINCESS
                      ? (j == 1 ? 0x4240FF : j == 2 ? 0x994E00 : 0)
                  : i < SPRITE_LADDER
                      ? (j == 1 ? 0xB53120 : j == 2 ? 0xEA9E22 : 0x994E00)
                      : (j == 1 ? 0 : j == 2 ? 0x4240FF : 0x994E00));
        }
        sprites[i].setRGB(0, y, 16, 1, pixels, 0, 16);
        sprites[i + SPRITE_FLIPPED].setRGB(0, y, 16, 1, pixels2, 0, 16);
      }
    }


    int[][] map = new int[81][81];
    int k;
    final int MAP_OFFSET = SPRITE_COUNT << 5;
    final int MAP_WALL = 8;
    final int MAP_EMPTY = 0;
    for(i = 0; i < 5; i++) {
      for(j = 0; j < 55; j++) {
        for(k = 0; k < 16; k++) {
          map[j][(i << 4) + k] = ((S.charAt(MAP_OFFSET + j + 55 * i) >> k) & 1)
              == 1 ? MAP_WALL : MAP_EMPTY;
        }
      }
    }


    final int MAP_SWORD = 1;
    map[45][31] = MAP_SWORD;
    final int MAP_CANDLE = 2;
    map[51][3] = MAP_CANDLE;
    final int MAP_KEY = 3;
    map[14][51] = MAP_KEY;
    map[27][60] = MAP_KEY;
    map[17][43] = MAP_KEY;
    map[3][53] = MAP_KEY;
    final int MAP_LOCK = 9;
    map[34][3] = MAP_LOCK;
    map[5][29] = MAP_LOCK;
    map[4][43] = MAP_LOCK;
    map[8][70] = MAP_LOCK;
    final int MAP_PRINCESS = 4;
    map[3][77] = MAP_PRINCESS;
    final int MAP_CRACKED_WALL = 10;
    map[27][15] = MAP_CRACKED_WALL;


    final int MAP_WATER = 11;
    for(i = 0; i < 23; i++) {
      if (i != 6) {
        map[i + 32][5] = map[i + 32][6] = MAP_WATER;
      }
    }
    for(i = 0; i < 3; i++) {
      for(j = 0; j < 7; j++) {
        map[i + 32][j + 7] = MAP_WATER;
      }
    }
    final int MAP_LADDER = 7;
    for(i = 0; i < 4; i++) {
      for(j = 0; j < 6; j++) {
        map[i + 25][j + 53] = MAP_WATER;
        map[j + 10][14] = MAP_LADDER;
      }
    }


    final int MAP_BRIDGE = 6;
    map[38][5] = map[38][6] = MAP_BRIDGE;

    BufferedImage image = new BufferedImage(256, 240, 1);
    Graphics2D g = (Graphics2D)image.getGraphics();
    Graphics2D g2 = null;

    long nextFrameStartTime = System.nanoTime();
    ArrayList<int[]> enemies = new ArrayList<>();
    boolean won = false;
    boolean fading = true;
    boolean flash = false;
    boolean scrolling = true;
    boolean acquiredKey = false;
    boolean acquiredCandle = false;
    boolean acquiredSword = true;
    boolean attackReleased = true;
    int attacking = 0;
    int fadeDelta = 1;
    int fadeIntensity = 255;
    int targetCameraY = 0;
    int cameraY = 0;
    int targetCameraX = 0;
    int cameraX = 0;
    int playerStunned = 0;
    int playerHealth = 0;
    int playerWalkCount = 0;
    int playerLastDirection = 0;
    int playerDirection = 0;
    int playerSprite = 0;
    int playerY = 0;
    int playerX = 0;
    int counter = 0;
    int s = 0;
    int r = 0;
    Color COLOR_RED = new Color(0xB53120);
    Color COLOR_MAP_2 = new Color(0x88D800);
    Color COLOR_MAP_1 = new Color(0x666666);
    Color COLOR_FLOOR = new Color(0xF7D8A5);
    final int ENEMY_DYING = 5;
    final int ENEMY_COUNTER = 4;
    final int ENEMY_DIRECTION = 3;
    final int ENEMY_SPRITE = 2;
    final int ENEMY_Y = 1;
    final int ENEMY_X = 0;
    final int SPRITE_LOCK = 17;
    final int SPRITE_TRIFORCE = 16;
    final int SPRITE_KEY = 15;
    final int SPRITE_HEART = 14;
    final int SPRITE_CANDLE = 13;
    final int SPRITE_OCTOROC_2 = 12;
    final int SPRITE_OCTOROC_1 = 11;
    final int SPRITE_SWORD = 8;
    final int SPRITE_LINK_LEFT_LUNGE = 5;
    final int SPRITE_LINK_RIGHT_LUNGE = SPRITE_LINK_LEFT_LUNGE + SPRITE_FLIPPED;
    final int SPRITE_LINK_LEFT_2 = 4;
    final int SPRITE_LINK_RIGHT_2 = SPRITE_LINK_LEFT_2 + SPRITE_FLIPPED;
    final int SPRITE_LINK_LEFT_1 = 3;
    final int SPRITE_LINK_RIGHT_1 = SPRITE_LINK_LEFT_1 + SPRITE_FLIPPED;
    final int SPRITE_LINK_UP_LUNGE = 7;
    final int SPRITE_LINK_UP_1 = 6;
    final int SPRITE_LINK_UP_2 = SPRITE_LINK_UP_1 + SPRITE_FLIPPED;
    final int SPRITE_LINK_DOWN_LUNGE = 2;
    final int SPRITE_LINK_DOWN_2 = 1;
    final int SPRITE_LINK_DOWN_1 = 0;
    final int MAP_HEART = 5;
    final int DIRECTION_UP = 3;
    final int DIRECTION_LEFT = 2;
    final int DIRECTION_DOWN = 1;
    final int DIRECTION_RIGHT = 0;
    final int VK_ATTACK = 0x42;
    final int VK_DOWN = 0x28;
    final int VK_UP = 0x26;
    final int VK_RIGHT = 0x27;
    final int VK_LEFT = 0x25;
    while(true) {

      int q;
      int p;
      do {
        nextFrameStartTime += 16666667;

        

        if (fading) {
          fadeIntensity += fadeDelta;
          if (fadeIntensity > 255) {
            
            fadeIntensity = 255;
            fadeDelta = -8;

            
            playerX = 640;
            playerY = 792;
            playerSprite = SPRITE_LINK_UP_1;
            playerLastDirection = playerDirection = DIRECTION_UP;
            playerWalkCount = 1;
            playerHealth = 6;
            playerStunned = 0;
            attacking = 0;
            cameraX = targetCameraX = 512;
            cameraY = targetCameraY = 704;
            enemies.clear();
          } else if (fadeIntensity < 0) {
            fading = false;
          }
          continue;
        }

        counter++;

        if (!(attackReleased || a[VK_ATTACK])) {
          attackReleased = true;
        }

        
        
        if (targetCameraX < cameraX) {
          cameraX -= 4;
          if ((playerX & 15) != 0) {
            playerX--;
          }
          continue;
        }
        if (targetCameraX > cameraX) {
          cameraX += 4;
          if ((playerX & 15) != 0) {
            playerX++;
          }
          continue;
        }
        if (targetCameraY < cameraY) {
          cameraY -= 4;
          if ((playerY & 15) != 0) {
            playerY--;
          }
          continue;
        }
        if (targetCameraY > cameraY) {
          cameraY += 4;
          if ((playerY & 15) != 0) {
            playerY++;
          }
          continue;
        }

        if (attacking > 0) {
          float angle = 1.57f * playerDirection;
          r = 8 + playerX + (int)(16 * Math.cos(angle));
          s = 8 + playerY + (int)(16 * Math.sin(angle));
          if (map[s >> 4][r >> 4] == MAP_CRACKED_WALL) {
            map[s >> 4][r >> 4] = MAP_EMPTY;
            int[] enemy = new int[32];
            enemy[ENEMY_X] = (r & ~15) + 8;
            enemy[ENEMY_Y] = (s & ~15) + 8;
            enemy[ENEMY_SPRITE] = SPRITE_OCTOROC_1;
            enemy[ENEMY_DYING] = 1;
            enemies.add(enemy);
          }
        }

        
        for(i = enemies.size() - 1; i >= 0; i--) {
          int[] enemy = enemies.get(i);
          if (enemy[ENEMY_DYING] > 0) {
            if (++enemy[ENEMY_DYING] == 16) {
              
              enemies.remove(i);

              if (random.nextInt(7) == 0) {
                x = enemy[ENEMY_X] >> 4;
                y = enemy[ENEMY_Y] >> 4;
                if (map[y][x] == MAP_EMPTY) {
                  map[y][x] = MAP_HEART;
                }
              }
            }
          } else {
            if ((counter & 7) == 0) {
              enemy[ENEMY_SPRITE] = enemy[ENEMY_SPRITE] != SPRITE_OCTOROC_1
                  ? SPRITE_OCTOROC_1 : SPRITE_OCTOROC_2;
            }
            if ((counter & 1) == 0) {
              x = enemy[ENEMY_X];
              y = enemy[ENEMY_Y];
              switch (enemy[ENEMY_DIRECTION]) {
                case DIRECTION_UP -> y--;
                case DIRECTION_DOWN -> y++;
                case DIRECTION_LEFT -> x--;
                default -> x++;
              }
              if (map[(y - 8) >> 4][x >> 4] == MAP_EMPTY
                  && map[(y + 7) >> 4][x >> 4] == MAP_EMPTY
                  && map[y >> 4][(x + 7) >> 4] == MAP_EMPTY
                  && map[(y - 8) >> 4][(x - 8) >> 4] == MAP_EMPTY
                  && x >= cameraX + 8 && x < cameraX + 248
                  && y >= cameraY + 8 && y < cameraY + 168) {
                enemy[ENEMY_X] = x;
                enemy[ENEMY_Y] = y;
              } else {
                enemy[ENEMY_DIRECTION] = random.nextInt(4);
                enemy[ENEMY_COUNTER] = random.nextInt(128);
              }
              if (--enemy[ENEMY_COUNTER] < 0 && (y & 15) == 8 && (x & 15) == 8) {
                enemy[ENEMY_DIRECTION] = random.nextInt(4);
                enemy[ENEMY_COUNTER] = random.nextInt(128);
              }

              if (attacking > 0) {
                x -= r;
                y -= s;
                if (x * x + y * y < 128) {
                  
                  ++enemy[ENEMY_DYING];
                  continue;
                }
              }

              
              if (playerStunned == 0 && attacking == 0
                  && playerX <= enemy[ENEMY_X] + 4
                  && playerX >= enemy[ENEMY_X] - 20
                  && playerY <= enemy[ENEMY_Y] + 4
                  && playerY >= enemy[ENEMY_Y] - 20) {
                
                playerStunned = 64;
                if (--playerHealth == 0) {
                  
                  fading = true;
                  fadeIntensity = 0;
                  fadeDelta = 2;
                }
              }

              if (enemy[ENEMY_X] < cameraX
                  || enemy[ENEMY_X] >= cameraX + 256
                  || enemy[ENEMY_Y] < cameraY
                  || enemy[ENEMY_Y] >= cameraY + 176) {
                
                enemies.remove(i);
              }
            }
          }
        }

        if (attacking > 0) {
          attacking--;
            playerSprite = switch (playerDirection) {
                case DIRECTION_UP -> SPRITE_LINK_UP_LUNGE;
                case DIRECTION_DOWN -> SPRITE_LINK_DOWN_LUNGE;
                case DIRECTION_LEFT -> SPRITE_LINK_LEFT_LUNGE;
                default -> SPRITE_LINK_RIGHT_LUNGE;
            };
        } else if (acquiredSword && attackReleased && a[VK_ATTACK]) {
          attackReleased = false;
          attacking = 10;
          playerWalkCount = 0;
        } else {
          
          i = 1 + (counter & 1);
          x = playerX;
          y = playerY;
          if (playerStunned > 56) {
            switch (playerDirection) {
              case DIRECTION_UP -> y += 4;
              case DIRECTION_DOWN -> y -= 4;
              case DIRECTION_LEFT -> x += 4;
              default -> x -= 4;
            }
          } else if (a[VK_UP]) {
            playerWalkCount++;
            y -= i;
            playerDirection = DIRECTION_UP;
          } else if (a[VK_DOWN]) {
            playerWalkCount++;
            y += i;
            playerDirection = DIRECTION_DOWN;
          } else if (a[VK_LEFT]) {
            playerWalkCount++;
            x -= i;
            playerDirection = DIRECTION_LEFT;
          } else if (a[VK_RIGHT]) {
            playerWalkCount++;
            x += i;
            playerDirection = DIRECTION_RIGHT;
          }
          if (playerStunned > 0) {
            playerStunned--;
          }

          if ((playerWalkCount & 7) == 0
              || playerLastDirection != playerDirection) {
            playerWalkCount++;
            playerLastDirection = playerDirection;


              playerSprite = switch (playerDirection) {
                  case DIRECTION_UP -> playerSprite != SPRITE_LINK_UP_1
                          ? SPRITE_LINK_UP_1 : SPRITE_LINK_UP_2;
                  case DIRECTION_DOWN -> playerSprite != SPRITE_LINK_DOWN_1
                          ? SPRITE_LINK_DOWN_1 : SPRITE_LINK_DOWN_2;
                  case DIRECTION_LEFT -> playerSprite != SPRITE_LINK_LEFT_1
                          ? SPRITE_LINK_LEFT_1 : SPRITE_LINK_LEFT_2;
                  default -> playerSprite != SPRITE_LINK_RIGHT_1
                          ? SPRITE_LINK_RIGHT_1 : SPRITE_LINK_RIGHT_2;
              };
          }

          
          if (playerDirection == DIRECTION_UP
                || playerDirection == DIRECTION_DOWN) {
            i = x & 7;
            if (i != 0) {
              if (i < 4) {
                x--;
              } else {
                x++;
              }
            }
          } else {
            i = y & 7;
            if (i != 0) {
              if (i < 4) {
                y--;
              } else {
                y++;
              }
            }
          }

          
          p = map[(y + 8) >> 4][x >> 4];
          q = map[(y + 15) >> 4][x >> 4];
          r = map[(y + 8) >> 4][(x + 14) >> 4];
          s = map[(y + 15) >> 4][(x + 14) >> 4];
          if (acquiredKey) {
            
            if (p == MAP_LOCK) {
              map[(y + 8) >> 4][x >> 4] = MAP_EMPTY;
              acquiredKey = false;
            } else if (q == MAP_LOCK) {
              map[(y + 15) >> 4][x >> 4] = MAP_EMPTY;
              acquiredKey = false;
            } else if (r == MAP_LOCK) {
              map[(y + 8) >> 4][(x + 14) >> 4] = MAP_EMPTY;
              acquiredKey = false;
            } else if (s == MAP_LOCK) {
              map[(y + 15) >> 4][(x + 14) >> 4] = MAP_EMPTY;
              acquiredKey = false;
            }
          }

          
          if (p < MAP_WALL && q < MAP_WALL && r < MAP_WALL && s < MAP_WALL) {
            playerX = x;
            playerY = y;
          }

          
          i = (y + 12) >> 4;
          j = (x + 8) >> 4;
          k = map[i][j];
          if (k != MAP_EMPTY && k < MAP_BRIDGE) {
            map[i][j] = MAP_EMPTY;
            if (k == MAP_SWORD) {
              acquiredSword = true;
            } else if (k == MAP_CANDLE) {
              acquiredCandle = true;
            } else if (k == MAP_KEY) {
              acquiredKey = true;
            } else if (k == MAP_PRINCESS) {
              won = true;
            } else if (playerHealth < 6) {
              playerHealth++;
            }
          }

          
          targetCameraX = (playerX + 8) & ~0xFF;
          targetCameraY = ((playerY + 12) / 176) * 176;

          if (cameraX != targetCameraX || cameraY != targetCameraY) {
            if (!scrolling) {
              scrolling = true;

              
              for(i = 3; i >= 0; i--) {
                do {
                  x = targetCameraX + ((2 + random.nextInt(12)) << 4);
                  y = targetCameraY + ((2 + random.nextInt(7)) << 4);
                } while(map[(y >> 4)][(x >> 4)] != MAP_EMPTY);
                int[] enemy = new int[32];
                enemy[ENEMY_X] = x + 8;
                enemy[ENEMY_Y] = y + 8;
                enemy[ENEMY_SPRITE] = SPRITE_OCTOROC_1;
                enemy[ENEMY_DIRECTION] = random.nextInt(4);
                enemy[ENEMY_COUNTER] = random.nextInt(128);
                enemies.add(enemy);
              }
            }
          } else {
            scrolling = false;
          }
        }

        

      } while(nextFrameStartTime < System.nanoTime());

      

      if (won) {
        
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, 256, 240);
        g.drawImage(sprites[SPRITE_TRIFORCE], 123, 84, null);
        g.drawImage(sprites[SPRITE_TRIFORCE], 118, 94, null);
        g.drawImage(sprites[SPRITE_TRIFORCE], 128, 94, null);
        g.drawImage(sprites[SPRITE_PRINCESS], 112, 112, null);
        g.drawImage(sprites[SPRITE_LINK_DOWN_1], 128, 112, null);
      } else {

        
        g.setColor(COLOR_FLOOR);
        g.fillRect(0, 64, 256, 176);

        
        x = cameraX >> 4;
        y = cameraY >> 4;
        r = playerX >> 4;
        s = playerY >> 4;
        int z = cameraX & 15;
        k = cameraY & 15;
        for(i = 0; i < 12; i++) {
          for(j = 0; j < 17; j++) {
            p = j + x;
            q = i + y;
            int u;
            if (((p >= 64 && q >= 22) || (p >= 16 && p < 64 && q < 11))
                  && (!acquiredCandle
                      || (p < r - 2 || p > r + 3 || q < s - 2 || q > s + 3))) {
              g.setColor(Color.BLACK);
              g.fillRect((j << 4) - z, (i << 4) - k + 64, 16, 16);
            } else if ((u = map[q][p]) != MAP_EMPTY) {
              g.drawImage(sprites[
                  u == MAP_WALL ? SPRITE_ROCK
                      : u == MAP_WATER ? SPRITE_WATER
                      : u == MAP_BRIDGE ? SPRITE_BRIDGE
                      : u == MAP_LADDER ? SPRITE_LADDER
                      : u == MAP_HEART ? SPRITE_HEART
                      : u == MAP_LOCK ? SPRITE_LOCK
                      : u == MAP_KEY ? SPRITE_KEY
                      : u == MAP_CRACKED_WALL ? SPRITE_CRACKED_ROCK
                      : u == MAP_SWORD ? SPRITE_SWORD
                      : u == MAP_CANDLE ? SPRITE_CANDLE
                      : SPRITE_PRINCESS],
                  (j << 4) - z, (i << 4) - k + 64, null);
            }
          }
        }

        if (attacking > 0) {
          
          i = playerX - cameraX + 8;
          j = playerY - cameraY + 72;
          float angle = 1.57f * (playerDirection - 1);
          switch (playerDirection) {
            case DIRECTION_UP -> i -= 2;
            case DIRECTION_DOWN -> i += 2;
            case DIRECTION_LEFT -> j += 2;
            default -> j++;
          }
          g.translate(i, j);
          g.rotate(angle);
          g.drawImage(sprites[SPRITE_SWORD], -8,
              attacking <= 4 ? (attacking << 1) - 4 : 4, null);
          g.setTransform(affineTransform);
        }

        flash = !flash;
        if (flash || playerStunned == 0) {
          
          g.drawImage(sprites[playerSprite],
              playerX - cameraX, playerY - cameraY + 64, null);
        }

        if ((x >= 64 && y >= 22) || (x >= 16 && x <= 48 && y == 0)) {
          
          g.drawImage(candleLight, playerX - cameraX - 48,
              playerY - cameraY + 16, null);
        }

        
        for(i = enemies.size() - 1; i >= 0; i--) {
          int[] enemy = enemies.get(i);
          g.translate(enemy[ENEMY_X] - cameraX, enemy[ENEMY_Y] - cameraY + 64);
          if (enemy[ENEMY_DYING] > 0) {
            j = enemy[ENEMY_DYING] < 8 ? enemy[ENEMY_DYING]
                : 15 - enemy[ENEMY_DYING];
            g.setColor(enemy[ENEMY_DYING] < 8 ? Color.WHITE : COLOR_RED);
            g.drawLine(-j, -j, j, j);
            g.drawLine(-j, j, j, -j);
            g.drawLine(0, -j, 0, j);
            g.drawLine(-j, 0, j, 0);
          } else {
            g.rotate(1.57f * enemy[ENEMY_DIRECTION]);
            g.drawImage(sprites[enemy[ENEMY_SPRITE]], -8, -8, null);
          }
          g.setTransform(affineTransform);
        }

        
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, 256, 64);

        
        g.setColor(COLOR_MAP_1);
        g.fillRect(16, 24, 64, 32);
        g.setColor(COLOR_MAP_2);
        g.fillRect(15 + ((cameraX + 128) / 20),
            23 + (((cameraY + 88) << 1) / 55), 3, 3);

        if (acquiredSword) {
          
          g.drawImage(sprites[SPRITE_SWORD], 176, 24, null);
        }
        if (acquiredCandle) {
          
          g.drawImage(sprites[SPRITE_CANDLE], 192, 24, null);
        }
        if (acquiredKey) {
          
          g.drawImage(sprites[SPRITE_KEY], 208, 24, null);
        }

        
        for(i = 0; i < playerHealth; i++) {
          g.drawImage(sprites[SPRITE_HEART], 171 + (i << 3), 44, null);
        }

        if (fading) {
          
          g.setColor(new Color(0, 0, 0, fadeIntensity));
          g.fillRect(0, 0, 256, 240);
        }
      }

      

      
      if (g2 == null) {
        g2 = (Graphics2D)getGraphics();
        requestFocus();
      } else {
        g2.drawImage(image, 0, 0, 512, 480, null);
      }

      
      while(nextFrameStartTime - System.nanoTime() > 0) {
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
      k = k == VK_W ? VK_UP : k == VK_D ? VK_RIGHT : k == VK_A ? VK_LEFT
          : k == VK_S ? VK_DOWN : k;
      final int VK_ATTACK = 0x42;
      a[(k >= VK_LEFT && k <= VK_DOWN) ? k : VK_ATTACK]
          = keyEvent.getID() != 402;
    }
  }

  
  public static void main(String[] args) throws Throwable {
    JFrame frame = new JFrame("Legend of Zelda 4K");
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    a applet = new a();
    applet.setPreferredSize(new Dimension(512, 480));
    frame.add(applet, BorderLayout.CENTER);
    frame.setResizable(false);
    frame.pack();
    frame.setLocationRelativeTo(null);
    frame.setVisible(true);
    Thread.sleep(250);
    applet.start();
  }
}
