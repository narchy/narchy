/*
 * Copyright (C) 1997-2001 Id Software, Inc.
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.
 * 
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA 02111-1307, USA.
 *  
 */



package jake2.game.monsters;

import jake2.Defines;
import jake2.client.M;
import jake2.game.*;
import jake2.util.Lib;
import jake2.util.Math3D;

public class M_Infantry {

    

    public static final int FRAME_gun02 = 0;

    private static final int FRAME_stand01 = 1;

    public static final int FRAME_stand02 = 2;

    public static final int FRAME_stand03 = 3;

    public static final int FRAME_stand04 = 4;

    public static final int FRAME_stand05 = 5;

    public static final int FRAME_stand06 = 6;

    public static final int FRAME_stand07 = 7;

    public static final int FRAME_stand08 = 8;

    public static final int FRAME_stand09 = 9;

    public static final int FRAME_stand10 = 10;

    public static final int FRAME_stand11 = 11;

    public static final int FRAME_stand12 = 12;

    public static final int FRAME_stand13 = 13;

    public static final int FRAME_stand14 = 14;

    public static final int FRAME_stand15 = 15;

    public static final int FRAME_stand16 = 16;

    public static final int FRAME_stand17 = 17;

    public static final int FRAME_stand18 = 18;

    public static final int FRAME_stand19 = 19;

    public static final int FRAME_stand20 = 20;

    public static final int FRAME_stand21 = 21;

    public static final int FRAME_stand22 = 22;

    public static final int FRAME_stand23 = 23;

    public static final int FRAME_stand24 = 24;

    public static final int FRAME_stand25 = 25;

    public static final int FRAME_stand26 = 26;

    public static final int FRAME_stand27 = 27;

    public static final int FRAME_stand28 = 28;

    public static final int FRAME_stand29 = 29;

    public static final int FRAME_stand30 = 30;

    public static final int FRAME_stand31 = 31;

    public static final int FRAME_stand32 = 32;

    public static final int FRAME_stand33 = 33;

    public static final int FRAME_stand34 = 34;

    public static final int FRAME_stand35 = 35;

    public static final int FRAME_stand36 = 36;

    public static final int FRAME_stand37 = 37;

    public static final int FRAME_stand38 = 38;

    public static final int FRAME_stand39 = 39;

    public static final int FRAME_stand40 = 40;

    public static final int FRAME_stand41 = 41;

    public static final int FRAME_stand42 = 42;

    public static final int FRAME_stand43 = 43;

    public static final int FRAME_stand44 = 44;

    public static final int FRAME_stand45 = 45;

    public static final int FRAME_stand46 = 46;

    public static final int FRAME_stand47 = 47;

    public static final int FRAME_stand48 = 48;

    private static final int FRAME_stand49 = 49;

    private static final int FRAME_stand50 = 50;

    public static final int FRAME_stand51 = 51;

    public static final int FRAME_stand52 = 52;

    public static final int FRAME_stand53 = 53;

    public static final int FRAME_stand54 = 54;

    public static final int FRAME_stand55 = 55;

    public static final int FRAME_stand56 = 56;

    public static final int FRAME_stand57 = 57;

    public static final int FRAME_stand58 = 58;

    public static final int FRAME_stand59 = 59;

    public static final int FRAME_stand60 = 60;

    public static final int FRAME_stand61 = 61;

    public static final int FRAME_stand62 = 62;

    public static final int FRAME_stand63 = 63;

    public static final int FRAME_stand64 = 64;

    public static final int FRAME_stand65 = 65;

    public static final int FRAME_stand66 = 66;

    public static final int FRAME_stand67 = 67;

    public static final int FRAME_stand68 = 68;

    public static final int FRAME_stand69 = 69;

    public static final int FRAME_stand70 = 70;

    private static final int FRAME_stand71 = 71;

    public static final int FRAME_walk01 = 72;

    public static final int FRAME_walk02 = 73;

    private static final int FRAME_walk03 = 74;

    public static final int FRAME_walk04 = 75;

    public static final int FRAME_walk05 = 76;

    public static final int FRAME_walk06 = 77;

    public static final int FRAME_walk07 = 78;

    public static final int FRAME_walk08 = 79;

    public static final int FRAME_walk09 = 80;

    public static final int FRAME_walk10 = 81;

    public static final int FRAME_walk11 = 82;

    public static final int FRAME_walk12 = 83;

    public static final int FRAME_walk13 = 84;

    private static final int FRAME_walk14 = 85;

    public static final int FRAME_walk15 = 86;

    public static final int FRAME_walk16 = 87;

    public static final int FRAME_walk17 = 88;

    public static final int FRAME_walk18 = 89;

    public static final int FRAME_walk19 = 90;

    public static final int FRAME_walk20 = 91;

    private static final int FRAME_run01 = 92;

    public static final int FRAME_run02 = 93;

    public static final int FRAME_run03 = 94;

    public static final int FRAME_run04 = 95;

    public static final int FRAME_run05 = 96;

    public static final int FRAME_run06 = 97;

    public static final int FRAME_run07 = 98;

    private static final int FRAME_run08 = 99;

    private static final int FRAME_pain101 = 100;

    public static final int FRAME_pain102 = 101;

    public static final int FRAME_pain103 = 102;

    public static final int FRAME_pain104 = 103;

    public static final int FRAME_pain105 = 104;

    public static final int FRAME_pain106 = 105;

    public static final int FRAME_pain107 = 106;

    public static final int FRAME_pain108 = 107;

    public static final int FRAME_pain109 = 108;

    private static final int FRAME_pain110 = 109;

    private static final int FRAME_pain201 = 110;

    public static final int FRAME_pain202 = 111;

    public static final int FRAME_pain203 = 112;

    public static final int FRAME_pain204 = 113;

    public static final int FRAME_pain205 = 114;

    public static final int FRAME_pain206 = 115;

    public static final int FRAME_pain207 = 116;

    public static final int FRAME_pain208 = 117;

    public static final int FRAME_pain209 = 118;

    private static final int FRAME_pain210 = 119;

    private static final int FRAME_duck01 = 120;

    public static final int FRAME_duck02 = 121;

    public static final int FRAME_duck03 = 122;

    public static final int FRAME_duck04 = 123;

    private static final int FRAME_duck05 = 124;

    private static final int FRAME_death101 = 125;

    public static final int FRAME_death102 = 126;

    public static final int FRAME_death103 = 127;

    public static final int FRAME_death104 = 128;

    public static final int FRAME_death105 = 129;

    public static final int FRAME_death106 = 130;

    public static final int FRAME_death107 = 131;

    public static final int FRAME_death108 = 132;

    public static final int FRAME_death109 = 133;

    public static final int FRAME_death110 = 134;

    public static final int FRAME_death111 = 135;

    public static final int FRAME_death112 = 136;

    public static final int FRAME_death113 = 137;

    public static final int FRAME_death114 = 138;

    public static final int FRAME_death115 = 139;

    public static final int FRAME_death116 = 140;

    public static final int FRAME_death117 = 141;

    public static final int FRAME_death118 = 142;

    public static final int FRAME_death119 = 143;

    private static final int FRAME_death120 = 144;

    private static final int FRAME_death201 = 145;

    public static final int FRAME_death202 = 146;

    public static final int FRAME_death203 = 147;

    public static final int FRAME_death204 = 148;

    public static final int FRAME_death205 = 149;

    public static final int FRAME_death206 = 150;

    public static final int FRAME_death207 = 151;

    public static final int FRAME_death208 = 152;

    public static final int FRAME_death209 = 153;

    public static final int FRAME_death210 = 154;

    private static final int FRAME_death211 = 155;

    public static final int FRAME_death212 = 156;

    public static final int FRAME_death213 = 157;

    public static final int FRAME_death214 = 158;

    public static final int FRAME_death215 = 159;

    public static final int FRAME_death216 = 160;

    public static final int FRAME_death217 = 161;

    public static final int FRAME_death218 = 162;

    public static final int FRAME_death219 = 163;

    public static final int FRAME_death220 = 164;

    public static final int FRAME_death221 = 165;

    public static final int FRAME_death222 = 166;

    public static final int FRAME_death223 = 167;

    public static final int FRAME_death224 = 168;

    private static final int FRAME_death225 = 169;

    private static final int FRAME_death301 = 170;

    public static final int FRAME_death302 = 171;

    public static final int FRAME_death303 = 172;

    public static final int FRAME_death304 = 173;

    public static final int FRAME_death305 = 174;

    public static final int FRAME_death306 = 175;

    public static final int FRAME_death307 = 176;

    public static final int FRAME_death308 = 177;

    private static final int FRAME_death309 = 178;

    public static final int FRAME_block01 = 179;

    public static final int FRAME_block02 = 180;

    public static final int FRAME_block03 = 181;

    public static final int FRAME_block04 = 182;

    public static final int FRAME_block05 = 183;

    private static final int FRAME_attak101 = 184;

    public static final int FRAME_attak102 = 185;

    public static final int FRAME_attak103 = 186;

    public static final int FRAME_attak104 = 187;

    public static final int FRAME_attak105 = 188;

    public static final int FRAME_attak106 = 189;

    public static final int FRAME_attak107 = 190;

    public static final int FRAME_attak108 = 191;

    public static final int FRAME_attak109 = 192;

    public static final int FRAME_attak110 = 193;

    private static final int FRAME_attak111 = 194;

    public static final int FRAME_attak112 = 195;

    public static final int FRAME_attak113 = 196;

    public static final int FRAME_attak114 = 197;

    private static final int FRAME_attak115 = 198;

    private static final int FRAME_attak201 = 199;

    public static final int FRAME_attak202 = 200;

    public static final int FRAME_attak203 = 201;

    public static final int FRAME_attak204 = 202;

    public static final int FRAME_attak205 = 203;

    public static final int FRAME_attak206 = 204;

    public static final int FRAME_attak207 = 205;

    private static final int FRAME_attak208 = 206;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_die1;

    private static int sound_die2;

    private static int sound_weapon_cock;

    private static int sound_punch_swing;

    private static int sound_punch_hit;

    private static int sound_sight;

    private static int sound_idle;

    private static final mframe_t[] infantry_frames_stand = {
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null) };

    private static final mmove_t infantry_move_stand = new mmove_t(FRAME_stand50,
            FRAME_stand71, infantry_frames_stand, null);

    public static final EntThinkAdapter infantry_stand = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_stand"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = infantry_move_stand;
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_fidget = {
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 3, null),
            new mframe_t(GameAI.ai_stand, 6, null),
            new mframe_t(GameAI.ai_stand, 3, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -2, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, -1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -1, null),
            new mframe_t(GameAI.ai_stand, -1, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -3, null),
            new mframe_t(GameAI.ai_stand, -2, null),
            new mframe_t(GameAI.ai_stand, -3, null),
            new mframe_t(GameAI.ai_stand, -3, null),
            new mframe_t(GameAI.ai_stand, -2, null) };

    private static final mmove_t infantry_move_fidget = new mmove_t(FRAME_stand01,
            FRAME_stand49, infantry_frames_fidget, infantry_stand);

    private static final EntThinkAdapter infantry_fidget = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_fidget"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = infantry_move_fidget;
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_idle, 1,
                    Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_walk = {
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 6, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 5, null) };

    private static final mmove_t infantry_move_walk = new mmove_t(FRAME_walk03, FRAME_walk14,
            infantry_frames_walk, null);

    private static final EntThinkAdapter infantry_walk = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_walk"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = infantry_move_walk;
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_run = {
            new mframe_t(GameAI.ai_run, 10, null),
            new mframe_t(GameAI.ai_run, 20, null),
            new mframe_t(GameAI.ai_run, 5, null),
            new mframe_t(GameAI.ai_run, 7, null),
            new mframe_t(GameAI.ai_run, 30, null),
            new mframe_t(GameAI.ai_run, 35, null),
            new mframe_t(GameAI.ai_run, 2, null),
            new mframe_t(GameAI.ai_run, 6, null) };

    private static final mmove_t infantry_move_run = new mmove_t(FRAME_run01, FRAME_run08,
            infantry_frames_run, null);

    private static final EntThinkAdapter infantry_run = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_run"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = infantry_move_stand;
            else
                self.monsterinfo.currentmove = infantry_move_run;
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_pain1 = {
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 6, null),
            new mframe_t(GameAI.ai_move, 2, null) };

    private static final mmove_t infantry_move_pain1 = new mmove_t(FRAME_pain101,
            FRAME_pain110, infantry_frames_pain1, infantry_run);

    private static final mframe_t[] infantry_frames_pain2 = {
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 5, null),
            new mframe_t(GameAI.ai_move, 2, null) };

    private static final mmove_t infantry_move_pain2 = new mmove_t(FRAME_pain201,
            FRAME_pain210, infantry_frames_pain2, infantry_run);

    private static final EntPainAdapter infantry_pain = new EntPainAdapter() {
    	@Override
        public String getID() { return "infantry_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {

            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            if (GameBase.level.time < self.pain_debounce_time)
                return;

            self.pain_debounce_time = GameBase.level.time + 3;

            if (GameBase.skill.value == 3)
                return;

            int n = Lib.rand() % 2;
            if (n == 0) {
                self.monsterinfo.currentmove = infantry_move_pain1;
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
            } else {
                self.monsterinfo.currentmove = infantry_move_pain2;
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NORM, 0);
            }
        }
    };

    private static final float[][] aimangles = { { 0.0f, 5.0f, 0.0f },
            { 10.0f, 15.0f, 0.0f }, { 20.0f, 25.0f, 0.0f },
            { 25.0f, 35.0f, 0.0f }, { 30.0f, 40.0f, 0.0f },
            { 30.0f, 45.0f, 0.0f }, { 25.0f, 50.0f, 0.0f },
            { 20.0f, 40.0f, 0.0f }, { 15.0f, 35.0f, 0.0f },
            { 40.0f, 35.0f, 0.0f }, { 70.0f, 35.0f, 0.0f },
            { 90.0f, 35.0f, 0.0f } };

    private static final EntThinkAdapter InfantryMachineGun = new EntThinkAdapter() {
    	@Override
        public String getID() { return "InfantryMachineGun"; }
        @Override
        public boolean think(edict_t self) {
            float[] start = { 0, 0, 0 };
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };
            int flash_number;

            if (self.s.frame == FRAME_attak111) {
                flash_number = Defines.MZ2_INFANTRY_MACHINEGUN_1;
                Math3D.AngleVectors(self.s.angles, forward, right, null);
                Math3D.G_ProjectSource(self.s.origin,
                        M_Flash.monster_flash_offset[flash_number], forward,
                        right, start);

                if (self.enemy != null) {
                    float[] target = {0, 0, 0};
                    Math3D.VectorMA(self.enemy.s.origin, -0.2f,
                            self.enemy.velocity, target);
                    target[2] += self.enemy.viewheight;
                    Math3D.VectorSubtract(target, start, forward);
                    Math3D.VectorNormalize(forward);
                } else {
                    Math3D.AngleVectors(self.s.angles, forward, right, null);
                }
            } else {
                flash_number = Defines.MZ2_INFANTRY_MACHINEGUN_2
                        + (self.s.frame - FRAME_death211);

                Math3D.AngleVectors(self.s.angles, forward, right, null);
                Math3D.G_ProjectSource(self.s.origin,
                        M_Flash.monster_flash_offset[flash_number], forward,
                        right, start);

                float[] vec = {0, 0, 0};
                Math3D.VectorSubtract(self.s.angles, aimangles[flash_number
                        - Defines.MZ2_INFANTRY_MACHINEGUN_2], vec);
                Math3D.AngleVectors(vec, forward, null, null);
            }

            Monster.monster_fire_bullet(self, start, forward, 3, 4,
                    Defines.DEFAULT_BULLET_HSPREAD,
                    Defines.DEFAULT_BULLET_VSPREAD, flash_number);
            return true;
        }
    };

    private static final EntInteractAdapter infantry_sight = new EntInteractAdapter() {
    	@Override
        public String getID() { return "infantry_sight"; }
        @Override
        public boolean interact(edict_t self, edict_t other) {
            game_import_t.sound(self, Defines.CHAN_BODY, sound_sight, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    

    private static final EntThinkAdapter infantry_dead = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_dead"; }
        @Override
        public boolean think(edict_t self) {
            Math3D.VectorSet(self.mins, -16, -16, -24);
            Math3D.VectorSet(self.maxs, 16, 16, -8);
            self.movetype = Defines.MOVETYPE_TOSS;
            self.svflags |= Defines.SVF_DEADMONSTER;
            game_import_t.linkentity(self);

            M.M_FlyCheck.think(self);
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_death1 = {
            new mframe_t(GameAI.ai_move, -4, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -4, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 9, null),
            new mframe_t(GameAI.ai_move, 9, null),
            new mframe_t(GameAI.ai_move, 5, null),
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -3, null) };

    private static final mmove_t infantry_move_death1 = new mmove_t(FRAME_death101,
            FRAME_death120, infantry_frames_death1, infantry_dead);

    
    private static final mframe_t[] infantry_frames_death2 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 5, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 4, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -2, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -2, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -3, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -1, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -2, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, 0, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, 2, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, 2, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, 3, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -10, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -7, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -8, InfantryMachineGun),
            new mframe_t(GameAI.ai_move, -6, null),
            new mframe_t(GameAI.ai_move, 4, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t infantry_move_death2 = new mmove_t(FRAME_death201,
            FRAME_death225, infantry_frames_death2, infantry_dead);

    private static final mframe_t[] infantry_frames_death3 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -6, null),
            new mframe_t(GameAI.ai_move, -11, null),
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -11, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t infantry_move_death3 = new mmove_t(FRAME_death301,
            FRAME_death309, infantry_frames_death3, infantry_dead);

    public static final EntDieAdapter infantry_die = new EntDieAdapter() {
    	@Override
        public String getID() { return "infantry_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {

            int n;

            
            if (self.health <= self.gib_health) {
                game_import_t
                        .sound(self, Defines.CHAN_VOICE, game_import_t
                                .soundindex("misc/udeath.wav"), 1,
                                Defines.ATTN_NORM, 0);
                for (n = 0; n < 2; n++)
                    GameMisc.ThrowGib(self, "models/objects/gibs/bone/tris.md2",
                            damage, Defines.GIB_ORGANIC);
                for (n = 0; n < 4; n++)
                    GameMisc.ThrowGib(self,
                            "models/objects/gibs/sm_meat/tris.md2", damage,
                            Defines.GIB_ORGANIC);
                GameMisc.ThrowHead(self, "models/objects/gibs/head2/tris.md2",
                        damage, Defines.GIB_ORGANIC);
                self.deadflag = Defines.DEAD_DEAD;
                return;
            }

            if (self.deadflag == Defines.DEAD_DEAD)
                return;

            
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_YES;

            n = Lib.rand() % 3;
            switch (n) {
                case 0 -> {
                    self.monsterinfo.currentmove = infantry_move_death1;
                    game_import_t.sound(self, Defines.CHAN_VOICE, sound_die2, 1,
                            Defines.ATTN_NORM, 0);
                }
                case 1 -> {
                    self.monsterinfo.currentmove = infantry_move_death2;
                    game_import_t.sound(self, Defines.CHAN_VOICE, sound_die1, 1,
                            Defines.ATTN_NORM, 0);
                }
                default -> {
                    self.monsterinfo.currentmove = infantry_move_death3;
                    game_import_t.sound(self, Defines.CHAN_VOICE, sound_die2, 1,
                            Defines.ATTN_NORM, 0);
                }
            }
        }
    };

    private static final EntThinkAdapter infantry_duck_down = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_duck_down"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_DUCKED) != 0)
                return true;
            self.monsterinfo.aiflags |= Defines.AI_DUCKED;
            self.maxs[2] -= 32;
            self.takedamage = Defines.DAMAGE_YES;
            self.monsterinfo.pausetime = GameBase.level.time + 1;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final EntThinkAdapter infantry_duck_hold = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_duck_hold"; }
        @Override
        public boolean think(edict_t self) {
            if (GameBase.level.time >= self.monsterinfo.pausetime)
                self.monsterinfo.aiflags &= ~Defines.AI_HOLD_FRAME;
            else
                self.monsterinfo.aiflags |= Defines.AI_HOLD_FRAME;
            return true;
        }
    };

    private static final EntThinkAdapter infantry_duck_up = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_duck_up"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.aiflags &= ~Defines.AI_DUCKED;
            self.maxs[2] += 32;
            self.takedamage = Defines.DAMAGE_AIM;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_duck = {
            new mframe_t(GameAI.ai_move, -2, infantry_duck_down),
            new mframe_t(GameAI.ai_move, -5, infantry_duck_hold),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 4, infantry_duck_up),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t infantry_move_duck = new mmove_t(FRAME_duck01, FRAME_duck05,
            infantry_frames_duck, infantry_run);

    private static final EntDodgeAdapter infantry_dodge = new EntDodgeAdapter() {
    	@Override
        public String getID() { return "infantry_dodge"; }
        @Override
        public void dodge(edict_t self, edict_t attacker, float eta) {
            if (Lib.random() > 0.25)
                return;

            if (null == self.enemy)
                self.enemy = attacker;

            self.monsterinfo.currentmove = infantry_move_duck;
        }
    };

    private static final EntThinkAdapter infantry_cock_gun = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_cock_gun"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_WEAPON, sound_weapon_cock, 1,
                    Defines.ATTN_NORM, 0);
            int n = (Lib.rand() & 15) + 3 + 7;
            self.monsterinfo.pausetime = GameBase.level.time + n
                    * Defines.FRAMETIME;
            return true;
        }
    };

    private static final EntThinkAdapter infantry_fire = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_fire"; }
        @Override
        public boolean think(edict_t self) {
            InfantryMachineGun.think(self);

            if (GameBase.level.time >= self.monsterinfo.pausetime)
                self.monsterinfo.aiflags &= ~Defines.AI_HOLD_FRAME;
            else
                self.monsterinfo.aiflags |= Defines.AI_HOLD_FRAME;
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_attack1 = {
            new mframe_t(GameAI.ai_charge, 4, null),
            new mframe_t(GameAI.ai_charge, -1, null),
            new mframe_t(GameAI.ai_charge, -1, null),
            new mframe_t(GameAI.ai_charge, 0, infantry_cock_gun),
            new mframe_t(GameAI.ai_charge, -1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 2, null),
            new mframe_t(GameAI.ai_charge, -2, null),
            new mframe_t(GameAI.ai_charge, -3, null),
            new mframe_t(GameAI.ai_charge, 1, infantry_fire),
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, -1, null),
            new mframe_t(GameAI.ai_charge, -2, null),
            new mframe_t(GameAI.ai_charge, -3, null) };

    private static final mmove_t infantry_move_attack1 = new mmove_t(FRAME_attak101,
            FRAME_attak115, infantry_frames_attack1, infantry_run);

    private static final EntThinkAdapter infantry_swing = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_swing"; }

        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_WEAPON, sound_punch_swing, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter infantry_smack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_smack"; }
        @Override
        public boolean think(edict_t self) {
            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, 0, 0);
            if (GameWeapon.fire_hit(self, aim, (5 + (Lib.rand() % 5)), 50))
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_punch_hit,
                        1, Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final mframe_t[] infantry_frames_attack2 = {
            new mframe_t(GameAI.ai_charge, 3, null),
            new mframe_t(GameAI.ai_charge, 6, null),
            new mframe_t(GameAI.ai_charge, 0, infantry_swing),
            new mframe_t(GameAI.ai_charge, 8, null),
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, 8, infantry_smack),
            new mframe_t(GameAI.ai_charge, 6, null),
            new mframe_t(GameAI.ai_charge, 3, null), };

    private static final mmove_t infantry_move_attack2 = new mmove_t(FRAME_attak201,
            FRAME_attak208, infantry_frames_attack2, infantry_run);

    private static final EntThinkAdapter infantry_attack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "infantry_attack"; }
        @Override
        public boolean think(edict_t self) {
            if (GameUtil.range(self, self.enemy) == Defines.RANGE_MELEE)
                self.monsterinfo.currentmove = infantry_move_attack2;
            else
                self.monsterinfo.currentmove = infantry_move_attack1;
            return true;
        }
    };

    /*
     * QUAKED monster_infantry (1 .5 0) (-16 -16 -24) (16 16 32) Ambush
     * Trigger_Spawn Sight
     */
    public static void SP_monster_infantry(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_pain1 = game_import_t.soundindex("infantry/infpain1.wav");
        sound_pain2 = game_import_t.soundindex("infantry/infpain2.wav");
        sound_die1 = game_import_t.soundindex("infantry/infdeth1.wav");
        sound_die2 = game_import_t.soundindex("infantry/infdeth2.wav");

        int sound_gunshot = game_import_t.soundindex("infantry/infatck1.wav");
        sound_weapon_cock = game_import_t.soundindex("infantry/infatck3.wav");
        sound_punch_swing = game_import_t.soundindex("infantry/infatck2.wav");
        sound_punch_hit = game_import_t.soundindex("infantry/melee2.wav");

        sound_sight = game_import_t.soundindex("infantry/infsght1.wav");
        int sound_search = game_import_t.soundindex("infantry/infsrch1.wav");
        sound_idle = game_import_t.soundindex("infantry/infidle1.wav");

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/infantry/tris.md2");
        Math3D.VectorSet(self.mins, -16, -16, -24);
        Math3D.VectorSet(self.maxs, 16, 16, 32);

        self.health = 100;
        self.gib_health = -40;
        self.mass = 200;

        self.pain = infantry_pain;
        self.die = infantry_die;

        self.monsterinfo.stand = infantry_stand;
        self.monsterinfo.walk = infantry_walk;
        self.monsterinfo.run = infantry_run;
        self.monsterinfo.dodge = infantry_dodge;
        self.monsterinfo.attack = infantry_attack;
        self.monsterinfo.melee = null;
        self.monsterinfo.sight = infantry_sight;
        self.monsterinfo.idle = infantry_fidget;

        game_import_t.linkentity(self);

        self.monsterinfo.currentmove = infantry_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.walkmonster_start.think(self);
    }
}