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
import jake2.game.*;
import jake2.util.Lib;
import jake2.util.Math3D;

public class M_Insane {
    

    private static final int FRAME_stand1 = 0;

    public static final int FRAME_stand2 = 1;

    public static final int FRAME_stand3 = 2;

    public static final int FRAME_stand4 = 3;

    public static final int FRAME_stand5 = 4;

    public static final int FRAME_stand6 = 5;

    public static final int FRAME_stand7 = 6;

    public static final int FRAME_stand8 = 7;

    public static final int FRAME_stand9 = 8;

    public static final int FRAME_stand10 = 9;

    public static final int FRAME_stand11 = 10;

    public static final int FRAME_stand12 = 11;

    public static final int FRAME_stand13 = 12;

    public static final int FRAME_stand14 = 13;

    public static final int FRAME_stand15 = 14;

    public static final int FRAME_stand16 = 15;

    public static final int FRAME_stand17 = 16;

    public static final int FRAME_stand18 = 17;

    public static final int FRAME_stand19 = 18;

    public static final int FRAME_stand20 = 19;

    public static final int FRAME_stand21 = 20;

    public static final int FRAME_stand22 = 21;

    public static final int FRAME_stand23 = 22;

    public static final int FRAME_stand24 = 23;

    public static final int FRAME_stand25 = 24;

    public static final int FRAME_stand26 = 25;

    public static final int FRAME_stand27 = 26;

    public static final int FRAME_stand28 = 27;

    public static final int FRAME_stand29 = 28;

    public static final int FRAME_stand30 = 29;

    public static final int FRAME_stand31 = 30;

    public static final int FRAME_stand32 = 31;

    public static final int FRAME_stand33 = 32;

    public static final int FRAME_stand34 = 33;

    public static final int FRAME_stand35 = 34;

    public static final int FRAME_stand36 = 35;

    public static final int FRAME_stand37 = 36;

    public static final int FRAME_stand38 = 37;

    public static final int FRAME_stand39 = 38;

    private static final int FRAME_stand40 = 39;

    private static final int FRAME_stand41 = 40;

    public static final int FRAME_stand42 = 41;

    public static final int FRAME_stand43 = 42;

    public static final int FRAME_stand44 = 43;

    public static final int FRAME_stand45 = 44;

    public static final int FRAME_stand46 = 45;

    public static final int FRAME_stand47 = 46;

    public static final int FRAME_stand48 = 47;

    public static final int FRAME_stand49 = 48;

    public static final int FRAME_stand50 = 49;

    public static final int FRAME_stand51 = 50;

    public static final int FRAME_stand52 = 51;

    public static final int FRAME_stand53 = 52;

    public static final int FRAME_stand54 = 53;

    public static final int FRAME_stand55 = 54;

    public static final int FRAME_stand56 = 55;

    public static final int FRAME_stand57 = 56;

    public static final int FRAME_stand58 = 57;

    private static final int FRAME_stand59 = 58;

    private static final int FRAME_stand60 = 59;

    public static final int FRAME_stand61 = 60;

    public static final int FRAME_stand62 = 61;

    public static final int FRAME_stand63 = 62;

    public static final int FRAME_stand64 = 63;

    private static final int FRAME_stand65 = 64;

    public static final int FRAME_stand66 = 65;

    public static final int FRAME_stand67 = 66;

    public static final int FRAME_stand68 = 67;

    public static final int FRAME_stand69 = 68;

    public static final int FRAME_stand70 = 69;

    public static final int FRAME_stand71 = 70;

    public static final int FRAME_stand72 = 71;

    public static final int FRAME_stand73 = 72;

    public static final int FRAME_stand74 = 73;

    public static final int FRAME_stand75 = 74;

    public static final int FRAME_stand76 = 75;

    public static final int FRAME_stand77 = 76;

    public static final int FRAME_stand78 = 77;

    public static final int FRAME_stand79 = 78;

    public static final int FRAME_stand80 = 79;

    public static final int FRAME_stand81 = 80;

    public static final int FRAME_stand82 = 81;

    public static final int FRAME_stand83 = 82;

    public static final int FRAME_stand84 = 83;

    public static final int FRAME_stand85 = 84;

    public static final int FRAME_stand86 = 85;

    public static final int FRAME_stand87 = 86;

    public static final int FRAME_stand88 = 87;

    public static final int FRAME_stand89 = 88;

    public static final int FRAME_stand90 = 89;

    public static final int FRAME_stand91 = 90;

    public static final int FRAME_stand92 = 91;

    public static final int FRAME_stand93 = 92;

    private static final int FRAME_stand94 = 93;

    public static final int FRAME_stand95 = 94;

    private static final int FRAME_stand96 = 95;

    public static final int FRAME_stand97 = 96;

    public static final int FRAME_stand98 = 97;

    private static final int FRAME_stand99 = 98;

    private static final int FRAME_stand100 = 99;

    public static final int FRAME_stand101 = 100;

    public static final int FRAME_stand102 = 101;

    public static final int FRAME_stand103 = 102;

    public static final int FRAME_stand104 = 103;

    public static final int FRAME_stand105 = 104;

    public static final int FRAME_stand106 = 105;

    public static final int FRAME_stand107 = 106;

    public static final int FRAME_stand108 = 107;

    public static final int FRAME_stand109 = 108;

    public static final int FRAME_stand110 = 109;

    public static final int FRAME_stand111 = 110;

    public static final int FRAME_stand112 = 111;

    public static final int FRAME_stand113 = 112;

    public static final int FRAME_stand114 = 113;

    public static final int FRAME_stand115 = 114;

    public static final int FRAME_stand116 = 115;

    public static final int FRAME_stand117 = 116;

    public static final int FRAME_stand118 = 117;

    public static final int FRAME_stand119 = 118;

    public static final int FRAME_stand120 = 119;

    public static final int FRAME_stand121 = 120;

    public static final int FRAME_stand122 = 121;

    public static final int FRAME_stand123 = 122;

    public static final int FRAME_stand124 = 123;

    public static final int FRAME_stand125 = 124;

    public static final int FRAME_stand126 = 125;

    public static final int FRAME_stand127 = 126;

    public static final int FRAME_stand128 = 127;

    public static final int FRAME_stand129 = 128;

    public static final int FRAME_stand130 = 129;

    public static final int FRAME_stand131 = 130;

    public static final int FRAME_stand132 = 131;

    public static final int FRAME_stand133 = 132;

    public static final int FRAME_stand134 = 133;

    public static final int FRAME_stand135 = 134;

    public static final int FRAME_stand136 = 135;

    public static final int FRAME_stand137 = 136;

    public static final int FRAME_stand138 = 137;

    public static final int FRAME_stand139 = 138;

    public static final int FRAME_stand140 = 139;

    public static final int FRAME_stand141 = 140;

    public static final int FRAME_stand142 = 141;

    public static final int FRAME_stand143 = 142;

    public static final int FRAME_stand144 = 143;

    public static final int FRAME_stand145 = 144;

    public static final int FRAME_stand146 = 145;

    public static final int FRAME_stand147 = 146;

    public static final int FRAME_stand148 = 147;

    public static final int FRAME_stand149 = 148;

    public static final int FRAME_stand150 = 149;

    public static final int FRAME_stand151 = 150;

    public static final int FRAME_stand152 = 151;

    public static final int FRAME_stand153 = 152;

    public static final int FRAME_stand154 = 153;

    public static final int FRAME_stand155 = 154;

    public static final int FRAME_stand156 = 155;

    public static final int FRAME_stand157 = 156;

    public static final int FRAME_stand158 = 157;

    public static final int FRAME_stand159 = 158;

    private static final int FRAME_stand160 = 159;

    private static final int FRAME_walk27 = 160;

    public static final int FRAME_walk28 = 161;

    public static final int FRAME_walk29 = 162;

    public static final int FRAME_walk30 = 163;

    public static final int FRAME_walk31 = 164;

    public static final int FRAME_walk32 = 165;

    public static final int FRAME_walk33 = 166;

    public static final int FRAME_walk34 = 167;

    public static final int FRAME_walk35 = 168;

    public static final int FRAME_walk36 = 169;

    public static final int FRAME_walk37 = 170;

    public static final int FRAME_walk38 = 171;

    private static final int FRAME_walk39 = 172;

    private static final int FRAME_walk1 = 173;

    public static final int FRAME_walk2 = 174;

    public static final int FRAME_walk3 = 175;

    public static final int FRAME_walk4 = 176;

    public static final int FRAME_walk5 = 177;

    public static final int FRAME_walk6 = 178;

    public static final int FRAME_walk7 = 179;

    public static final int FRAME_walk8 = 180;

    public static final int FRAME_walk9 = 181;

    public static final int FRAME_walk10 = 182;

    public static final int FRAME_walk11 = 183;

    public static final int FRAME_walk12 = 184;

    public static final int FRAME_walk13 = 185;

    public static final int FRAME_walk14 = 186;

    public static final int FRAME_walk15 = 187;

    public static final int FRAME_walk16 = 188;

    public static final int FRAME_walk17 = 189;

    public static final int FRAME_walk18 = 190;

    public static final int FRAME_walk19 = 191;

    public static final int FRAME_walk20 = 192;

    public static final int FRAME_walk21 = 193;

    public static final int FRAME_walk22 = 194;

    public static final int FRAME_walk23 = 195;

    public static final int FRAME_walk24 = 196;

    public static final int FRAME_walk25 = 197;

    private static final int FRAME_walk26 = 198;

    private static final int FRAME_st_pain2 = 199;

    public static final int FRAME_st_pain3 = 200;

    public static final int FRAME_st_pain4 = 201;

    public static final int FRAME_st_pain5 = 202;

    public static final int FRAME_st_pain6 = 203;

    public static final int FRAME_st_pain7 = 204;

    public static final int FRAME_st_pain8 = 205;

    public static final int FRAME_st_pain9 = 206;

    public static final int FRAME_st_pain10 = 207;

    public static final int FRAME_st_pain11 = 208;

    private static final int FRAME_st_pain12 = 209;

    private static final int FRAME_st_death2 = 210;

    public static final int FRAME_st_death3 = 211;

    public static final int FRAME_st_death4 = 212;

    public static final int FRAME_st_death5 = 213;

    public static final int FRAME_st_death6 = 214;

    public static final int FRAME_st_death7 = 215;

    public static final int FRAME_st_death8 = 216;

    public static final int FRAME_st_death9 = 217;

    public static final int FRAME_st_death10 = 218;

    public static final int FRAME_st_death11 = 219;

    public static final int FRAME_st_death12 = 220;

    public static final int FRAME_st_death13 = 221;

    public static final int FRAME_st_death14 = 222;

    public static final int FRAME_st_death15 = 223;

    public static final int FRAME_st_death16 = 224;

    public static final int FRAME_st_death17 = 225;

    private static final int FRAME_st_death18 = 226;

    private static final int FRAME_crawl1 = 227;

    public static final int FRAME_crawl2 = 228;

    public static final int FRAME_crawl3 = 229;

    public static final int FRAME_crawl4 = 230;

    public static final int FRAME_crawl5 = 231;

    public static final int FRAME_crawl6 = 232;

    public static final int FRAME_crawl7 = 233;

    public static final int FRAME_crawl8 = 234;

    private static final int FRAME_crawl9 = 235;

    private static final int FRAME_cr_pain2 = 236;

    public static final int FRAME_cr_pain3 = 237;

    public static final int FRAME_cr_pain4 = 238;

    public static final int FRAME_cr_pain5 = 239;

    public static final int FRAME_cr_pain6 = 240;

    public static final int FRAME_cr_pain7 = 241;

    public static final int FRAME_cr_pain8 = 242;

    public static final int FRAME_cr_pain9 = 243;

    private static final int FRAME_cr_pain10 = 244;

    private static final int FRAME_cr_death10 = 245;

    public static final int FRAME_cr_death11 = 246;

    public static final int FRAME_cr_death12 = 247;

    public static final int FRAME_cr_death13 = 248;

    public static final int FRAME_cr_death14 = 249;

    public static final int FRAME_cr_death15 = 250;

    private static final int FRAME_cr_death16 = 251;

    private static final int FRAME_cross1 = 252;

    public static final int FRAME_cross2 = 253;

    public static final int FRAME_cross3 = 254;

    public static final int FRAME_cross4 = 255;

    public static final int FRAME_cross5 = 256;

    public static final int FRAME_cross6 = 257;

    public static final int FRAME_cross7 = 258;

    public static final int FRAME_cross8 = 259;

    public static final int FRAME_cross9 = 260;

    public static final int FRAME_cross10 = 261;

    public static final int FRAME_cross11 = 262;

    public static final int FRAME_cross12 = 263;

    public static final int FRAME_cross13 = 264;

    public static final int FRAME_cross14 = 265;

    private static final int FRAME_cross15 = 266;

    private static final int FRAME_cross16 = 267;

    public static final int FRAME_cross17 = 268;

    public static final int FRAME_cross18 = 269;

    public static final int FRAME_cross19 = 270;

    public static final int FRAME_cross20 = 271;

    public static final int FRAME_cross21 = 272;

    public static final int FRAME_cross22 = 273;

    public static final int FRAME_cross23 = 274;

    public static final int FRAME_cross24 = 275;

    public static final int FRAME_cross25 = 276;

    public static final int FRAME_cross26 = 277;

    public static final int FRAME_cross27 = 278;

    public static final int FRAME_cross28 = 279;

    public static final int FRAME_cross29 = 280;

    private static final int FRAME_cross30 = 281;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_fist;

    private static int sound_shake;

    private static int sound_moan;

    private static final int[] sound_scream = { 0, 0, 0, 0, 0, 0, 0, 0 };

    private static final EntThinkAdapter insane_fist = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_fist"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_fist, 1,
                    Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    private static final EntThinkAdapter insane_shake = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_shake"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_shake, 1,
                    Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    private static final EntThinkAdapter insane_moan = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_moan"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_moan, 1,
                    Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    private static final EntThinkAdapter insane_scream = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_scream"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE,
                    sound_scream[Lib.rand() % 8], 1, Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    private static final EntThinkAdapter insane_cross = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_cross"; }
        @Override
        public boolean think(edict_t self) {
            if (Lib.random() < 0.8)
                self.monsterinfo.currentmove = insane_move_cross;
            else
                self.monsterinfo.currentmove = insane_move_struggle_cross;
            return true;
        }
    };

    private static final EntThinkAdapter insane_walk = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_walk"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.spawnflags & 16) != 0) 
                if (self.s.frame == FRAME_cr_pain10) {
                    self.monsterinfo.currentmove = insane_move_down;
                    return true;
                }
            if ((self.spawnflags & 4) != 0)
                self.monsterinfo.currentmove = insane_move_crawl;
            else if (Lib.random() <= 0.5)
                self.monsterinfo.currentmove = insane_move_walk_normal;
            else
                self.monsterinfo.currentmove = insane_move_walk_insane;
            return true;
        }
    };

    private static final EntThinkAdapter insane_run = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_run"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.spawnflags & 16) != 0) 
                if (self.s.frame == FRAME_cr_pain10) {
                    self.monsterinfo.currentmove = insane_move_down;
                    return true;
                }
            if ((self.spawnflags & 4) != 0) 
                self.monsterinfo.currentmove = insane_move_runcrawl;
            else if (Lib.random() <= 0.5) 
                self.monsterinfo.currentmove = insane_move_run_normal;
            else
                self.monsterinfo.currentmove = insane_move_run_insane;
            return true;
        }
    };

    private static final EntPainAdapter insane_pain = new EntPainAdapter() {
    	@Override
        public String getID() { return "insane_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {


            if (GameBase.level.time < self.pain_debounce_time)
                return;

            self.pain_debounce_time = GameBase.level.time + 3;

            int r = 1 + (Lib.rand() & 1);
            int l = switch (self.health) {
                case int i when i < 25 -> 25;
                case int i when i < 50 -> 50;
                case int i when i < 75 -> 75;
                default -> 100;
            };
            game_import_t.sound(self, Defines.CHAN_VOICE, game_import_t
                    .soundindex("player/male/pain" + l + '_' + r + ".wav"), 1,
                    Defines.ATTN_IDLE, 0);

            if (GameBase.skill.value == 3)
                return; 

            
            if ((self.spawnflags & 8) != 0) {
                self.monsterinfo.currentmove = insane_move_struggle_cross;
                return;
            }

            if (((self.s.frame >= FRAME_crawl1) && (self.s.frame <= FRAME_crawl9))
                    || ((self.s.frame >= FRAME_stand99) && (self.s.frame <= FRAME_stand160))) {
                self.monsterinfo.currentmove = insane_move_crawl_pain;
            } else
                self.monsterinfo.currentmove = insane_move_stand_pain;
        }
    };

    private static final EntThinkAdapter insane_onground = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_onground"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = insane_move_down;
            return true;
        }
    };

    private static final EntThinkAdapter insane_checkdown = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_checkdown"; }
        @Override
        public boolean think(edict_t self) {
            
            
            if ((self.spawnflags & 32) != 0) 
                return true;
            if (Lib.random() < 0.3)
                if (Lib.random() < 0.5)
                    self.monsterinfo.currentmove = insane_move_uptodown;
                else
                    self.monsterinfo.currentmove = insane_move_jumpdown;
            return true;
        }
    };

    private static final EntThinkAdapter insane_checkup = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_checkup"; }
        @Override
        public boolean think(edict_t self) {
            
            if ((self.spawnflags & 4) != 0 && (self.spawnflags & 16) != 0)
                return true;
            if (Lib.random() < 0.5)
                self.monsterinfo.currentmove = insane_move_downtoup;
            return true;
        }
    };

    private static final EntThinkAdapter insane_stand = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_stand"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.spawnflags & 8) != 0) 
            {
                self.monsterinfo.currentmove = insane_move_cross;
                self.monsterinfo.aiflags |= Defines.AI_STAND_GROUND;
            }
            
            else if ((self.spawnflags & 4) != 0 && (self.spawnflags & 16) != 0)
                self.monsterinfo.currentmove = insane_move_down;
            else if (Lib.random() < 0.5)
                self.monsterinfo.currentmove = insane_move_stand_normal;
            else
                self.monsterinfo.currentmove = insane_move_stand_insane;
            return true;
        }
    };

    private static final EntThinkAdapter insane_dead = new EntThinkAdapter() {
    	@Override
        public String getID() { return "insane_dead"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.spawnflags & 8) != 0) {
                self.flags |= Defines.FL_FLY;
            } else {
                Math3D.VectorSet(self.mins, -16, -16, -24);
                Math3D.VectorSet(self.maxs, 16, 16, -8);
                self.movetype = Defines.MOVETYPE_TOSS;
            }
            self.svflags |= Defines.SVF_DEADMONSTER;
            self.nextthink = 0;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final EntDieAdapter insane_die = new EntDieAdapter() {
    	@Override
        public String getID() { return "insane_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {

            if (self.health <= self.gib_health) {
                game_import_t
                        .sound(self, Defines.CHAN_VOICE, game_import_t
                                .soundindex("misc/udeath.wav"), 1,
                                Defines.ATTN_IDLE, 0);
                int n;
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

            game_import_t.sound(self, Defines.CHAN_VOICE, game_import_t
                    .soundindex("player/male/death" + ((Lib.rand() % 4) + 1)
                            + ".wav"), 1, Defines.ATTN_IDLE, 0);

            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_YES;

            if ((self.spawnflags & 8) != 0) {
                insane_dead.think(self);
            } else {
                if (((self.s.frame >= FRAME_crawl1) && (self.s.frame <= FRAME_crawl9))
                        || ((self.s.frame >= FRAME_stand99) && (self.s.frame <= FRAME_stand160)))
                    self.monsterinfo.currentmove = insane_move_crawl_death;
                else
                    self.monsterinfo.currentmove = insane_move_stand_death;
            }
        }

    };

    private static final mframe_t[] insane_frames_stand_normal = {
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, insane_checkdown) };

    private static final mmove_t insane_move_stand_normal = new mmove_t(FRAME_stand60,
            FRAME_stand65, insane_frames_stand_normal, insane_stand);

    private static final mframe_t[] insane_frames_stand_insane = {
            new mframe_t(GameAI.ai_stand, 0, insane_shake),
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
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, insane_checkdown) };

    private static final mmove_t insane_move_stand_insane = new mmove_t(FRAME_stand65,
            FRAME_stand94, insane_frames_stand_insane, insane_stand);

    private static final mframe_t[] insane_frames_uptodown = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, insane_moan),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 2.7f, null),
            new mframe_t(GameAI.ai_move, 4.1f, null),
            new mframe_t(GameAI.ai_move, 6f, null),
            new mframe_t(GameAI.ai_move, 7.6f, null),
            new mframe_t(GameAI.ai_move, 3.6f, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, insane_fist),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, insane_fist),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_uptodown = new mmove_t(FRAME_stand1,
            FRAME_stand40, insane_frames_uptodown, insane_onground);

    private static final mframe_t[] insane_frames_downtoup = {
            new mframe_t(GameAI.ai_move, -0.7f, null), 
            new mframe_t(GameAI.ai_move, -1.2f, null), 
            new mframe_t(GameAI.ai_move, -1.5f, null), 
            new mframe_t(GameAI.ai_move, -4.5f, null), 
            new mframe_t(GameAI.ai_move, -3.5f, null), 
            new mframe_t(GameAI.ai_move, -0.2f, null), 
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, -1.3f, null), 
            new mframe_t(GameAI.ai_move, -3, null), 
            new mframe_t(GameAI.ai_move, -2, null), 
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, -3.3f, null), 
            new mframe_t(GameAI.ai_move, -1.6f, null), 
            new mframe_t(GameAI.ai_move, -0.3f, null), 
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, 0, null) 
    };

    private static final mmove_t insane_move_downtoup = new mmove_t(FRAME_stand41,
            FRAME_stand59, insane_frames_downtoup, insane_stand);

    private static final mframe_t[] insane_frames_jumpdown = {
            new mframe_t(GameAI.ai_move, 0.2f, null),
            new mframe_t(GameAI.ai_move, 11.5f, null),
            new mframe_t(GameAI.ai_move, 5.1f, null),
            new mframe_t(GameAI.ai_move, 7.1f, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_jumpdown = new mmove_t(FRAME_stand96,
            FRAME_stand100, insane_frames_jumpdown, insane_onground);

    private static final mframe_t[] insane_frames_down = {
            new mframe_t(GameAI.ai_move, 0, null), 
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            
            new mframe_t(GameAI.ai_move, -1.7f, null),
            new mframe_t(GameAI.ai_move, -1.6f, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, insane_fist),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, insane_moan),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            
            new mframe_t(GameAI.ai_move, 0.5f, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -0.2f, insane_scream),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0.2f, null),
            new mframe_t(GameAI.ai_move, 0.4f, null),
            new mframe_t(GameAI.ai_move, 0.6f, null),
            new mframe_t(GameAI.ai_move, 0.8f, null),
            new mframe_t(GameAI.ai_move, 0.7f, null),
            new mframe_t(GameAI.ai_move, 0, insane_checkup) 
    };

    private static final mmove_t insane_move_down = new mmove_t(FRAME_stand100,
            FRAME_stand160, insane_frames_down, insane_onground);

    private static final mframe_t[] insane_frames_walk_normal = {
            new mframe_t(GameAI.ai_walk, 0, insane_scream),
            new mframe_t(GameAI.ai_walk, 2.5f, null),
            new mframe_t(GameAI.ai_walk, 3.5f, null),
            new mframe_t(GameAI.ai_walk, 1.7f, null),
            new mframe_t(GameAI.ai_walk, 2.3f, null),
            new mframe_t(GameAI.ai_walk, 2.4f, null),
            new mframe_t(GameAI.ai_walk, 2.2f, null),
            new mframe_t(GameAI.ai_walk, 4.2f, null),
            new mframe_t(GameAI.ai_walk, 5.6f, null),
            new mframe_t(GameAI.ai_walk, 3.3f, null),
            new mframe_t(GameAI.ai_walk, 2.4f, null),
            new mframe_t(GameAI.ai_walk, 0.9f, null),
            new mframe_t(GameAI.ai_walk, 0, null) };

    private static final mmove_t insane_move_walk_normal = new mmove_t(FRAME_walk27,
            FRAME_walk39, insane_frames_walk_normal, insane_walk);

    private static final mmove_t insane_move_run_normal = new mmove_t(FRAME_walk27,
            FRAME_walk39, insane_frames_walk_normal, insane_run);

    private static final mframe_t[] insane_frames_walk_insane = {
            new mframe_t(GameAI.ai_walk, 0, insane_scream), 
            new mframe_t(GameAI.ai_walk, 3.4f, null), 
            new mframe_t(GameAI.ai_walk, 3.6f, null), 
            new mframe_t(GameAI.ai_walk, 2.9f, null), 
            new mframe_t(GameAI.ai_walk, 2.2f, null), 
            new mframe_t(GameAI.ai_walk, 2.6f, null), 
            new mframe_t(GameAI.ai_walk, 0, null), 
            new mframe_t(GameAI.ai_walk, 0.7f, null), 
            new mframe_t(GameAI.ai_walk, 4.8f, null), 
            new mframe_t(GameAI.ai_walk, 5.3f, null), 
            new mframe_t(GameAI.ai_walk, 1.1f, null), 
            new mframe_t(GameAI.ai_walk, 2, null), 
            new mframe_t(GameAI.ai_walk, 0.5f, null), 
            new mframe_t(GameAI.ai_walk, 0, null), 
            new mframe_t(GameAI.ai_walk, 0, null), 
            new mframe_t(GameAI.ai_walk, 4.9f, null), 
            new mframe_t(GameAI.ai_walk, 6.7f, null), 
            new mframe_t(GameAI.ai_walk, 3.8f, null), 
            new mframe_t(GameAI.ai_walk, 2, null), 
            new mframe_t(GameAI.ai_walk, 0.2f, null), 
            new mframe_t(GameAI.ai_walk, 0, null), 
            new mframe_t(GameAI.ai_walk, 3.4f, null), 
            new mframe_t(GameAI.ai_walk, 6.4f, null), 
            new mframe_t(GameAI.ai_walk, 5, null), 
            new mframe_t(GameAI.ai_walk, 1.8f, null), 
            new mframe_t(GameAI.ai_walk, 0, null) 
    };

    private static final mmove_t insane_move_walk_insane = new mmove_t(FRAME_walk1,
            FRAME_walk26, insane_frames_walk_insane, insane_walk);

    private static final mmove_t insane_move_run_insane = new mmove_t(FRAME_walk1,
            FRAME_walk26, insane_frames_walk_insane, insane_run);

    private static final mframe_t[] insane_frames_stand_pain = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_stand_pain = new mmove_t(FRAME_st_pain2,
            FRAME_st_pain12, insane_frames_stand_pain, insane_run);

    private static final mframe_t[] insane_frames_stand_death = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_stand_death = new mmove_t(FRAME_st_death2,
            FRAME_st_death18, insane_frames_stand_death, insane_dead);

    private static final mframe_t[] insane_frames_crawl = {
            new mframe_t(GameAI.ai_walk, 0, insane_scream),
            new mframe_t(GameAI.ai_walk, 1.5f, null),
            new mframe_t(GameAI.ai_walk, 2.1f, null),
            new mframe_t(GameAI.ai_walk, 3.6f, null),
            new mframe_t(GameAI.ai_walk, 2f, null),
            new mframe_t(GameAI.ai_walk, 0.9f, null),
            new mframe_t(GameAI.ai_walk, 3f, null),
            new mframe_t(GameAI.ai_walk, 3.4f, null),
            new mframe_t(GameAI.ai_walk, 2.4f, null) };

    private static final mmove_t insane_move_crawl = new mmove_t(FRAME_crawl1, FRAME_crawl9,
            insane_frames_crawl, null);

    private static final mmove_t insane_move_runcrawl = new mmove_t(FRAME_crawl1,
            FRAME_crawl9, insane_frames_crawl, null);

    private static final mframe_t[] insane_frames_crawl_pain = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_crawl_pain = new mmove_t(FRAME_cr_pain2,
            FRAME_cr_pain10, insane_frames_crawl_pain, insane_run);

    private static final mframe_t[] insane_frames_crawl_death = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_crawl_death = new mmove_t(FRAME_cr_death10,
            FRAME_cr_death16, insane_frames_crawl_death, insane_dead);

    private static final mframe_t[] insane_frames_cross = {
            new mframe_t(GameAI.ai_move, 0, insane_moan),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_cross = new mmove_t(FRAME_cross1, FRAME_cross15,
            insane_frames_cross, insane_cross);

    private static final mframe_t[] insane_frames_struggle_cross = {
            new mframe_t(GameAI.ai_move, 0, insane_scream),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t insane_move_struggle_cross = new mmove_t(FRAME_cross16,
            FRAME_cross30, insane_frames_struggle_cross, insane_cross);

    /*
     * QUAKED misc_insane (1 .5 0) (-16 -16 -24) (16 16 32) Ambush Trigger_Spawn
     * CRAWL CRUCIFIED STAND_GROUND ALWAYS_STAND
     */
    public static void SP_misc_insane(edict_t self) {
        

        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_fist = game_import_t.soundindex("insane/insane11.wav");
        sound_shake = game_import_t.soundindex("insane/insane5.wav");
        sound_moan = game_import_t.soundindex("insane/insane7.wav");
        sound_scream[0] = game_import_t.soundindex("insane/insane1.wav");
        sound_scream[1] = game_import_t.soundindex("insane/insane2.wav");
        sound_scream[2] = game_import_t.soundindex("insane/insane3.wav");
        sound_scream[3] = game_import_t.soundindex("insane/insane4.wav");
        sound_scream[4] = game_import_t.soundindex("insane/insane6.wav");
        sound_scream[5] = game_import_t.soundindex("insane/insane8.wav");
        sound_scream[6] = game_import_t.soundindex("insane/insane9.wav");
        sound_scream[7] = game_import_t.soundindex("insane/insane10.wav");

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/insane/tris.md2");

        Math3D.VectorSet(self.mins, -16, -16, -24);
        Math3D.VectorSet(self.maxs, 16, 16, 32);

        self.health = 100;
        self.gib_health = -50;
        self.mass = 300;

        self.pain = insane_pain;
        self.die = insane_die;

        self.monsterinfo.stand = insane_stand;
        self.monsterinfo.walk = insane_walk;
        self.monsterinfo.run = insane_run;
        self.monsterinfo.dodge = null;
        self.monsterinfo.attack = null;
        self.monsterinfo.melee = null;
        self.monsterinfo.sight = null;
        self.monsterinfo.aiflags |= Defines.AI_GOOD_GUY;

        
        
        
        
        

        game_import_t.linkentity(self);

        if ((self.spawnflags & 16) != 0) 
            self.monsterinfo.aiflags |= Defines.AI_STAND_GROUND;

        self.monsterinfo.currentmove = insane_move_stand_normal;

        self.monsterinfo.scale = MODEL_SCALE;
        if ((self.spawnflags & 8) != 0) 
        {
            Math3D.VectorSet(self.mins, -16, 0, 0);
            Math3D.VectorSet(self.maxs, 16, 8, 32);
            self.flags |= Defines.FL_NO_KNOCKBACK;
            GameAI.flymonster_start.think(self);
        } else {
            GameAI.walkmonster_start.think(self);
            self.s.skinnum = Lib.rand() % 3;
        }
    }
}