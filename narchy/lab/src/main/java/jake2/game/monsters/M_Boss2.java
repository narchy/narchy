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

public class M_Boss2 {

    private static final int FRAME_stand30 = 0;

    public static final int FRAME_stand31 = 1;

    public static final int FRAME_stand32 = 2;

    public static final int FRAME_stand33 = 3;

    public static final int FRAME_stand34 = 4;

    public static final int FRAME_stand35 = 5;

    public static final int FRAME_stand36 = 6;

    public static final int FRAME_stand37 = 7;

    public static final int FRAME_stand38 = 8;

    public static final int FRAME_stand39 = 9;

    public static final int FRAME_stand40 = 10;

    public static final int FRAME_stand41 = 11;

    public static final int FRAME_stand42 = 12;

    public static final int FRAME_stand43 = 13;

    public static final int FRAME_stand44 = 14;

    public static final int FRAME_stand45 = 15;

    public static final int FRAME_stand46 = 16;

    public static final int FRAME_stand47 = 17;

    public static final int FRAME_stand48 = 18;

    public static final int FRAME_stand49 = 19;

    private static final int FRAME_stand50 = 20;

    private static final int FRAME_stand1 = 21;

    public static final int FRAME_stand2 = 22;

    public static final int FRAME_stand3 = 23;

    public static final int FRAME_stand4 = 24;

    public static final int FRAME_stand5 = 25;

    public static final int FRAME_stand6 = 26;

    public static final int FRAME_stand7 = 27;

    public static final int FRAME_stand8 = 28;

    public static final int FRAME_stand9 = 29;

    public static final int FRAME_stand10 = 30;

    public static final int FRAME_stand11 = 31;

    public static final int FRAME_stand12 = 32;

    public static final int FRAME_stand13 = 33;

    public static final int FRAME_stand14 = 34;

    public static final int FRAME_stand15 = 35;

    public static final int FRAME_stand16 = 36;

    public static final int FRAME_stand17 = 37;

    public static final int FRAME_stand18 = 38;

    public static final int FRAME_stand19 = 39;

    public static final int FRAME_stand20 = 40;

    public static final int FRAME_stand21 = 41;

    public static final int FRAME_stand22 = 42;

    public static final int FRAME_stand23 = 43;

    public static final int FRAME_stand24 = 44;

    public static final int FRAME_stand25 = 45;

    public static final int FRAME_stand26 = 46;

    public static final int FRAME_stand27 = 47;

    public static final int FRAME_stand28 = 48;

    public static final int FRAME_stand29 = 49;

    private static final int FRAME_walk1 = 50;

    public static final int FRAME_walk2 = 51;

    public static final int FRAME_walk3 = 52;

    public static final int FRAME_walk4 = 53;

    public static final int FRAME_walk5 = 54;

    public static final int FRAME_walk6 = 55;

    public static final int FRAME_walk7 = 56;

    public static final int FRAME_walk8 = 57;

    public static final int FRAME_walk9 = 58;

    public static final int FRAME_walk10 = 59;

    public static final int FRAME_walk11 = 60;

    public static final int FRAME_walk12 = 61;

    public static final int FRAME_walk13 = 62;

    public static final int FRAME_walk14 = 63;

    public static final int FRAME_walk15 = 64;

    public static final int FRAME_walk16 = 65;

    public static final int FRAME_walk17 = 66;

    public static final int FRAME_walk18 = 67;

    public static final int FRAME_walk19 = 68;

    private static final int FRAME_walk20 = 69;

    private static final int FRAME_attack1 = 70;

    public static final int FRAME_attack2 = 71;

    public static final int FRAME_attack3 = 72;

    public static final int FRAME_attack4 = 73;

    public static final int FRAME_attack5 = 74;

    public static final int FRAME_attack6 = 75;

    public static final int FRAME_attack7 = 76;

    public static final int FRAME_attack8 = 77;

    private static final int FRAME_attack9 = 78;

    private static final int FRAME_attack10 = 79;

    public static final int FRAME_attack11 = 80;

    public static final int FRAME_attack12 = 81;

    public static final int FRAME_attack13 = 82;

    public static final int FRAME_attack14 = 83;

    private static final int FRAME_attack15 = 84;

    private static final int FRAME_attack16 = 85;

    public static final int FRAME_attack17 = 86;

    public static final int FRAME_attack18 = 87;

    private static final int FRAME_attack19 = 88;

    private static final int FRAME_attack20 = 89;

    public static final int FRAME_attack21 = 90;

    public static final int FRAME_attack22 = 91;

    public static final int FRAME_attack23 = 92;

    public static final int FRAME_attack24 = 93;

    public static final int FRAME_attack25 = 94;

    public static final int FRAME_attack26 = 95;

    public static final int FRAME_attack27 = 96;

    public static final int FRAME_attack28 = 97;

    public static final int FRAME_attack29 = 98;

    public static final int FRAME_attack30 = 99;

    public static final int FRAME_attack31 = 100;

    public static final int FRAME_attack32 = 101;

    public static final int FRAME_attack33 = 102;

    public static final int FRAME_attack34 = 103;

    public static final int FRAME_attack35 = 104;

    public static final int FRAME_attack36 = 105;

    public static final int FRAME_attack37 = 106;

    public static final int FRAME_attack38 = 107;

    public static final int FRAME_attack39 = 108;

    private static final int FRAME_attack40 = 109;

    private static final int FRAME_pain2 = 110;

    public static final int FRAME_pain3 = 111;

    public static final int FRAME_pain4 = 112;

    public static final int FRAME_pain5 = 113;

    public static final int FRAME_pain6 = 114;

    public static final int FRAME_pain7 = 115;

    public static final int FRAME_pain8 = 116;

    public static final int FRAME_pain9 = 117;

    public static final int FRAME_pain10 = 118;

    public static final int FRAME_pain11 = 119;

    public static final int FRAME_pain12 = 120;

    public static final int FRAME_pain13 = 121;

    public static final int FRAME_pain14 = 122;

    public static final int FRAME_pain15 = 123;

    public static final int FRAME_pain16 = 124;

    public static final int FRAME_pain17 = 125;

    public static final int FRAME_pain18 = 126;

    private static final int FRAME_pain19 = 127;

    private static final int FRAME_pain20 = 128;

    public static final int FRAME_pain21 = 129;

    public static final int FRAME_pain22 = 130;

    private static final int FRAME_pain23 = 131;

    private static final int FRAME_death2 = 132;

    public static final int FRAME_death3 = 133;

    public static final int FRAME_death4 = 134;

    public static final int FRAME_death5 = 135;

    public static final int FRAME_death6 = 136;

    public static final int FRAME_death7 = 137;

    public static final int FRAME_death8 = 138;

    public static final int FRAME_death9 = 139;

    public static final int FRAME_death10 = 140;

    public static final int FRAME_death11 = 141;

    public static final int FRAME_death12 = 142;

    public static final int FRAME_death13 = 143;

    public static final int FRAME_death14 = 144;

    public static final int FRAME_death15 = 145;

    public static final int FRAME_death16 = 146;

    public static final int FRAME_death17 = 147;

    public static final int FRAME_death18 = 148;

    public static final int FRAME_death19 = 149;

    public static final int FRAME_death20 = 150;

    public static final int FRAME_death21 = 151;

    public static final int FRAME_death22 = 152;

    public static final int FRAME_death23 = 153;

    public static final int FRAME_death24 = 154;

    public static final int FRAME_death25 = 155;

    public static final int FRAME_death26 = 156;

    public static final int FRAME_death27 = 157;

    public static final int FRAME_death28 = 158;

    public static final int FRAME_death29 = 159;

    public static final int FRAME_death30 = 160;

    public static final int FRAME_death31 = 161;

    public static final int FRAME_death32 = 162;

    public static final int FRAME_death33 = 163;

    public static final int FRAME_death34 = 164;

    public static final int FRAME_death35 = 165;

    public static final int FRAME_death36 = 166;

    public static final int FRAME_death37 = 167;

    public static final int FRAME_death38 = 168;

    public static final int FRAME_death39 = 169;

    public static final int FRAME_death40 = 170;

    public static final int FRAME_death41 = 171;

    public static final int FRAME_death42 = 172;

    public static final int FRAME_death43 = 173;

    public static final int FRAME_death44 = 174;

    public static final int FRAME_death45 = 175;

    public static final int FRAME_death46 = 176;

    public static final int FRAME_death47 = 177;

    public static final int FRAME_death48 = 178;

    public static final int FRAME_death49 = 179;

    private static final int FRAME_death50 = 180;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_pain3;

    private static int sound_death;

    private static int sound_search1;

    private static final EntThinkAdapter boss2_stand = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_stand"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = boss2_move_stand;
            return true;
        }
    };

    private static final EntThinkAdapter boss2_run = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_run"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = boss2_move_stand;
            else
                self.monsterinfo.currentmove = boss2_move_run;
            return true;
        }
    };

    private static final EntThinkAdapter boss2_walk = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_walk"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = boss2_move_stand;

            self.monsterinfo.currentmove = boss2_move_walk;
            return true;
        }
    };

    private static final EntThinkAdapter boss2_attack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_attack"; }
        @Override
        public boolean think(edict_t self) {
            float[] vec = { 0, 0, 0 };

            Math3D.VectorSubtract(self.enemy.s.origin, self.s.origin, vec);
            float range = Math3D.VectorLength(vec);

            if (range <= 125) {
                self.monsterinfo.currentmove = boss2_move_attack_pre_mg;
            } else {
                if (Lib.random() <= 0.6)
                    self.monsterinfo.currentmove = boss2_move_attack_pre_mg;
                else
                    self.monsterinfo.currentmove = boss2_move_attack_rocket;
            }
            return true;
        }
    };

    private static final EntThinkAdapter boss2_attack_mg = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_attack_mg"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = boss2_move_attack_mg;
            return true;
        }
    };

    private static final EntThinkAdapter boss2_reattack_mg = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_reattack_mg"; }
        @Override
        public boolean think(edict_t self) {
            if (GameUtil.infront(self, self.enemy))
                if (Lib.random() <= 0.7)
                    self.monsterinfo.currentmove = boss2_move_attack_mg;
                else
                    self.monsterinfo.currentmove = boss2_move_attack_post_mg;
            else
                self.monsterinfo.currentmove = boss2_move_attack_post_mg;
            return true;
        }
    };

    private static final EntPainAdapter boss2_pain = new EntPainAdapter() {
    	@Override
        public String getID() { return "boss2_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {
            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            if (GameBase.level.time < self.pain_debounce_time)
                return;

            self.pain_debounce_time = GameBase.level.time + 3;
            
            if (damage < 10) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain3, 1,
                        Defines.ATTN_NONE, 0);
                self.monsterinfo.currentmove = boss2_move_pain_light;
            } else if (damage < 30) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NONE, 0);
                self.monsterinfo.currentmove = boss2_move_pain_light;
            } else {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NONE, 0);
                self.monsterinfo.currentmove = boss2_move_pain_heavy;
            }
        }
    };

    private static final EntThinkAdapter boss2_dead = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_dead"; }
        @Override
        public boolean think(edict_t self) {
            Math3D.VectorSet(self.mins, -56, -56, 0);
            Math3D.VectorSet(self.maxs, 56, 56, 80);
            self.movetype = Defines.MOVETYPE_TOSS;
            self.svflags |= Defines.SVF_DEADMONSTER;
            self.nextthink = 0;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final EntDieAdapter boss2_die = new EntDieAdapter() {
    	@Override
        public String getID() { return "boss2_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_death, 1,
                    Defines.ATTN_NONE, 0);
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_NO;
            self.count = 0;
            self.monsterinfo.currentmove = boss2_move_death;

        }
    };

    private static final EntThinkAdapter Boss2_CheckAttack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "Boss2_CheckAttack"; }
        @Override
        public boolean think(edict_t self) {

            if (self.enemy.health > 0) {

                float[] spot1 = {0, 0, 0};
                Math3D.VectorCopy(self.s.origin, spot1);
                spot1[2] += self.viewheight;
                float[] spot2 = {0, 0, 0};
                Math3D.VectorCopy(self.enemy.s.origin, spot2);
                spot2[2] += self.enemy.viewheight;

                trace_t tr = game_import_t.trace(spot1, null, null, spot2, self,
                        Defines.CONTENTS_SOLID | Defines.CONTENTS_MONSTER
                                | Defines.CONTENTS_SLIME
                                | Defines.CONTENTS_LAVA);


                if (tr.ent != self.enemy)
                    return false;
            }

            int enemy_range = GameUtil.range(self, self.enemy);
            float[] temp = {0, 0, 0};
            Math3D.VectorSubtract(self.enemy.s.origin, self.s.origin, temp);

            self.ideal_yaw = Math3D.vectoyaw(temp);

            
            if (enemy_range == Defines.RANGE_MELEE) {
                if (self.monsterinfo.melee != null)
                    self.monsterinfo.attack_state = Defines.AS_MELEE;
                else
                    self.monsterinfo.attack_state = Defines.AS_MISSILE;
                return true;
            }

            
            if (self.monsterinfo.attack == null)
                return false;

            if (GameBase.level.time < self.monsterinfo.attack_finished)
                return false;

            if (enemy_range == Defines.RANGE_FAR)
                return false;

            float chance;
            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0) {
                chance = 0.4f;
            } else if (enemy_range == Defines.RANGE_MELEE) {
                chance = 0.8f;
            } else if (enemy_range == Defines.RANGE_NEAR) {
                chance = 0.8f;
            } else if (enemy_range == Defines.RANGE_MID) {
                chance = 0.8f;
            } else {
                return false;
            }

            if (Lib.random() < chance) {
                self.monsterinfo.attack_state = Defines.AS_MISSILE;
                self.monsterinfo.attack_finished = GameBase.level.time + 2
                        * Lib.random();
                return true;
            }

            if ((self.flags & Defines.FL_FLY) != 0) {
                if (Lib.random() < 0.3)
                    self.monsterinfo.attack_state = Defines.AS_SLIDING;
                else
                    self.monsterinfo.attack_state = Defines.AS_STRAIGHT;
            }

            return false;
        }
    };

    private static final EntThinkAdapter boss2_search = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_search"; }
        @Override
        public boolean think(edict_t self) {
            if (Lib.random() < 0.5)
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_search1, 1,
                        Defines.ATTN_NONE, 0);
            return true;
        }
    };

    private static final EntThinkAdapter Boss2Rocket = new EntThinkAdapter() {
    	@Override
        public String getID() { return "Boss2Rocket"; }
        @Override
        public boolean think(edict_t self) {
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);


            float[] start = {0, 0, 0};
            Math3D.G_ProjectSource(self.s.origin,
                    M_Flash.monster_flash_offset[Defines.MZ2_BOSS2_ROCKET_1],
                    forward, right, start);
            float[] vec = {0, 0, 0};
            Math3D.VectorCopy(self.enemy.s.origin, vec);
            vec[2] += self.enemy.viewheight;
            float[] dir = {0, 0, 0};
            Math3D.VectorSubtract(vec, start, dir);
            Math3D.VectorNormalize(dir);
            Monster.monster_fire_rocket(self, start, dir, 50, 500,
                    Defines.MZ2_BOSS2_ROCKET_1);

            
            Math3D.G_ProjectSource(self.s.origin,
                    M_Flash.monster_flash_offset[Defines.MZ2_BOSS2_ROCKET_2],
                    forward, right, start);
            Math3D.VectorCopy(self.enemy.s.origin, vec);
            vec[2] += self.enemy.viewheight;
            Math3D.VectorSubtract(vec, start, dir);
            Math3D.VectorNormalize(dir);
            Monster.monster_fire_rocket(self, start, dir, 50, 500,
                    Defines.MZ2_BOSS2_ROCKET_2);

            
            Math3D.G_ProjectSource(self.s.origin,
                    M_Flash.monster_flash_offset[Defines.MZ2_BOSS2_ROCKET_3],
                    forward, right, start);
            Math3D.VectorCopy(self.enemy.s.origin, vec);
            vec[2] += self.enemy.viewheight;
            Math3D.VectorSubtract(vec, start, dir);
            Math3D.VectorNormalize(dir);
            Monster.monster_fire_rocket(self, start, dir, 50, 500,
                    Defines.MZ2_BOSS2_ROCKET_3);

            
            Math3D.G_ProjectSource(self.s.origin,
                    M_Flash.monster_flash_offset[Defines.MZ2_BOSS2_ROCKET_4],
                    forward, right, start);
            Math3D.VectorCopy(self.enemy.s.origin, vec);
            vec[2] += self.enemy.viewheight;
            Math3D.VectorSubtract(vec, start, dir);
            Math3D.VectorNormalize(dir);
            Monster.monster_fire_rocket(self, start, dir, 50, 500,
                    Defines.MZ2_BOSS2_ROCKET_4);
            return true;
        }
    };

    private static final EntThinkAdapter boss2_firebullet_right = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_firebullet_right"; }
        @Override
        public boolean think(edict_t self) {
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D
                    .G_ProjectSource(
                            self.s.origin,
                            M_Flash.monster_flash_offset[Defines.MZ2_BOSS2_MACHINEGUN_R1],
                            forward, right, start);

            float[] target = {0,
                    0, 0};
            Math3D.VectorMA(self.enemy.s.origin, -0.2f, self.enemy.velocity,
                    target);
            target[2] += self.enemy.viewheight;
            Math3D.VectorSubtract(target, start, forward);
            Math3D.VectorNormalize(forward);

            Monster.monster_fire_bullet(self, start, forward, 6, 4,
                    Defines.DEFAULT_BULLET_HSPREAD,
                    Defines.DEFAULT_BULLET_VSPREAD,
                    Defines.MZ2_BOSS2_MACHINEGUN_R1);

            return true;
        }
    };

    private static final EntThinkAdapter boss2_firebullet_left = new EntThinkAdapter() {
    	@Override
        public String getID() { return "boss2_firebullet_left"; }
        @Override
        public boolean think(edict_t self) {
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D
                    .G_ProjectSource(
                            self.s.origin,
                            M_Flash.monster_flash_offset[Defines.MZ2_BOSS2_MACHINEGUN_L1],
                            forward, right, start);

            float[] target = {0,
                    0, 0};
            Math3D.VectorMA(self.enemy.s.origin, -0.2f, self.enemy.velocity,
                    target);

            target[2] += self.enemy.viewheight;
            Math3D.VectorSubtract(target, start, forward);
            Math3D.VectorNormalize(forward);

            Monster.monster_fire_bullet(self, start, forward, 6, 4,
                    Defines.DEFAULT_BULLET_HSPREAD,
                    Defines.DEFAULT_BULLET_VSPREAD,
                    Defines.MZ2_BOSS2_MACHINEGUN_L1);

            return true;
        }
    };

    private static final EntThinkAdapter Boss2MachineGun = new EntThinkAdapter() {
    	@Override
        public String getID() { return "Boss2MachineGun"; }
        @Override
        public boolean think(edict_t self) {
            /*
             * RST: this was disabled ! float[] forward={0,0,0}, right={0,0,0};
             * float[] start={0,0,0}; float[] dir={0,0,0}; float[] vec={0,0,0};
             * int flash_number;
             * 
             * AngleVectors (self.s.angles, forward, right, null);
             * 
             * flash_number = MZ2_BOSS2_MACHINEGUN_1 + (self.s.frame -
             * FRAME_attack10); G_ProjectSource (self.s.origin,
             * monster_flash_offset[flash_number], forward, right, start);
             * 
             * VectorCopy (self.enemy.s.origin, vec); vec[2] +=
             * self.enemy.viewheight; VectorSubtract (vec, start, dir);
             * VectorNormalize (dir); monster_fire_bullet (self, start, dir, 3,
             * 4, DEFAULT_BULLET_HSPREAD, DEFAULT_BULLET_VSPREAD, flash_number);
             */
            boss2_firebullet_left.think(self);
            boss2_firebullet_right.think(self);
            return true;
        }
    };

    private static final mframe_t[] boss2_frames_stand = {
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

    private static final mmove_t boss2_move_stand = new mmove_t(FRAME_stand30, FRAME_stand50,
            boss2_frames_stand, null);

    private static final mframe_t[] boss2_frames_fidget = {
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
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null) };

    static mmove_t boss2_move_fidget = new mmove_t(FRAME_stand1, FRAME_stand30,
            boss2_frames_fidget, null);

    private static final mframe_t[] boss2_frames_walk = {
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 8, null) };

    private static final mmove_t boss2_move_walk = new mmove_t(FRAME_walk1, FRAME_walk20,
            boss2_frames_walk, null);

    private static final mframe_t[] boss2_frames_run = {
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 8, null) };

    private static final mmove_t boss2_move_run = new mmove_t(FRAME_walk1, FRAME_walk20,
            boss2_frames_run, null);

    private static final mframe_t[] boss2_frames_attack_pre_mg = {
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, boss2_attack_mg) };

    private static final mmove_t boss2_move_attack_pre_mg = new mmove_t(FRAME_attack1,
            FRAME_attack9, boss2_frames_attack_pre_mg, null);

    
    private static final mframe_t[] boss2_frames_attack_mg = {
            new mframe_t(GameAI.ai_charge, 1, Boss2MachineGun),
            new mframe_t(GameAI.ai_charge, 1, Boss2MachineGun),
            new mframe_t(GameAI.ai_charge, 1, Boss2MachineGun),
            new mframe_t(GameAI.ai_charge, 1, Boss2MachineGun),
            new mframe_t(GameAI.ai_charge, 1, Boss2MachineGun),
            new mframe_t(GameAI.ai_charge, 1, boss2_reattack_mg) };

    private static final mmove_t boss2_move_attack_mg = new mmove_t(FRAME_attack10,
            FRAME_attack15, boss2_frames_attack_mg, null);

    private static final mframe_t[] boss2_frames_attack_post_mg = {
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null) };

    private static final mmove_t boss2_move_attack_post_mg = new mmove_t(FRAME_attack16,
            FRAME_attack19, boss2_frames_attack_post_mg, boss2_run);

    private static final mframe_t[] boss2_frames_attack_rocket = {
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_move, -20, Boss2Rocket),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 1, null) };

    private static final mmove_t boss2_move_attack_rocket = new mmove_t(FRAME_attack20,
            FRAME_attack40, boss2_frames_attack_rocket, boss2_run);

    private static final mframe_t[] boss2_frames_pain_heavy = {
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
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t boss2_move_pain_heavy = new mmove_t(FRAME_pain2,
            FRAME_pain19, boss2_frames_pain_heavy, boss2_run);

    private static final mframe_t[] boss2_frames_pain_light = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t boss2_move_pain_light = new mmove_t(FRAME_pain20,
            FRAME_pain23, boss2_frames_pain_light, boss2_run);

    private static final mframe_t[] boss2_frames_death = {
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
            new mframe_t(GameAI.ai_move, 0, M_Supertank.BossExplode) };

    /*
     * static EntThinkAdapter xxx = new EntThinkAdapter() { public boolean
     * think(edict_t self) { return true; } };
     */

    private static final mmove_t boss2_move_death = new mmove_t(FRAME_death2, FRAME_death50,
            boss2_frames_death, boss2_dead);

    /*
     * QUAKED monster_boss2 (1 .5 0) (-56 -56 0) (56 56 80) Ambush Trigger_Spawn
     * Sight
     */
    public static void SP_monster_boss2(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_pain1 = game_import_t.soundindex("bosshovr/bhvpain1.wav");
        sound_pain2 = game_import_t.soundindex("bosshovr/bhvpain2.wav");
        sound_pain3 = game_import_t.soundindex("bosshovr/bhvpain3.wav");
        sound_death = game_import_t.soundindex("bosshovr/bhvdeth1.wav");
        sound_search1 = game_import_t.soundindex("bosshovr/bhvunqv1.wav");

        self.s.sound = game_import_t.soundindex("bosshovr/bhvengn1.wav");

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/boss2/tris.md2");
        Math3D.VectorSet(self.mins, -56, -56, 0);
        Math3D.VectorSet(self.maxs, 56, 56, 80);

        self.health = 2000;
        self.gib_health = -200;
        self.mass = 1000;

        self.flags |= Defines.FL_IMMUNE_LASER;

        self.pain = boss2_pain;
        self.die = boss2_die;

        self.monsterinfo.stand = boss2_stand;
        self.monsterinfo.walk = boss2_walk;
        self.monsterinfo.run = boss2_run;
        self.monsterinfo.attack = boss2_attack;
        self.monsterinfo.search = boss2_search;
        self.monsterinfo.checkattack = Boss2_CheckAttack;
        game_import_t.linkentity(self);

        self.monsterinfo.currentmove = boss2_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.flymonster_start.think(self);
    }
}