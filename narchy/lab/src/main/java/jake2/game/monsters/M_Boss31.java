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

public class M_Boss31 {

    private static final int FRAME_attak101 = 0;

    public static final int FRAME_attak102 = 1;

    public static final int FRAME_attak103 = 2;

    public static final int FRAME_attak104 = 3;

    public static final int FRAME_attak105 = 4;

    public static final int FRAME_attak106 = 5;

    public static final int FRAME_attak107 = 6;

    private static final int FRAME_attak108 = 7;

    private static final int FRAME_attak109 = 8;

    public static final int FRAME_attak110 = 9;

    public static final int FRAME_attak111 = 10;

    public static final int FRAME_attak112 = 11;

    public static final int FRAME_attak113 = 12;

    private static final int FRAME_attak114 = 13;

    private static final int FRAME_attak115 = 14;

    public static final int FRAME_attak116 = 15;

    public static final int FRAME_attak117 = 16;

    private static final int FRAME_attak118 = 17;

    private static final int FRAME_attak201 = 18;

    public static final int FRAME_attak202 = 19;

    public static final int FRAME_attak203 = 20;

    public static final int FRAME_attak204 = 21;

    public static final int FRAME_attak205 = 22;

    public static final int FRAME_attak206 = 23;

    public static final int FRAME_attak207 = 24;

    private static final int FRAME_attak208 = 25;

    public static final int FRAME_attak209 = 26;

    public static final int FRAME_attak210 = 27;

    public static final int FRAME_attak211 = 28;

    public static final int FRAME_attak212 = 29;

    private static final int FRAME_attak213 = 30;

    private static final int FRAME_death01 = 31;

    public static final int FRAME_death02 = 32;

    public static final int FRAME_death03 = 33;

    public static final int FRAME_death04 = 34;

    public static final int FRAME_death05 = 35;

    public static final int FRAME_death06 = 36;

    public static final int FRAME_death07 = 37;

    public static final int FRAME_death08 = 38;

    public static final int FRAME_death09 = 39;

    public static final int FRAME_death10 = 40;

    public static final int FRAME_death11 = 41;

    public static final int FRAME_death12 = 42;

    public static final int FRAME_death13 = 43;

    public static final int FRAME_death14 = 44;

    public static final int FRAME_death15 = 45;

    public static final int FRAME_death16 = 46;

    public static final int FRAME_death17 = 47;

    public static final int FRAME_death18 = 48;

    public static final int FRAME_death19 = 49;

    public static final int FRAME_death20 = 50;

    public static final int FRAME_death21 = 51;

    public static final int FRAME_death22 = 52;

    public static final int FRAME_death23 = 53;

    public static final int FRAME_death24 = 54;

    public static final int FRAME_death25 = 55;

    public static final int FRAME_death26 = 56;

    public static final int FRAME_death27 = 57;

    public static final int FRAME_death28 = 58;

    public static final int FRAME_death29 = 59;

    public static final int FRAME_death30 = 60;

    public static final int FRAME_death31 = 61;

    public static final int FRAME_death32 = 62;

    public static final int FRAME_death33 = 63;

    public static final int FRAME_death34 = 64;

    public static final int FRAME_death35 = 65;

    public static final int FRAME_death36 = 66;

    public static final int FRAME_death37 = 67;

    public static final int FRAME_death38 = 68;

    public static final int FRAME_death39 = 69;

    public static final int FRAME_death40 = 70;

    public static final int FRAME_death41 = 71;

    public static final int FRAME_death42 = 72;

    public static final int FRAME_death43 = 73;

    public static final int FRAME_death44 = 74;

    public static final int FRAME_death45 = 75;

    public static final int FRAME_death46 = 76;

    public static final int FRAME_death47 = 77;

    public static final int FRAME_death48 = 78;

    public static final int FRAME_death49 = 79;

    private static final int FRAME_death50 = 80;

    private static final int FRAME_pain101 = 81;

    public static final int FRAME_pain102 = 82;

    private static final int FRAME_pain103 = 83;

    private static final int FRAME_pain201 = 84;

    public static final int FRAME_pain202 = 85;

    private static final int FRAME_pain203 = 86;

    private static final int FRAME_pain301 = 87;

    public static final int FRAME_pain302 = 88;

    public static final int FRAME_pain303 = 89;

    public static final int FRAME_pain304 = 90;

    public static final int FRAME_pain305 = 91;

    public static final int FRAME_pain306 = 92;

    public static final int FRAME_pain307 = 93;

    public static final int FRAME_pain308 = 94;

    public static final int FRAME_pain309 = 95;

    public static final int FRAME_pain310 = 96;

    public static final int FRAME_pain311 = 97;

    public static final int FRAME_pain312 = 98;

    public static final int FRAME_pain313 = 99;

    public static final int FRAME_pain314 = 100;

    public static final int FRAME_pain315 = 101;

    public static final int FRAME_pain316 = 102;

    public static final int FRAME_pain317 = 103;

    public static final int FRAME_pain318 = 104;

    public static final int FRAME_pain319 = 105;

    public static final int FRAME_pain320 = 106;

    public static final int FRAME_pain321 = 107;

    public static final int FRAME_pain322 = 108;

    public static final int FRAME_pain323 = 109;

    public static final int FRAME_pain324 = 110;

    private static final int FRAME_pain325 = 111;

    private static final int FRAME_stand01 = 112;

    public static final int FRAME_stand02 = 113;

    public static final int FRAME_stand03 = 114;

    public static final int FRAME_stand04 = 115;

    public static final int FRAME_stand05 = 116;

    public static final int FRAME_stand06 = 117;

    public static final int FRAME_stand07 = 118;

    public static final int FRAME_stand08 = 119;

    public static final int FRAME_stand09 = 120;

    public static final int FRAME_stand10 = 121;

    public static final int FRAME_stand11 = 122;

    public static final int FRAME_stand12 = 123;

    public static final int FRAME_stand13 = 124;

    public static final int FRAME_stand14 = 125;

    public static final int FRAME_stand15 = 126;

    public static final int FRAME_stand16 = 127;

    public static final int FRAME_stand17 = 128;

    public static final int FRAME_stand18 = 129;

    public static final int FRAME_stand19 = 130;

    public static final int FRAME_stand20 = 131;

    public static final int FRAME_stand21 = 132;

    public static final int FRAME_stand22 = 133;

    public static final int FRAME_stand23 = 134;

    public static final int FRAME_stand24 = 135;

    public static final int FRAME_stand25 = 136;

    public static final int FRAME_stand26 = 137;

    public static final int FRAME_stand27 = 138;

    public static final int FRAME_stand28 = 139;

    public static final int FRAME_stand29 = 140;

    public static final int FRAME_stand30 = 141;

    public static final int FRAME_stand31 = 142;

    public static final int FRAME_stand32 = 143;

    public static final int FRAME_stand33 = 144;

    public static final int FRAME_stand34 = 145;

    public static final int FRAME_stand35 = 146;

    public static final int FRAME_stand36 = 147;

    public static final int FRAME_stand37 = 148;

    public static final int FRAME_stand38 = 149;

    public static final int FRAME_stand39 = 150;

    public static final int FRAME_stand40 = 151;

    public static final int FRAME_stand41 = 152;

    public static final int FRAME_stand42 = 153;

    public static final int FRAME_stand43 = 154;

    public static final int FRAME_stand44 = 155;

    public static final int FRAME_stand45 = 156;

    public static final int FRAME_stand46 = 157;

    public static final int FRAME_stand47 = 158;

    public static final int FRAME_stand48 = 159;

    public static final int FRAME_stand49 = 160;

    public static final int FRAME_stand50 = 161;

    private static final int FRAME_stand51 = 162;

    private static final int FRAME_walk01 = 163;

    public static final int FRAME_walk02 = 164;

    public static final int FRAME_walk03 = 165;

    public static final int FRAME_walk04 = 166;

    private static final int FRAME_walk05 = 167;

    private static final int FRAME_walk06 = 168;

    public static final int FRAME_walk07 = 169;

    public static final int FRAME_walk08 = 170;

    public static final int FRAME_walk09 = 171;

    public static final int FRAME_walk10 = 172;

    public static final int FRAME_walk11 = 173;

    public static final int FRAME_walk12 = 174;

    public static final int FRAME_walk13 = 175;

    public static final int FRAME_walk14 = 176;

    public static final int FRAME_walk15 = 177;

    public static final int FRAME_walk16 = 178;

    public static final int FRAME_walk17 = 179;

    public static final int FRAME_walk18 = 180;

    private static final int FRAME_walk19 = 181;

    private static final int FRAME_walk20 = 182;

    public static final int FRAME_walk21 = 183;

    public static final int FRAME_walk22 = 184;

    public static final int FRAME_walk23 = 185;

    public static final int FRAME_walk24 = 186;

    private static final int FRAME_walk25 = 187;

    private static final float MODEL_SCALE = 1.000000f;

    /*
     * ==============================================================================
     * 
     * jorg
     * 
     * ==============================================================================
     */

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_pain3;

    private static int sound_idle;

    private static int sound_death;

    private static int sound_search1;

    private static int sound_search2;

    private static int sound_search3;

    private static int sound_attack1;

    private static int sound_attack2;

    private static int sound_step_left;

    private static int sound_step_right;

    private static int sound_death_hit;

    /*
     * static EntThinkAdapter xxx = new EntThinkAdapter() { public boolean
     * think(edict_t self) { return true; } };
     */

    private static final EntThinkAdapter jorg_search = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_search"; }
        @Override
        public boolean think(edict_t self) {

            float r = Lib.random();

            if (r <= 0.3)
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_search1, 1,
                        Defines.ATTN_NORM, 0);
            else if (r <= 0.6)
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_search2, 1,
                        Defines.ATTN_NORM, 0);
            else
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_search3, 1,
                        Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_idle = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_idle"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_idle, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    static EntThinkAdapter jorg_death_hit = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_death_hit"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_BODY, sound_death_hit, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_step_left = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_step_left"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_BODY, sound_step_left, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_step_right = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_step_right"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_BODY, sound_step_right, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_stand = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_stand"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = jorg_move_stand;
            return true;
        }
    };

    private static final EntThinkAdapter jorg_reattack1 = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_reattack1"; }
        @Override
        public boolean think(edict_t self) {
            if (GameUtil.visible(self, self.enemy))
                if (Lib.random() < 0.9)
                    self.monsterinfo.currentmove = jorg_move_attack1;
                else {
                    self.s.sound = 0;
                    self.monsterinfo.currentmove = jorg_move_end_attack1;
                }
            else {
                self.s.sound = 0;
                self.monsterinfo.currentmove = jorg_move_end_attack1;
            }
            return true;
        }
    };

    private static final EntThinkAdapter jorg_attack1 = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_attack1"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = jorg_move_attack1;
            return true;
        }
    };

    private static final EntPainAdapter jorg_pain = new EntPainAdapter() {
    	@Override
        public String getID() { return "jorg_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {
            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            self.s.sound = 0;

            if (GameBase.level.time < self.pain_debounce_time)
                return;

            
            
            if (damage <= 40)
                if (Lib.random() <= 0.6)
                    return;

            /*
             * If he's entering his attack1 or using attack1, lessen the chance
             * of him going into pain
             */

            if ((self.s.frame >= FRAME_attak101)
                    && (self.s.frame <= FRAME_attak108))
                if (Lib.random() <= 0.005)
                    return;

            if ((self.s.frame >= FRAME_attak109)
                    && (self.s.frame <= FRAME_attak114))
                if (Lib.random() <= 0.00005)
                    return;

            if ((self.s.frame >= FRAME_attak201)
                    && (self.s.frame <= FRAME_attak208))
                if (Lib.random() <= 0.005)
                    return;

            self.pain_debounce_time = GameBase.level.time + 3;
            if (GameBase.skill.value == 3)
                return; 

            if (damage <= 50) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = jorg_move_pain1;
            } else if (damage <= 100) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = jorg_move_pain2;
            } else {
                if (Lib.random() <= 0.3) {
                    game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain3, 1,
                            Defines.ATTN_NORM, 0);
                    self.monsterinfo.currentmove = jorg_move_pain3;
                }
            }

        }
    };

    private static final EntThinkAdapter jorgBFG = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorgBFG"; }
        @Override
        public boolean think(edict_t self) {
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D.G_ProjectSource(self.s.origin,
                    M_Flash.monster_flash_offset[Defines.MZ2_JORG_BFG_1],
                    forward, right, start);

            float[] vec = {0, 0, 0};
            Math3D.VectorCopy(self.enemy.s.origin, vec);
            vec[2] += self.enemy.viewheight;
            float[] dir = {0, 0, 0};
            Math3D.VectorSubtract(vec, start, dir);
            Math3D.VectorNormalize(dir);
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_attack2, 1,
                    Defines.ATTN_NORM, 0);
            /*
             * void monster_fire_bfg (edict_t self, float [] start, float []
             * aimdir, int damage, int speed, int kick, float damage_radius, int
             * flashtype)
             */
            Monster.monster_fire_bfg(self, start, dir, 50, 300, 100, 200,
                    Defines.MZ2_JORG_BFG_1);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_firebullet_right = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_firebullet_right"; }
        @Override
        public boolean think(edict_t self) {
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D
                    .G_ProjectSource(
                            self.s.origin,
                            M_Flash.monster_flash_offset[Defines.MZ2_JORG_MACHINEGUN_R1],
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
                    Defines.MZ2_JORG_MACHINEGUN_R1);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_firebullet_left = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_firebullet_left"; }
        @Override
        public boolean think(edict_t self) {
            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D
                    .G_ProjectSource(
                            self.s.origin,
                            M_Flash.monster_flash_offset[Defines.MZ2_JORG_MACHINEGUN_L1],
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
                    Defines.MZ2_JORG_MACHINEGUN_L1);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_firebullet = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_firebullet"; }
        @Override
        public boolean think(edict_t self) {
            jorg_firebullet_left.think(self);
            jorg_firebullet_right.think(self);
            return true;
        }
    };

    private static final EntThinkAdapter jorg_attack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_attack"; }
        @Override
        public boolean think(edict_t self) {
            float[] vec = { 0, 0, 0 };

            Math3D.VectorSubtract(self.enemy.s.origin, self.s.origin, vec);
            float range = Math3D.VectorLength(vec);

            if (Lib.random() <= 0.75) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_attack1, 1,
                        Defines.ATTN_NORM, 0);
                self.s.sound = game_import_t.soundindex("boss3/w_loop.wav");
                self.monsterinfo.currentmove = jorg_move_start_attack1;
            } else {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_attack2, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = jorg_move_attack2;
            }
            return true;
        }
    };

    /** Was disabled. RST. */
    private static final EntThinkAdapter jorg_dead = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_dead"; }
        @Override
        public boolean think(edict_t self) {
            /*
             * edict_t tempent;
             * 
             * 
             * 16, 16, -8); 
             * VectorSet( self.mins, -60, -60, 0); VectorSet(self.maxs, 60, 60,
             * 72); self.movetype= MOVETYPE_TOSS; self.nextthink= 0;
             * gi.linkentity(self);
             * 
             * tempent= G_Spawn(); VectorCopy(self.s.origin, tempent.s.origin);
             * VectorCopy(self.s.angles, tempent.s.angles); tempent.killtarget=
             * self.killtarget; tempent.target= self.target; tempent.activator=
             * self.enemy; self.killtarget= 0; self.target= 0;
             * SP_monster_makron(tempent);
             *  
             */
            return true;
        }
    };

    private static final EntDieAdapter jorg_die = new EntDieAdapter() {
    	@Override
        public String getID() { return "jorg_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_death, 1,
                    Defines.ATTN_NORM, 0);
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_NO;
            self.s.sound = 0;
            self.count = 0;
            self.monsterinfo.currentmove = jorg_move_death;
        }
    };

    private static final EntThinkAdapter Jorg_CheckAttack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "Jorg_CheckAttack"; }
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
                chance = 0.4f;
            } else if (enemy_range == Defines.RANGE_MID) {
                chance = 0.2f;
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

    
    
    

    private static final mframe_t[] jorg_frames_stand = {
            new mframe_t(GameAI.ai_stand, 0, jorg_idle),
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
            
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 19, null),
            new mframe_t(GameAI.ai_stand, 11, jorg_step_left),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 6, null),
            new mframe_t(GameAI.ai_stand, 9, jorg_step_right),
            new mframe_t(GameAI.ai_stand, 0, null),
            
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -2, null),
            new mframe_t(GameAI.ai_stand, -17, jorg_step_left),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, -12, null),
            
            new mframe_t(GameAI.ai_stand, -14, jorg_step_right) 
    };

    private static final mmove_t jorg_move_stand = new mmove_t(FRAME_stand01, FRAME_stand51,
            jorg_frames_stand, null);

    private static final mframe_t[] jorg_frames_run = {
            new mframe_t(GameAI.ai_run, 17, jorg_step_left),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 12, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 10, null),
            new mframe_t(GameAI.ai_run, 33, jorg_step_right),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 9, null),
            new mframe_t(GameAI.ai_run, 9, null),
            new mframe_t(GameAI.ai_run, 9, null) };

    private static final mmove_t jorg_move_run = new mmove_t(FRAME_walk06, FRAME_walk19,
            jorg_frames_run, null);

    
    
    

    private static final mframe_t[] jorg_frames_start_walk = {
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 6, null),
            new mframe_t(GameAI.ai_walk, 7, null),
            new mframe_t(GameAI.ai_walk, 9, null),
            new mframe_t(GameAI.ai_walk, 15, null) };

    static mmove_t jorg_move_start_walk = new mmove_t(FRAME_walk01,
            FRAME_walk05, jorg_frames_start_walk, null);

    private static final mframe_t[] jorg_frames_walk = {
            new mframe_t(GameAI.ai_walk, 17, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 12, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 10, null),
            new mframe_t(GameAI.ai_walk, 33, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 9, null),
            new mframe_t(GameAI.ai_walk, 9, null),
            new mframe_t(GameAI.ai_walk, 9, null) };

    private static final mmove_t jorg_move_walk = new mmove_t(FRAME_walk06, FRAME_walk19,
            jorg_frames_walk, null);

    private static final mframe_t[] jorg_frames_end_walk = {
            new mframe_t(GameAI.ai_walk, 11, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, -8, null) };

    static mmove_t jorg_move_end_walk = new mmove_t(FRAME_walk20, FRAME_walk25,
            jorg_frames_end_walk, null);

    private static final EntThinkAdapter jorg_walk = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_walk"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = jorg_move_walk;
            return true;
        }
    };

    private static final EntThinkAdapter jorg_run = new EntThinkAdapter() {
    	@Override
        public String getID() { return "jorg_run"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = jorg_move_stand;
            else
                self.monsterinfo.currentmove = jorg_move_run;
            return true;
        }
    };

    private static final mframe_t[] jorg_frames_pain3 = {
            new mframe_t(GameAI.ai_move, -28, null),
            new mframe_t(GameAI.ai_move, -6, null),
            new mframe_t(GameAI.ai_move, -3, jorg_step_left),
            new mframe_t(GameAI.ai_move, -9, null),
            new mframe_t(GameAI.ai_move, 0, jorg_step_right),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -7, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, -11, null),
            new mframe_t(GameAI.ai_move, -4, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 10, null),
            new mframe_t(GameAI.ai_move, 11, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 10, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 10, null),
            new mframe_t(GameAI.ai_move, 7, jorg_step_left),
            new mframe_t(GameAI.ai_move, 17, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, jorg_step_right) };

    private static final mmove_t jorg_move_pain3 = new mmove_t(FRAME_pain301, FRAME_pain325,
            jorg_frames_pain3, jorg_run);

    private static final mframe_t[] jorg_frames_pain2 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t jorg_move_pain2 = new mmove_t(FRAME_pain201, FRAME_pain203,
            jorg_frames_pain2, jorg_run);

    private static final mframe_t[] jorg_frames_pain1 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t jorg_move_pain1 = new mmove_t(FRAME_pain101, FRAME_pain103,
            jorg_frames_pain1, jorg_run);

    private static final mframe_t[] jorg_frames_death1 = {
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
            new mframe_t(GameAI.ai_move, 0, M_Boss32.MakronToss),
            new mframe_t(GameAI.ai_move, 0, M_Supertank.BossExplode) 
    };

    private static final mmove_t jorg_move_death = new mmove_t(FRAME_death01, FRAME_death50,
            jorg_frames_death1, jorg_dead);

    private static final mframe_t[] jorg_frames_attack2 = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, jorgBFG),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t jorg_move_attack2 = new mmove_t(FRAME_attak201,
            FRAME_attak213, jorg_frames_attack2, jorg_run);

    private static final mframe_t[] jorg_frames_start_attack1 = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null) };

    private static final mmove_t jorg_move_start_attack1 = new mmove_t(FRAME_attak101,
            FRAME_attak108, jorg_frames_start_attack1, jorg_attack1);

    private static final mframe_t[] jorg_frames_attack1 = {
            new mframe_t(GameAI.ai_charge, 0, jorg_firebullet),
            new mframe_t(GameAI.ai_charge, 0, jorg_firebullet),
            new mframe_t(GameAI.ai_charge, 0, jorg_firebullet),
            new mframe_t(GameAI.ai_charge, 0, jorg_firebullet),
            new mframe_t(GameAI.ai_charge, 0, jorg_firebullet),
            new mframe_t(GameAI.ai_charge, 0, jorg_firebullet) };

    private static final mmove_t jorg_move_attack1 = new mmove_t(FRAME_attak109,
            FRAME_attak114, jorg_frames_attack1, jorg_reattack1);

    private static final mframe_t[] jorg_frames_end_attack1 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t jorg_move_end_attack1 = new mmove_t(FRAME_attak115,
            FRAME_attak118, jorg_frames_end_attack1, jorg_run);

    /*
     * QUAKED monster_jorg (1 .5 0) (-80 -80 0) (90 90 140) Ambush Trigger_Spawn
     * Sight
     */
    public static void SP_monster_jorg(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_pain1 = game_import_t.soundindex("boss3/bs3pain1.wav");
        sound_pain2 = game_import_t.soundindex("boss3/bs3pain2.wav");
        sound_pain3 = game_import_t.soundindex("boss3/bs3pain3.wav");
        sound_death = game_import_t.soundindex("boss3/bs3deth1.wav");
        sound_attack1 = game_import_t.soundindex("boss3/bs3atck1.wav");
        sound_attack2 = game_import_t.soundindex("boss3/bs3atck2.wav");
        sound_search1 = game_import_t.soundindex("boss3/bs3srch1.wav");
        sound_search2 = game_import_t.soundindex("boss3/bs3srch2.wav");
        sound_search3 = game_import_t.soundindex("boss3/bs3srch3.wav");
        sound_idle = game_import_t.soundindex("boss3/bs3idle1.wav");
        sound_step_left = game_import_t.soundindex("boss3/step1.wav");
        sound_step_right = game_import_t.soundindex("boss3/step2.wav");
        int sound_firegun = game_import_t.soundindex("boss3/xfire.wav");
        sound_death_hit = game_import_t.soundindex("boss3/d_hit.wav");

        M_Boss32.MakronPrecache();

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/boss3/rider/tris.md2");
        self.s.modelindex2 = game_import_t
                .modelindex("models/monsters/boss3/jorg/tris.md2");
        Math3D.VectorSet(self.mins, -80, -80, 0);
        Math3D.VectorSet(self.maxs, 80, 80, 140);

        self.health = 3000;
        self.gib_health = -2000;
        self.mass = 1000;

        self.pain = jorg_pain;
        self.die = jorg_die;
        self.monsterinfo.stand = jorg_stand;
        self.monsterinfo.walk = jorg_walk;
        self.monsterinfo.run = jorg_run;
        self.monsterinfo.dodge = null;
        self.monsterinfo.attack = jorg_attack;
        self.monsterinfo.search = jorg_search;
        self.monsterinfo.melee = null;
        self.monsterinfo.sight = null;
        self.monsterinfo.checkattack = Jorg_CheckAttack;
        game_import_t.linkentity(self);

        self.monsterinfo.currentmove = jorg_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.walkmonster_start.think(self);
    }
}