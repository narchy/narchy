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

public class M_Mutant {

    

    private static final int FRAME_attack01 = 0;

    private static final int FRAME_attack02 = 1;

    public static final int FRAME_attack03 = 2;

    public static final int FRAME_attack04 = 3;

    private static final int FRAME_attack05 = 4;

    public static final int FRAME_attack06 = 5;

    public static final int FRAME_attack07 = 6;

    private static final int FRAME_attack08 = 7;

    private static final int FRAME_attack09 = 8;

    public static final int FRAME_attack10 = 9;

    public static final int FRAME_attack11 = 10;

    public static final int FRAME_attack12 = 11;

    public static final int FRAME_attack13 = 12;

    public static final int FRAME_attack14 = 13;

    private static final int FRAME_attack15 = 14;

    private static final int FRAME_death101 = 15;

    public static final int FRAME_death102 = 16;

    public static final int FRAME_death103 = 17;

    public static final int FRAME_death104 = 18;

    public static final int FRAME_death105 = 19;

    public static final int FRAME_death106 = 20;

    public static final int FRAME_death107 = 21;

    public static final int FRAME_death108 = 22;

    private static final int FRAME_death109 = 23;

    private static final int FRAME_death201 = 24;

    public static final int FRAME_death202 = 25;

    public static final int FRAME_death203 = 26;

    public static final int FRAME_death204 = 27;

    public static final int FRAME_death205 = 28;

    public static final int FRAME_death206 = 29;

    public static final int FRAME_death207 = 30;

    public static final int FRAME_death208 = 31;

    public static final int FRAME_death209 = 32;

    private static final int FRAME_death210 = 33;

    private static final int FRAME_pain101 = 34;

    public static final int FRAME_pain102 = 35;

    public static final int FRAME_pain103 = 36;

    public static final int FRAME_pain104 = 37;

    private static final int FRAME_pain105 = 38;

    private static final int FRAME_pain201 = 39;

    public static final int FRAME_pain202 = 40;

    public static final int FRAME_pain203 = 41;

    public static final int FRAME_pain204 = 42;

    public static final int FRAME_pain205 = 43;

    private static final int FRAME_pain206 = 44;

    private static final int FRAME_pain301 = 45;

    public static final int FRAME_pain302 = 46;

    public static final int FRAME_pain303 = 47;

    public static final int FRAME_pain304 = 48;

    public static final int FRAME_pain305 = 49;

    public static final int FRAME_pain306 = 50;

    public static final int FRAME_pain307 = 51;

    public static final int FRAME_pain308 = 52;

    public static final int FRAME_pain309 = 53;

    public static final int FRAME_pain310 = 54;

    private static final int FRAME_pain311 = 55;

    private static final int FRAME_run03 = 56;

    public static final int FRAME_run04 = 57;

    public static final int FRAME_run05 = 58;

    public static final int FRAME_run06 = 59;

    public static final int FRAME_run07 = 60;

    private static final int FRAME_run08 = 61;

    private static final int FRAME_stand101 = 62;

    public static final int FRAME_stand102 = 63;

    public static final int FRAME_stand103 = 64;

    public static final int FRAME_stand104 = 65;

    public static final int FRAME_stand105 = 66;

    public static final int FRAME_stand106 = 67;

    public static final int FRAME_stand107 = 68;

    public static final int FRAME_stand108 = 69;

    public static final int FRAME_stand109 = 70;

    public static final int FRAME_stand110 = 71;

    public static final int FRAME_stand111 = 72;

    public static final int FRAME_stand112 = 73;

    public static final int FRAME_stand113 = 74;

    public static final int FRAME_stand114 = 75;

    public static final int FRAME_stand115 = 76;

    public static final int FRAME_stand116 = 77;

    public static final int FRAME_stand117 = 78;

    public static final int FRAME_stand118 = 79;

    public static final int FRAME_stand119 = 80;

    public static final int FRAME_stand120 = 81;

    public static final int FRAME_stand121 = 82;

    public static final int FRAME_stand122 = 83;

    public static final int FRAME_stand123 = 84;

    public static final int FRAME_stand124 = 85;

    public static final int FRAME_stand125 = 86;

    public static final int FRAME_stand126 = 87;

    public static final int FRAME_stand127 = 88;

    public static final int FRAME_stand128 = 89;

    public static final int FRAME_stand129 = 90;

    public static final int FRAME_stand130 = 91;

    public static final int FRAME_stand131 = 92;

    public static final int FRAME_stand132 = 93;

    public static final int FRAME_stand133 = 94;

    public static final int FRAME_stand134 = 95;

    public static final int FRAME_stand135 = 96;

    public static final int FRAME_stand136 = 97;

    public static final int FRAME_stand137 = 98;

    public static final int FRAME_stand138 = 99;

    public static final int FRAME_stand139 = 100;

    public static final int FRAME_stand140 = 101;

    public static final int FRAME_stand141 = 102;

    public static final int FRAME_stand142 = 103;

    public static final int FRAME_stand143 = 104;

    public static final int FRAME_stand144 = 105;

    public static final int FRAME_stand145 = 106;

    public static final int FRAME_stand146 = 107;

    public static final int FRAME_stand147 = 108;

    public static final int FRAME_stand148 = 109;

    public static final int FRAME_stand149 = 110;

    public static final int FRAME_stand150 = 111;

    private static final int FRAME_stand151 = 112;

    private static final int FRAME_stand152 = 113;

    public static final int FRAME_stand153 = 114;

    public static final int FRAME_stand154 = 115;

    private static final int FRAME_stand155 = 116;

    public static final int FRAME_stand156 = 117;

    public static final int FRAME_stand157 = 118;

    public static final int FRAME_stand158 = 119;

    public static final int FRAME_stand159 = 120;

    public static final int FRAME_stand160 = 121;

    public static final int FRAME_stand161 = 122;

    public static final int FRAME_stand162 = 123;

    public static final int FRAME_stand163 = 124;

    private static final int FRAME_stand164 = 125;

    private static final int FRAME_walk01 = 126;

    public static final int FRAME_walk02 = 127;

    public static final int FRAME_walk03 = 128;

    private static final int FRAME_walk04 = 129;

    private static final int FRAME_walk05 = 130;

    public static final int FRAME_walk06 = 131;

    public static final int FRAME_walk07 = 132;

    public static final int FRAME_walk08 = 133;

    public static final int FRAME_walk09 = 134;

    public static final int FRAME_walk10 = 135;

    public static final int FRAME_walk11 = 136;

    public static final int FRAME_walk12 = 137;

    public static final int FRAME_walk13 = 138;

    public static final int FRAME_walk14 = 139;

    public static final int FRAME_walk15 = 140;

    private static final int FRAME_walk16 = 141;

    public static final int FRAME_walk17 = 142;

    public static final int FRAME_walk18 = 143;

    public static final int FRAME_walk19 = 144;

    public static final int FRAME_walk20 = 145;

    public static final int FRAME_walk21 = 146;

    public static final int FRAME_walk22 = 147;

    public static final int FRAME_walk23 = 148;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_swing;

    private static int sound_hit;

    private static int sound_hit2;

    private static int sound_death;

    private static int sound_idle;

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_sight;

    private static int sound_search;

    private static int sound_step1;

    private static int sound_step2;

    private static int sound_step3;

    private static int sound_thud;

    
    
    
    private static final EntThinkAdapter mutant_step = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_step"; }
        @Override
        public boolean think(edict_t self) {
            int n = (Lib.rand() + 1) % 3;
            game_import_t.sound(self, Defines.CHAN_VOICE, switch (n) {
                        case 0 -> sound_step1;
                        case 1 -> sound_step2;
                        default -> sound_step3;
                    }, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntInteractAdapter mutant_sight = new EntInteractAdapter() {
    	@Override
        public String getID(){ return "mutant_sight"; }
        @Override
        public boolean interact(edict_t self, edict_t other) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_sight, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter mutant_search = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_search"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_search, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    static EntThinkAdapter mutant_swing = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_swing"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_swing, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    
    
    

    private static final mframe_t[] mutant_frames_stand = {
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

    private static final mmove_t mutant_move_stand = new mmove_t(FRAME_stand101,
            FRAME_stand151, mutant_frames_stand, null);

    private static final EntThinkAdapter mutant_stand = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_stand"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = mutant_move_stand;
            return true;
        }
    };

    
    
    

    private static final EntThinkAdapter mutant_idle_loop = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_idle_loop"; }
        @Override
        public boolean think(edict_t self) {
            if (Lib.random() < 0.75)
                self.monsterinfo.nextframe = FRAME_stand155;
            return true;
        }
    };

    private static final mframe_t[] mutant_frames_idle = {
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, mutant_idle_loop),
            
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null) };

    private static final mmove_t mutant_move_idle = new mmove_t(FRAME_stand152,
            FRAME_stand164, mutant_frames_idle, mutant_stand);

    private static final EntThinkAdapter mutant_idle = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_idle"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = mutant_move_idle;
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_idle, 1,
                    Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    
    
    

    private static final mframe_t[] mutant_frames_walk = {
            new mframe_t(GameAI.ai_walk, 3, null),
            new mframe_t(GameAI.ai_walk, 1, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 10, null),
            new mframe_t(GameAI.ai_walk, 13, null),
            new mframe_t(GameAI.ai_walk, 10, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 6, null),
            new mframe_t(GameAI.ai_walk, 16, null),
            new mframe_t(GameAI.ai_walk, 15, null),
            new mframe_t(GameAI.ai_walk, 6, null) };

    private static final mmove_t mutant_move_walk = new mmove_t(FRAME_walk05, FRAME_walk16,
            mutant_frames_walk, null);

    private static final EntThinkAdapter mutant_walk_loop = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_walk_loop"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = mutant_move_walk;
            return true;
        }
    };

    private static final mframe_t[] mutant_frames_start_walk = {
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, -2, null),
            new mframe_t(GameAI.ai_walk, 1, null) };

    private static final mmove_t mutant_move_start_walk = new mmove_t(FRAME_walk01,
            FRAME_walk04, mutant_frames_start_walk, mutant_walk_loop);

    private static final EntThinkAdapter mutant_walk = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_walk"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = mutant_move_start_walk;
            return true;
        }
    };

    
    
    

    private static final mframe_t[] mutant_frames_run = {
            new mframe_t(GameAI.ai_run, 40, null),
            new mframe_t(GameAI.ai_run, 40, mutant_step),
            new mframe_t(GameAI.ai_run, 24, null),
            new mframe_t(GameAI.ai_run, 5, mutant_step),
            new mframe_t(GameAI.ai_run, 17, null),
            new mframe_t(GameAI.ai_run, 10, null) };

    private static final mmove_t mutant_move_run = new mmove_t(FRAME_run03, FRAME_run08,
            mutant_frames_run, null);

    private static final EntThinkAdapter mutant_run = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_run"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = mutant_move_stand;
            else
                self.monsterinfo.currentmove = mutant_move_run;

            return true;
        }
    };

    
    
    

    private static final EntThinkAdapter mutant_hit_left = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_hit_left"; }
        @Override
        public boolean think(edict_t self) {
            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, self.mins[0], 8);
            if (GameWeapon.fire_hit(self, aim, (10 + (Lib.rand() % 5)), 100))
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_hit, 1,
                        Defines.ATTN_NORM, 0);
            else
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_swing, 1,
                        Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter mutant_hit_right = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_hit_right"; }
        @Override
        public boolean think(edict_t self) {
            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, self.maxs[0], 8);
            if (GameWeapon.fire_hit(self, aim, (10 + (Lib.rand() % 5)), 100))
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_hit2, 1,
                        Defines.ATTN_NORM, 0);
            else
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_swing, 1,
                        Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter mutant_check_refire = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_check_refire"; }
        @Override
        public boolean think(edict_t self) {
            if (null == self.enemy || !self.enemy.inuse
                    || self.enemy.health <= 0)
                return true;

            if (((GameBase.skill.value == 3) && (Lib.random() < 0.5))
                    || (GameUtil.range(self, self.enemy) == Defines.RANGE_MELEE))
                self.monsterinfo.nextframe = FRAME_attack09;
            return true;
        }
    };

    private static final mframe_t[] mutant_frames_attack = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, mutant_hit_left),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, mutant_hit_right),
            new mframe_t(GameAI.ai_charge, 0, mutant_check_refire) };

    private static final mmove_t mutant_move_attack = new mmove_t(FRAME_attack09,
            FRAME_attack15, mutant_frames_attack, mutant_run);

    private static final EntThinkAdapter mutant_melee = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_melee"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = mutant_move_attack;
            return true;
        }
    };

    
    
    

    private static final EntTouchAdapter mutant_jump_touch = new EntTouchAdapter() {
    	@Override
        public String getID(){ return "mutant_jump_touch"; }

        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (self.health <= 0) {
                self.touch = null;
                return;
            }

            if (other.takedamage != 0) {
                if (Math3D.VectorLength(self.velocity) > 400) {
                    float[] normal = { 0, 0, 0 };

                    Math3D.VectorCopy(self.velocity, normal);
                    Math3D.VectorNormalize(normal);
                    float[] point = {0, 0, 0};
                    Math3D.VectorMA(self.s.origin, self.maxs[0], normal, point);
                    int damage = (int) (40 + 10 * Lib.random());
                    GameCombat.T_Damage(other, self, self, self.velocity, point,
                            normal, damage, damage, 0, Defines.MOD_UNKNOWN);
                }
            }

            if (!M.M_CheckBottom(self)) {
                if (self.groundentity != null) {
                    self.monsterinfo.nextframe = FRAME_attack02;
                    self.touch = null;
                }
                return;
            }

            self.touch = null;
        }
    };

    private static final EntThinkAdapter mutant_jump_takeoff = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_jump_takeoff"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_sight, 1,
                    Defines.ATTN_NORM, 0);
            float[] forward = {0, 0, 0};
            Math3D.AngleVectors(self.s.angles, forward, null, null);
            self.s.origin[2] += 1;
            Math3D.VectorScale(forward, 600, self.velocity);
            self.velocity[2] = 250;
            self.groundentity = null;
            self.monsterinfo.aiflags |= Defines.AI_DUCKED;
            self.monsterinfo.attack_finished = GameBase.level.time + 3;
            self.touch = mutant_jump_touch;
            return true;
        }
    };

    private static final EntThinkAdapter mutant_check_landing = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_check_landing"; }
        @Override
        public boolean think(edict_t self) {
            if (self.groundentity != null) {
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_thud, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.attack_finished = 0;
                self.monsterinfo.aiflags &= ~Defines.AI_DUCKED;
                return true;
            }

            if (GameBase.level.time > self.monsterinfo.attack_finished)
                self.monsterinfo.nextframe = FRAME_attack02;
            else
                self.monsterinfo.nextframe = FRAME_attack05;
            return true;
        }
    };

    private static final mframe_t[] mutant_frames_jump = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 17, null),
            new mframe_t(GameAI.ai_charge, 15, mutant_jump_takeoff),
            new mframe_t(GameAI.ai_charge, 15, null),
            new mframe_t(GameAI.ai_charge, 15, mutant_check_landing),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 3, null),
            new mframe_t(GameAI.ai_charge, 0, null) };

    private static final mmove_t mutant_move_jump = new mmove_t(FRAME_attack01,
            FRAME_attack08, mutant_frames_jump, mutant_run);

    private static final EntThinkAdapter mutant_jump = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_jump"; }
        @Override
        public boolean think(edict_t self) {

            self.monsterinfo.currentmove = mutant_move_jump;
            return true;
        }
    };

    
    
    
    private static final EntThinkAdapter mutant_check_melee = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_check_melee"; }
        @Override
        public boolean think(edict_t self) {
            return GameUtil.range(self, self.enemy) == Defines.RANGE_MELEE;

        }
    };

    private static final EntThinkAdapter mutant_check_jump = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_check_jump"; }
        @Override
        public boolean think(edict_t self) {

            if (self.absmin[2] > (self.enemy.absmin[2] + 0.75 * self.enemy.size[2]))
                return false;

            if (self.absmax[2] < (self.enemy.absmin[2] + 0.25 * self.enemy.size[2]))
                return false;

            float[] v = {0, 0, 0};
            v[0] = self.s.origin[0] - self.enemy.s.origin[0];
            v[1] = self.s.origin[1] - self.enemy.s.origin[1];
            v[2] = 0;
            float distance = Math3D.VectorLength(v);

            if (distance < 100)
                return false;
            if (distance > 100) {
                return !(Lib.random() < 0.9);
            }

            return true;
        }
    };

    private static final EntThinkAdapter mutant_checkattack = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_checkattack"; }
        @Override
        public boolean think(edict_t self) {

            if (null == self.enemy || self.enemy.health <= 0)
                return false;

            if (mutant_check_melee.think(self)) {
                self.monsterinfo.attack_state = Defines.AS_MELEE;
                return true;
            }

            if (mutant_check_jump.think(self)) {
                self.monsterinfo.attack_state = Defines.AS_MISSILE;
                
                return true;
            }

            return false;
        }
    };

    
    
    

    private static final mframe_t[] mutant_frames_pain1 = {
            new mframe_t(GameAI.ai_move, 4, null),
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -8, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 5, null) };

    private static final mmove_t mutant_move_pain1 = new mmove_t(FRAME_pain101,
            FRAME_pain105, mutant_frames_pain1, mutant_run);

    private static final mframe_t[] mutant_frames_pain2 = {
            new mframe_t(GameAI.ai_move, -24, null),
            new mframe_t(GameAI.ai_move, 11, null),
            new mframe_t(GameAI.ai_move, 5, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 6, null),
            new mframe_t(GameAI.ai_move, 4, null) };

    private static final mmove_t mutant_move_pain2 = new mmove_t(FRAME_pain201,
            FRAME_pain206, mutant_frames_pain2, mutant_run);

    private static final mframe_t[] mutant_frames_pain3 = {
            new mframe_t(GameAI.ai_move, -22, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 6, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 1, null) };

    private static final mmove_t mutant_move_pain3 = new mmove_t(FRAME_pain301,
            FRAME_pain311, mutant_frames_pain3, mutant_run);

    private static final EntPainAdapter mutant_pain = new EntPainAdapter() {
    	@Override
        public String getID(){ return "mutant_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {

            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            if (GameBase.level.time < self.pain_debounce_time)
                return;

            self.pain_debounce_time = GameBase.level.time + 3;

            if (GameBase.skill.value == 3)
                return;

            float r = Lib.random();
            if (r < 0.33) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = mutant_move_pain1;
            } else if (r < 0.66) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = mutant_move_pain2;
            } else {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = mutant_move_pain3;
            }
        }
    };

    
    
    
    private static final EntThinkAdapter mutant_dead = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "mutant_dead"; }
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

    private static final mframe_t[] mutant_frames_death1 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t mutant_move_death1 = new mmove_t(FRAME_death101,
            FRAME_death109, mutant_frames_death1, mutant_dead);

    private static final mframe_t[] mutant_frames_death2 = {
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

    private static final mmove_t mutant_move_death2 = new mmove_t(FRAME_death201,
            FRAME_death210, mutant_frames_death2, mutant_dead);

    private static final EntDieAdapter mutant_die = new EntDieAdapter() {
    	@Override
        public String getID(){ return "mutant_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {

            if (self.health <= self.gib_health) {
                game_import_t
                        .sound(self, Defines.CHAN_VOICE, game_import_t
                                .soundindex("misc/udeath.wav"), 1,
                                Defines.ATTN_NORM, 0);
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

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_death, 1,
                    Defines.ATTN_NORM, 0);
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_YES;
            self.s.skinnum = 1;

            if (Lib.random() < 0.5)
                self.monsterinfo.currentmove = mutant_move_death1;
            else
                self.monsterinfo.currentmove = mutant_move_death2;
        }
    };

    
    
    

    /*
     * QUAKED monster_mutant (1 .5 0) (-32 -32 -24) (32 32 32) Ambush
     * Trigger_Spawn Sight
     */
    public static final EntThinkAdapter SP_monster_mutant = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "SP_monster_mutant"; }
        @Override
        public boolean think(edict_t self) {
            if (GameBase.deathmatch.value != 0) {
                GameUtil.G_FreeEdict(self);
                return false;
            }

            sound_swing = game_import_t.soundindex("mutant/mutatck1.wav");
            sound_hit = game_import_t.soundindex("mutant/mutatck2.wav");
            sound_hit2 = game_import_t.soundindex("mutant/mutatck3.wav");
            sound_death = game_import_t.soundindex("mutant/mutdeth1.wav");
            sound_idle = game_import_t.soundindex("mutant/mutidle1.wav");
            sound_pain1 = game_import_t.soundindex("mutant/mutpain1.wav");
            sound_pain2 = game_import_t.soundindex("mutant/mutpain2.wav");
            sound_sight = game_import_t.soundindex("mutant/mutsght1.wav");
            sound_search = game_import_t.soundindex("mutant/mutsrch1.wav");
            sound_step1 = game_import_t.soundindex("mutant/step1.wav");
            sound_step2 = game_import_t.soundindex("mutant/step2.wav");
            sound_step3 = game_import_t.soundindex("mutant/step3.wav");
            sound_thud = game_import_t.soundindex("mutant/thud1.wav");

            self.movetype = Defines.MOVETYPE_STEP;
            self.solid = Defines.SOLID_BBOX;
            self.s.modelindex = game_import_t
                    .modelindex("models/monsters/mutant/tris.md2");
            Math3D.VectorSet(self.mins, -32, -32, -24);
            Math3D.VectorSet(self.maxs, 32, 32, 48);

            self.health = 300;
            self.gib_health = -120;
            self.mass = 300;

            self.pain = mutant_pain;
            self.die = mutant_die;

            self.monsterinfo.stand = mutant_stand;
            self.monsterinfo.walk = mutant_walk;
            self.monsterinfo.run = mutant_run;
            self.monsterinfo.dodge = null;
            self.monsterinfo.attack = mutant_jump;
            self.monsterinfo.melee = mutant_melee;
            self.monsterinfo.sight = mutant_sight;
            self.monsterinfo.search = mutant_search;
            self.monsterinfo.idle = mutant_idle;
            self.monsterinfo.checkattack = mutant_checkattack;

            game_import_t.linkentity(self);

            self.monsterinfo.currentmove = mutant_move_stand;

            self.monsterinfo.scale = MODEL_SCALE;
            GameAI.walkmonster_start.think(self);
            return true;
        }
    };
}