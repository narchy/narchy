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

import java.util.stream.IntStream;

public class M_Medic {
    

    private static final int FRAME_walk1 = 0;

    public static final int FRAME_walk2 = 1;

    public static final int FRAME_walk3 = 2;

    public static final int FRAME_walk4 = 3;

    public static final int FRAME_walk5 = 4;

    public static final int FRAME_walk6 = 5;

    public static final int FRAME_walk7 = 6;

    public static final int FRAME_walk8 = 7;

    public static final int FRAME_walk9 = 8;

    public static final int FRAME_walk10 = 9;

    public static final int FRAME_walk11 = 10;

    private static final int FRAME_walk12 = 11;

    private static final int FRAME_wait1 = 12;

    public static final int FRAME_wait2 = 13;

    public static final int FRAME_wait3 = 14;

    public static final int FRAME_wait4 = 15;

    public static final int FRAME_wait5 = 16;

    public static final int FRAME_wait6 = 17;

    public static final int FRAME_wait7 = 18;

    public static final int FRAME_wait8 = 19;

    public static final int FRAME_wait9 = 20;

    public static final int FRAME_wait10 = 21;

    public static final int FRAME_wait11 = 22;

    public static final int FRAME_wait12 = 23;

    public static final int FRAME_wait13 = 24;

    public static final int FRAME_wait14 = 25;

    public static final int FRAME_wait15 = 26;

    public static final int FRAME_wait16 = 27;

    public static final int FRAME_wait17 = 28;

    public static final int FRAME_wait18 = 29;

    public static final int FRAME_wait19 = 30;

    public static final int FRAME_wait20 = 31;

    public static final int FRAME_wait21 = 32;

    public static final int FRAME_wait22 = 33;

    public static final int FRAME_wait23 = 34;

    public static final int FRAME_wait24 = 35;

    public static final int FRAME_wait25 = 36;

    public static final int FRAME_wait26 = 37;

    public static final int FRAME_wait27 = 38;

    public static final int FRAME_wait28 = 39;

    public static final int FRAME_wait29 = 40;

    public static final int FRAME_wait30 = 41;

    public static final int FRAME_wait31 = 42;

    public static final int FRAME_wait32 = 43;

    public static final int FRAME_wait33 = 44;

    public static final int FRAME_wait34 = 45;

    public static final int FRAME_wait35 = 46;

    public static final int FRAME_wait36 = 47;

    public static final int FRAME_wait37 = 48;

    public static final int FRAME_wait38 = 49;

    public static final int FRAME_wait39 = 50;

    public static final int FRAME_wait40 = 51;

    public static final int FRAME_wait41 = 52;

    public static final int FRAME_wait42 = 53;

    public static final int FRAME_wait43 = 54;

    public static final int FRAME_wait44 = 55;

    public static final int FRAME_wait45 = 56;

    public static final int FRAME_wait46 = 57;

    public static final int FRAME_wait47 = 58;

    public static final int FRAME_wait48 = 59;

    public static final int FRAME_wait49 = 60;

    public static final int FRAME_wait50 = 61;

    public static final int FRAME_wait51 = 62;

    public static final int FRAME_wait52 = 63;

    public static final int FRAME_wait53 = 64;

    public static final int FRAME_wait54 = 65;

    public static final int FRAME_wait55 = 66;

    public static final int FRAME_wait56 = 67;

    public static final int FRAME_wait57 = 68;

    public static final int FRAME_wait58 = 69;

    public static final int FRAME_wait59 = 70;

    public static final int FRAME_wait60 = 71;

    public static final int FRAME_wait61 = 72;

    public static final int FRAME_wait62 = 73;

    public static final int FRAME_wait63 = 74;

    public static final int FRAME_wait64 = 75;

    public static final int FRAME_wait65 = 76;

    public static final int FRAME_wait66 = 77;

    public static final int FRAME_wait67 = 78;

    public static final int FRAME_wait68 = 79;

    public static final int FRAME_wait69 = 80;

    public static final int FRAME_wait70 = 81;

    public static final int FRAME_wait71 = 82;

    public static final int FRAME_wait72 = 83;

    public static final int FRAME_wait73 = 84;

    public static final int FRAME_wait74 = 85;

    public static final int FRAME_wait75 = 86;

    public static final int FRAME_wait76 = 87;

    public static final int FRAME_wait77 = 88;

    public static final int FRAME_wait78 = 89;

    public static final int FRAME_wait79 = 90;

    public static final int FRAME_wait80 = 91;

    public static final int FRAME_wait81 = 92;

    public static final int FRAME_wait82 = 93;

    public static final int FRAME_wait83 = 94;

    public static final int FRAME_wait84 = 95;

    public static final int FRAME_wait85 = 96;

    public static final int FRAME_wait86 = 97;

    public static final int FRAME_wait87 = 98;

    public static final int FRAME_wait88 = 99;

    public static final int FRAME_wait89 = 100;

    private static final int FRAME_wait90 = 101;

    private static final int FRAME_run1 = 102;

    public static final int FRAME_run2 = 103;

    public static final int FRAME_run3 = 104;

    public static final int FRAME_run4 = 105;

    public static final int FRAME_run5 = 106;

    private static final int FRAME_run6 = 107;

    private static final int FRAME_paina1 = 108;

    public static final int FRAME_paina2 = 109;

    public static final int FRAME_paina3 = 110;

    public static final int FRAME_paina4 = 111;

    public static final int FRAME_paina5 = 112;

    public static final int FRAME_paina6 = 113;

    public static final int FRAME_paina7 = 114;

    private static final int FRAME_paina8 = 115;

    private static final int FRAME_painb1 = 116;

    public static final int FRAME_painb2 = 117;

    public static final int FRAME_painb3 = 118;

    public static final int FRAME_painb4 = 119;

    public static final int FRAME_painb5 = 120;

    public static final int FRAME_painb6 = 121;

    public static final int FRAME_painb7 = 122;

    public static final int FRAME_painb8 = 123;

    public static final int FRAME_painb9 = 124;

    public static final int FRAME_painb10 = 125;

    public static final int FRAME_painb11 = 126;

    public static final int FRAME_painb12 = 127;

    public static final int FRAME_painb13 = 128;

    public static final int FRAME_painb14 = 129;

    private static final int FRAME_painb15 = 130;

    private static final int FRAME_duck1 = 131;

    public static final int FRAME_duck2 = 132;

    public static final int FRAME_duck3 = 133;

    public static final int FRAME_duck4 = 134;

    public static final int FRAME_duck5 = 135;

    public static final int FRAME_duck6 = 136;

    public static final int FRAME_duck7 = 137;

    public static final int FRAME_duck8 = 138;

    public static final int FRAME_duck9 = 139;

    public static final int FRAME_duck10 = 140;

    public static final int FRAME_duck11 = 141;

    public static final int FRAME_duck12 = 142;

    public static final int FRAME_duck13 = 143;

    public static final int FRAME_duck14 = 144;

    public static final int FRAME_duck15 = 145;

    private static final int FRAME_duck16 = 146;

    private static final int FRAME_death1 = 147;

    public static final int FRAME_death2 = 148;

    public static final int FRAME_death3 = 149;

    public static final int FRAME_death4 = 150;

    public static final int FRAME_death5 = 151;

    public static final int FRAME_death6 = 152;

    public static final int FRAME_death7 = 153;

    public static final int FRAME_death8 = 154;

    public static final int FRAME_death9 = 155;

    public static final int FRAME_death10 = 156;

    public static final int FRAME_death11 = 157;

    public static final int FRAME_death12 = 158;

    public static final int FRAME_death13 = 159;

    public static final int FRAME_death14 = 160;

    public static final int FRAME_death15 = 161;

    public static final int FRAME_death16 = 162;

    public static final int FRAME_death17 = 163;

    public static final int FRAME_death18 = 164;

    public static final int FRAME_death19 = 165;

    public static final int FRAME_death20 = 166;

    public static final int FRAME_death21 = 167;

    public static final int FRAME_death22 = 168;

    public static final int FRAME_death23 = 169;

    public static final int FRAME_death24 = 170;

    public static final int FRAME_death25 = 171;

    public static final int FRAME_death26 = 172;

    public static final int FRAME_death27 = 173;

    public static final int FRAME_death28 = 174;

    public static final int FRAME_death29 = 175;

    private static final int FRAME_death30 = 176;

    private static final int FRAME_attack1 = 177;

    public static final int FRAME_attack2 = 178;

    public static final int FRAME_attack3 = 179;

    public static final int FRAME_attack4 = 180;

    public static final int FRAME_attack5 = 181;

    public static final int FRAME_attack6 = 182;

    public static final int FRAME_attack7 = 183;

    public static final int FRAME_attack8 = 184;

    private static final int FRAME_attack9 = 185;

    public static final int FRAME_attack10 = 186;

    public static final int FRAME_attack11 = 187;

    private static final int FRAME_attack12 = 188;

    public static final int FRAME_attack13 = 189;

    private static final int FRAME_attack14 = 190;

    private static final int FRAME_attack15 = 191;

    public static final int FRAME_attack16 = 192;

    public static final int FRAME_attack17 = 193;

    public static final int FRAME_attack18 = 194;

    private static final int FRAME_attack19 = 195;

    public static final int FRAME_attack20 = 196;

    public static final int FRAME_attack21 = 197;

    private static final int FRAME_attack22 = 198;

    public static final int FRAME_attack23 = 199;

    public static final int FRAME_attack24 = 200;

    private static final int FRAME_attack25 = 201;

    public static final int FRAME_attack26 = 202;

    public static final int FRAME_attack27 = 203;

    private static final int FRAME_attack28 = 204;

    public static final int FRAME_attack29 = 205;

    private static final int FRAME_attack30 = 206;

    public static final int FRAME_attack31 = 207;

    public static final int FRAME_attack32 = 208;

    private static final int FRAME_attack33 = 209;

    public static final int FRAME_attack34 = 210;

    public static final int FRAME_attack35 = 211;

    public static final int FRAME_attack36 = 212;

    public static final int FRAME_attack37 = 213;

    public static final int FRAME_attack38 = 214;

    public static final int FRAME_attack39 = 215;

    public static final int FRAME_attack40 = 216;

    public static final int FRAME_attack41 = 217;

    private static final int FRAME_attack42 = 218;

    private static final int FRAME_attack43 = 219;

    private static final int FRAME_attack44 = 220;

    public static final int FRAME_attack45 = 221;

    public static final int FRAME_attack46 = 222;

    public static final int FRAME_attack47 = 223;

    public static final int FRAME_attack48 = 224;

    public static final int FRAME_attack49 = 225;

    private static final int FRAME_attack50 = 226;

    public static final int FRAME_attack51 = 227;

    public static final int FRAME_attack52 = 228;

    public static final int FRAME_attack53 = 229;

    public static final int FRAME_attack54 = 230;

    public static final int FRAME_attack55 = 231;

    public static final int FRAME_attack56 = 232;

    public static final int FRAME_attack57 = 233;

    public static final int FRAME_attack58 = 234;

    public static final int FRAME_attack59 = 235;

    private static final int FRAME_attack60 = 236;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_idle1;

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_die;

    private static int sound_sight;

    private static int sound_search;

    private static int sound_hook_launch;

    private static int sound_hook_hit;

    private static int sound_hook_heal;

    private static int sound_hook_retract;

    private static edict_t medic_FindDeadMonster(edict_t self) {
        edict_t ent;
        edict_t best = null;
        EdictIterator edit = null;

        while ((edit = GameBase.findradius(edit, self.s.origin, 1024)) != null) {
            ent = edit.o;
            if (ent == self)
                continue;
            if (0 == (ent.svflags & Defines.SVF_MONSTER))
                continue;
            if ((ent.monsterinfo.aiflags & Defines.AI_GOOD_GUY) != 0)
                continue;
            if (ent.owner == null)
                continue;
            if (ent.health > 0)
                continue;
            if (ent.nextthink == 0)
                continue;
            if (!GameUtil.visible(self, ent))
                continue;
            if (best == null) {
                best = ent;
                continue;
            }
            if (ent.max_health <= best.max_health)
                continue;
            best = ent;
        }

        return best;
    }

    private static final EntThinkAdapter medic_idle = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_idle"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_idle1, 1,
                    Defines.ATTN_IDLE, 0);

            edict_t ent = medic_FindDeadMonster(self);
            if (ent != null) {
                self.enemy = ent;
                self.enemy.owner = self;
                self.monsterinfo.aiflags |= Defines.AI_MEDIC;
                GameUtil.FoundTarget(self);
            }
            return true;
        }
    };

    private static final EntThinkAdapter medic_search = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_search"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_search, 1,
                    Defines.ATTN_IDLE, 0);

            if (self.oldenemy == null) {
                edict_t ent = medic_FindDeadMonster(self);
                if (ent != null) {
                    self.oldenemy = self.enemy;
                    self.enemy = ent;
                    self.enemy.owner = self;
                    self.monsterinfo.aiflags |= Defines.AI_MEDIC;
                    GameUtil.FoundTarget(self);
                }
            }
            return true;
        }
    };

    private static final EntInteractAdapter medic_sight = new EntInteractAdapter() {
    	@Override
        public String getID(){ return "medic_sight"; }
        @Override
        public boolean interact(edict_t self, edict_t other) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_sight, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final mframe_t[] medic_frames_stand = {
            new mframe_t(GameAI.ai_stand, 0, medic_idle),
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
            new mframe_t(GameAI.ai_stand, 0, null), };

    private static final mmove_t medic_move_stand = new mmove_t(FRAME_wait1, FRAME_wait90,
            medic_frames_stand, null);

    private static final EntThinkAdapter medic_stand = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_stand"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = medic_move_stand;
            return true;
        }
    };

    private static final mframe_t[] medic_frames_walk = {
            new mframe_t(GameAI.ai_walk, 6.2f, null),
            new mframe_t(GameAI.ai_walk, 18.1f, null),
            new mframe_t(GameAI.ai_walk, 1, null),
            new mframe_t(GameAI.ai_walk, 9, null),
            new mframe_t(GameAI.ai_walk, 10, null),
            new mframe_t(GameAI.ai_walk, 9, null),
            new mframe_t(GameAI.ai_walk, 11, null),
            new mframe_t(GameAI.ai_walk, 11.6f, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 9.9f, null),
            new mframe_t(GameAI.ai_walk, 14, null),
            new mframe_t(GameAI.ai_walk, 9.3f, null) };

    private static final mmove_t medic_move_walk = new mmove_t(FRAME_walk1, FRAME_walk12,
            medic_frames_walk, null);

    private static final EntThinkAdapter medic_walk = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_walk"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = medic_move_walk;
            return true;
        }
    };

    private static final mframe_t[] medic_frames_run = {
            new mframe_t(GameAI.ai_run, 18, null),
            new mframe_t(GameAI.ai_run, 22.5f, null),
            new mframe_t(GameAI.ai_run, 25.4f, null),
            new mframe_t(GameAI.ai_run, 23.4f, null),
            new mframe_t(GameAI.ai_run, 24, null),
            new mframe_t(GameAI.ai_run, 35.6f, null) };

    private static final mmove_t medic_move_run = new mmove_t(FRAME_run1, FRAME_run6,
            medic_frames_run, null);

    private static final EntThinkAdapter medic_run = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_run"; }
        @Override
        public boolean think(edict_t self) {
            if (0 == (self.monsterinfo.aiflags & Defines.AI_MEDIC)) {

                edict_t ent = medic_FindDeadMonster(self);
                if (ent != null) {
                    self.oldenemy = self.enemy;
                    self.enemy = ent;
                    self.enemy.owner = self;
                    self.monsterinfo.aiflags |= Defines.AI_MEDIC;
                    GameUtil.FoundTarget(self);
                    return true;
                }
            }

            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = medic_move_stand;
            else
                self.monsterinfo.currentmove = medic_move_run;
            return true;
        }
    };

    private static final mframe_t[] medic_frames_pain1 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t medic_move_pain1 = new mmove_t(FRAME_paina1, FRAME_paina8,
            medic_frames_pain1, medic_run);

    private static final mframe_t[] medic_frames_pain2 = {
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

    private static final mmove_t medic_move_pain2 = new mmove_t(FRAME_painb1, FRAME_painb15,
            medic_frames_pain2, medic_run);

    private static final EntPainAdapter medic_pain = new EntPainAdapter() {
    	@Override
        public String getID(){ return "medic_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {

            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            if (GameBase.level.time < self.pain_debounce_time)
                return;

            self.pain_debounce_time = GameBase.level.time + 3;

            if (GameBase.skill.value == 3)
                return; 

            if (Lib.random() < 0.5) {
                self.monsterinfo.currentmove = medic_move_pain1;
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
            } else {
                self.monsterinfo.currentmove = medic_move_pain2;
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NORM, 0);
            }
        }
    };

    private static final EntThinkAdapter medic_fire_blaster = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_fire_blaster"; }
        @Override
        public boolean think(edict_t self) {
            int effect;

            if ((self.s.frame == FRAME_attack9)
                    || (self.s.frame == FRAME_attack12))
                effect = Defines.EF_BLASTER;
            else {
                boolean b = IntStream.of(FRAME_attack19, FRAME_attack22, FRAME_attack25, FRAME_attack28).anyMatch(i -> (self.s.frame == i));
                if (b)
                    effect = Defines.EF_HYPERBLASTER;
                else
                    effect = 0;
            }

            float[] right = {0, 0, 0};
            float[] forward = {0, 0, 0};
            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D.G_ProjectSource(self.s.origin,
                    M_Flash.monster_flash_offset[Defines.MZ2_MEDIC_BLASTER_1],
                    forward, right, start);

            float[] end = {0, 0, 0};
            Math3D.VectorCopy(self.enemy.s.origin, end);
            end[2] += self.enemy.viewheight;
            float[] dir = {0, 0, 0};
            Math3D.VectorSubtract(end, start, dir);

            Monster.monster_fire_blaster(self, start, dir, 2, 1000,
                    Defines.MZ2_MEDIC_BLASTER_1, effect);
            return true;
        }
    };

    private static final EntThinkAdapter medic_dead = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_dead"; }
        @Override
        public boolean think(edict_t self) {
            Math3D.VectorSet(self.mins, -16, -16, -24);
            Math3D.VectorSet(self.maxs, 16, 16, -8);
            self.movetype = Defines.MOVETYPE_TOSS;
            self.svflags |= Defines.SVF_DEADMONSTER;
            self.nextthink = 0;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final mframe_t[] medic_frames_death = {
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
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t medic_move_death = new mmove_t(FRAME_death1, FRAME_death30,
            medic_frames_death, medic_dead);

    private static final EntDieAdapter medic_die = new EntDieAdapter() {
    	@Override
        public String getID(){ return "medic_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {


            if ((self.enemy != null) && (self.enemy.owner == self))
                self.enemy.owner = null;

            
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

            
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_die, 1,
                    Defines.ATTN_NORM, 0);
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_YES;

            self.monsterinfo.currentmove = medic_move_death;
        }
    };

    private static final EntThinkAdapter medic_duck_down = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_duck_down"; }
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

    private static final EntThinkAdapter medic_duck_hold = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_duck_hold"; }
        @Override
        public boolean think(edict_t self) {
            if (GameBase.level.time >= self.monsterinfo.pausetime)
                self.monsterinfo.aiflags &= ~Defines.AI_HOLD_FRAME;
            else
                self.monsterinfo.aiflags |= Defines.AI_HOLD_FRAME;
            return true;
        }
    };

    private static final EntThinkAdapter medic_duck_up = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_duck_up"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.aiflags &= ~Defines.AI_DUCKED;
            self.maxs[2] += 32;
            self.takedamage = Defines.DAMAGE_AIM;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final mframe_t[] medic_frames_duck = {
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, medic_duck_down),
            new mframe_t(GameAI.ai_move, -1, medic_duck_hold),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, medic_duck_up),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -1, null) };

    private static final mmove_t medic_move_duck = new mmove_t(FRAME_duck1, FRAME_duck16,
            medic_frames_duck, medic_run);

    private static final EntDodgeAdapter medic_dodge = new EntDodgeAdapter() {
    	@Override
        public String getID(){ return "medic_dodge"; }
        @Override
        public void dodge(edict_t self, edict_t attacker, float eta) {
            if (Lib.random() > 0.25)
                return;

            if (self.enemy == null)
                self.enemy = attacker;

            self.monsterinfo.currentmove = medic_move_duck;
        }
    };

    private static final mframe_t[] medic_frames_attackHyperBlaster = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster) };

    private static final mmove_t medic_move_attackHyperBlaster = new mmove_t(FRAME_attack15,
            FRAME_attack30, medic_frames_attackHyperBlaster, medic_run);

    private static final EntThinkAdapter medic_continue = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_continue"; }
        @Override
        public boolean think(edict_t self) {
            if (GameUtil.visible(self, self.enemy))
                if (Lib.random() <= 0.95)
                    self.monsterinfo.currentmove = medic_move_attackHyperBlaster;
            return true;
        }
    };

    private static final mframe_t[] medic_frames_attackBlaster = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, 3, null),
            new mframe_t(GameAI.ai_charge, 2, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, medic_fire_blaster),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, medic_continue) 
                                                              
                                                              
                                                              
    };

    private static final mmove_t medic_move_attackBlaster = new mmove_t(FRAME_attack1,
            FRAME_attack14, medic_frames_attackBlaster, medic_run);

    private static final EntThinkAdapter medic_hook_launch = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_hook_launch"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_WEAPON, sound_hook_launch, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final float[][] medic_cable_offsets = { { 45.0f, -9.2f, 15.5f },
            { 48.4f, -9.7f, 15.2f }, { 47.8f, -9.8f, 15.8f },
            { 47.3f, -9.3f, 14.3f }, { 45.4f, -10.1f, 13.1f },
            { 41.9f, -12.7f, 12.0f }, { 37.8f, -15.8f, 11.2f },
            { 34.3f, -18.4f, 10.7f }, { 32.7f, -19.7f, 10.4f },
            { 32.7f, -19.7f, 10.4f } };

    private static final EntThinkAdapter medic_cable_attack = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_cable_attack"; }
        @Override
        public boolean think(edict_t self) {

            if (!self.enemy.inuse)
                return true;

            float[] r = {0, 0, 0};
            float[] f = {
                    0, 0, 0};
            Math3D.AngleVectors(self.s.angles, f, r, null);
            float[] offset = {0, 0, 0};
            Math3D.VectorCopy(
                    medic_cable_offsets[self.s.frame - FRAME_attack42], offset);
            float[] start = {0, 0, 0};
            Math3D.G_ProjectSource(self.s.origin, offset, f, r, start);


            float[] dir = {0, 0, 0};
            Math3D.VectorSubtract(start, self.enemy.s.origin, dir);
            float distance = Math3D.VectorLength(dir);
            if (distance > 256)
                return true;


            float[] angles = {0, 0, 0};
            Math3D.vectoangles(dir, angles);
            if (angles[0] < -180)
                angles[0] += 360;
            if (Math.abs(angles[0]) > 45)
                return true;

            trace_t tr = game_import_t.trace(start, null, null, self.enemy.s.origin,
                    self, Defines.MASK_SHOT);
            if (tr.fraction != 1.0 && tr.ent != self.enemy)
                return true;

            switch (self.s.frame) {
                case FRAME_attack43:
                    game_import_t.sound(self.enemy, Defines.CHAN_AUTO,
                            sound_hook_hit, 1, Defines.ATTN_NORM, 0);
                    self.enemy.monsterinfo.aiflags |= Defines.AI_RESURRECTING;
                    break;
                case FRAME_attack50:
                    self.enemy.spawnflags = 0;
                    self.enemy.monsterinfo.aiflags = 0;
                    self.enemy.target = null;
                    self.enemy.targetname = null;
                    self.enemy.combattarget = null;
                    self.enemy.deathtarget = null;
                    self.enemy.owner = self;
                    GameSpawn.ED_CallSpawn(self.enemy);
                    self.enemy.owner = null;
                    if (self.enemy.think != null) {
                        self.enemy.nextthink = GameBase.level.time;
                        self.enemy.think.think(self.enemy);
                    }
                    self.enemy.monsterinfo.aiflags |= Defines.AI_RESURRECTING;
                    if (self.oldenemy != null && self.oldenemy.client != null) {
                        self.enemy.enemy = self.oldenemy;
                        GameUtil.FoundTarget(self.enemy);
                    }
                    break;
                default:
                    if (self.s.frame == FRAME_attack44)
                        game_import_t.sound(self, Defines.CHAN_WEAPON,
                                sound_hook_heal, 1, Defines.ATTN_NORM, 0);
                    break;
            }

            
            Math3D.VectorMA(start, 8, f, start);


            float[] end = {0, 0, 0};
            Math3D.VectorCopy(self.enemy.s.origin, end);
            end[2] = self.enemy.absmin[2] + self.enemy.size[2] / 2;

            game_import_t.WriteByte(Defines.svc_temp_entity);
            game_import_t.WriteByte(Defines.TE_MEDIC_CABLE_ATTACK);
            game_import_t.WriteShort(self.index);
            game_import_t.WritePosition(start);
            game_import_t.WritePosition(end);
            game_import_t.multicast(self.s.origin, Defines.MULTICAST_PVS);
            return true;
        }
    };

    private static final EntThinkAdapter medic_hook_retract = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_hook_retract"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_WEAPON, sound_hook_retract, 1,
                    Defines.ATTN_NORM, 0);
            self.enemy.monsterinfo.aiflags &= ~Defines.AI_RESURRECTING;
            return true;
        }
    };

    private static final mframe_t[] medic_frames_attackCable = {
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 5, null),
            new mframe_t(GameAI.ai_move, 4.4f, null),
            new mframe_t(GameAI.ai_charge, 4.7f, null),
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, 6, null),
            new mframe_t(GameAI.ai_charge, 4, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_move, 0, medic_hook_launch),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, 0, medic_cable_attack),
            new mframe_t(GameAI.ai_move, -15, medic_hook_retract),
            new mframe_t(GameAI.ai_move, -1.5f, null),
            new mframe_t(GameAI.ai_move, -1.2f, null),
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 0.3f, null),
            new mframe_t(GameAI.ai_move, 0.7f, null),
            new mframe_t(GameAI.ai_move, 1.2f, null),
            new mframe_t(GameAI.ai_move, 1.3f, null) };

    private static final mmove_t medic_move_attackCable = new mmove_t(FRAME_attack33,
            FRAME_attack60, medic_frames_attackCable, medic_run);

    private static final EntThinkAdapter medic_attack = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_attack"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_MEDIC) != 0)
                self.monsterinfo.currentmove = medic_move_attackCable;
            else
                self.monsterinfo.currentmove = medic_move_attackBlaster;
            return true;
        }
    };

    private static final EntThinkAdapter medic_checkattack = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "medic_checkattack"; }
        @Override
        public boolean think(edict_t self) {
            if ((self.monsterinfo.aiflags & Defines.AI_MEDIC) != 0) {
                medic_attack.think(self);
                return true;
            }

            return GameUtil.M_CheckAttack.think(self);

        }
    };

    /*
     * QUAKED monster_medic (1 .5 0) (-16 -16 -24) (16 16 32) Ambush
     * Trigger_Spawn Sight
     */
    public static void SP_monster_medic(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_idle1 = game_import_t.soundindex("medic/idle.wav");
        sound_pain1 = game_import_t.soundindex("medic/medpain1.wav");
        sound_pain2 = game_import_t.soundindex("medic/medpain2.wav");
        sound_die = game_import_t.soundindex("medic/meddeth1.wav");
        sound_sight = game_import_t.soundindex("medic/medsght1.wav");
        sound_search = game_import_t.soundindex("medic/medsrch1.wav");
        sound_hook_launch = game_import_t.soundindex("medic/medatck2.wav");
        sound_hook_hit = game_import_t.soundindex("medic/medatck3.wav");
        sound_hook_heal = game_import_t.soundindex("medic/medatck4.wav");
        sound_hook_retract = game_import_t.soundindex("medic/medatck5.wav");

        game_import_t.soundindex("medic/medatck1.wav");

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/medic/tris.md2");
        Math3D.VectorSet(self.mins, -24, -24, -24);
        Math3D.VectorSet(self.maxs, 24, 24, 32);

        self.health = 300;
        self.gib_health = -130;
        self.mass = 400;

        self.pain = medic_pain;
        self.die = medic_die;

        self.monsterinfo.stand = medic_stand;
        self.monsterinfo.walk = medic_walk;
        self.monsterinfo.run = medic_run;
        self.monsterinfo.dodge = medic_dodge;
        self.monsterinfo.attack = medic_attack;
        self.monsterinfo.melee = null;
        self.monsterinfo.sight = medic_sight;
        self.monsterinfo.idle = medic_idle;
        self.monsterinfo.search = medic_search;
        self.monsterinfo.checkattack = medic_checkattack;

        game_import_t.linkentity(self);

        self.monsterinfo.currentmove = medic_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.walkmonster_start.think(self);
    }
}