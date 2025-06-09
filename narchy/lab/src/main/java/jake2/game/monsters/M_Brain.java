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

public class M_Brain {

    private static final int FRAME_walk101 = 0;

    public static final int FRAME_walk102 = 1;

    public static final int FRAME_walk103 = 2;

    public static final int FRAME_walk104 = 3;

    public static final int FRAME_walk105 = 4;

    public static final int FRAME_walk106 = 5;

    public static final int FRAME_walk107 = 6;

    public static final int FRAME_walk108 = 7;

    public static final int FRAME_walk109 = 8;

    public static final int FRAME_walk110 = 9;

    private static final int FRAME_walk111 = 10;

    public static final int FRAME_walk112 = 11;

    public static final int FRAME_walk113 = 12;

    public static final int FRAME_walk201 = 13;

    public static final int FRAME_walk202 = 14;

    public static final int FRAME_walk203 = 15;

    public static final int FRAME_walk204 = 16;

    public static final int FRAME_walk205 = 17;

    public static final int FRAME_walk206 = 18;

    public static final int FRAME_walk207 = 19;

    public static final int FRAME_walk208 = 20;

    public static final int FRAME_walk209 = 21;

    public static final int FRAME_walk210 = 22;

    public static final int FRAME_walk211 = 23;

    public static final int FRAME_walk212 = 24;

    public static final int FRAME_walk213 = 25;

    public static final int FRAME_walk214 = 26;

    public static final int FRAME_walk215 = 27;

    public static final int FRAME_walk216 = 28;

    public static final int FRAME_walk217 = 29;

    public static final int FRAME_walk218 = 30;

    public static final int FRAME_walk219 = 31;

    public static final int FRAME_walk220 = 32;

    public static final int FRAME_walk221 = 33;

    public static final int FRAME_walk222 = 34;

    public static final int FRAME_walk223 = 35;

    public static final int FRAME_walk224 = 36;

    public static final int FRAME_walk225 = 37;

    public static final int FRAME_walk226 = 38;

    public static final int FRAME_walk227 = 39;

    public static final int FRAME_walk228 = 40;

    public static final int FRAME_walk229 = 41;

    public static final int FRAME_walk230 = 42;

    public static final int FRAME_walk231 = 43;

    public static final int FRAME_walk232 = 44;

    public static final int FRAME_walk233 = 45;

    public static final int FRAME_walk234 = 46;

    public static final int FRAME_walk235 = 47;

    public static final int FRAME_walk236 = 48;

    public static final int FRAME_walk237 = 49;

    public static final int FRAME_walk238 = 50;

    public static final int FRAME_walk239 = 51;

    public static final int FRAME_walk240 = 52;

    private static final int FRAME_attak101 = 53;

    public static final int FRAME_attak102 = 54;

    public static final int FRAME_attak103 = 55;

    public static final int FRAME_attak104 = 56;

    public static final int FRAME_attak105 = 57;

    public static final int FRAME_attak106 = 58;

    public static final int FRAME_attak107 = 59;

    public static final int FRAME_attak108 = 60;

    public static final int FRAME_attak109 = 61;

    public static final int FRAME_attak110 = 62;

    public static final int FRAME_attak111 = 63;

    public static final int FRAME_attak112 = 64;

    public static final int FRAME_attak113 = 65;

    public static final int FRAME_attak114 = 66;

    public static final int FRAME_attak115 = 67;

    public static final int FRAME_attak116 = 68;

    public static final int FRAME_attak117 = 69;

    private static final int FRAME_attak118 = 70;

    private static final int FRAME_attak201 = 71;

    public static final int FRAME_attak202 = 72;

    public static final int FRAME_attak203 = 73;

    public static final int FRAME_attak204 = 74;

    public static final int FRAME_attak205 = 75;

    public static final int FRAME_attak206 = 76;

    public static final int FRAME_attak207 = 77;

    public static final int FRAME_attak208 = 78;

    public static final int FRAME_attak209 = 79;

    public static final int FRAME_attak210 = 80;

    public static final int FRAME_attak211 = 81;

    public static final int FRAME_attak212 = 82;

    public static final int FRAME_attak213 = 83;

    public static final int FRAME_attak214 = 84;

    public static final int FRAME_attak215 = 85;

    public static final int FRAME_attak216 = 86;

    private static final int FRAME_attak217 = 87;

    private static final int FRAME_pain101 = 88;

    public static final int FRAME_pain102 = 89;

    public static final int FRAME_pain103 = 90;

    public static final int FRAME_pain104 = 91;

    public static final int FRAME_pain105 = 92;

    public static final int FRAME_pain106 = 93;

    public static final int FRAME_pain107 = 94;

    public static final int FRAME_pain108 = 95;

    public static final int FRAME_pain109 = 96;

    public static final int FRAME_pain110 = 97;

    public static final int FRAME_pain111 = 98;

    public static final int FRAME_pain112 = 99;

    public static final int FRAME_pain113 = 100;

    public static final int FRAME_pain114 = 101;

    public static final int FRAME_pain115 = 102;

    public static final int FRAME_pain116 = 103;

    public static final int FRAME_pain117 = 104;

    public static final int FRAME_pain118 = 105;

    public static final int FRAME_pain119 = 106;

    public static final int FRAME_pain120 = 107;

    private static final int FRAME_pain121 = 108;

    private static final int FRAME_pain201 = 109;

    public static final int FRAME_pain202 = 110;

    public static final int FRAME_pain203 = 111;

    public static final int FRAME_pain204 = 112;

    public static final int FRAME_pain205 = 113;

    public static final int FRAME_pain206 = 114;

    public static final int FRAME_pain207 = 115;

    private static final int FRAME_pain208 = 116;

    private static final int FRAME_pain301 = 117;

    public static final int FRAME_pain302 = 118;

    public static final int FRAME_pain303 = 119;

    public static final int FRAME_pain304 = 120;

    public static final int FRAME_pain305 = 121;

    private static final int FRAME_pain306 = 122;

    private static final int FRAME_death101 = 123;

    public static final int FRAME_death102 = 124;

    public static final int FRAME_death103 = 125;

    public static final int FRAME_death104 = 126;

    public static final int FRAME_death105 = 127;

    public static final int FRAME_death106 = 128;

    public static final int FRAME_death107 = 129;

    public static final int FRAME_death108 = 130;

    public static final int FRAME_death109 = 131;

    public static final int FRAME_death110 = 132;

    public static final int FRAME_death111 = 133;

    public static final int FRAME_death112 = 134;

    public static final int FRAME_death113 = 135;

    public static final int FRAME_death114 = 136;

    public static final int FRAME_death115 = 137;

    public static final int FRAME_death116 = 138;

    public static final int FRAME_death117 = 139;

    private static final int FRAME_death118 = 140;

    private static final int FRAME_death201 = 141;

    public static final int FRAME_death202 = 142;

    public static final int FRAME_death203 = 143;

    public static final int FRAME_death204 = 144;

    private static final int FRAME_death205 = 145;

    private static final int FRAME_duck01 = 146;

    public static final int FRAME_duck02 = 147;

    public static final int FRAME_duck03 = 148;

    public static final int FRAME_duck04 = 149;

    public static final int FRAME_duck05 = 150;

    public static final int FRAME_duck06 = 151;

    public static final int FRAME_duck07 = 152;

    private static final int FRAME_duck08 = 153;

    private static final int FRAME_defens01 = 154;

    public static final int FRAME_defens02 = 155;

    public static final int FRAME_defens03 = 156;

    public static final int FRAME_defens04 = 157;

    public static final int FRAME_defens05 = 158;

    public static final int FRAME_defens06 = 159;

    public static final int FRAME_defens07 = 160;

    private static final int FRAME_defens08 = 161;

    private static final int FRAME_stand01 = 162;

    public static final int FRAME_stand02 = 163;

    public static final int FRAME_stand03 = 164;

    public static final int FRAME_stand04 = 165;

    public static final int FRAME_stand05 = 166;

    public static final int FRAME_stand06 = 167;

    public static final int FRAME_stand07 = 168;

    public static final int FRAME_stand08 = 169;

    public static final int FRAME_stand09 = 170;

    public static final int FRAME_stand10 = 171;

    public static final int FRAME_stand11 = 172;

    public static final int FRAME_stand12 = 173;

    public static final int FRAME_stand13 = 174;

    public static final int FRAME_stand14 = 175;

    public static final int FRAME_stand15 = 176;

    public static final int FRAME_stand16 = 177;

    public static final int FRAME_stand17 = 178;

    public static final int FRAME_stand18 = 179;

    public static final int FRAME_stand19 = 180;

    public static final int FRAME_stand20 = 181;

    public static final int FRAME_stand21 = 182;

    public static final int FRAME_stand22 = 183;

    public static final int FRAME_stand23 = 184;

    public static final int FRAME_stand24 = 185;

    public static final int FRAME_stand25 = 186;

    public static final int FRAME_stand26 = 187;

    public static final int FRAME_stand27 = 188;

    public static final int FRAME_stand28 = 189;

    public static final int FRAME_stand29 = 190;

    private static final int FRAME_stand30 = 191;

    private static final int FRAME_stand31 = 192;

    public static final int FRAME_stand32 = 193;

    public static final int FRAME_stand33 = 194;

    public static final int FRAME_stand34 = 195;

    public static final int FRAME_stand35 = 196;

    public static final int FRAME_stand36 = 197;

    public static final int FRAME_stand37 = 198;

    public static final int FRAME_stand38 = 199;

    public static final int FRAME_stand39 = 200;

    public static final int FRAME_stand40 = 201;

    public static final int FRAME_stand41 = 202;

    public static final int FRAME_stand42 = 203;

    public static final int FRAME_stand43 = 204;

    public static final int FRAME_stand44 = 205;

    public static final int FRAME_stand45 = 206;

    public static final int FRAME_stand46 = 207;

    public static final int FRAME_stand47 = 208;

    public static final int FRAME_stand48 = 209;

    public static final int FRAME_stand49 = 210;

    public static final int FRAME_stand50 = 211;

    public static final int FRAME_stand51 = 212;

    public static final int FRAME_stand52 = 213;

    public static final int FRAME_stand53 = 214;

    public static final int FRAME_stand54 = 215;

    public static final int FRAME_stand55 = 216;

    public static final int FRAME_stand56 = 217;

    public static final int FRAME_stand57 = 218;

    public static final int FRAME_stand58 = 219;

    public static final int FRAME_stand59 = 220;

    private static final int FRAME_stand60 = 221;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_chest_open;

    private static int sound_tentacles_retract;

    private static int sound_death;

    private static int sound_idle3;

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_sight;

    private static int sound_search;

    private static int sound_melee1;

    private static int sound_melee2;

    private static int sound_melee3;

    private static final EntInteractAdapter brain_sight = new EntInteractAdapter() {
    	@Override
        public String getID() { return "brain_sight"; }
        @Override
        public boolean interact(edict_t self, edict_t other) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_sight, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter brain_search = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_search"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_search, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    
    
    

    private static final mframe_t[] brain_frames_stand = {
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

    private static final mmove_t brain_move_stand = new mmove_t(FRAME_stand01, FRAME_stand30,
            brain_frames_stand, null);

    private static final EntThinkAdapter brain_stand = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_stand"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = brain_move_stand;
            return true;
        }
    };

    
    
    

    private static final mframe_t[] brain_frames_idle = {
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

    private static final mmove_t brain_move_idle = new mmove_t(FRAME_stand31, FRAME_stand60,
            brain_frames_idle, brain_stand);

    private static final EntThinkAdapter brain_idle = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_idle"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_AUTO, sound_idle3, 1,
                    Defines.ATTN_IDLE, 0);
            self.monsterinfo.currentmove = brain_move_idle;
            return true;
        }
    };

    
    
    
    private static final mframe_t[] brain_frames_walk1 = {
            new mframe_t(GameAI.ai_walk, 7, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 3, null),
            new mframe_t(GameAI.ai_walk, 3, null),
            new mframe_t(GameAI.ai_walk, 1, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 9, null),
            new mframe_t(GameAI.ai_walk, -4, null),
            new mframe_t(GameAI.ai_walk, -1, null),
            new mframe_t(GameAI.ai_walk, 2, null) };

    private static final mmove_t brain_move_walk1 = new mmove_t(FRAME_walk101, FRAME_walk111,
            brain_frames_walk1, null);

    
    /*
     * # if 0 void brain_walk2_cycle(edict_t self) { if (random() > 0.1)
     * self.monsterinfo.nextframe= FRAME_walk220; }
     * 
     * static mframe_t brain_frames_walk2[]= new mframe_t[] { new
     * mframe_t(ai_walk, 3, null), new mframe_t(ai_walk, -2, null), new
     * mframe_t(ai_walk, -4, null), new mframe_t(ai_walk, -3, null), new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 1, null), new
     * mframe_t(ai_walk, 12, null), new mframe_t(ai_walk, 0, null), new
     * mframe_t(ai_walk, -3, null), new mframe_t(ai_walk, 0, null), new
     * mframe_t(ai_walk, -2, null), new mframe_t(ai_walk, 0, null), new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 1, null), new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 0, null), new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 0, null), new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 10, null, 
     * Start)
     * 
     * new mframe_t(ai_walk, -1, null), new mframe_t(ai_walk, 7, null), new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 3, null), new
     * mframe_t(ai_walk, -3, null), new mframe_t(ai_walk, 2, null), new
     * mframe_t(ai_walk, 4, null), new mframe_t(ai_walk, -3, null), new
     * mframe_t(ai_walk, 2, null), new mframe_t(ai_walk, 0, null), new
     * mframe_t(ai_walk, 4, brain_walk2_cycle), new mframe_t(ai_walk, -1, null),
     * new mframe_t(ai_walk, -1, null), new mframe_t(ai_walk, -8, null,) new
     * mframe_t(ai_walk, 0, null), new mframe_t(ai_walk, 1, null), new
     * mframe_t(ai_walk, 5, null), new mframe_t(ai_walk, 2, null), new
     * mframe_t(ai_walk, -1, null), new mframe_t(ai_walk, -5, null)}; static
     * mmove_t brain_move_walk2= new mmove_t(FRAME_walk201, FRAME_walk240,
     * brain_frames_walk2, null);
     *  # endif
     */
    private static final EntThinkAdapter brain_walk = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_walk"; }
        @Override
        public boolean think(edict_t self) {
            
            self.monsterinfo.currentmove = brain_move_walk1;
            
            
            return true;
        }
    };

    
    
    

    private static final EntThinkAdapter brain_duck_down = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_duck_down"; }
        @Override
        public boolean think(edict_t self) {

            if ((self.monsterinfo.aiflags & Defines.AI_DUCKED) != 0)
                return true;
            self.monsterinfo.aiflags |= Defines.AI_DUCKED;
            self.maxs[2] -= 32;
            self.takedamage = Defines.DAMAGE_YES;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final EntThinkAdapter brain_duck_hold = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_duck_hold"; }
        @Override
        public boolean think(edict_t self) {
            if (GameBase.level.time >= self.monsterinfo.pausetime)
                self.monsterinfo.aiflags &= ~Defines.AI_HOLD_FRAME;
            else
                self.monsterinfo.aiflags |= Defines.AI_HOLD_FRAME;
            return true;
        }
    };

    private static final EntThinkAdapter brain_duck_up = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_duck_up"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.aiflags &= ~Defines.AI_DUCKED;
            self.maxs[2] += 32;
            self.takedamage = Defines.DAMAGE_AIM;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final EntDodgeAdapter brain_dodge = new EntDodgeAdapter() {
    	@Override
        public String getID() { return "brain_dodge"; }
        @Override
        public void dodge(edict_t self, edict_t attacker, float eta) {
            if (Lib.random() > 0.25)
                return;

            if (self.enemy == null)
                self.enemy = attacker;

            self.monsterinfo.pausetime = GameBase.level.time + eta + 0.5f;
            self.monsterinfo.currentmove = brain_move_duck;
        }
    };

    private static final mframe_t[] brain_frames_death2 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 9, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final EntThinkAdapter brain_dead = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_dead"; }
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

    private static final mmove_t brain_move_death2 = new mmove_t(FRAME_death201,
            FRAME_death205, brain_frames_death2, brain_dead);

    private static final mframe_t[] brain_frames_death1 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 9, null),
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

    private static final mmove_t brain_move_death1 = new mmove_t(FRAME_death101,
            FRAME_death118, brain_frames_death1, brain_dead);

    
    
    

    private static final EntThinkAdapter brain_swing_right = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_swing_right"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_BODY, sound_melee1, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter brain_hit_right = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_hit_right"; }
        @Override
        public boolean think(edict_t self) {
            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, self.maxs[0], 8);
            if (GameWeapon.fire_hit(self, aim, (15 + (Lib.rand() % 5)), 40))
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_melee3, 1,
                        Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter brain_swing_left = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_swing_left"; }
        @Override
        public boolean think(edict_t self) {
            game_import_t.sound(self, Defines.CHAN_BODY, sound_melee2, 1,
                    Defines.ATTN_NORM, 0);

            return true;
        }
    };

    private static final EntThinkAdapter brain_hit_left = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_hit_left"; }
        @Override
        public boolean think(edict_t self) {
            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, self.mins[0], 8);
            if (GameWeapon.fire_hit(self, aim, (15 + (Lib.rand() % 5)), 40))
                game_import_t.sound(self, Defines.CHAN_WEAPON, sound_melee3, 1,
                        Defines.ATTN_NORM, 0);

            return true;
        }
    };

    private static final EntThinkAdapter brain_chest_open = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_chest_open"; }
        @Override
        public boolean think(edict_t self) {
            self.spawnflags &= ~65536;
            self.monsterinfo.power_armor_type = Defines.POWER_ARMOR_NONE;
            game_import_t.sound(self, Defines.CHAN_BODY, sound_chest_open, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter brain_tentacle_attack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_tentacle_attack"; }
        @Override
        public boolean think(edict_t self) {

            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, 0, 8);
            if (GameWeapon.fire_hit(self, aim, (10 + (Lib.rand() % 5)), -600)
                    && GameBase.skill.value > 0)
                self.spawnflags |= 65536;
            game_import_t.sound(self, Defines.CHAN_WEAPON,
                    sound_tentacles_retract, 1, Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final mframe_t[] brain_frames_attack1 = {
            new mframe_t(GameAI.ai_charge, 8, null),
            new mframe_t(GameAI.ai_charge, 3, null),
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, -3, brain_swing_right),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, -5, null),
            new mframe_t(GameAI.ai_charge, -7, brain_hit_right),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 6, brain_swing_left),
            new mframe_t(GameAI.ai_charge, 1, null),
            new mframe_t(GameAI.ai_charge, 2, brain_hit_left),
            new mframe_t(GameAI.ai_charge, -3, null),
            new mframe_t(GameAI.ai_charge, 6, null),
            new mframe_t(GameAI.ai_charge, -1, null),
            new mframe_t(GameAI.ai_charge, -3, null),
            new mframe_t(GameAI.ai_charge, 2, null),
            new mframe_t(GameAI.ai_charge, -11, null) };

    private static final EntThinkAdapter brain_chest_closed = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_chest_closed"; }
        @Override
        public boolean think(edict_t self) {

            self.monsterinfo.power_armor_type = Defines.POWER_ARMOR_SCREEN;
            if ((self.spawnflags & 65536) != 0) {
                self.spawnflags &= ~65536;
                self.monsterinfo.currentmove = brain_move_attack1;
            }
            return true;
        }
    };

    private static final mframe_t[] brain_frames_attack2 = {
            new mframe_t(GameAI.ai_charge, 5, null),
            new mframe_t(GameAI.ai_charge, -4, null),
            new mframe_t(GameAI.ai_charge, -4, null),
            new mframe_t(GameAI.ai_charge, -3, null),
            new mframe_t(GameAI.ai_charge, 0, brain_chest_open),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 13, brain_tentacle_attack),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 2, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, -9, brain_chest_closed),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 4, null),
            new mframe_t(GameAI.ai_charge, 3, null),
            new mframe_t(GameAI.ai_charge, 2, null),
            new mframe_t(GameAI.ai_charge, -3, null),
            new mframe_t(GameAI.ai_charge, -6, null) };

    private static final EntThinkAdapter brain_melee = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_melee"; }
        @Override
        public boolean think(edict_t self) {
            if (Lib.random() <= 0.5)
                self.monsterinfo.currentmove = brain_move_attack1;
            else
                self.monsterinfo.currentmove = brain_move_attack2;

            return true;
        }
    };

    
    
    

    private static final mframe_t[] brain_frames_run = {
            new mframe_t(GameAI.ai_run, 9, null),
            new mframe_t(GameAI.ai_run, 2, null),
            new mframe_t(GameAI.ai_run, 3, null),
            new mframe_t(GameAI.ai_run, 3, null),
            new mframe_t(GameAI.ai_run, 1, null),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 0, null),
            new mframe_t(GameAI.ai_run, 10, null),
            new mframe_t(GameAI.ai_run, -4, null),
            new mframe_t(GameAI.ai_run, -1, null),
            new mframe_t(GameAI.ai_run, 2, null) };

    private static final mmove_t brain_move_run = new mmove_t(FRAME_walk101, FRAME_walk111,
            brain_frames_run, null);

    private static final EntThinkAdapter brain_run = new EntThinkAdapter() {
    	@Override
        public String getID() { return "brain_run"; }
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.power_armor_type = Defines.POWER_ARMOR_SCREEN;
            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = brain_move_stand;
            else
                self.monsterinfo.currentmove = brain_move_run;
            return true;
        }
    };

    private static final mframe_t[] brain_frames_defense = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    static mmove_t brain_move_defense = new mmove_t(FRAME_defens01,
            FRAME_defens08, brain_frames_defense, null);

    private static final mframe_t[] brain_frames_pain3 = {
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -4, null) };

    private static final mmove_t brain_move_pain3 = new mmove_t(FRAME_pain301, FRAME_pain306,
            brain_frames_pain3, brain_run);

    private static final mframe_t[] brain_frames_pain2 = {
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, -2, null) };

    private static final mmove_t brain_move_pain2 = new mmove_t(FRAME_pain201, FRAME_pain208,
            brain_frames_pain2, brain_run);

    private static final mframe_t[] brain_frames_pain1 = {
            new mframe_t(GameAI.ai_move, -6, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, -6, null),
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
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 2, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 7, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, -1, null) };

    private static final mmove_t brain_move_pain1 = new mmove_t(FRAME_pain101, FRAME_pain121,
            brain_frames_pain1, brain_run);

    private static final mframe_t[] brain_frames_duck = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -2, brain_duck_down),
            new mframe_t(GameAI.ai_move, 17, brain_duck_hold),
            new mframe_t(GameAI.ai_move, -3, null),
            new mframe_t(GameAI.ai_move, -1, brain_duck_up),
            new mframe_t(GameAI.ai_move, -5, null),
            new mframe_t(GameAI.ai_move, -6, null),
            new mframe_t(GameAI.ai_move, -6, null) };

    private static final mmove_t brain_move_duck = new mmove_t(FRAME_duck01, FRAME_duck08,
            brain_frames_duck, brain_run);

    private static final EntPainAdapter brain_pain = new EntPainAdapter() {
    	@Override
        public String getID() { return "brain_pain"; }
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
                self.monsterinfo.currentmove = brain_move_pain1;
            } else if (r < 0.66) {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = brain_move_pain2;
            } else {
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
                self.monsterinfo.currentmove = brain_move_pain3;
            }
        }

    };

    private static final EntDieAdapter brain_die = new EntDieAdapter() {
    	@Override
        public String getID() { return "brain_die"; }
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {

            self.s.effects = 0;
            self.monsterinfo.power_armor_type = Defines.POWER_ARMOR_NONE;

            
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
            if (Lib.random() <= 0.5)
                self.monsterinfo.currentmove = brain_move_death1;
            else
                self.monsterinfo.currentmove = brain_move_death2;
        }
    };

    private static final mmove_t brain_move_attack1 = new mmove_t(FRAME_attak101,
            FRAME_attak118, brain_frames_attack1, brain_run);

    private static final mmove_t brain_move_attack2 = new mmove_t(FRAME_attak201,
            FRAME_attak217, brain_frames_attack2, brain_run);

    /*
     * QUAKED monster_brain (1 .5 0) (-16 -16 -24) (16 16 32) Ambush
     * Trigger_Spawn Sight
     */
    public static void SP_monster_brain(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_chest_open = game_import_t.soundindex("brain/brnatck1.wav");
        int sound_tentacles_extend = game_import_t.soundindex("brain/brnatck2.wav");
        sound_tentacles_retract = game_import_t.soundindex("brain/brnatck3.wav");
        sound_death = game_import_t.soundindex("brain/brndeth1.wav");
        int sound_idle1 = game_import_t.soundindex("brain/brnidle1.wav");
        int sound_idle2 = game_import_t.soundindex("brain/brnidle2.wav");
        sound_idle3 = game_import_t.soundindex("brain/brnlens1.wav");
        sound_pain1 = game_import_t.soundindex("brain/brnpain1.wav");
        sound_pain2 = game_import_t.soundindex("brain/brnpain2.wav");
        sound_sight = game_import_t.soundindex("brain/brnsght1.wav");
        sound_search = game_import_t.soundindex("brain/brnsrch1.wav");
        sound_melee1 = game_import_t.soundindex("brain/melee1.wav");
        sound_melee2 = game_import_t.soundindex("brain/melee2.wav");
        sound_melee3 = game_import_t.soundindex("brain/melee3.wav");

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/brain/tris.md2");
        Math3D.VectorSet(self.mins, -16, -16, -24);
        Math3D.VectorSet(self.maxs, 16, 16, 32);

        self.health = 300;
        self.gib_health = -150;
        self.mass = 400;

        self.pain = brain_pain;
        self.die = brain_die;

        self.monsterinfo.stand = brain_stand;
        self.monsterinfo.walk = brain_walk;
        self.monsterinfo.run = brain_run;
        self.monsterinfo.dodge = brain_dodge;
        
        self.monsterinfo.melee = brain_melee;
        self.monsterinfo.sight = brain_sight;
        self.monsterinfo.search = brain_search;
        self.monsterinfo.idle = brain_idle;

        self.monsterinfo.power_armor_type = Defines.POWER_ARMOR_SCREEN;
        self.monsterinfo.power_armor_power = 100;

        game_import_t.linkentity(self);

        self.monsterinfo.currentmove = brain_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.walkmonster_start.think(self);
    }
}