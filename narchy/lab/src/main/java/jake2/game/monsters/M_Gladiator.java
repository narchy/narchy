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

public class M_Gladiator {

    

    private static final int FRAME_stand1 = 0;

    public static final int FRAME_stand2 = 1;

    public static final int FRAME_stand3 = 2;

    public static final int FRAME_stand4 = 3;

    public static final int FRAME_stand5 = 4;

    public static final int FRAME_stand6 = 5;

    private static final int FRAME_stand7 = 6;

    private static final int FRAME_walk1 = 7;

    public static final int FRAME_walk2 = 8;

    public static final int FRAME_walk3 = 9;

    public static final int FRAME_walk4 = 10;

    public static final int FRAME_walk5 = 11;

    public static final int FRAME_walk6 = 12;

    public static final int FRAME_walk7 = 13;

    public static final int FRAME_walk8 = 14;

    public static final int FRAME_walk9 = 15;

    public static final int FRAME_walk10 = 16;

    public static final int FRAME_walk11 = 17;

    public static final int FRAME_walk12 = 18;

    public static final int FRAME_walk13 = 19;

    public static final int FRAME_walk14 = 20;

    public static final int FRAME_walk15 = 21;

    private static final int FRAME_walk16 = 22;

    private static final int FRAME_run1 = 23;

    public static final int FRAME_run2 = 24;

    public static final int FRAME_run3 = 25;

    public static final int FRAME_run4 = 26;

    public static final int FRAME_run5 = 27;

    private static final int FRAME_run6 = 28;

    private static final int FRAME_melee1 = 29;

    public static final int FRAME_melee2 = 30;

    public static final int FRAME_melee3 = 31;

    public static final int FRAME_melee4 = 32;

    public static final int FRAME_melee5 = 33;

    public static final int FRAME_melee6 = 34;

    public static final int FRAME_melee7 = 35;

    public static final int FRAME_melee8 = 36;

    public static final int FRAME_melee9 = 37;

    public static final int FRAME_melee10 = 38;

    public static final int FRAME_melee11 = 39;

    public static final int FRAME_melee12 = 40;

    public static final int FRAME_melee13 = 41;

    public static final int FRAME_melee14 = 42;

    public static final int FRAME_melee15 = 43;

    public static final int FRAME_melee16 = 44;

    private static final int FRAME_melee17 = 45;

    private static final int FRAME_attack1 = 46;

    public static final int FRAME_attack2 = 47;

    public static final int FRAME_attack3 = 48;

    public static final int FRAME_attack4 = 49;

    public static final int FRAME_attack5 = 50;

    public static final int FRAME_attack6 = 51;

    public static final int FRAME_attack7 = 52;

    public static final int FRAME_attack8 = 53;

    private static final int FRAME_attack9 = 54;

    private static final int FRAME_pain1 = 55;

    public static final int FRAME_pain2 = 56;

    public static final int FRAME_pain3 = 57;

    public static final int FRAME_pain4 = 58;

    public static final int FRAME_pain5 = 59;

    private static final int FRAME_pain6 = 60;

    private static final int FRAME_death1 = 61;

    public static final int FRAME_death2 = 62;

    public static final int FRAME_death3 = 63;

    public static final int FRAME_death4 = 64;

    public static final int FRAME_death5 = 65;

    public static final int FRAME_death6 = 66;

    public static final int FRAME_death7 = 67;

    public static final int FRAME_death8 = 68;

    public static final int FRAME_death9 = 69;

    public static final int FRAME_death10 = 70;

    public static final int FRAME_death11 = 71;

    public static final int FRAME_death12 = 72;

    public static final int FRAME_death13 = 73;

    public static final int FRAME_death14 = 74;

    public static final int FRAME_death15 = 75;

    public static final int FRAME_death16 = 76;

    public static final int FRAME_death17 = 77;

    public static final int FRAME_death18 = 78;

    public static final int FRAME_death19 = 79;

    public static final int FRAME_death20 = 80;

    public static final int FRAME_death21 = 81;

    private static final int FRAME_death22 = 82;

    private static final int FRAME_painup1 = 83;

    public static final int FRAME_painup2 = 84;

    public static final int FRAME_painup3 = 85;

    public static final int FRAME_painup4 = 86;

    public static final int FRAME_painup5 = 87;

    public static final int FRAME_painup6 = 88;

    private static final int FRAME_painup7 = 89;

    private static final float MODEL_SCALE = 1.000000f;

    private static int sound_pain1;

    private static int sound_pain2;

    private static int sound_die;

    private static int sound_gun;

    private static int sound_cleaver_swing;

    private static int sound_cleaver_hit;

    private static int sound_cleaver_miss;

    private static int sound_idle;

    private static int sound_search;

    private static int sound_sight;

    private static final EntThinkAdapter gladiator_idle = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_idle"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_idle, 1,
                    Defines.ATTN_IDLE, 0);
            return true;
        }
    };

    private static final EntInteractAdapter gladiator_sight = new EntInteractAdapter() {
    	@Override
        public String getID() { return "gladiator_sight"; }
        @Override
        public boolean interact(edict_t self, edict_t other) {

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_sight, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter gladiator_search = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_search"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_VOICE, sound_search, 1,
                    Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final EntThinkAdapter gladiator_cleaver_swing = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_cleaver_swing"; }
        @Override
        public boolean think(edict_t self) {

            game_import_t.sound(self, Defines.CHAN_WEAPON, sound_cleaver_swing,
                    1, Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final mframe_t[] gladiator_frames_stand = {
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null),
            new mframe_t(GameAI.ai_stand, 0, null) };

    private static final mmove_t gladiator_move_stand = new mmove_t(FRAME_stand1,
            FRAME_stand7, gladiator_frames_stand, null);

    private static final EntThinkAdapter gladiator_stand = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_stand"; }
        @Override
        public boolean think(edict_t self) {

            self.monsterinfo.currentmove = gladiator_move_stand;
            return true;
        }
    };

    private static final mframe_t[] gladiator_frames_walk = {
            new mframe_t(GameAI.ai_walk, 15, null),
            new mframe_t(GameAI.ai_walk, 7, null),
            new mframe_t(GameAI.ai_walk, 6, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 12, null),
            new mframe_t(GameAI.ai_walk, 8, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 5, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 1, null),
            new mframe_t(GameAI.ai_walk, 8, null) };

    private static final mmove_t gladiator_move_walk = new mmove_t(FRAME_walk1, FRAME_walk16,
            gladiator_frames_walk, null);

    private static final EntThinkAdapter gladiator_walk = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_walk"; }
        @Override
        public boolean think(edict_t self) {

            self.monsterinfo.currentmove = gladiator_move_walk;

            return true;
        }
    };

    private static final mframe_t[] gladiator_frames_run = {
            new mframe_t(GameAI.ai_run, 23, null),
            new mframe_t(GameAI.ai_run, 14, null),
            new mframe_t(GameAI.ai_run, 14, null),
            new mframe_t(GameAI.ai_run, 21, null),
            new mframe_t(GameAI.ai_run, 12, null),
            new mframe_t(GameAI.ai_run, 13, null) };

    private static final mmove_t gladiator_move_run = new mmove_t(FRAME_run1, FRAME_run6,
            gladiator_frames_run, null);

    private static final EntThinkAdapter gladiator_run = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_run"; }
        @Override
        public boolean think(edict_t self) {

            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0)
                self.monsterinfo.currentmove = gladiator_move_stand;
            else
                self.monsterinfo.currentmove = gladiator_move_run;

            return true;
        }
    };

    private static final EntThinkAdapter GaldiatorMelee = new EntThinkAdapter() {
    	@Override
        public String getID() { return "GaldiatorMelee"; }
        @Override
        public boolean think(edict_t self) {

            float[] aim = { 0, 0, 0 };

            Math3D.VectorSet(aim, Defines.MELEE_DISTANCE, self.mins[0], -4);
            if (GameWeapon.fire_hit(self, aim, (20 + (Lib.rand() % 5)), 300))
                game_import_t.sound(self, Defines.CHAN_AUTO, sound_cleaver_hit,
                        1, Defines.ATTN_NORM, 0);
            else
                game_import_t.sound(self, Defines.CHAN_AUTO, sound_cleaver_miss,
                        1, Defines.ATTN_NORM, 0);
            return true;
        }
    };

    private static final mframe_t[] gladiator_frames_attack_melee = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, gladiator_cleaver_swing),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, GaldiatorMelee),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, gladiator_cleaver_swing),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, GaldiatorMelee),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null) };

    private static final mmove_t gladiator_move_attack_melee = new mmove_t(FRAME_melee1,
            FRAME_melee17, gladiator_frames_attack_melee, gladiator_run);

    private static final EntThinkAdapter gladiator_melee = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_melee"; }
        @Override
        public boolean think(edict_t self) {

            self.monsterinfo.currentmove = gladiator_move_attack_melee;
            return true;
        }
    };

    private static final EntThinkAdapter GladiatorGun = new EntThinkAdapter() {
    	@Override
        public String getID() { return "GladiatorGun"; }
        @Override
        public boolean think(edict_t self) {

            float[] forward = { 0, 0, 0 }, right = { 0, 0, 0 };

            Math3D.AngleVectors(self.s.angles, forward, right, null);
            float[] start = {0, 0, 0};
            Math3D
                    .G_ProjectSource(
                            self.s.origin,
                            M_Flash.monster_flash_offset[Defines.MZ2_GLADIATOR_RAILGUN_1],
                            forward, right, start);


            float[] dir = {0, 0, 0};
            Math3D.VectorSubtract(self.pos1, start, dir);
            Math3D.VectorNormalize(dir);

            Monster.monster_fire_railgun(self, start, dir, 50, 100,
                    Defines.MZ2_GLADIATOR_RAILGUN_1);

            return true;
        }
    };

    private static final mframe_t[] gladiator_frames_attack_gun = {
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, GladiatorGun),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null),
            new mframe_t(GameAI.ai_charge, 0, null) };

    private static final mmove_t gladiator_move_attack_gun = new mmove_t(FRAME_attack1,
            FRAME_attack9, gladiator_frames_attack_gun, gladiator_run);

    private static final EntThinkAdapter gladiator_attack = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_attack"; }
        @Override
        public boolean think(edict_t self) {

            float[] v = { 0, 0, 0 };

            
            Math3D.VectorSubtract(self.s.origin, self.enemy.s.origin, v);
            float range = Math3D.VectorLength(v);
            if (range <= (Defines.MELEE_DISTANCE + 32))
                return true;

            
            game_import_t.sound(self, Defines.CHAN_WEAPON, sound_gun, 1,
                    Defines.ATTN_NORM, 0);
            Math3D.VectorCopy(self.enemy.s.origin, self.pos1);
            
            self.pos1[2] += self.enemy.viewheight;
            self.monsterinfo.currentmove = gladiator_move_attack_gun;
            return true;
        }
    };

    private static final mframe_t[] gladiator_frames_pain = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t gladiator_move_pain = new mmove_t(FRAME_pain1, FRAME_pain6,
            gladiator_frames_pain, gladiator_run);

    private static final mframe_t[] gladiator_frames_pain_air = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t gladiator_move_pain_air = new mmove_t(FRAME_painup1,
            FRAME_painup7, gladiator_frames_pain_air, gladiator_run);

    private static final EntPainAdapter gladiator_pain = new EntPainAdapter() {
    	@Override
        public String getID() { return "gladiator_pain"; }
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {

            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            if (GameBase.level.time < self.pain_debounce_time) {
                if ((self.velocity[2] > 100)
                        && (self.monsterinfo.currentmove == gladiator_move_pain))
                    self.monsterinfo.currentmove = gladiator_move_pain_air;
                return;
            }

            self.pain_debounce_time = GameBase.level.time + 3;

            if (Lib.random() < 0.5)
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain1, 1,
                        Defines.ATTN_NORM, 0);
            else
                game_import_t.sound(self, Defines.CHAN_VOICE, sound_pain2, 1,
                        Defines.ATTN_NORM, 0);

            if (GameBase.skill.value == 3)
                return; 

            if (self.velocity[2] > 100)
                self.monsterinfo.currentmove = gladiator_move_pain_air;
            else
                self.monsterinfo.currentmove = gladiator_move_pain;

        }
    };

    private static final EntThinkAdapter gladiator_dead = new EntThinkAdapter() {
    	@Override
        public String getID() { return "gladiator_dead"; }
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

    private static final mframe_t[] gladiator_frames_death = {
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

    private static final mmove_t gladiator_move_death = new mmove_t(FRAME_death1,
            FRAME_death22, gladiator_frames_death, gladiator_dead);

    private static final EntDieAdapter gladiator_die = new EntDieAdapter() {
    	@Override
        public String getID() { return "gladiator_die"; }
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

            
            game_import_t.sound(self, Defines.CHAN_VOICE, sound_die, 1,
                    Defines.ATTN_NORM, 0);
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_YES;

            self.monsterinfo.currentmove = gladiator_move_death;

        }
    };

    /*
     * QUAKED monster_gladiator (1 .5 0) (-32 -32 -24) (32 32 64) Ambush
     * Trigger_Spawn Sight
     */
    public static void SP_monster_gladiator(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        sound_pain1 = game_import_t.soundindex("gladiator/pain.wav");
        sound_pain2 = game_import_t.soundindex("gladiator/gldpain2.wav");
        sound_die = game_import_t.soundindex("gladiator/glddeth2.wav");
        sound_gun = game_import_t.soundindex("gladiator/railgun.wav");
        sound_cleaver_swing = game_import_t.soundindex("gladiator/melee1.wav");
        sound_cleaver_hit = game_import_t.soundindex("gladiator/melee2.wav");
        sound_cleaver_miss = game_import_t.soundindex("gladiator/melee3.wav");
        sound_idle = game_import_t.soundindex("gladiator/gldidle1.wav");
        sound_search = game_import_t.soundindex("gladiator/gldsrch1.wav");
        sound_sight = game_import_t.soundindex("gladiator/sight.wav");

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t
                .modelindex("models/monsters/gladiatr/tris.md2");
        Math3D.VectorSet(self.mins, -32, -32, -24);
        Math3D.VectorSet(self.maxs, 32, 32, 64);

        self.health = 400;
        self.gib_health = -175;
        self.mass = 400;

        self.pain = gladiator_pain;
        self.die = gladiator_die;

        self.monsterinfo.stand = gladiator_stand;
        self.monsterinfo.walk = gladiator_walk;
        self.monsterinfo.run = gladiator_run;
        self.monsterinfo.dodge = null;
        self.monsterinfo.attack = gladiator_attack;
        self.monsterinfo.melee = gladiator_melee;
        self.monsterinfo.sight = gladiator_sight;
        self.monsterinfo.idle = gladiator_idle;
        self.monsterinfo.search = gladiator_search;

        game_import_t.linkentity(self);
        self.monsterinfo.currentmove = gladiator_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.walkmonster_start.think(self);
    }
}