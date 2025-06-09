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



package jake2.game;

import jake2.Defines;
import jake2.Globals;
import jake2.util.Lib;
import jake2.util.Math3D;

import java.util.stream.IntStream;

class GameFunc {

    private static void Move_Calc(edict_t ent, float[] dest, EntThinkAdapter func) {
        Math3D.VectorClear(ent.velocity);
        Math3D.VectorSubtract(dest, ent.s.origin, ent.moveinfo.dir);
        ent.moveinfo.remaining_distance = Math3D
                .VectorNormalize(ent.moveinfo.dir);

        ent.moveinfo.endfunc = func;

        if (ent.moveinfo.speed == ent.moveinfo.accel
                && ent.moveinfo.speed == ent.moveinfo.decel) {
            if (GameBase.level.current_entity == ((ent.flags & Defines.FL_TEAMSLAVE) != 0 ? ent.teammaster
                    : ent)) {
                Move_Begin.think(ent);
            } else {
                ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
                ent.think = Move_Begin;
            }
        } else {
            
            ent.moveinfo.current_speed = 0;
            ent.think = Think_AccelMove;
            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
        }
    }

    private static void AngleMove_Calc(edict_t ent, EntThinkAdapter func) {
        Math3D.VectorClear(ent.avelocity);
        ent.moveinfo.endfunc = func;
        if (GameBase.level.current_entity == ((ent.flags & Defines.FL_TEAMSLAVE) != 0 ? ent.teammaster
                : ent)) {
            AngleMove_Begin.think(ent);
        } else {
            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            ent.think = AngleMove_Begin;
        }
    }

    /**
     * Think_AccelMove
     * 
     * The team has completed a frame of movement, so change the speed for the
     * next frame.
     */
    private static float AccelerationDistance(float target, float rate) {
        return target * ((target / rate) + 1) / 2;
    }

    private static void plat_CalcAcceleratedMove(moveinfo_t moveinfo) {

        moveinfo.move_speed = moveinfo.speed;

        if (moveinfo.remaining_distance < moveinfo.accel) {
            moveinfo.current_speed = moveinfo.remaining_distance;
            return;
        }

        float accel_dist = AccelerationDistance(moveinfo.speed, moveinfo.accel);
        float decel_dist = AccelerationDistance(moveinfo.speed, moveinfo.decel);

        if ((moveinfo.remaining_distance - accel_dist - decel_dist) < 0) {

            double f = (moveinfo.accel + moveinfo.decel)
                    / (moveinfo.accel * moveinfo.decel);
            moveinfo.move_speed = (float) ((-2 + Math.sqrt(4 - 4 * f
                    * (-2 * moveinfo.remaining_distance))) / (2 * f));
            decel_dist = AccelerationDistance(moveinfo.move_speed,
                    moveinfo.decel);
        }

        moveinfo.decel_distance = decel_dist;
    }

    private static void plat_Accelerate(moveinfo_t moveinfo) {
        
        if (moveinfo.remaining_distance <= moveinfo.decel_distance) {
            if (moveinfo.remaining_distance < moveinfo.decel_distance) {
                if (moveinfo.next_speed != 0) {
                    moveinfo.current_speed = moveinfo.next_speed;
                    moveinfo.next_speed = 0;
                    return;
                }
                if (moveinfo.current_speed > moveinfo.decel)
                    moveinfo.current_speed -= moveinfo.decel;
            }
            return;
        }

        
        if (moveinfo.current_speed == moveinfo.move_speed)
            if ((moveinfo.remaining_distance - moveinfo.current_speed) < moveinfo.decel_distance) {

                float p1_distance = moveinfo.remaining_distance
                        - moveinfo.decel_distance;
                float p2_distance = moveinfo.move_speed
                        * (1.0f - (p1_distance / moveinfo.move_speed));
                moveinfo.current_speed = moveinfo.move_speed;
                float distance = p1_distance + p2_distance;
                moveinfo.next_speed = moveinfo.move_speed - moveinfo.decel
                        * (p2_distance / distance);
                return;
            }

        
        if (moveinfo.current_speed < moveinfo.speed) {

            float old_speed = moveinfo.current_speed;

            
            moveinfo.current_speed += moveinfo.accel;
            if (moveinfo.current_speed > moveinfo.speed)
                moveinfo.current_speed = moveinfo.speed;

            
            if ((moveinfo.remaining_distance - moveinfo.current_speed) >= moveinfo.decel_distance)
                return;


            float p1_distance = moveinfo.remaining_distance - moveinfo.decel_distance;
            float p1_speed = (old_speed + moveinfo.move_speed) / 2.0f;
            float p2_distance = moveinfo.move_speed
                    * (1.0f - (p1_distance / p1_speed));
            float distance = p1_distance + p2_distance;
            moveinfo.current_speed = (p1_speed * (p1_distance / distance))
                    + (moveinfo.move_speed * (p2_distance / distance));
            moveinfo.next_speed = moveinfo.move_speed - moveinfo.decel
                    * (p2_distance / distance);
        }

        
    }

    private static void plat_go_up(edict_t ent) {
        if (0 == (ent.flags & Defines.FL_TEAMSLAVE)) {
            if (ent.moveinfo.sound_start != 0)
                game_import_t.sound(ent, Defines.CHAN_NO_PHS_ADD
                        + Defines.CHAN_VOICE, ent.moveinfo.sound_start, 1,
                        Defines.ATTN_STATIC, 0);
            ent.s.sound = ent.moveinfo.sound_middle;
        }
        ent.moveinfo.state = STATE_UP;
        Move_Calc(ent, ent.moveinfo.start_origin, plat_hit_top);
    }

    private static void plat_spawn_inside_trigger(edict_t ent) {


        edict_t trigger = GameUtil.G_Spawn();
        trigger.touch = Touch_Plat_Center;
        trigger.movetype = Defines.MOVETYPE_NONE;
        trigger.solid = Defines.SOLID_TRIGGER;
        trigger.enemy = ent;

        float[] tmin = {0, 0, 0};
        tmin[0] = ent.mins[0] + 25;
        tmin[1] = ent.mins[1] + 25;
        tmin[2] = ent.mins[2];

        float[] tmax = {0, 0, 0};
        tmax[0] = ent.maxs[0] - 25;
        tmax[1] = ent.maxs[1] - 25;
        tmax[2] = ent.maxs[2] + 8;

        tmin[2] = tmax[2] - (ent.pos1[2] - ent.pos2[2] + GameBase.st.lip);

        if ((ent.spawnflags & PLAT_LOW_TRIGGER) != 0)
            tmax[2] = tmin[2] + 8;

        if (tmax[0] - tmin[0] <= 0) {
            tmin[0] = (ent.mins[0] + ent.maxs[0]) * 0.5f;
            tmax[0] = tmin[0] + 1;
        }
        if (tmax[1] - tmin[1] <= 0) {
            tmin[1] = (ent.mins[1] + ent.maxs[1]) * 0.5f;
            tmax[1] = tmin[1] + 1;
        }

        Math3D.VectorCopy(tmin, trigger.mins);
        Math3D.VectorCopy(tmax, trigger.maxs);

        game_import_t.linkentity(trigger);
    }

    /**
     * QUAKED func_plat (0 .5 .8) ? PLAT_LOW_TRIGGER speed default 150
     * 
     * Plats are always drawn in the extended position, so they will light
     * correctly.
     * 
     * If the plat is the target of another trigger or button, it will start out
     * disabled in the extended position until it is trigger, when it will lower
     * and become a normal plat.
     * 
     * "speed" overrides default 200. "accel" overrides default 500 "lip"
     * overrides default 8 pixel lip
     * 
     * If the "height" key is setAt, that will determine the amount the plat
     * moves, instead of being implicitly determoveinfoned by the model's
     * height.
     * 
     * Set "sounds" to one of the following: 1) base fast 2) chain slow
     */
    static void SP_func_plat(edict_t ent) {
        Math3D.VectorClear(ent.s.angles);
        ent.solid = Defines.SOLID_BSP;
        ent.movetype = Defines.MOVETYPE_PUSH;

        game_import_t.setmodel(ent, ent.model);

        ent.blocked = plat_blocked;

        if (0 == ent.speed)
            ent.speed = 20;
        else
            ent.speed *= 0.1;

        if (ent.accel == 0)
            ent.accel = 5;
        else
            ent.accel *= 0.1;

        if (ent.decel == 0)
            ent.decel = 5;
        else
            ent.decel *= 0.1;

        if (ent.dmg == 0)
            ent.dmg = 2;

        if (GameBase.st.lip == 0)
            GameBase.st.lip = 8;

        
        Math3D.VectorCopy(ent.s.origin, ent.pos1);
        Math3D.VectorCopy(ent.s.origin, ent.pos2);
        if (GameBase.st.height != 0)
            ent.pos2[2] -= GameBase.st.height;
        else
            ent.pos2[2] -= (ent.maxs[2] - ent.mins[2]) - GameBase.st.lip;

        ent.use = Use_Plat;

        plat_spawn_inside_trigger(ent); 

        if (ent.targetname != null) {
            ent.moveinfo.state = STATE_UP;
        } else {
            Math3D.VectorCopy(ent.pos2, ent.s.origin);
            game_import_t.linkentity(ent);
            ent.moveinfo.state = STATE_BOTTOM;
        }

        ent.moveinfo.speed = ent.speed;
        ent.moveinfo.accel = ent.accel;
        ent.moveinfo.decel = ent.decel;
        ent.moveinfo.wait = ent.wait;
        Math3D.VectorCopy(ent.pos1, ent.moveinfo.start_origin);
        Math3D.VectorCopy(ent.s.angles, ent.moveinfo.start_angles);
        Math3D.VectorCopy(ent.pos2, ent.moveinfo.end_origin);
        Math3D.VectorCopy(ent.s.angles, ent.moveinfo.end_angles);

        ent.moveinfo.sound_start = game_import_t.soundindex("plats/pt1_strt.wav");
        ent.moveinfo.sound_middle = game_import_t.soundindex("plats/pt1_mid.wav");
        ent.moveinfo.sound_end = game_import_t.soundindex("plats/pt1_end.wav");
    }

    /**
     * DOORS
     * 
     * spawn a trigger surrounding the entire team unless it is already targeted
     * by another.
     * 
     */

    /**
     * QUAKED func_door (0 .5 .8) ? START_OPEN x CRUSHER NOMONSTER ANIMATED
     * TOGGLE ANIMATED_FAST TOGGLE wait in both the start and end states for a
     * trigger event. START_OPEN the door to moves to its destination when
     * spawned, and operate in reverse. It is used to temporarily or permanently
     * close off an area when triggered (not useful for touch or takedamage
     * doors). NOMONSTER monsters will not trigger this door
     * 
     * "message" is printed when the door is touched if it is a trigger door and
     * it hasn't been fired yet "angle" determines the opening direction
     * "targetname" if setAt, no touch field will be spawned and a remote button
     * or trigger field activates the door. "health" if setAt, door must be shot
     * open "speed" movement speed (100 default) "wait" wait before returning (3
     * default, -1 = never return) "lip" lip remaining at end of move (8
     * default) "dmg" damage to inflict when blocked (2 default) "sounds" 1)
     * silent 2) light 3) medium 4) heavy
     */

    private static void door_use_areaportals(edict_t self, boolean open) {

        if (self.target == null)
            return;

        EdictIterator edit = null;

        edict_t t;
        while ((edit = GameBase
                .G_Find(edit, GameBase.findByTarget, self.target)) != null) {
            t = edit.o;
            if (Lib.Q_stricmp(t.classname, "func_areaportal") == 0) {
                game_import_t.SetAreaPortalState(t.style, open);
            }
        }
    }

    private static void door_go_up(edict_t self, edict_t activator) {
        if (self.moveinfo.state == STATE_UP)
            return; 

        if (self.moveinfo.state == STATE_TOP) {
            
            if (self.moveinfo.wait >= 0)
                self.nextthink = GameBase.level.time + self.moveinfo.wait;
            return;
        }

        if (0 == (self.flags & Defines.FL_TEAMSLAVE)) {
            if (self.moveinfo.sound_start != 0)
                game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                        + Defines.CHAN_VOICE, self.moveinfo.sound_start, 1,
                        Defines.ATTN_STATIC, 0);
            self.s.sound = self.moveinfo.sound_middle;
        }
        self.moveinfo.state = STATE_UP;
        if (Lib.strcmp(self.classname, "func_door") == 0)
            Move_Calc(self, self.moveinfo.end_origin, door_hit_top);
        else if (Lib.strcmp(self.classname, "func_door_rotating") == 0)
            AngleMove_Calc(self, door_hit_top);

        GameUtil.G_UseTargets(self, activator);
        door_use_areaportals(self, true);
    }

    /**
     * QUAKED func_water (0 .5 .8) ? START_OPEN func_water is a moveable water
     * brush. It must be targeted to operate. Use a non-water texture at your
     * own risk.
     * 
     * START_OPEN causes the water to move to its destination when spawned and
     * operate in reverse.
     * 
     * "angle" determines the opening direction (up or down only) "speed"
     * movement speed (25 default) "wait" wait before returning (-1 default, -1 =
     * TOGGLE) "lip" lip remaining at end of move (0 default) "sounds" (yes,
     * these need to be changed) 0) no sound 1) water 2) lava
     */

    static void SP_func_water(edict_t self) {

        GameBase.G_SetMovedir(self.s.angles, self.movedir);
        self.movetype = Defines.MOVETYPE_PUSH;
        self.solid = Defines.SOLID_BSP;
        game_import_t.setmodel(self, self.model);

        switch (self.sounds) {
        default:
            break;

        case 1:

            case 2:
                self.moveinfo.sound_start = game_import_t
                    .soundindex("world/mov_watr.wav");
            self.moveinfo.sound_end = game_import_t
                    .soundindex("world/stp_watr.wav");
            break;
        }

        
        Math3D.VectorCopy(self.s.origin, self.pos1);
        float[] abs_movedir = {0, 0, 0};
        abs_movedir[0] = Math.abs(self.movedir[0]);
        abs_movedir[1] = Math.abs(self.movedir[1]);
        abs_movedir[2] = Math.abs(self.movedir[2]);
        self.moveinfo.distance = abs_movedir[0] * self.size[0] + abs_movedir[1]
                * self.size[1] + abs_movedir[2] * self.size[2]
                - GameBase.st.lip;
        Math3D.VectorMA(self.pos1, self.moveinfo.distance, self.movedir,
                self.pos2);

        
        if ((self.spawnflags & DOOR_START_OPEN) != 0) {
            Math3D.VectorCopy(self.pos2, self.s.origin);
            Math3D.VectorCopy(self.pos1, self.pos2);
            Math3D.VectorCopy(self.s.origin, self.pos1);
        }

        Math3D.VectorCopy(self.pos1, self.moveinfo.start_origin);
        Math3D.VectorCopy(self.s.angles, self.moveinfo.start_angles);
        Math3D.VectorCopy(self.pos2, self.moveinfo.end_origin);
        Math3D.VectorCopy(self.s.angles, self.moveinfo.end_angles);

        self.moveinfo.state = STATE_BOTTOM;

        if (0 == self.speed)
            self.speed = 25;
        self.moveinfo.accel = self.moveinfo.decel = self.moveinfo.speed = self.speed;

        if (0 == self.wait)
            self.wait = -1;
        self.moveinfo.wait = self.wait;

        self.use = door_use;

        if (self.wait == -1)
            self.spawnflags |= DOOR_TOGGLE;

        self.classname = "func_door";

        game_import_t.linkentity(self);
    }

    private static void train_resume(edict_t self) {
        float[] dest = { 0, 0, 0 };

        edict_t ent = self.target_ent;

        Math3D.VectorSubtract(ent.s.origin, self.mins, dest);
        self.moveinfo.state = STATE_TOP;
        Math3D.VectorCopy(self.s.origin, self.moveinfo.start_origin);
        Math3D.VectorCopy(dest, self.moveinfo.end_origin);
        Move_Calc(self, dest, train_wait);
        self.spawnflags |= TRAIN_START_ON;

    }

    static void SP_func_train(edict_t self) {
        self.movetype = Defines.MOVETYPE_PUSH;

        Math3D.VectorClear(self.s.angles);
        self.blocked = train_blocked;
        if ((self.spawnflags & TRAIN_BLOCK_STOPS) != 0)
            self.dmg = 0;
        else {
            if (0 == self.dmg)
                self.dmg = 100;
        }
        self.solid = Defines.SOLID_BSP;
        game_import_t.setmodel(self, self.model);

        if (GameBase.st.noise != null)
            self.moveinfo.sound_middle = game_import_t
                    .soundindex(GameBase.st.noise);

        if (0 == self.speed)
            self.speed = 100;

        self.moveinfo.speed = self.speed;
        self.moveinfo.accel = self.moveinfo.decel = self.moveinfo.speed;

        self.use = train_use;

        game_import_t.linkentity(self);

        if (self.target != null) {
            
            
            
            self.nextthink = GameBase.level.time + Defines.FRAMETIME;
            self.think = func_train_find;
        } else {
            game_import_t.dprintf("func_train without a target at "
                    + Lib.vtos(self.absmin) + '\n');
        }
    }

    static void SP_func_timer(edict_t self) {
        if (0 == self.wait)
            self.wait = 1.0f;

        self.use = func_timer_use;
        self.think = func_timer_think;

        if (self.random >= self.wait) {
            self.random = self.wait - Defines.FRAMETIME;
            game_import_t.dprintf("func_timer at " + Lib.vtos(self.s.origin)
                    + " has random >= wait\n");
        }

        if ((self.spawnflags & 1) != 0) {
            self.nextthink = GameBase.level.time + 1.0f + GameBase.st.pausetime
                    + self.delay + self.wait + Lib.crandom() * self.random;
            self.activator = self;
        }

        self.svflags = Defines.SVF_NOCLIENT;
    }

    /**
     * PLATS
     * 
     * movement options:
     * 
     * linear smooth start, hard stop smooth start, smooth stop
     * 
     * start end acceleration speed deceleration begin sound end sound target
     * fired when reaching end wait at end
     * 
     * object characteristics that use move segments
     * --------------------------------------------- movetype_push, or
     * movetype_stop action when touched action when blocked action when used
     * disabled? auto trigger spawning
     * 
     */

    private static final int PLAT_LOW_TRIGGER = 1;

    private static final int STATE_TOP = 0;

    private static final int STATE_BOTTOM = 1;

    private static final int STATE_UP = 2;

    private static final int STATE_DOWN = 3;

    private static final int DOOR_START_OPEN = 1;

    private static final int DOOR_REVERSE = 2;

    private static final int DOOR_CRUSHER = 4;

    private static final int DOOR_NOMONSTER = 8;

    private static final int DOOR_TOGGLE = 32;

    private static final int DOOR_X_AXIS = 64;

    private static final int DOOR_Y_AXIS = 128;

    
    
    

    private static final EntThinkAdapter Move_Done = new EntThinkAdapter() {
        @Override
        public String getID() { return "move_done";}
        @Override
        public boolean think(edict_t ent) {
            Math3D.VectorClear(ent.velocity);
            ent.moveinfo.endfunc.think(ent);
            return true;
        }
    };

    private static final EntThinkAdapter Move_Final = new EntThinkAdapter() {
        @Override
        public String getID() { return "move_final";}
        @Override
        public boolean think(edict_t ent) {

            if (ent.moveinfo.remaining_distance == 0) {
                Move_Done.think(ent);
                return true;
            }

            Math3D.VectorScale(ent.moveinfo.dir,
                    ent.moveinfo.remaining_distance / Defines.FRAMETIME,
                    ent.velocity);

            ent.think = Move_Done;
            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            return true;
        }
    };

    private static final EntThinkAdapter Move_Begin = new EntThinkAdapter() {
        @Override
        public String getID() { return "move_begin";}
        @Override
        public boolean think(edict_t ent) {

            if ((ent.moveinfo.speed * Defines.FRAMETIME) >= ent.moveinfo.remaining_distance) {
                Move_Final.think(ent);
                return true;
            }
            Math3D.VectorScale(ent.moveinfo.dir, ent.moveinfo.speed,
                    ent.velocity);
            float frames = (float) Math
                    .floor((ent.moveinfo.remaining_distance / ent.moveinfo.speed)
                            / Defines.FRAMETIME);
            ent.moveinfo.remaining_distance -= frames * ent.moveinfo.speed
                    * Defines.FRAMETIME;
            ent.nextthink = GameBase.level.time + (frames * Defines.FRAMETIME);
            ent.think = Move_Final;
            return true;
        }
    };

    
    
    

    private static final EntThinkAdapter AngleMove_Done = new EntThinkAdapter() {
        @Override
        public String getID() { return "agnle_move_done";}
        @Override
        public boolean think(edict_t ent) {
            Math3D.VectorClear(ent.avelocity);
            ent.moveinfo.endfunc.think(ent);
            return true;
        }
    };

    private static final EntThinkAdapter AngleMove_Final = new EntThinkAdapter() {
        @Override
        public String getID() { return "angle_move_final";}
        @Override
        public boolean think(edict_t ent) {
            float[] move = { 0, 0, 0 };

            if (ent.moveinfo.state == STATE_UP)
                Math3D.VectorSubtract(ent.moveinfo.end_angles, ent.s.angles,
                        move);
            else
                Math3D.VectorSubtract(ent.moveinfo.start_angles, ent.s.angles,
                        move);

            if (Math3D.VectorEquals(move, Globals.vec3_origin)) {
                AngleMove_Done.think(ent);
                return true;
            }

            Math3D.VectorScale(move, 1.0f / Defines.FRAMETIME, ent.avelocity);

            ent.think = AngleMove_Done;
            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            return true;
        }
    };

    private static final EntThinkAdapter AngleMove_Begin = new EntThinkAdapter() {
        @Override
        public String getID() { return "angle_move_begin";}
        @Override
        public boolean think(edict_t ent) {
            float[] destdelta = { 0, 0, 0 };


            if (ent.moveinfo.state == STATE_UP)
                Math3D.VectorSubtract(ent.moveinfo.end_angles, ent.s.angles,
                        destdelta);
            else
                Math3D.VectorSubtract(ent.moveinfo.start_angles, ent.s.angles,
                        destdelta);


            float len = Math3D.VectorLength(destdelta);


            float traveltime = len / ent.moveinfo.speed;

            if (traveltime < Defines.FRAMETIME) {
                AngleMove_Final.think(ent);
                return true;
            }

            float frames = (float) (Math.floor(traveltime / Defines.FRAMETIME));

            
            
            Math3D.VectorScale(destdelta, 1.0f / traveltime, ent.avelocity);

            
            ent.nextthink = GameBase.level.time + frames * Defines.FRAMETIME;
            ent.think = AngleMove_Final;
            return true;
        }
    };

    private static final EntThinkAdapter Think_AccelMove = new EntThinkAdapter() {
        @Override
        public String getID() { return "thinc_accelmove";}
        @Override
        public boolean think(edict_t ent) {
            ent.moveinfo.remaining_distance -= ent.moveinfo.current_speed;

            if (ent.moveinfo.current_speed == 0) 
                plat_CalcAcceleratedMove(ent.moveinfo);

            plat_Accelerate(ent.moveinfo);

            
            if (ent.moveinfo.remaining_distance <= ent.moveinfo.current_speed) {
                Move_Final.think(ent);
                return true;
            }

            Math3D.VectorScale(ent.moveinfo.dir,
                    ent.moveinfo.current_speed * 10, ent.velocity);
            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            ent.think = Think_AccelMove;
            return true;
        }
    };

    private static final EntThinkAdapter plat_hit_top = new EntThinkAdapter() {
        @Override
        public String getID() { return "plat_hit_top";}
        @Override
        public boolean think(edict_t ent) {
            if (0 == (ent.flags & Defines.FL_TEAMSLAVE)) {
                if (ent.moveinfo.sound_end != 0)
                    game_import_t.sound(ent, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, ent.moveinfo.sound_end, 1,
                            Defines.ATTN_STATIC, 0);
                ent.s.sound = 0;
            }
            ent.moveinfo.state = STATE_TOP;

            ent.think = plat_go_down;
            ent.nextthink = GameBase.level.time + 3;
            return true;
        }
    };

    private static final EntThinkAdapter plat_hit_bottom = new EntThinkAdapter() {
        @Override
        public String getID() { return "plat_hit_bottom";}
        @Override
        public boolean think(edict_t ent) {

            if (0 == (ent.flags & Defines.FL_TEAMSLAVE)) {
                if (ent.moveinfo.sound_end != 0)
                    game_import_t.sound(ent, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, ent.moveinfo.sound_end, 1,
                            Defines.ATTN_STATIC, 0);
                ent.s.sound = 0;
            }
            ent.moveinfo.state = STATE_BOTTOM;
            return true;
        }
    };

    private static final EntThinkAdapter plat_go_down = new EntThinkAdapter() {
        @Override
        public String getID() { return "plat_go_down";}
        @Override
        public boolean think(edict_t ent) {
            if (0 == (ent.flags & Defines.FL_TEAMSLAVE)) {
                if (ent.moveinfo.sound_start != 0)
                    game_import_t.sound(ent, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, ent.moveinfo.sound_start, 1,
                            Defines.ATTN_STATIC, 0);
                ent.s.sound = ent.moveinfo.sound_middle;
            }
            ent.moveinfo.state = STATE_DOWN;
            Move_Calc(ent, ent.moveinfo.end_origin, plat_hit_bottom);
            return true;
        }
    };

    private static final EntBlockedAdapter plat_blocked = new EntBlockedAdapter() {
        @Override
        public String getID() { return "plat_blocked";}
        @Override
        public void blocked(edict_t self, edict_t other) {
            if (0 == (other.svflags & Defines.SVF_MONSTER)
                    && (null == other.client)) {
                
                GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                        other.s.origin, Globals.vec3_origin, 100000, 1, 0,
                        Defines.MOD_CRUSH);
                
                if (other != null)
                    GameMisc.BecomeExplosion1(other);
                return;
            }

            GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                    other.s.origin, Globals.vec3_origin, self.dmg, 1, 0,
                    Defines.MOD_CRUSH);

            switch (self.moveinfo.state) {
                case STATE_UP -> plat_go_down.think(self);
                case STATE_DOWN -> plat_go_up(self);
            }

        }
    };

    private static final EntUseAdapter Use_Plat = new EntUseAdapter() {
        @Override
        public String getID() { return "use_plat";}
        @Override
        public void use(edict_t ent, edict_t other, edict_t activator) {
            if (ent.think != null)
                return; 
            plat_go_down.think(ent);
        }
    };

    private static final EntTouchAdapter Touch_Plat_Center = new EntTouchAdapter() {
        @Override
        public String getID() { return "touch_plat_center";}
        @Override
        public void touch(edict_t ent, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (other.client == null)
                return;

            if (other.health <= 0)
                return;

            ent = ent.enemy;
            switch (ent.moveinfo.state) {
                case STATE_BOTTOM -> plat_go_up(ent);
                case STATE_TOP -> ent.nextthink = GameBase.level.time + 1;
            }
        }
    };

    /**
     * QUAKED func_rotating (0 .5 .8) ? START_ON REVERSE X_AXIS Y_AXIS
     * TOUCH_PAIN STOP ANIMATED ANIMATED_FAST You need to have an origin brush
     * as part of this entity. The center of that brush will be the point around
     * which it is rotated. It will rotate around the Z axis by default. You can
     * check either the X_AXIS or Y_AXIS box to change that.
     * 
     * "speed" determines how fast it moves; default value is 100. "dmg" damage
     * to inflict when blocked (2 default)
     * 
     * REVERSE will cause the it to rotate in the opposite direction. STOP mean
     * it will stop moving instead of pushing entities
     */

    private static final EntBlockedAdapter rotating_blocked = new EntBlockedAdapter() {
        @Override
        public String getID() { return "rotating_blocked";}
        @Override
        public void blocked(edict_t self, edict_t other) {
            GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                    other.s.origin, Globals.vec3_origin, self.dmg, 1, 0,
                    Defines.MOD_CRUSH);
        }
    };

    private static final EntTouchAdapter rotating_touch = new EntTouchAdapter() {
        @Override
        public String getID() { return "rotating_touch";}
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (IntStream.of(0, 1, 2).anyMatch(i -> self.avelocity[i] != 0)) {
                GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                        other.s.origin, Globals.vec3_origin, self.dmg, 1, 0,
                        Defines.MOD_CRUSH);
            }
        }
    };

    private static final EntUseAdapter rotating_use = new EntUseAdapter() {
        @Override
        public String getID() { return "rotating_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            if (!Math3D.VectorEquals(self.avelocity, Globals.vec3_origin)) {
                self.s.sound = 0;
                Math3D.VectorClear(self.avelocity);
                self.touch = null;
            } else {
                self.s.sound = self.moveinfo.sound_middle;
                Math3D.VectorScale(self.movedir, self.speed, self.avelocity);
                if ((self.spawnflags & 16) != 0)
                    self.touch = rotating_touch;
            }
        }
    };

    static final EntThinkAdapter SP_func_rotating = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_rotating";}
        @Override
        public boolean think(edict_t ent) {
            ent.solid = Defines.SOLID_BSP;
            if ((ent.spawnflags & 32) != 0)
                ent.movetype = Defines.MOVETYPE_STOP;
            else
                ent.movetype = Defines.MOVETYPE_PUSH;

            
            Math3D.VectorClear(ent.movedir);
            if ((ent.spawnflags & 4) != 0)
                ent.movedir[2] = 1.0f;
            else if ((ent.spawnflags & 8) != 0)
                ent.movedir[0] = 1.0f;
            else
                
                ent.movedir[1] = 1.0f;

            
            if ((ent.spawnflags & 2) != 0)
                Math3D.VectorNegate(ent.movedir, ent.movedir);

            if (0 == ent.speed)
                ent.speed = 100;
            if (0 == ent.dmg)
                ent.dmg = 2;

            

            ent.use = rotating_use;
            if (ent.dmg != 0)
                ent.blocked = rotating_blocked;

            if ((ent.spawnflags & 1) != 0)
                ent.use.use(ent, null, null);

            if ((ent.spawnflags & 64) != 0)
                ent.s.effects |= Defines.EF_ANIM_ALL;
            if ((ent.spawnflags & 128) != 0)
                ent.s.effects |= Defines.EF_ANIM_ALLFAST;

            game_import_t.setmodel(ent, ent.model);
            game_import_t.linkentity(ent);
            return true;
        }
    };

    /*
     * ======================================================================
     * 
     * BUTTONS
     * 
     * ======================================================================
     */

    /*
     * QUAKED func_button (0 .5 .8) ? When a button is touched, it moves some
     * distance in the direction of it's angle, triggers all of it's targets,
     * waits some time, then returns to it's original position where it can be
     * triggered again.
     * 
     * "angle" determines the opening direction "target" all entities with a
     * matching targetname will be used "speed" override the default 40 speed
     * "wait" override the default 1 second wait (-1 = never return) "lip"
     * override the default 4 pixel lip remaining at end of move "health" if
     * setAt, the button must be killed instead of touched "sounds" 1) silent 2)
     * steam metal 3) wooden clunk 4) metallic click 5) in-out
     */

    private static final EntThinkAdapter button_done = new EntThinkAdapter() {
        @Override
        public String getID() { return "button_done";}
        @Override
        public boolean think(edict_t self) {

            self.moveinfo.state = STATE_BOTTOM;
            self.s.effects &= ~Defines.EF_ANIM23;
            self.s.effects |= Defines.EF_ANIM01;
            return true;
        }
    };

    private static final EntThinkAdapter button_return = new EntThinkAdapter() {
        @Override
        public String getID() { return "button_return";}
        @Override
        public boolean think(edict_t self) {
            self.moveinfo.state = STATE_DOWN;

            Move_Calc(self, self.moveinfo.start_origin, button_done);

            self.s.frame = 0;

            if (self.health != 0)
                self.takedamage = Defines.DAMAGE_YES;
            return true;
        }
    };

    private static final EntThinkAdapter button_wait = new EntThinkAdapter() {
        @Override
        public String getID() { return "button_wait";}
        @Override
        public boolean think(edict_t self) {
            self.moveinfo.state = STATE_TOP;
            self.s.effects &= ~Defines.EF_ANIM01;
            self.s.effects |= Defines.EF_ANIM23;

            GameUtil.G_UseTargets(self, self.activator);
            self.s.frame = 1;
            if (self.moveinfo.wait >= 0) {
                self.nextthink = GameBase.level.time + self.moveinfo.wait;
                self.think = button_return;
            }
            return true;
        }
    };

    private static final EntThinkAdapter button_fire = new EntThinkAdapter() {
        @Override
        public String getID() { return "button_fire";}
        @Override
        public boolean think(edict_t self) {
            if (self.moveinfo.state == STATE_UP
                    || self.moveinfo.state == STATE_TOP)
                return true;

            self.moveinfo.state = STATE_UP;
            if (self.moveinfo.sound_start != 0
                    && 0 == (self.flags & Defines.FL_TEAMSLAVE))
                game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                        + Defines.CHAN_VOICE, self.moveinfo.sound_start, 1,
                        Defines.ATTN_STATIC, 0);
            Move_Calc(self, self.moveinfo.end_origin, button_wait);
            return true;
        }
    };

    private static final EntUseAdapter button_use = new EntUseAdapter() {
        @Override
        public String getID() { return "button_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            self.activator = activator;
            button_fire.think(self);
        }
    };

    private static final EntTouchAdapter button_touch = new EntTouchAdapter() {
        @Override
        public String getID() { return "button_touch";}
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (null == other.client)
                return;

            if (other.health <= 0)
                return;

            self.activator = other;
            button_fire.think(self);

        }
    };

    private static final EntDieAdapter button_killed = new EntDieAdapter() {
        @Override
        public String getID() { return "button_killed";}
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {
            self.activator = attacker;
            self.health = self.max_health;
            self.takedamage = Defines.DAMAGE_NO;
            button_fire.think(self);

        }
    };

    static final EntThinkAdapter SP_func_button = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_button";}
        @Override
        public boolean think(edict_t ent) {

            GameBase.G_SetMovedir(ent.s.angles, ent.movedir);
            ent.movetype = Defines.MOVETYPE_STOP;
            ent.solid = Defines.SOLID_BSP;
            game_import_t.setmodel(ent, ent.model);

            if (ent.sounds != 1)
                ent.moveinfo.sound_start = game_import_t
                        .soundindex("switches/butn2.wav");

            if (0 == ent.speed)
                ent.speed = 40;
            if (0 == ent.accel)
                ent.accel = ent.speed;
            if (0 == ent.decel)
                ent.decel = ent.speed;

            if (0 == ent.wait)
                ent.wait = 3;
            if (0 == GameBase.st.lip)
                GameBase.st.lip = 4;

            Math3D.VectorCopy(ent.s.origin, ent.pos1);
            float[] abs_movedir = {0, 0, 0};
            abs_movedir[0] = Math.abs(ent.movedir[0]);
            abs_movedir[1] = Math.abs(ent.movedir[1]);
            abs_movedir[2] = Math.abs(ent.movedir[2]);
            float dist = abs_movedir[0] * ent.size[0] + abs_movedir[1] * ent.size[1]
                    + abs_movedir[2] * ent.size[2] - GameBase.st.lip;
            Math3D.VectorMA(ent.pos1, dist, ent.movedir, ent.pos2);

            ent.use = button_use;
            ent.s.effects |= Defines.EF_ANIM01;

            if (ent.health != 0) {
                ent.max_health = ent.health;
                ent.die = button_killed;
                ent.takedamage = Defines.DAMAGE_YES;
            } else if (null == ent.targetname)
                ent.touch = button_touch;

            ent.moveinfo.state = STATE_BOTTOM;

            ent.moveinfo.speed = ent.speed;
            ent.moveinfo.accel = ent.accel;
            ent.moveinfo.decel = ent.decel;
            ent.moveinfo.wait = ent.wait;
            Math3D.VectorCopy(ent.pos1, ent.moveinfo.start_origin);
            Math3D.VectorCopy(ent.s.angles, ent.moveinfo.start_angles);
            Math3D.VectorCopy(ent.pos2, ent.moveinfo.end_origin);
            Math3D.VectorCopy(ent.s.angles, ent.moveinfo.end_angles);

            game_import_t.linkentity(ent);
            return true;
        }
    };

    private static final EntThinkAdapter door_hit_top = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_hit_top";}
        @Override
        public boolean think(edict_t self) {
            if (0 == (self.flags & Defines.FL_TEAMSLAVE)) {
                if (self.moveinfo.sound_end != 0)
                    game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, self.moveinfo.sound_end, 1,
                            Defines.ATTN_STATIC, 0);
                self.s.sound = 0;
            }
            self.moveinfo.state = STATE_TOP;
            if ((self.spawnflags & DOOR_TOGGLE) != 0)
                return true;
            if (self.moveinfo.wait >= 0) {
                self.think = door_go_down;
                self.nextthink = GameBase.level.time + self.moveinfo.wait;
            }
            return true;
        }
    };

    private static final EntThinkAdapter door_hit_bottom = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_hit_bottom";}
        @Override
        public boolean think(edict_t self) {
            if (0 == (self.flags & Defines.FL_TEAMSLAVE)) {
                if (self.moveinfo.sound_end != 0)
                    game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, self.moveinfo.sound_end, 1,
                            Defines.ATTN_STATIC, 0);
                self.s.sound = 0;
            }
            self.moveinfo.state = STATE_BOTTOM;
            door_use_areaportals(self, false);
            return true;
        }
    };

    private static final EntThinkAdapter door_go_down = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_go_down";}
        @Override
        public boolean think(edict_t self) {
            if (0 == (self.flags & Defines.FL_TEAMSLAVE)) {
                if (self.moveinfo.sound_start != 0)
                    game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, self.moveinfo.sound_start, 1,
                            Defines.ATTN_STATIC, 0);
                self.s.sound = self.moveinfo.sound_middle;
            }
            if (self.max_health != 0) {
                self.takedamage = Defines.DAMAGE_YES;
                self.health = self.max_health;
            }

            self.moveinfo.state = STATE_DOWN;
            if (Lib.strcmp(self.classname, "func_door") == 0)
                Move_Calc(self, self.moveinfo.start_origin,
                        door_hit_bottom);
            else if (Lib.strcmp(self.classname, "func_door_rotating") == 0)
                AngleMove_Calc(self, door_hit_bottom);
            return true;
        }
    };

    private static final EntUseAdapter door_use = new EntUseAdapter() {
        @Override
        public String getID() { return "door_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {

            if ((self.flags & Defines.FL_TEAMSLAVE) != 0)
                return;

            edict_t ent;
            if ((self.spawnflags & DOOR_TOGGLE) != 0) {
                if (self.moveinfo.state == STATE_UP
                        || self.moveinfo.state == STATE_TOP) {
                    
                    for (ent = self; ent != null; ent = ent.teamchain) {
                        ent.message = null;
                        ent.touch = null;
                        door_go_down.think(ent);
                    }
                    return;
                }
            }

            
            for (ent = self; ent != null; ent = ent.teamchain) {
                ent.message = null;
                ent.touch = null;
                door_go_up(ent, activator);
            }
        }
    };

    private static final EntTouchAdapter Touch_DoorTrigger = new EntTouchAdapter() {
        @Override
        public String getID() { return "touch_door_trigger";}
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (other.health <= 0)
                return;

            if (0 == (other.svflags & Defines.SVF_MONSTER)
                    && (null == other.client))
                return;

            if (0 != (self.owner.spawnflags & DOOR_NOMONSTER)
                    && 0 != (other.svflags & Defines.SVF_MONSTER))
                return;

            if (GameBase.level.time < self.touch_debounce_time)
                return;
            self.touch_debounce_time = GameBase.level.time + 1.0f;

            door_use.use(self.owner, other, other);
        }
    };

    private static final EntThinkAdapter Think_CalcMoveSpeed = new EntThinkAdapter() {
        @Override
        public String getID() { return "think_calc_movespeed";}
        @Override
        public boolean think(edict_t self) {

            if ((self.flags & Defines.FL_TEAMSLAVE) != 0)
                return true;


            float min = Math.abs(self.moveinfo.distance);
            edict_t ent;
            for (ent = self.teamchain; ent != null; ent = ent.teamchain) {
                float dist = Math.abs(ent.moveinfo.distance);
                if (dist < min)
                    min = dist;
            }

            float time = min / self.moveinfo.speed;

            
            for (ent = self; ent != null; ent = ent.teamchain) {
                float newspeed = Math.abs(ent.moveinfo.distance) / time;
                float ratio = newspeed / ent.moveinfo.speed;
                if (ent.moveinfo.accel == ent.moveinfo.speed)
                    ent.moveinfo.accel = newspeed;
                else
                    ent.moveinfo.accel *= ratio;
                if (ent.moveinfo.decel == ent.moveinfo.speed)
                    ent.moveinfo.decel = newspeed;
                else
                    ent.moveinfo.decel *= ratio;
                ent.moveinfo.speed = newspeed;
            }
            return true;
        }
    };

    private static final EntThinkAdapter Think_SpawnDoorTrigger = new EntThinkAdapter() {
        @Override
        public String getID() { return "think_spawn_door_trigger";}
        @Override
        public boolean think(edict_t ent) {

            if ((ent.flags & Defines.FL_TEAMSLAVE) != 0)
                return true;

            float[] mins = {0, 0, 0};
            Math3D.VectorCopy(ent.absmin, mins);
            float[] maxs = {0, 0, 0};
            Math3D.VectorCopy(ent.absmax, maxs);

            edict_t other;
            for (other = ent.teamchain; other != null; other = other.teamchain) {
                GameBase.AddPointToBounds(other.absmin, mins, maxs);
                GameBase.AddPointToBounds(other.absmax, mins, maxs);
            }

            
            mins[0] -= 60;
            mins[1] -= 60;
            maxs[0] += 60;
            maxs[1] += 60;

            other = GameUtil.G_Spawn();
            Math3D.VectorCopy(mins, other.mins);
            Math3D.VectorCopy(maxs, other.maxs);
            other.owner = ent;
            other.solid = Defines.SOLID_TRIGGER;
            other.movetype = Defines.MOVETYPE_NONE;
            other.touch = Touch_DoorTrigger;
            game_import_t.linkentity(other);

            if ((ent.spawnflags & DOOR_START_OPEN) != 0)
                door_use_areaportals(ent, true);

            Think_CalcMoveSpeed.think(ent);
            return true;
        }
    };

    private static final EntBlockedAdapter door_blocked = new EntBlockedAdapter() {
        @Override
        public String getID() { return "door_blocked";}
        @Override
        public void blocked(edict_t self, edict_t other) {

            if (0 == (other.svflags & Defines.SVF_MONSTER)
                    && (null == other.client)) {
                
                GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                        other.s.origin, Globals.vec3_origin, 100000, 1, 0,
                        Defines.MOD_CRUSH);
                
                if (other != null)
                    GameMisc.BecomeExplosion1(other);
                return;
            }

            GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                    other.s.origin, Globals.vec3_origin, self.dmg, 1, 0,
                    Defines.MOD_CRUSH);

            if ((self.spawnflags & DOOR_CRUSHER) != 0)
                return;

            
            
            
            if (self.moveinfo.wait >= 0) {
                edict_t ent;
                if (self.moveinfo.state == STATE_DOWN) {
                    for (ent = self.teammaster; ent != null; ent = ent.teamchain)
                        door_go_up(ent, ent.activator);
                } else {
                    for (ent = self.teammaster; ent != null; ent = ent.teamchain)
                        door_go_down.think(ent);
                }
            }
        }
    };

    private static final EntDieAdapter door_killed = new EntDieAdapter() {
        @Override
        public String getID() { return "door_killed";}
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {

            for (edict_t ent = self.teammaster; ent != null; ent = ent.teamchain) {
                ent.health = ent.max_health;
                ent.takedamage = Defines.DAMAGE_NO;
            }
            door_use.use(self.teammaster, attacker, attacker);
        }
    };

    private static final EntTouchAdapter door_touch = new EntTouchAdapter() {
        @Override
        public String getID() { return "door_touch";}
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (null == other.client)
                return;

            if (GameBase.level.time < self.touch_debounce_time)
                return;
            self.touch_debounce_time = GameBase.level.time + 5.0f;

            game_import_t.centerprintf(other, self.message);
            game_import_t.sound(other, Defines.CHAN_AUTO, game_import_t
                    .soundindex("misc/talk1.wav"), 1, Defines.ATTN_NORM, 0);
        }
    };

    static final EntThinkAdapter SP_func_door = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_door";}
        @Override
        public boolean think(edict_t ent) {

            if (ent.sounds != 1) {
                ent.moveinfo.sound_start = game_import_t
                        .soundindex("doors/dr1_strt.wav");
                ent.moveinfo.sound_middle = game_import_t
                        .soundindex("doors/dr1_mid.wav");
                ent.moveinfo.sound_end = game_import_t
                        .soundindex("doors/dr1_end.wav");
            }

            GameBase.G_SetMovedir(ent.s.angles, ent.movedir);
            ent.movetype = Defines.MOVETYPE_PUSH;
            ent.solid = Defines.SOLID_BSP;
            game_import_t.setmodel(ent, ent.model);

            ent.blocked = door_blocked;
            ent.use = door_use;

            if (0 == ent.speed)
                ent.speed = 100;
            if (GameBase.deathmatch.value != 0)
                ent.speed *= 2;

            if (0 == ent.accel)
                ent.accel = ent.speed;
            if (0 == ent.decel)
                ent.decel = ent.speed;

            if (0 == ent.wait)
                ent.wait = 3;
            if (0 == GameBase.st.lip)
                GameBase.st.lip = 8;
            if (0 == ent.dmg)
                ent.dmg = 2;

            
            Math3D.VectorCopy(ent.s.origin, ent.pos1);
            float[] abs_movedir = {0, 0, 0};
            abs_movedir[0] = Math.abs(ent.movedir[0]);
            abs_movedir[1] = Math.abs(ent.movedir[1]);
            abs_movedir[2] = Math.abs(ent.movedir[2]);
            ent.moveinfo.distance = abs_movedir[0] * ent.size[0]
                    + abs_movedir[1] * ent.size[1] + abs_movedir[2]
                    * ent.size[2] - GameBase.st.lip;

            Math3D.VectorMA(ent.pos1, ent.moveinfo.distance, ent.movedir,
                    ent.pos2);

            
            if ((ent.spawnflags & DOOR_START_OPEN) != 0) {
                Math3D.VectorCopy(ent.pos2, ent.s.origin);
                Math3D.VectorCopy(ent.pos1, ent.pos2);
                Math3D.VectorCopy(ent.s.origin, ent.pos1);
            }

            ent.moveinfo.state = STATE_BOTTOM;

            if (ent.health != 0) {
                ent.takedamage = Defines.DAMAGE_YES;
                ent.die = door_killed;
                ent.max_health = ent.health;
            } else if (ent.targetname != null && ent.message != null) {
                game_import_t.soundindex("misc/talk.wav");
                ent.touch = door_touch;
            }

            ent.moveinfo.speed = ent.speed;
            ent.moveinfo.accel = ent.accel;
            ent.moveinfo.decel = ent.decel;
            ent.moveinfo.wait = ent.wait;
            Math3D.VectorCopy(ent.pos1, ent.moveinfo.start_origin);
            Math3D.VectorCopy(ent.s.angles, ent.moveinfo.start_angles);
            Math3D.VectorCopy(ent.pos2, ent.moveinfo.end_origin);
            Math3D.VectorCopy(ent.s.angles, ent.moveinfo.end_angles);

            if ((ent.spawnflags & 16) != 0)
                ent.s.effects |= Defines.EF_ANIM_ALL;
            if ((ent.spawnflags & 64) != 0)
                ent.s.effects |= Defines.EF_ANIM_ALLFAST;

            
            
            if (null == ent.team)
                ent.teammaster = ent;

            game_import_t.linkentity(ent);

            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            if (ent.health != 0 || ent.targetname != null)
                ent.think = Think_CalcMoveSpeed;
            else
                ent.think = Think_SpawnDoorTrigger;
            return true;
        }
    };

    /*
     * QUAKED func_door_rotating (0 .5 .8) ? START_OPEN REVERSE CRUSHER
     * NOMONSTER ANIMATED TOGGLE X_AXIS Y_AXIS TOGGLE causes the door to wait in
     * both the start and end states for a trigger event.
     * 
     * START_OPEN the door to moves to its destination when spawned, and operate
     * in reverse. It is used to temporarily or permanently close off an area
     * when triggered (not useful for touch or takedamage doors). NOMONSTER
     * monsters will not trigger this door
     * 
     * You need to have an origin brush as part of this entity. The center of
     * that brush will be the point around which it is rotated. It will rotate
     * around the Z axis by default. You can check either the X_AXIS or Y_AXIS
     * box to change that.
     * 
     * "distance" is how many degrees the door will be rotated. "speed"
     * determines how fast the door moves; default value is 100.
     * 
     * REVERSE will cause the door to rotate in the opposite direction.
     * 
     * "message" is printed when the door is touched if it is a trigger door and
     * it hasn't been fired yet "angle" determines the opening direction
     * "targetname" if setAt, no touch field will be spawned and a remote button
     * or trigger field activates the door. "health" if setAt, door must be shot
     * open "speed" movement speed (100 default) "wait" wait before returning (3
     * default, -1 = never return) "dmg" damage to inflict when blocked (2
     * default) "sounds" 1) silent 2) light 3) medium 4) heavy
     */

    static final EntThinkAdapter SP_func_door_rotating = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_door_rotating";}
        @Override
        public boolean think(edict_t ent) {
            Math3D.VectorClear(ent.s.angles);

            
            Math3D.VectorClear(ent.movedir);
            if ((ent.spawnflags & DOOR_X_AXIS) != 0)
                ent.movedir[2] = 1.0f;
            else if ((ent.spawnflags & DOOR_Y_AXIS) != 0)
                ent.movedir[0] = 1.0f;
            else
                
                ent.movedir[1] = 1.0f;

            
            if ((ent.spawnflags & DOOR_REVERSE) != 0)
                Math3D.VectorNegate(ent.movedir, ent.movedir);

            if (0 == GameBase.st.distance) {
                game_import_t.dprintf(ent.classname + " at "
                        + Lib.vtos(ent.s.origin) + " with no distance setAt\n");
                GameBase.st.distance = 90;
            }

            Math3D.VectorCopy(ent.s.angles, ent.pos1);
            Math3D.VectorMA(ent.s.angles, GameBase.st.distance, ent.movedir,
                    ent.pos2);
            ent.moveinfo.distance = GameBase.st.distance;

            ent.movetype = Defines.MOVETYPE_PUSH;
            ent.solid = Defines.SOLID_BSP;
            game_import_t.setmodel(ent, ent.model);

            ent.blocked = door_blocked;
            ent.use = door_use;

            if (0 == ent.speed)
                ent.speed = 100;
            if (0 == ent.accel)
                ent.accel = ent.speed;
            if (0 == ent.decel)
                ent.decel = ent.speed;

            if (0 == ent.wait)
                ent.wait = 3;
            if (0 == ent.dmg)
                ent.dmg = 2;

            if (ent.sounds != 1) {
                ent.moveinfo.sound_start = game_import_t
                        .soundindex("doors/dr1_strt.wav");
                ent.moveinfo.sound_middle = game_import_t
                        .soundindex("doors/dr1_mid.wav");
                ent.moveinfo.sound_end = game_import_t
                        .soundindex("doors/dr1_end.wav");
            }

            
            if ((ent.spawnflags & DOOR_START_OPEN) != 0) {
                Math3D.VectorCopy(ent.pos2, ent.s.angles);
                Math3D.VectorCopy(ent.pos1, ent.pos2);
                Math3D.VectorCopy(ent.s.angles, ent.pos1);
                Math3D.VectorNegate(ent.movedir, ent.movedir);
            }

            if (ent.health != 0) {
                ent.takedamage = Defines.DAMAGE_YES;
                ent.die = door_killed;
                ent.max_health = ent.health;
            }

            if (ent.targetname != null && ent.message != null) {
                game_import_t.soundindex("misc/talk.wav");
                ent.touch = door_touch;
            }

            ent.moveinfo.state = STATE_BOTTOM;
            ent.moveinfo.speed = ent.speed;
            ent.moveinfo.accel = ent.accel;
            ent.moveinfo.decel = ent.decel;
            ent.moveinfo.wait = ent.wait;
            Math3D.VectorCopy(ent.s.origin, ent.moveinfo.start_origin);
            Math3D.VectorCopy(ent.pos1, ent.moveinfo.start_angles);
            Math3D.VectorCopy(ent.s.origin, ent.moveinfo.end_origin);
            Math3D.VectorCopy(ent.pos2, ent.moveinfo.end_angles);

            if ((ent.spawnflags & 16) != 0)
                ent.s.effects |= Defines.EF_ANIM_ALL;

            
            
            if (ent.team == null)
                ent.teammaster = ent;

            game_import_t.linkentity(ent);

            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            if (ent.health != 0 || ent.targetname != null)
                ent.think = Think_CalcMoveSpeed;
            else
                ent.think = Think_SpawnDoorTrigger;
            return true;
        }
    };

    private static final int TRAIN_START_ON = 1;

    private static final int TRAIN_TOGGLE = 2;

    private static final int TRAIN_BLOCK_STOPS = 4;

    /*
     * QUAKED func_train (0 .5 .8) ? START_ON TOGGLE BLOCK_STOPS Trains are
     * moving platforms that players can ride. The targets origin specifies the
     * min point of the train at each corner. The train spawns at the first
     * target it is pointing at. If the train is the target of a button or
     * trigger, it will not begin moving until activated. speed default 100 dmg
     * default 2 noise looping sound to play when the train is in motion
     *  
     */

    private static final EntBlockedAdapter train_blocked = new EntBlockedAdapter() {
        @Override
        public String getID() { return "train_blocked";}
        @Override
        public void blocked(edict_t self, edict_t other) {
            if (0 == (other.svflags & Defines.SVF_MONSTER)
                    && (null == other.client)) {
                
                GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                        other.s.origin, Globals.vec3_origin, 100000, 1, 0,
                        Defines.MOD_CRUSH);
                
                if (other != null)
                    GameMisc.BecomeExplosion1(other);
                return;
            }

            if (GameBase.level.time < self.touch_debounce_time)
                return;

            if (self.dmg == 0)
                return;
            self.touch_debounce_time = GameBase.level.time + 0.5f;
            GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                    other.s.origin, Globals.vec3_origin, self.dmg, 1, 0,
                    Defines.MOD_CRUSH);
        }
    };

    private static final EntThinkAdapter train_wait = new EntThinkAdapter() {
        @Override
        public String getID() { return "train_wait";}
        @Override
        public boolean think(edict_t self) {
            if (self.target_ent.pathtarget != null) {

                edict_t ent = self.target_ent;
                String savetarget = ent.target;
                ent.target = ent.pathtarget;
                GameUtil.G_UseTargets(ent, self.activator);
                ent.target = savetarget;

                
                if (!self.inuse)
                    return true;
            }

            if (self.moveinfo.wait != 0) {
                if (self.moveinfo.wait > 0) {
                    self.nextthink = GameBase.level.time + self.moveinfo.wait;
                    self.think = train_next;
                } else if (0 != (self.spawnflags & TRAIN_TOGGLE)) 
                {
                    train_next.think(self);
                    self.spawnflags &= ~TRAIN_START_ON;
                    Math3D.VectorClear(self.velocity);
                    self.nextthink = 0;
                }

                if (0 == (self.flags & Defines.FL_TEAMSLAVE)) {
                    if (self.moveinfo.sound_end != 0)
                        game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                                + Defines.CHAN_VOICE, self.moveinfo.sound_end,
                                1, Defines.ATTN_STATIC, 0);
                    self.s.sound = 0;
                }
            } else {
                train_next.think(self);
            }
            return true;
        }
    };

    private static final EntThinkAdapter train_next = new EntThinkAdapter() {
        @Override
        public String getID() { return "train_next";}
        @Override
        public boolean think(edict_t self) {
            edict_t ent = null;

            boolean first = true;

            boolean dogoto = true;
            while (dogoto) {
                if (null == self.target) {
                    
                    return true;
                }

                ent = GameBase.G_PickTarget(self.target);
                if (null == ent) {
                    game_import_t.dprintf("train_next: bad target " + self.target
                            + '\n');
                    return true;
                }

                self.target = ent.target;
                dogoto = false;
                
                if ((ent.spawnflags & 1) != 0) {
                    if (!first) {
                        game_import_t
                                .dprintf("connected teleport path_corners, see "
                                        + ent.classname
                                        + " at "
                                        + Lib.vtos(ent.s.origin) + '\n');
                        return true;
                    }
                    first = false;
                    Math3D.VectorSubtract(ent.s.origin, self.mins,
                            self.s.origin);
                    Math3D.VectorCopy(self.s.origin, self.s.old_origin);
                    self.s.event = Defines.EV_OTHER_TELEPORT;
                    game_import_t.linkentity(self);
                    dogoto = true;
                }
            }
            self.moveinfo.wait = ent.wait;
            self.target_ent = ent;

            if (0 == (self.flags & Defines.FL_TEAMSLAVE)) {
                if (self.moveinfo.sound_start != 0)
                    game_import_t.sound(self, Defines.CHAN_NO_PHS_ADD
                            + Defines.CHAN_VOICE, self.moveinfo.sound_start, 1,
                            Defines.ATTN_STATIC, 0);
                self.s.sound = self.moveinfo.sound_middle;
            }

            float[] dest = {0, 0, 0};
            Math3D.VectorSubtract(ent.s.origin, self.mins, dest);
            self.moveinfo.state = STATE_TOP;
            Math3D.VectorCopy(self.s.origin, self.moveinfo.start_origin);
            Math3D.VectorCopy(dest, self.moveinfo.end_origin);
            Move_Calc(self, dest, train_wait);
            self.spawnflags |= TRAIN_START_ON;
            return true;
        }
    };

    public static final EntThinkAdapter func_train_find = new EntThinkAdapter() {
        @Override
        public String getID() { return "func_train_find";}
        @Override
        public boolean think(edict_t self) {

            if (null == self.target) {
                game_import_t.dprintf("train_find: no target\n");
                return true;
            }
            edict_t ent = GameBase.G_PickTarget(self.target);
            if (null == ent) {
                game_import_t.dprintf("train_find: target " + self.target
                        + " not found\n");
                return true;
            }
            self.target = ent.target;

            Math3D.VectorSubtract(ent.s.origin, self.mins, self.s.origin);
            game_import_t.linkentity(self);

            
            if (null == self.targetname)
                self.spawnflags |= TRAIN_START_ON;

            if ((self.spawnflags & TRAIN_START_ON) != 0) {
                self.nextthink = GameBase.level.time + Defines.FRAMETIME;
                self.think = train_next;
                self.activator = self;
            }
            return true;
        }
    };

    public static final EntUseAdapter train_use = new EntUseAdapter() {
        @Override
        public String getID() { return "train_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            self.activator = activator;

            if ((self.spawnflags & TRAIN_START_ON) != 0) {
                if (0 == (self.spawnflags & TRAIN_TOGGLE))
                    return;
                self.spawnflags &= ~TRAIN_START_ON;
                Math3D.VectorClear(self.velocity);
                self.nextthink = 0;
            } else {
                if (self.target_ent != null)
                    train_resume(self);
                else
                    train_next.think(self);
            }
        }
    };

    /*
     * QUAKED trigger_elevator (0.3 0.1 0.6) (-8 -8 -8) (8 8 8)
     */
    private static final EntUseAdapter trigger_elevator_use = new EntUseAdapter() {
        @Override
        public String getID() { return "trigger_elevator_use";}

        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {

            if (0 != self.movetarget.nextthink) {
                
                return;
            }

            if (null == other.pathtarget) {
                game_import_t.dprintf("elevator used with no pathtarget\n");
                return;
            }

            edict_t target = GameBase.G_PickTarget(other.pathtarget);
            if (null == target) {
                game_import_t.dprintf("elevator used with bad pathtarget: "
                        + other.pathtarget + '\n');
                return;
            }

            self.movetarget.target_ent = target;
            train_resume(self.movetarget);
        }
    };

    private static final EntThinkAdapter trigger_elevator_init = new EntThinkAdapter() {
        @Override
        public String getID() { return "trigger_elevator_init";}
        @Override
        public boolean think(edict_t self) {
            if (null == self.target) {
                game_import_t.dprintf("trigger_elevator has no target\n");
                return true;
            }
            self.movetarget = GameBase.G_PickTarget(self.target);
            if (null == self.movetarget) {
                game_import_t.dprintf("trigger_elevator unable to find target "
                        + self.target + '\n');
                return true;
            }
            if (Lib.strcmp(self.movetarget.classname, "func_train") != 0) {
                game_import_t.dprintf("trigger_elevator target " + self.target
                        + " is not a train\n");
                return true;
            }

            self.use = trigger_elevator_use;
            self.svflags = Defines.SVF_NOCLIENT;
            return true;
        }
    };

    static final EntThinkAdapter SP_trigger_elevator = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_trigger_elevator";}
        @Override
        public boolean think(edict_t self) {
            self.think = trigger_elevator_init;
            self.nextthink = GameBase.level.time + Defines.FRAMETIME;
            return true;
        }
    };

    /*
     * QUAKED func_timer (0.3 0.1 0.6) (-8 -8 -8) (8 8 8) START_ON "wait" base
     * time between triggering all targets, default is 1 "random" wait variance,
     * default is 0
     * 
     * so, the basic time between firing is a random time between (wait -
     * random) and (wait + random)
     * 
     * "delay" delay before first firing when turned on, default is 0
     * 
     * "pausetime" additional delay used only the very first time and only if
     * spawned with START_ON
     * 
     * These can used but not touched.
     */

    private static final EntThinkAdapter func_timer_think = new EntThinkAdapter() {
        @Override
        public String getID() { return "func_timer_think";}
        @Override
        public boolean think(edict_t self) {
            GameUtil.G_UseTargets(self, self.activator);
            self.nextthink = GameBase.level.time + self.wait + Lib.crandom()
                    * self.random;
            return true;
        }
    };

    private static final EntUseAdapter func_timer_use = new EntUseAdapter() {
        @Override
        public String getID() { return "func_timer_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            self.activator = activator;

            
            if (self.nextthink != 0) {
                self.nextthink = 0;
                return;
            }

            
            if (self.delay != 0)
                self.nextthink = GameBase.level.time + self.delay;
            else
                func_timer_think.think(self);
        }
    };

    /*
     * QUAKED func_conveyor (0 .5 .8) ? START_ON TOGGLE Conveyors are stationary
     * brushes that move what's on them. The brush should be have a surface with
     * at least one current content enabled. speed default 100
     */

    private static final EntUseAdapter func_conveyor_use = new EntUseAdapter() {
        @Override
        public String getID() { return "func_conveyor_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            if ((self.spawnflags & 1) != 0) {
                self.speed = 0;
                self.spawnflags &= ~1;
            } else {
                self.speed = self.count;
                self.spawnflags |= 1;
            }

            if (0 == (self.spawnflags & 2))
                self.count = 0;
        }
    };

    static final EntThinkAdapter SP_func_conveyor = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_conveyor";}
        @Override
        public boolean think(edict_t self) {

            if (0 == self.speed)
                self.speed = 100;

            if (0 == (self.spawnflags & 1)) {
                self.count = (int) self.speed;
                self.speed = 0;
            }

            self.use = func_conveyor_use;

            game_import_t.setmodel(self, self.model);
            self.solid = Defines.SOLID_BSP;
            game_import_t.linkentity(self);
            return true;
        }
    };

    /*
     * QUAKED func_door_secret (0 .5 .8) ? always_shoot 1st_left 1st_down A
     * secret door. Slide back and then to the side.
     * 
     * open_once doors never closes 1st_left 1st move is left of arrow 1st_down
     * 1st move is down from arrow always_shoot door is shootebale even if
     * targeted
     * 
     * "angle" determines the direction "dmg" damage to inflic when blocked
     * (default 2) "wait" how long to hold in the open position (default 5, -1
     * means hold)
     */

    private static final int SECRET_ALWAYS_SHOOT = 1;

    private static final int SECRET_1ST_LEFT = 2;

    private static final int SECRET_1ST_DOWN = 4;

    private static final EntUseAdapter door_secret_use = new EntUseAdapter() {
        @Override
        public String getID() { return "door_secret_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            
            if (!Math3D.VectorEquals(self.s.origin, Globals.vec3_origin))
                return;

            Move_Calc(self, self.pos1, door_secret_move1);
            door_use_areaportals(self, true);
        }
    };

    private static final EntThinkAdapter door_secret_move1 = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move1";}
        @Override
        public boolean think(edict_t self) {
            self.nextthink = GameBase.level.time + 1.0f;
            self.think = door_secret_move2;
            return true;
        }
    };

    private static final EntThinkAdapter door_secret_move2 = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move2";}
        @Override
        public boolean think(edict_t self) {
            Move_Calc(self, self.pos2, door_secret_move3);
            return true;
        }
    };

    private static final EntThinkAdapter door_secret_move3 = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move3";}
        @Override
        public boolean think(edict_t self) {
            if (self.wait == -1)
                return true;
            self.nextthink = GameBase.level.time + self.wait;
            self.think = door_secret_move4;
            return true;
        }
    };

    private static final EntThinkAdapter door_secret_move4 = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move4";}
        @Override
        public boolean think(edict_t self) {
            Move_Calc(self, self.pos1, door_secret_move5);
            return true;
        }
    };

    private static final EntThinkAdapter door_secret_move5 = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move5";}
        @Override
        public boolean think(edict_t self) {
            self.nextthink = GameBase.level.time + 1.0f;
            self.think = door_secret_move6;
            return true;
        }
    };

    private static final EntThinkAdapter door_secret_move6 = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move6";}
        @Override
        public boolean think(edict_t self) {
            Move_Calc(self, Globals.vec3_origin, door_secret_done);
            return true;
        }
    };

    private static final EntThinkAdapter door_secret_done = new EntThinkAdapter() {
        @Override
        public String getID() { return "door_secret_move7";}
        @Override
        public boolean think(edict_t self) {
            if (null == (self.targetname)
                    || 0 != (self.spawnflags & SECRET_ALWAYS_SHOOT)) {
                self.health = 0;
                self.takedamage = Defines.DAMAGE_YES;
            }
            door_use_areaportals(self, false);
            return true;
        }
    };

    private static final EntBlockedAdapter door_secret_blocked = new EntBlockedAdapter() {
        @Override
        public String getID() { return "door_secret_blocked";}
        @Override
        public void blocked(edict_t self, edict_t other) {
            if (0 == (other.svflags & Defines.SVF_MONSTER)
                    && (null == other.client)) {
                
                GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                        other.s.origin, Globals.vec3_origin, 100000, 1, 0,
                        Defines.MOD_CRUSH);
                
                if (other != null)
                    GameMisc.BecomeExplosion1(other);
                return;
            }

            if (GameBase.level.time < self.touch_debounce_time)
                return;
            self.touch_debounce_time = GameBase.level.time + 0.5f;

            GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                    other.s.origin, Globals.vec3_origin, self.dmg, 1, 0,
                    Defines.MOD_CRUSH);
        }
    };

    private static final EntDieAdapter door_secret_die = new EntDieAdapter() {
        @Override
        public String getID() { return "door_secret_die";}
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {
            self.takedamage = Defines.DAMAGE_NO;
            door_secret_use.use(self, attacker, attacker);
        }
    };

    static final EntThinkAdapter SP_func_door_secret = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_door_secret";}
        @Override
        public boolean think(edict_t ent) {

            ent.moveinfo.sound_start = game_import_t
                    .soundindex("doors/dr1_strt.wav");
            ent.moveinfo.sound_middle = game_import_t
                    .soundindex("doors/dr1_mid.wav");
            ent.moveinfo.sound_end = game_import_t
                    .soundindex("doors/dr1_end.wav");

            ent.movetype = Defines.MOVETYPE_PUSH;
            ent.solid = Defines.SOLID_BSP;
            game_import_t.setmodel(ent, ent.model);

            ent.blocked = door_secret_blocked;
            ent.use = door_secret_use;

            if (null == (ent.targetname)
                    || 0 != (ent.spawnflags & SECRET_ALWAYS_SHOOT)) {
                ent.health = 0;
                ent.takedamage = Defines.DAMAGE_YES;
                ent.die = door_secret_die;
            }

            if (0 == ent.dmg)
                ent.dmg = 2;

            if (0 == ent.wait)
                ent.wait = 5;

            ent.moveinfo.accel = ent.moveinfo.decel = ent.moveinfo.speed = 50;


            float[] up = {0, 0, 0};
            float[] right = {0, 0, 0};
            float[] forward = {0, 0, 0};
            Math3D.AngleVectors(ent.s.angles, forward, right, up);
            Math3D.VectorClear(ent.s.angles);
            float side = 1.0f - (ent.spawnflags & SECRET_1ST_LEFT);
            float width;
            if ((ent.spawnflags & SECRET_1ST_DOWN) != 0)
                width = Math.abs(Math3D.DotProduct(up, ent.size));
            else
                width = Math.abs(Math3D.DotProduct(right, ent.size));
            float length = Math.abs(Math3D.DotProduct(forward, ent.size));
            if ((ent.spawnflags & SECRET_1ST_DOWN) != 0)
                Math3D.VectorMA(ent.s.origin, -1 * width, up, ent.pos1);
            else
                Math3D.VectorMA(ent.s.origin, side * width, right, ent.pos1);
            Math3D.VectorMA(ent.pos1, length, forward, ent.pos2);

            if (ent.health != 0) {
                ent.takedamage = Defines.DAMAGE_YES;
                ent.die = door_killed;
                ent.max_health = ent.health;
            } else if (ent.targetname != null && ent.message != null) {
                game_import_t.soundindex("misc/talk.wav");
                ent.touch = door_touch;
            }

            ent.classname = "func_door";

            game_import_t.linkentity(ent);
            return true;
        }
    };

    /**
     * QUAKED func_killbox (1 0 0) ? Kills everything inside when fired,
     * irrespective of protection.
     */
    private static final EntUseAdapter use_killbox = new EntUseAdapter() {
        @Override
        public String getID() { return "use_killbox";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            GameUtil.KillBox(self);
        }
    };

    static final EntThinkAdapter SP_func_killbox = new EntThinkAdapter() {
        @Override
        public String getID() { return "sp_func_killbox";}
        @Override
        public boolean think(edict_t ent) {
            game_import_t.setmodel(ent, ent.model);
            ent.use = use_killbox;
            ent.svflags = Defines.SVF_NOCLIENT;
            return true;
        }
    };
}