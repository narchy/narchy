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

class GameTrigger {

    private static void InitTrigger(edict_t self) {
        if (!Math3D.VectorEquals(self.s.angles, Globals.vec3_origin))
            GameBase.G_SetMovedir(self.s.angles, self.movedir);

        self.solid = Defines.SOLID_TRIGGER;
        self.movetype = Defines.MOVETYPE_NONE;
        game_import_t.setmodel(self, self.model);
        self.svflags = Defines.SVF_NOCLIENT;
    }

    
    
    
    private static void multi_trigger(edict_t ent) {
        if (ent.nextthink != 0)
            return; 

        GameUtil.G_UseTargets(ent, ent.activator);

        if (ent.wait > 0) {
            ent.think = multi_wait;
            ent.nextthink = GameBase.level.time + ent.wait;
        } else { 
                 
            
            ent.touch = null;
            ent.nextthink = GameBase.level.time + Defines.FRAMETIME;
            ent.think = GameUtil.G_FreeEdictA;
        }
    }

    public static void SP_trigger_multiple(edict_t ent) {
        switch (ent.sounds) {
            case 1 -> ent.noise_index = game_import_t.soundindex("misc/secret.wav");
            case 2 -> ent.noise_index = game_import_t.soundindex("misc/talk.wav");
            case 3 -> ent.noise_index = game_import_t.soundindex("misc/trigger1.wav");
        }

        if (ent.wait == 0)
            ent.wait = 0.2f;

        ent.touch = Touch_Multi;
        ent.movetype = Defines.MOVETYPE_NONE;
        ent.svflags |= Defines.SVF_NOCLIENT;

        if ((ent.spawnflags & 4) != 0) {
            ent.solid = Defines.SOLID_NOT;
            ent.use = trigger_enable;
        } else {
            ent.solid = Defines.SOLID_TRIGGER;
            ent.use = Use_Multi;
        }

        if (!Math3D.VectorEquals(ent.s.angles, Globals.vec3_origin))
            GameBase.G_SetMovedir(ent.s.angles, ent.movedir);

        game_import_t.setmodel(ent, ent.model);
        game_import_t.linkentity(ent);
    }

    /**
     * QUAKED trigger_once (.5 .5 .5) ? x x TRIGGERED Triggers once, then
     * removes itself. You must set the key "target" to the name of another
     * object in the level that has a matching "targetname".
     * 
     * If TRIGGERED, this trigger must be triggered before it is live.
     * 
     * sounds 1) secret 2) beep beep 3) large switch 4)
     * 
     * "message" string to be displayed when triggered
     */

    public static void SP_trigger_once(edict_t ent) {
        
        
        if ((ent.spawnflags & 1) != 0) {
            float[] v = { 0, 0, 0 };

            Math3D.VectorMA(ent.mins, 0.5f, ent.size, v);
            ent.spawnflags &= ~1;
            ent.spawnflags |= 4;
            game_import_t.dprintf("fixed TRIGGERED flag on " + ent.classname
                    + " at " + Lib.vtos(v) + '\n');
        }

        ent.wait = -1;
        SP_trigger_multiple(ent);
    }

    public static void SP_trigger_relay(edict_t self) {
        self.use = trigger_relay_use;
    }

    public static void SP_trigger_key(edict_t self) {
        if (GameBase.st.item == null) {
            game_import_t.dprintf("no key item for trigger_key at "
                    + Lib.vtos(self.s.origin) + '\n');
            return;
        }
        self.item = GameItems.FindItemByClassname(GameBase.st.item);

        if (null == self.item) {
            game_import_t.dprintf("item " + GameBase.st.item
                    + " not found for trigger_key at "
                    + Lib.vtos(self.s.origin) + '\n');
            return;
        }

        if (self.target == null) {
            game_import_t.dprintf(self.classname + " at "
                    + Lib.vtos(self.s.origin) + " has no target\n");
            return;
        }

        game_import_t.soundindex("misc/keytry.wav");
        game_import_t.soundindex("misc/keyuse.wav");

        self.use = trigger_key_use;
    }

    public static void SP_trigger_counter(edict_t self) {
        self.wait = -1;
        if (0 == self.count)
            self.count = 2;

        self.use = trigger_counter_use;
    }

    /*
     * ==============================================================================
     * 
     * trigger_always
     * 
     * ==============================================================================
     */

    /*
     * QUAKED trigger_always (.5 .5 .5) (-8 -8 -8) (8 8 8) This trigger will
     * always fire. It is activated by the world.
     */
    public static void SP_trigger_always(edict_t ent) {
        
        if (ent.delay < 0.2f)
            ent.delay = 0.2f;
        GameUtil.G_UseTargets(ent, ent);
    }

    /*
     * QUAKED trigger_push (.5 .5 .5) ? PUSH_ONCE Pushes the player "speed"
     * defaults to 1000
     */
    public static void SP_trigger_push(edict_t self) {
        InitTrigger(self);
        windsound = game_import_t.soundindex("misc/windfly.wav");
        self.touch = trigger_push_touch;
        if (0 == self.speed)
            self.speed = 1000;
        game_import_t.linkentity(self);
    }

    public static void SP_trigger_hurt(edict_t self) {
        InitTrigger(self);

        self.noise_index = game_import_t.soundindex("world/electro.wav");
        self.touch = hurt_touch;

        if (0 == self.dmg)
            self.dmg = 5;

        if ((self.spawnflags & 1) != 0)
            self.solid = Defines.SOLID_NOT;
        else
            self.solid = Defines.SOLID_TRIGGER;

        if ((self.spawnflags & 2) != 0)
            self.use = hurt_use;

        game_import_t.linkentity(self);
    }

    public static void SP_trigger_gravity(edict_t self) {
        if (GameBase.st.gravity == null) {
            game_import_t.dprintf("trigger_gravity without gravity setAt at "
                    + Lib.vtos(self.s.origin) + '\n');
            GameUtil.G_FreeEdict(self);
            return;
        }

        InitTrigger(self);
        self.gravity = Lib.atoi(GameBase.st.gravity);
        self.touch = trigger_gravity_touch;
    }

    public static void SP_trigger_monsterjump(edict_t self) {
        if (0 == self.speed)
            self.speed = 200;
        if (0 == GameBase.st.height)
            GameBase.st.height = 200;
        if (self.s.angles[Defines.YAW] == 0)
            self.s.angles[Defines.YAW] = 360;
        InitTrigger(self);
        self.touch = trigger_monsterjump_touch;
        self.movedir[2] = GameBase.st.height;
    }

    
    private static final EntThinkAdapter multi_wait = new EntThinkAdapter() {
    	@Override
        public String getID(){ return "multi_wait"; }
        @Override
        public boolean think(edict_t ent) {

            ent.nextthink = 0;
            return true;
        }
    };

    private static final EntUseAdapter Use_Multi = new EntUseAdapter() {
    	@Override
        public String getID(){ return "Use_Multi"; }
        @Override
        public void use(edict_t ent, edict_t other, edict_t activator) {
            ent.activator = activator;
            multi_trigger(ent);
        }
    };

    private static final EntTouchAdapter Touch_Multi = new EntTouchAdapter() {
    	@Override
        public String getID(){ return "Touch_Multi"; }
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (other.client != null) {
                if ((self.spawnflags & 2) != 0)
                    return;
            } else if ((other.svflags & Defines.SVF_MONSTER) != 0) {
                if (0 == (self.spawnflags & 1))
                    return;
            } else
                return;

            if (!Math3D.VectorEquals(self.movedir, Globals.vec3_origin)) {
                float[] forward = { 0, 0, 0 };

                Math3D.AngleVectors(other.s.angles, forward, null, null);
                if (Math3D.DotProduct(forward, self.movedir) < 0)
                    return;
            }

            self.activator = other;
            multi_trigger(self);
        }
    };

    /**
     * QUAKED trigger_multiple (.5 .5 .5) ? MONSTER NOT_PLAYER TRIGGERED
     * Variable sized repeatable trigger. Must be targeted at one or more
     * entities. If "delay" is setAt, the trigger waits some time after activating
     * before firing. "wait" : Seconds between triggerings. (.2 default) sounds
     * 1) secret 2) beep beep 3) large switch 4) set "message" to text string
     */
    private static final EntUseAdapter trigger_enable = new EntUseAdapter() {
    	@Override
        public String getID(){ return "trigger_enable"; }
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            self.solid = Defines.SOLID_TRIGGER;
            self.use = Use_Multi;
            game_import_t.linkentity(self);
        }
    };

    /**
     * QUAKED trigger_relay (.5 .5 .5) (-8 -8 -8) (8 8 8) This fixed size
     * trigger cannot be touched, it can only be fired by other events.
     */
    private static final EntUseAdapter trigger_relay_use = new EntUseAdapter() {
    	@Override
        public String getID(){ return "trigger_relay_use"; }
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            GameUtil.G_UseTargets(self, activator);
        }
    };

    /*
     * ==============================================================================
     * 
     * trigger_key
     * 
     * ==============================================================================
     */

    /**
     * QUAKED trigger_key (.5 .5 .5) (-8 -8 -8) (8 8 8) A relay trigger that
     * only fires it's targets if player has the proper key. Use "item" to
     * specify the required key, for example "key_data_cd"
     */

    private static final EntUseAdapter trigger_key_use = new EntUseAdapter() {
    	@Override
        public String getID(){ return "trigger_key_use"; }
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {

            if (self.item == null)
                return;
            if (activator.client == null)
                return;

            int index = GameItems.ITEM_INDEX(self.item);
            if (activator.client.pers.inventory[index] == 0) {
                if (GameBase.level.time < self.touch_debounce_time)
                    return;
                self.touch_debounce_time = GameBase.level.time + 5.0f;
                game_import_t.centerprintf(activator, "You need the "
                        + self.item.pickup_name);
                game_import_t.sound(activator, Defines.CHAN_AUTO,
                		game_import_t.soundindex("misc/keytry.wav"), 1,
                                Defines.ATTN_NORM, 0);
                return;
            }

            game_import_t.sound(activator, Defines.CHAN_AUTO, game_import_t
                    .soundindex("misc/keyuse.wav"), 1, Defines.ATTN_NORM, 0);
            if (GameBase.coop.value != 0) {
                int player;
                edict_t ent;

                if (Lib.strcmp(self.item.classname, "key_power_cube") == 0) {
                    int cube;

                    for (cube = 0; cube < 8; cube++)
                        if ((activator.client.pers.power_cubes & (1 << cube)) != 0)
                            break;
                    for (player = 1; player <= GameBase.game.maxclients; player++) {
                        ent = GameBase.g_edicts[player];
                        if (!ent.inuse)
                            continue;
                        if (null == ent.client)
                            continue;
                        if ((ent.client.pers.power_cubes & (1 << cube)) != 0) {
                            ent.client.pers.inventory[index]--;
                            ent.client.pers.power_cubes &= ~(1 << cube);
                        }
                    }
                } else {
                    for (player = 1; player <= GameBase.game.maxclients; player++) {
                        ent = GameBase.g_edicts[player];
                        if (!ent.inuse)
                            continue;
                        if (ent.client == null)
                            continue;
                        ent.client.pers.inventory[index] = 0;
                    }
                }
            } else {
                activator.client.pers.inventory[index]--;
            }

            GameUtil.G_UseTargets(self, activator);

            self.use = null;
        }
    };

    /**
     * QUAKED trigger_counter (.5 .5 .5) ? nomessage Acts as an intermediary for
     * an action that takes multiple inputs.
     * 
     * If nomessage is not setAt, t will print "1 more.. " etc when triggered and
     * "sequence complete" when finished.
     * 
     * After the counter has been triggered "count" times (default 2), it will
     * fire all of it's targets and remove itself.
     */
    private static final EntUseAdapter trigger_counter_use = new EntUseAdapter() {
    	@Override
        public String getID(){ return "trigger_counter_use"; }

        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            if (self.count == 0)
                return;

            self.count--;

            if (self.count != 0) {
                if (0 == (self.spawnflags & 1)) {
                    game_import_t.centerprintf(activator, self.count
                            + " more to go...");
                    game_import_t.sound(activator, Defines.CHAN_AUTO, game_import_t
                            .soundindex("misc/talk1.wav"), 1,
                            Defines.ATTN_NORM, 0);
                }
                return;
            }

            if (0 == (self.spawnflags & 1)) {
                game_import_t.centerprintf(activator, "Sequence completed!");
                game_import_t.sound(activator, Defines.CHAN_AUTO, game_import_t
                        .soundindex("misc/talk1.wav"), 1, Defines.ATTN_NORM, 0);
            }
            self.activator = activator;
            multi_trigger(self);
        }
    };

    /*
     * ==============================================================================
     * 
     * trigger_push
     * 
     * ==============================================================================
     */

    private static final int PUSH_ONCE = 1;

    private static int windsound;

    private static final EntTouchAdapter trigger_push_touch = new EntTouchAdapter() {
    	@Override
        public String getID(){ return "trigger_push_touch"; }
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if (Lib.strcmp(other.classname, "grenade") == 0) {
                Math3D.VectorScale(self.movedir, self.speed * 10,
                        other.velocity);
            } else if (other.health > 0) {
                Math3D.VectorScale(self.movedir, self.speed * 10,
                        other.velocity);

                if (other.client != null) {
                    
                    Math3D.VectorCopy(other.velocity, other.client.oldvelocity);
                    if (other.fly_sound_debounce_time < GameBase.level.time) {
                        other.fly_sound_debounce_time = GameBase.level.time + 1.5f;
                        game_import_t.sound(other, Defines.CHAN_AUTO, windsound,
                                1, Defines.ATTN_NORM, 0);
                    }
                }
            }
            if ((self.spawnflags & PUSH_ONCE) != 0)
                GameUtil.G_FreeEdict(self);
        }
    };


    /**
     * QUAKED trigger_hurt (.5 .5 .5) ? START_OFF TOGGLE SILENT NO_PROTECTION
     * SLOW Any entity that touches this will be hurt.
     * 
     * It does dmg points of damage each server frame
     * 
     * SILENT supresses playing the sound SLOW changes the damage rate to once
     * per second NO_PROTECTION *nothing* stops the damage
     * 
     * "dmg" default 5 (whole numbers only)
     *  
     */
    private static final EntUseAdapter hurt_use = new EntUseAdapter() {
    	@Override
        public String getID(){ return "hurt_use"; }

        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {
            if (self.solid == Defines.SOLID_NOT)
                self.solid = Defines.SOLID_TRIGGER;
            else
                self.solid = Defines.SOLID_NOT;
            game_import_t.linkentity(self);

            if (0 == (self.spawnflags & 2))
                self.use = null;
        }
    };

    private static final EntTouchAdapter hurt_touch = new EntTouchAdapter() {
    	@Override
        public String getID(){ return "hurt_touch"; }
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {

            if (other.takedamage == 0)
                return;

            if (self.timestamp > GameBase.level.time)
                return;

            if ((self.spawnflags & 16) != 0)
                self.timestamp = GameBase.level.time + 1;
            else
                self.timestamp = GameBase.level.time + Defines.FRAMETIME;

            if (0 == (self.spawnflags & 4)) {
                if ((GameBase.level.framenum % 10) == 0)
                    game_import_t.sound(other, Defines.CHAN_AUTO,
                            self.noise_index, 1, Defines.ATTN_NORM, 0);
            }

            int dflags;
            if ((self.spawnflags & 8) != 0)
                dflags = Defines.DAMAGE_NO_PROTECTION;
            else
                dflags = 0;
            GameCombat.T_Damage(other, self, self, Globals.vec3_origin,
                    other.s.origin, Globals.vec3_origin, self.dmg, self.dmg,
                    dflags, Defines.MOD_TRIGGER_HURT);
        }
    };

    /*
     * ==============================================================================
     * 
     * trigger_gravity
     * 
     * ==============================================================================
     */

    /**
     * QUAKED trigger_gravity (.5 .5 .5) ? Changes the touching entites gravity
     * to the value of "gravity". 1.0 is standard gravity for the level.
     */

    private static final EntTouchAdapter trigger_gravity_touch = new EntTouchAdapter() {
    	@Override
        public String getID(){ return "trigger_gravity_touch"; }

        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            other.gravity = self.gravity;
        }
    };

    /*
     * ==============================================================================
     * 
     * trigger_monsterjump
     * 
     * ==============================================================================
     */

    /**
     * QUAKED trigger_monsterjump (.5 .5 .5) ? Walking monsters that touch this
     * will jump in the direction of the trigger's angle "speed" default to 200,
     * the speed thrown forward "height" default to 200, the speed thrown
     * upwards
     */

    private static final EntTouchAdapter trigger_monsterjump_touch = new EntTouchAdapter() {
    	@Override
        public String getID(){ return "trigger_monsterjump_touch"; }
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {
            if ((other.flags & (Defines.FL_FLY | Defines.FL_SWIM)) != 0)
                return;
            if ((other.svflags & Defines.SVF_DEADMONSTER) != 0)
                return;
            if (0 == (other.svflags & Defines.SVF_MONSTER))
                return;

            
            other.velocity[0] = self.movedir[0] * self.speed;
            other.velocity[1] = self.movedir[1] * self.speed;

            if (other.groundentity != null)
                return;

            other.groundentity = null;
            other.velocity[2] = self.movedir[2];
        }
    };
}