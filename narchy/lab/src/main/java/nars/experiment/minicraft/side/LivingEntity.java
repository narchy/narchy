/*
 * Copyright 2012 Jonathan Leahey
 *
 * This file is part of Minicraft
 *
 * Minicraft is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * Minicraft is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Minicraft. If not, see http:
 */

package nars.experiment.minicraft.side;

public abstract class LivingEntity extends Entity {
    private static final long serialVersionUID = 1L;
    protected static final int maxHP = 20;

    public int hitPoints;
    public boolean climbing;
    public boolean facingRight = true;
    public final Inventory inventory;

    protected static final float walkSpeed = .5f;
    protected static final float swimSpeed = .08f;
    protected static final float armLength = 4.5f;
    protected float moveDirection;
    protected long ticksAlive;
    protected int ticksUnderwater;
    protected boolean jumping;

    protected LivingEntity(boolean gravityApplies, float x, float y, int width, int height) {
        super(null, gravityApplies, x, y, width, height);
        this.hitPoints = maxHP;
        inventory = new Inventory(10, 4, 3);
    }

    public void giveItem(Item item, int count) {
        inventory.addItem(item, count);
    }

    public int airRemaining() {
        return Math.max(10 - (ticksUnderwater / 50), 0);
    }

    public void jump(World world, int tileSize) {
        if (jumping) {
            return;
        }

        if (!this.isInWater(world, tileSize)) {
            dy = -.3f;
            jumping = true;
        } else {
            dy = -maxWaterDY - .000001f;
        }
    }

    @Override
    public void updatePosition(World world, int tileSize) {
        ticksAlive++;
        boolean isSwimClimb = this.isInWaterOrClimbable(world, tileSize);
        if (isSwimClimb) {
            dx = moveDirection * swimSpeed;
        } else {
            dx = moveDirection * walkSpeed;
        }
        if (climbing) {
            if (isSwimClimb) {
                jumping = false;
                dy = -maxWaterDY - .000001f;
            } else {
                jump(world, tileSize);
            }
        }
        super.updatePosition(world, tileSize);
        if (this.dy == 0) {
            jumping = false;
        }
        if (this.isInWater(world, tileSize)) {
            jumping = false;
        }

        if (this.isHeadUnderWater(world, tileSize)) {
            ticksUnderwater++;
            if (this.airRemaining() == 0) {
                this.takeDamage(5);

                ticksUnderwater = 300;
            }
        } else {
            ticksUnderwater = 0;
        }
    }

    public void startLeft(boolean slow) {
        facingRight = false;
        if (slow) {
            moveDirection = -.2f;
        } else {
            moveDirection = -1;
        }
    }

    public void stopLeft() {
        if (moveDirection < 0) {
            moveDirection = 0;
        }
    }

    public void startRight(boolean slow) {
        facingRight = true;
        if (slow) {
            moveDirection = .2f;
        } else {
            moveDirection = 1;
        }
    }

    public void stopRight() {
        if (moveDirection > 0) {
            moveDirection = 0;
        }
    }

    public void startClimb() {
        climbing = true;
    }

    public void stopClimb() {
        climbing = false;
    }

    public static float findIntersection(float rayOx, float rayOy, float m, float p1x, float p1y,
                                         float p2x, float p2y) {
        float freeVar = -1;
        if (p1x == p2x) {
            freeVar = -m * (rayOx - p1x) + rayOy;
            if ((freeVar < p1y && freeVar < p2y) || (freeVar > p1y && freeVar > p2y)) {
                return -1;
            }
        } else if (p1y == p2y) {
            freeVar = -(rayOy - p1y) / m + rayOx;
            if ((freeVar < p1x && freeVar < p2x) || (freeVar > p1x && freeVar > p2x)) {
                return -1;
            }
        } else {
            System.err.println("Find intersection -- bad arguments");
        }
        return freeVar;
    }

    @Override
    public void takeDamage(int amount) {
        this.hitPoints -= amount;

        System.out.println("Took " + amount + " damage. Current health = " + this.hitPoints);
    }

    @Override
    public void heal(int amount) {
        int newHP = this.hitPoints + amount;
        this.hitPoints = Math.min(newHP, maxHP);
        System.out.println("Healed " + amount + ". Current health = " + this.hitPoints);
    }
}