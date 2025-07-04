/*******************************************************************************
 * Copyright (c) 2013, Daniel Murphy
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 	* Redistributions of source code must retain the above copyright notice,
 * 	  this list of conditions and the following disclaimer.
 * 	* Redistributions in binary form must reproduce the above copyright notice,
 * 	  this list of conditions and the following disclaimer in the documentation
 * 	  and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
/*
 * JBox2D - A Java Port of Erin Catto's Box2D
 *
 * JBox2D homepage: http:
 * Box2D homepage: http:
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

package spacegraph.space2d.phys.dynamics.joints;

import jcog.math.v2;
import spacegraph.space2d.phys.dynamics.Body2D;

/**
 * Revolute joint definition. This requires defining an anchor point where the bodies are joined.
 * The definition uses local anchor points so that the initial configuration can violate the
 * constraint slightly. You also need to specify the initial relative angle for joint limits. This
 * helps when saving and loading a game. The local anchor points are measured from the body's origin
 * rather than the center of mass because:<br/>
 * <ul>
 * <li>you might not know where the center of mass will be.</li>
 * <li>if you addAt/remove shapes from a body and recompute the mass, the joints will be broken.</li>
 * </ul>
 */
public class RevoluteJointDef extends JointDef {

    /**
     * The local anchor point relative to body1's origin.
     */
    public final v2 localAnchorA;

    /**
     * The local anchor point relative to body2's origin.
     */
    public final v2 localAnchorB;

    /**
     * The body2 angle minus body1 angle in the reference state (radians).
     */
    public float referenceAngle;

    /**
     * A flag to enable joint limits.
     */
    public boolean enableLimit;

    /**
     * The lower angle for the joint limit (radians).
     */
    public float lowerAngle;

    /**
     * The upper angle for the joint limit (radians).
     */
    public float upperAngle;

    /**
     * A flag to enable the joint motor.
     */
    public boolean enableMotor;

    /**
     * The desired motor speed. Usually in radians per second.
     */
    public float motorSpeed;

    /**
     * The maximum motor torque used to achieve the desired motor speed. Usually in N-m.
     */
    public float maxMotorTorque;

    public RevoluteJointDef() { this(false); }

    public RevoluteJointDef(boolean motorized) {
        super(JointType.REVOLUTE);
        localAnchorA = new v2(0, 0);
        localAnchorB = new v2(0, 0);
        referenceAngle = upperAngle = lowerAngle = 0;
        maxMotorTorque = motorized ? 1 : 0;
        motorSpeed = motorized ? 1 : 0;
        enableMotor = enableLimit = motorized;
    }

    @Deprecated public RevoluteJointDef(Body2D b1, Body2D b2) {
        this(b1, b2, false);
    }

    /** uses the midpoint of their centers */
    public RevoluteJointDef(Body2D b1, Body2D b2, boolean motorized) {
        this(motorized);
        bodyA = b1;
        localA(0,0);
        bodyB = b2;
        localB(0,0);
        referenceAngle = 0;
    }

    public final RevoluteJointDef localA(float x, float y) {
        localAnchorA.set(x, y);
        return this;
    }
    public final RevoluteJointDef localB(float x, float y) {
        localAnchorB.set(x, y);
        return this;
    }

    /**
     * Initialize the bodies, anchors, and reference angle using the world anchor.
     *
     * @param b1
     * @param b2
     * @param anchor
     */
    public RevoluteJointDef initialize(Body2D b1, Body2D b2, v2 anchor) {
        bodyA = b1;
        bodyB = b2;
        bodyA.getLocalPointToOut(anchor, localAnchorA);
        bodyB.getLocalPointToOut(anchor, localAnchorB);
        referenceAngle = bodyB.getAngle() - bodyA.getAngle();
        return this;
    }

    public RevoluteJointDef enableLimit(boolean e) {
        this.enableLimit = e;
        return this;
    }
    public RevoluteJointDef enableMotor(boolean e) {
        this.enableMotor = e;
        return this;
    }
    public RevoluteJointDef limit(float lower, float upper) {
        this.enableLimit = true;
        this.lowerAngle = lower; this.upperAngle = upper;
        return this;
    }
    public RevoluteJointDef maxMotorTorque(float t) {
        this.maxMotorTorque = t;
        return this;
    }



}