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
package spacegraph.space2d.phys.common;

import com.jogamp.opengl.math.FloatUtil;
import jcog.Util;

/**
 * Global tuning constants based on MKS units and various integer maximums (vertices per shape,
 * pairs, etc.).
 */
public enum Settings {
	;

	/**
     * A "close to zero" float epsilon value for use
     */
    public static final float EPSILON = 1.1920928955078125E-7f;
    public static final float EPSILONsqr = Util.sqr(EPSILON);
    public static final float EPSILONsqrt = FloatUtil.sqrt(EPSILON);

    /**
     * Pi.
     */
    public static final float PI = (float) Math.PI;

    
//    public static final boolean FAST_ABS = true;
//    public static final boolean FAST_FLOOR = true;
//    public static final boolean FAST_CEIL = true;
//    public static final boolean FAST_ROUND = true;
//    public static final boolean FAST_ATAN2 = true;
//    public static final boolean FAST_POW = true;
    public static final int CONTACT_STACK_INIT_SIZE = 32;
//    public static final boolean SINCOS_LUT_ENABLED = true;
    /**
     * smaller the precision, the larger the table. If a small table is used (eg, precision is .006 or
     * greater), make sure you set the table to lerp it's results. Accuracy chart is in the MathUtils
     * source. Or, run the tests yourself in {@link SinCosTest}.</br> </br> Good lerp precision
     * values:
     * <ul>
     * <li>.0092</li>
     * <li>.008201</li>
     * <li>.005904</li>
     * <li>.005204</li>
     * <li>.004305</li>
     * <li>.002807</li>
     * <li>.001508</li>
     * <li>9.32500E-4</li>
     * <li>7.48000E-4</li>
     * <li>8.47000E-4</li>
     * <li>.0005095</li>
     * <li>.0001098</li>
     * <li>9.50499E-5</li>
     * <li>6.08500E-5</li>
     * <li>3.07000E-5</li>
     * <li>1.53999E-5</li>
     * </ul>
     */
    public static final float SINCOS_LUT_PRECISION = 0.00011f;
//    public static final int SINCOS_LUT_LENGTH = (int) Math.ceil(Math.PI * 2 / SINCOS_LUT_PRECISION);
//    /**
//     * Use if the table's precision is large (eg .006 or greater). Although it is more expensive, it
//     * greatly increases accuracy. Look in the MathUtils source for some test results on the accuracy
//     * and speed of lerp vs non lerp. Or, run the tests yourself in {@link SinCosTest}.
//     */
//    public static final boolean SINCOS_LUT_LERP = false;


    

    /**
     * The maximum number of contact points between two convex shapes.
     */
    public static final int maxManifoldPoints = 2;

    /**
     * The maximum number of vertices on a convex polygon.
     */
    public static final int maxPolygonVertices = 16;

    /**
     * This is used to fatten AABBs in the dynamic tree. This allows proxies to move by a small amount
     * without triggering a tree adjustment. This is in meters.
     */
    public static final float aabbExtension = 0.1f;

    /**
     * This is used to fatten AABBs in the dynamic tree. This is used to predict the future position
     * based on the current displacement. This is a dimensionless multiplier.
     */
    public static final float aabbMultiplier = 2.0f;

    /**
     * A small length used as a collision and constraint tolerance. Usually it is chosen to be
     * numerically significant, but visually insignificant.
     */
    public static final float linearSlop = 0.000005f;

    /**
     * A small angle used as a collision and constraint tolerance. Usually it is chosen to be
     * numerically significant, but visually insignificant.
     */
    public static final float angularSlop = (float) (0.001 / 180 * PI);

    /**
     * The radius of the polygon/edge shape skin. This should not be modified. Making this smaller
     * means polygons will have and insufficient for continuous collision. Making it larger may create
     * artifacts for vertex collision.
     */
    public static final float polygonRadius = 2 * linearSlop;

    /**
     * Maximum number of sub-steps per contact in continuous physics simulation.
     */
    public static final int maxSubSteps = 8;

    

    /**
     * Maximum number of contacts to be handled to solve a TOI island.
     */
    public static final int maxTOIContacts = 128;

    /**
     * A velocity threshold for elastic collisions. Any collision with a relative linear velocity
     * below this threshold will be treated as inelastic.
     */
    public static final float velocityThreshold = 1.0f;

    /**
     * The maximum linear position correction used when solving constraints. This helps to prevent
     * overshoot.
     */
    public static final float maxLinearCorrection = 0.2f;

    /**
     * The maximum angular position correction used when solving constraints. This helps to prevent
     * overshoot.
     */
    public static final float maxAngularCorrection = (float) (1.0 / 180.0 * PI);

    /**
     * The maximum linear velocity of a body. This limit is very large and is used to prevent
     * numerical problems. You shouldn't need to adjust this.
     */
    public static final float maxTranslation = 64.0f;
    public static final float maxTranslationSquared = (maxTranslation * maxTranslation);

    /**
     * The maximum angular velocity of a body. This limit is very large and is used to prevent
     * numerical problems. You shouldn't need to adjust this.
     */
    public static final float maxRotation = (0.5f * PI);
    public static final float maxRotationSquared = (maxRotation * maxRotation);

    /**
     * This scale factor controls how fast overlap is resolved. Ideally this would be 1 so that
     * overlap is removed in one time step. However using values close to 1 often lead to overshoot.
     */
    public static final float baumgarte = 0.2f;
    public static final float toiBaugarte = 0.75f;


    

    /**
     * The time that a body must be still before it will go to sleep.
     */
    public static final float timeToSleep = 0.5f;

    /**
     * A body cannot sleep if its linear velocity is above this tolerance.
     */
    public static final float linearSleepTolerance = 0.01f;

    /**
     * A body cannot sleep if its angular velocity is above this tolerance.
     */
    public static final float angularSleepTolerance = (1.0f / 180.0f * PI);

    

    /**
     * A symbolic constant that stands for particle allocation error.
     */
    public static final int invalidParticleIndex = (-1);

    /**
     * The standard distance between particles, divided by the particle radius.
     */
    public static final float particleStride = 0.75f;

    /**
     * The minimum particle weight that produces pressure.
     */
    public static final float minParticleWeight = 1.0f;

    /**
     * The upper limit for particle weight used in pressure calculation.
     */
    public static final float maxParticleWeight = 5.0f;

    /**
     * The maximum distance between particles in a triad, divided by the particle radius.
     */
    private static final int maxTriadDistance = 2;
    public static final int maxTriadDistanceSquared = (maxTriadDistance * maxTriadDistance);

    /**
     * The initial size of particle data buffers.
     */
    public static final int minParticleBufferCapacity = 1024;


//
//    /**
//     * Friction mixing law. Feel free to customize this. TODO djm: add customization
//     *
//     * @param friction1
//     * @param friction2
//     * @return
//     */
//    public static float mixFriction(float friction1, float friction2) {
//        return (float) Math.sqrt(friction1 * friction2);
//    }

//    /**
//     * Restitution mixing law. Feel free to customize this. TODO djm: add customization
//     *
//     * @param restitution1
//     * @param restitution2
//     * @return
//     */
//    public static float mixRestitution(float restitution1, float restitution2) {
//        return Math.max(restitution1, restitution2);
//    }
}