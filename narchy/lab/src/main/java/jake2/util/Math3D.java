/*
Copyright (C) 1997-2001 Id Software, Inc.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

*/


package jake2.util;

import jake2.Defines;
import jake2.game.cplane_t;
import jake2.qcommon.Com;
import jcog.Util;

import java.util.Arrays;

import static jcog.Util.sqr;

public class Math3D {

    private static final float shortratio = 360.0f / 65536.0f;
    private static final float piratio = (float) (Math.PI / 360.0);
    private static final float[][] m = new float[3][3];
    private static final float[][] im = new float[3][3];
    private static final float[][] tmpmat = new float[3][3];
    private static final float[][] zrot = new float[3][3];
    private static final float[] vr = {0, 0, 0};
    private static final float[] vup = {0, 0, 0};
    private static final float[] vf = {0, 0, 0};
    private static final float[][] PLANE_XYZ = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    public static void set(float[] v1, float[] v2) {
        v1[0] = v2[0];
        v1[1] = v2[1];
        v1[2] = v2[2];
    }

    public static void VectorSubtract(float[] a, float[] b, float[] c) {
        c[0] = a[0] - b[0];
        c[1] = a[1] - b[1];
        c[2] = a[2] - b[2];
    }

    public static void VectorSubtract(short[] a, short[] b, int[] c) {
        c[0] = a[0] - b[0];
        c[1] = a[1] - b[1];
        c[2] = a[2] - b[2];
    }

    public static void VectorAdd(float[] a, float[] b, float[] to) {
        to[0] = a[0] + b[0];
        to[1] = a[1] + b[1];
        to[2] = a[2] + b[2];
    }

    public static void VectorCopy(float[] from, float[] to) {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
    }

    public static void VectorCopy(short[] from, short[] to) {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
    }

    public static void VectorCopy(short[] from, float[] to) {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
    }

    public static void VectorCopy(float[] from, short[] to) {
        to[0] = (short) from[0];
        to[1] = (short) from[1];
        to[2] = (short) from[2];
    }

    public static void VectorClear(float[] a) {
        a[0] = a[1] = a[2] = 0;
    }

    public static boolean VectorEquals(float[] v1, float[] v2) {
        return !(v1[0] != v2[0] || v1[1] != v2[1] || v1[2] != v2[2]);

    }

    public static void VectorNegate(float[] from, float[] to) {
        to[0] = -from[0];
        to[1] = -from[1];
        to[2] = -from[2];
    }

    public static void VectorSet(float[] v, float x, float y, float z) {
        v[0] = (x);
        v[1] = (y);
        v[2] = (z);
    }

    public static void VectorMA(float[] veca, float scale, float[] vecb, float[] to) {
        to[0] = jcog.Util.fma(scale, vecb[0], veca[0]);
        to[1] = jcog.Util.fma(scale, vecb[1], veca[1]);
        to[2] = jcog.Util.fma(scale, vecb[2], veca[2]);
    }

    public static float VectorNormalize(float[] v) {

        double lengthSq = VectorLengthSq(v);
        if (Math.abs(lengthSq) > Util.sqr(Float.MIN_NORMAL)) {
            float length = (float)Math.sqrt(lengthSq);
            v[0] /= length;
            v[1] /= length;
            v[2] /= length;
			return length;
        } else {
			Arrays.fill(v, 0);
			return 0;
		}
    }

    public static float VectorLength(float[] v) {
        return (float) Math.sqrt(VectorLengthSq(v));
    }

    public static double VectorLengthSq(float[] v) {
        return sqr(((double)v[0])) + sqr(v[1]) + sqr(v[2]);
    }

//    public static void VectorInverse(float[] v) {
//        v[0] = -v[0];
//        v[1] = -v[1];
//        v[2] = -v[2];
//    }

    public static void VectorScale(float[] in, float scale, float[] out) {
        out[0] = in[0] * scale;
        out[1] = in[1] * scale;
        out[2] = in[2] * scale;
    }

    public static float vectoyaw(float[] vec) {
        float yaw;

        if (/*vec[YAW] == 0 &&*/
                vec[Defines.PITCH] == 0) {
            yaw = 0;
            if (vec[Defines.YAW] > 0)
                yaw = 90;
            else if (vec[Defines.YAW] < 0)
                yaw = -90;
        } else {

            yaw = (float) (Math.atan2(vec[Defines.YAW], vec[Defines.PITCH]) * 180 / Math.PI);
            if (yaw < 0)
                yaw += 360;
        }

        return yaw;
    }

    public static void vectoangles(float[] value1, float[] angles) {

        float yaw, pitch;

        if (value1[1] == 0 && value1[0] == 0) {
            yaw = 0;
            if (value1[2] > 0)
                pitch = 90;
            else
                pitch = 270;
        } else {
            if (value1[0] != 0)
                yaw = (int) (Math.atan2(value1[1], value1[0]) * 180 / Math.PI);
            else if (value1[1] > 0)
                yaw = 90;
            else
                yaw = -90;
            if (yaw < 0)
                yaw += 360;

            double forward = Math.sqrt(sqr(value1[0]) + sqr(value1[1] * value1[1]));
            pitch = (float) (Math.atan2(value1[2], forward) * 180 / Math.PI);
            if (pitch < 0)
                pitch += 360;
        }

        angles[Defines.PITCH] = -pitch;
        angles[Defines.YAW] = yaw;
        angles[Defines.ROLL] = 0;
    }

    public static void RotatePointAroundVector(float[] dst, float[] dir, float[] point, float degrees) {
        vf[0] = dir[0];
        vf[1] = dir[1];
        vf[2] = dir[2];

        PerpendicularVector(vr, dir);
        CrossProduct(vr, vf, vup);

        m[0][0] = vr[0];
        m[1][0] = vr[1];
        m[2][0] = vr[2];

        m[0][1] = vup[0];
        m[1][1] = vup[1];
        m[2][1] = vup[2];

        m[0][2] = vf[0];
        m[1][2] = vf[1];
        m[2][2] = vf[2];

        im[0][0] = m[0][0];
        im[0][1] = m[1][0];
        im[0][2] = m[2][0];
        im[1][0] = m[0][1];
        im[1][1] = m[1][1];
        im[1][2] = m[2][1];
        im[2][0] = m[0][2];
        im[2][1] = m[1][2];
        im[2][2] = m[2][2];

        float[][] zrot = Math3D.zrot;
        zrot[0][2] = zrot[1][2] = zrot[2][0] = zrot[2][1] = 0.0f;

        zrot[2][2] = 1.0F;

        zrot[0][0] = zrot[1][1] = (float) Math.cos(DEG2RAD(degrees));
        zrot[0][1] = (float) Math.sin(DEG2RAD(degrees));
        zrot[1][0] = -zrot[0][1];

        R_ConcatRotations(m, zrot, tmpmat);
        R_ConcatRotations(tmpmat, im, zrot);

        for (int i = 0; i < 3; i++) {
            float[] zr = zrot[i];
            dst[i] = zr[0] * point[0] + zr[1] * point[1] + zr[2] * point[2];
        }
    }

    public static void MakeNormalVectors(float[] forward, float[] right, float[] up) {


        right[1] = -forward[0];
        right[2] = forward[1];
        right[0] = forward[2];

        float d = DotProduct(right, forward);
        VectorMA(right, -d, forward, right);
        VectorNormalize(right);
        CrossProduct(right, forward, up);
    }

    public static float SHORT2ANGLE(int x) {
        return (x * shortratio);
    }

//    /*
//    ================
//    R_ConcatTransforms
//    ================
//    */
//    public static void R_ConcatTransforms(float[][] in1, float[][] in2, float[][] out) {
//        out[0][0] = in1[0][0] * in2[0][0] + in1[0][1] * in2[1][0] + in1[0][2] * in2[2][0];
//        out[0][1] = in1[0][0] * in2[0][1] + in1[0][1] * in2[1][1] + in1[0][2] * in2[2][1];
//        out[0][2] = in1[0][0] * in2[0][2] + in1[0][1] * in2[1][2] + in1[0][2] * in2[2][2];
//        out[0][3] = in1[0][0] * in2[0][3] + in1[0][1] * in2[1][3] + in1[0][2] * in2[2][3] + in1[0][3];
//        out[1][0] = in1[1][0] * in2[0][0] + in1[1][1] * in2[1][0] + in1[1][2] * in2[2][0];
//        out[1][1] = in1[1][0] * in2[0][1] + in1[1][1] * in2[1][1] + in1[1][2] * in2[2][1];
//        out[1][2] = in1[1][0] * in2[0][2] + in1[1][1] * in2[1][2] + in1[1][2] * in2[2][2];
//        out[1][3] = in1[1][0] * in2[0][3] + in1[1][1] * in2[1][3] + in1[1][2] * in2[2][3] + in1[1][3];
//        out[2][0] = in1[2][0] * in2[0][0] + in1[2][1] * in2[1][0] + in1[2][2] * in2[2][0];
//        out[2][1] = in1[2][0] * in2[0][1] + in1[2][1] * in2[1][1] + in1[2][2] * in2[2][1];
//        out[2][2] = in1[2][0] * in2[0][2] + in1[2][1] * in2[1][2] + in1[2][2] * in2[2][2];
//        out[2][3] = in1[2][0] * in2[0][3] + in1[2][1] * in2[1][3] + in1[2][2] * in2[2][3] + in1[2][3];
//    }

    /**
     * concatenates 2 matrices each [3][3].
     */
    private static void R_ConcatRotations(float[][] in1, float[][] in2, float[][] out) {
		R_concatRotationComponent(in2, in1[0], out[0]);
		R_concatRotationComponent(in2, in1[1], out[1]);
		R_concatRotationComponent(in2, in1[2], out[2]);
	}

	private static void R_concatRotationComponent(float[][] x, float[] y, float[] xy) {
		float[] x0 = x[0];
		float[] x1 = x[1];
		float[] x2 = x[2];
		xy[0] = y[0] * x0[0] + y[1] * x1[0] + y[2] * x2[0];
		xy[1] = y[0] * x0[1] + y[1] * x1[1] + y[2] * x2[1];
		xy[2] = y[0] * x0[2] + y[1] * x1[2] + y[2] * x2[2];
	}

	private static void ProjectPointOnPlane(float[] dst, float[] p, float[] normal) {

        float inv_denom = 1.0F / DotProduct(normal, normal);

        float d = DotProduct(normal, p) * inv_denom;

        dst[0] = normal[0] * inv_denom;
        dst[1] = normal[1] * inv_denom;
        dst[2] = normal[2] * inv_denom;

        dst[0] = p[0] - d * dst[0];
        dst[1] = p[1] - d * dst[1];
        dst[2] = p[2] - d * dst[2];
    }

    /**
     * assumes "src" is normalized
     */
    public static void PerpendicularVector(float[] dst, float[] src) {
        int pos;
        int i;
        float minelem = 1.0F;


        for (pos = 0, i = 0; i < 3; i++) {
            if (Math.abs(src[i]) < minelem) {
                pos = i;
                minelem = Math.abs(src[i]);
            }
        }

        ProjectPointOnPlane(dst, PLANE_XYZ[pos], src);


        VectorNormalize(dst);
    }

    /**
     * stellt fest, auf welcher Seite sich die Kiste befindet, wenn die Ebene
     * durch Entfernung und Senkrechten-Normale gegeben ist.
     * erste Version mit vec3_t...
     */
    public static int BoxOnPlaneSide(float[] emins, float[] emaxs, cplane_t p) {

        assert (emins.length == 3 && emaxs.length == 3) : "vec3_t bug";


        byte ptype = p.type;
        float pDist = p.dist;
        if (ptype < 3) {
            if (pDist <= emins[ptype])
                return 1;
            if (pDist >= emaxs[ptype])
                return 2;
            return 3;
        }


        float[] normal = p.normal;
        float n0 = normal[0];
        float n1 = normal[1];
        float n2 = normal[2];
        float max0 = emaxs[0];
        float max1 = emaxs[1];
        float max2 = emaxs[2];
        float min0 = emins[0];
        float min1 = emins[1];
        float min2 = emins[2];
        float dist2;
        float dist1;
		switch (p.signbits) {
			case 0 -> {
				dist1 = n0 * max0 + n1 * max1 + n2 * max2;
				dist2 = n0 * min0 + n1 * min1 + n2 * min2;
			}
			case 1 -> {
				dist1 = n0 * min0 + n1 * max1 + n2 * max2;
				dist2 = n0 * max0 + n1 * min1 + n2 * min2;
			}
			case 2 -> {
				dist1 = n0 * max0 + n1 * min1 + n2 * max2;
				dist2 = n0 * min0 + n1 * max1 + n2 * min2;
			}
			case 3 -> {
				dist1 = n0 * min0 + n1 * min1 + n2 * max2;
				dist2 = n0 * max0 + n1 * max1 + n2 * min2;
			}
			case 4 -> {
				dist1 = n0 * max0 + n1 * max1 + n2 * min2;
				dist2 = n0 * min0 + n1 * min1 + n2 * max2;
			}
			case 5 -> {
				dist1 = n0 * min0 + n1 * max1 + n2 * min2;
				dist2 = n0 * max0 + n1 * min1 + n2 * max2;
			}
			case 6 -> {
				dist1 = n0 * max0 + n1 * min1 + n2 * min2;
				dist2 = n0 * min0 + n1 * max1 + n2 * max2;
			}
			case 7 -> {
				dist1 = n0 * min0 + n1 * min1 + n2 * min2;
				dist2 = n0 * max0 + n1 * max1 + n2 * max2;
			}
			default -> {
				dist1 = dist2 = 0;
				assert (false) : "BoxOnPlaneSide bug";
			}
		}

        int sides = 0;
        if (dist1 >= pDist)
            sides = 1;
        if (dist2 < pDist)
            sides |= 2;

        assert (sides != 0) : "BoxOnPlaneSide(): sides == 0 bug";

        return sides;
    }


    public static void AngleVectors(float[] angles, float[] forward, float[] right, float[] up) {

        float cr = 2.0f * piratio;

        float angle = angles[Defines.YAW] * (cr);
        float sy = (float) Math.sin(angle);
        float cy = (float) Math.cos(angle);
        float angle2 = angles[Defines.PITCH] * (cr);
        float sp = (float) Math.sin(angle2);
        float cp = (float) Math.cos(angle2);

        if (forward != null) {
            forward[0] = cp * cy;
            forward[1] = cp * sy;
            forward[2] = -sp;
        }

        if (right != null || up != null) {
            float angle3 = angles[Defines.ROLL] * (cr);
            float sr = (float) Math.sin(angle3);
            cr = (float) Math.cos(angle3);

            if (right != null) {
                right[0] = (-sr * sp * cy + cr * sy);
                right[1] = (-sr * sp * sy + -cr * cy);
                right[2] = -sr * cp;
            }
            if (up != null) {
                up[0] = (cr * sp * cy + sr * sy);
                up[1] = (cr * sp * sy + -sr * cy);
                up[2] = cr * cp;
            }
        }
    }

    public static void G_ProjectSource(
            float[] point,
            float[] distance,
            float[] forward,
            float[] right,
            float[] result) {
        result[0] = point[0] + forward[0] * distance[0] + right[0] * distance[1];
        result[1] = point[1] + forward[1] * distance[0] + right[1] * distance[1];
        result[2] = point[2] + forward[2] * distance[0] + right[2] * distance[1] + distance[2];
    }

    public static float DotProduct(float[] x, float[] y) {
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    }

    public static void CrossProduct(float[] v1, float[] v2, float[] cross) {
        float v11 = v1[1];
        float v12 = v1[2];
        float v22 = v2[2];
        float v21 = v2[1];
        cross[0] = v11 * v22 - v12 * v21;
        float v10 = v1[0];
        float v20 = v2[0];
        cross[1] = v12 * v20 - v10 * v22;
        cross[2] = v10 * v21 - v11 * v20;
    }


    private static float DEG2RAD(float in) {
        return (in * (float) Math.PI) / 180.0f;
    }

    public static float anglemod(float a) {
        return shortratio * ((int) (a / (shortratio)) & 65535);
    }

    public static int ANGLE2SHORT(float x) {
        return ((int) ((x) / shortratio) & 65535);
    }

    public static float LerpAngle(float a2, float a1, float frac) {
        if (a1 - a2 > 180)
            a1 -= 360;
        if (a1 - a2 < -180)
            a1 += 360;
        return a2 + frac * (a1 - a2);
    }

    public static float CalcFov(float fov_x, float width, float height) {
        if (fov_x < 1.0f || fov_x > 179.0f)
            Com.Error(Defines.ERR_DROP, "Bad fov: " + fov_x);

        return (float) (Math.atan(height / (width / Math.tan(fov_x * piratio)))) / piratio;
    }
}