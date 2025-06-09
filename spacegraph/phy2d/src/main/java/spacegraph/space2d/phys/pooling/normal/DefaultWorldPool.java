/**
 * Copyright (c) 2013, Daniel Murphy
 * All rights reserved.
 * <p>
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * <p>
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
 * <p>
 * Created at 3:26:14 AM Jan 11, 2011
 */
/**
 * Created at 3:26:14 AM Jan 11, 2011
 */
package spacegraph.space2d.phys.pooling.normal;

import jcog.math.v2;
import jcog.math.v3;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;
import spacegraph.space2d.phys.collision.AABB;
import spacegraph.space2d.phys.collision.Collision;
import spacegraph.space2d.phys.collision.Distance;
import spacegraph.space2d.phys.collision.TimeOfImpact;
import spacegraph.space2d.phys.common.*;
import spacegraph.space2d.phys.dynamics.contacts.*;
import spacegraph.space2d.phys.pooling.IDynamicStack;
import spacegraph.space2d.phys.pooling.IWorldPool;

import java.util.stream.IntStream;

/**
 * Provides object pooling for all objects used in the engine. Objects retrieved from here should
 * only be used temporarily, and then pushed back (with the exception of arrays).
 *
 * @author Daniel Murphy
 */
public class DefaultWorldPool implements IWorldPool {

    private final OrderedStack<v2> vecs;
    private final OrderedStack<v3> vec3s;
    private final OrderedStack<Mat22> mats;
    private final OrderedStack<Mat33> mat33s;
    private final OrderedStack<AABB> aabbs;
    private final OrderedStack<Rot> rots;

    private final IntObjectHashMap<float[]> afloats = new IntObjectHashMap<>();
    private final IntObjectHashMap<int[]> aints = new IntObjectHashMap<>();
    private final IntObjectHashMap<v2[]> avecs = new IntObjectHashMap<>();

    private final IWorldPool world = this;

    private final IDynamicStack<Contact> pcstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new PolygonContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new PolygonContact[size];
                }
            };

    private final IDynamicStack<Contact> ccstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new CircleContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new CircleContact[size];
                }
            };

    private final IDynamicStack<Contact> cpstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new PolygonAndCircleContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new PolygonAndCircleContact[size];
                }
            };

    private final IDynamicStack<Contact> ecstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new EdgeAndCircleContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new EdgeAndCircleContact[size];
                }
            };

    private final IDynamicStack<Contact> epstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new EdgeAndPolygonContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new EdgeAndPolygonContact[size];
                }
            };

    private final IDynamicStack<Contact> chcstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new ChainAndCircleContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new ChainAndCircleContact[size];
                }
            };

    private final IDynamicStack<Contact> chpstack =
            new MutableStack<>(Settings.CONTACT_STACK_INIT_SIZE) {
                protected Contact newInstance() {
                    return new ChainAndPolygonContact(world);
                }

                protected Contact[] newArray(int size) {
                    return new ChainAndPolygonContact[size];
                }
            };

    private final Collision collision;
    private final TimeOfImpact toi;
    private final Distance dist;

    public DefaultWorldPool(int argSize, int argContainerSize) {
        vecs = new OrderedStack<>(argSize, argContainerSize) {
            protected v2 newInstance() {
                return new v2();
            }
        };
        vec3s = new OrderedStack<>(argSize, argContainerSize) {
            protected v3 newInstance() {
                return new Vec3();
            }
        };
        mats = new OrderedStack<>(argSize, argContainerSize) {
            protected Mat22 newInstance() {
                return new Mat22();
            }
        };
        aabbs = new OrderedStack<>(argSize, argContainerSize) {
            protected AABB newInstance() {
                return new AABB();
            }
        };
        rots = new OrderedStack<>(argSize, argContainerSize) {
            protected Rot newInstance() {
                return new Rot();
            }
        };
        mat33s = new OrderedStack<>(argSize, argContainerSize) {
            protected Mat33 newInstance() {
                return new Mat33();
            }
        };

        dist = new Distance();
        collision = new Collision(this);
        toi = new TimeOfImpact(this);
    }

    public final IDynamicStack<Contact> getPolyContactStack() {
        return pcstack;
    }

    public final IDynamicStack<Contact> getCircleContactStack() {
        return ccstack;
    }

    public final IDynamicStack<Contact> getPolyCircleContactStack() {
        return cpstack;
    }

    @Override
    public IDynamicStack<Contact> getEdgeCircleContactStack() {
        return ecstack;
    }

    @Override
    public IDynamicStack<Contact> getEdgePolyContactStack() {
        return epstack;
    }

    @Override
    public IDynamicStack<Contact> getChainCircleContactStack() {
        return chcstack;
    }

    @Override
    public IDynamicStack<Contact> getChainPolyContactStack() {
        return chpstack;
    }

    public final v2 popVec2() {
        return vecs.pop();
    }

    public final v2[] popVec2(int argNum) {
        return vecs.pop(argNum);
    }

    public final void pushVec2(int argNum) {
        vecs.push(argNum);
    }

    public final v3 popVec3() {
        return vec3s.pop();
    }

    public final v3[] popVec3(int argNum) {
        return vec3s.pop(argNum);
    }

    public final void pushVec3(int argNum) {
        vec3s.push(argNum);
    }

    public final Mat22 popMat22() {
        return mats.pop();
    }

    public final Mat22[] popMat22(int argNum) {
        return mats.pop(argNum);
    }

    public final void pushMat22(int argNum) {
        mats.push(argNum);
    }

    public final Mat33 popMat33() {
        return mat33s.pop();
    }

    public final void pushMat33(int argNum) {
        mat33s.push(argNum);
    }

    public final AABB popAABB() {
        return aabbs.pop();
    }

    public final AABB[] popAABB(int argNum) {
        return aabbs.pop(argNum);
    }

    public final void pushAABB(int argNum) {
        aabbs.push(argNum);
    }

    public final Rot popRot() {
        return rots.pop();
    }

    public final void pushRot(int num) {
        rots.push(num);
    }

    public final Collision getCollision() {
        return collision;
    }

    public final TimeOfImpact getTimeOfImpact() {
        return toi;
    }

    public final Distance getDistance() {
        return dist;
    }

    public final float[] getFloatArray(int argLength) {
        if (!afloats.containsKey(argLength)) {
            afloats.put(argLength, new float[argLength]);
        }

        assert (afloats.get(argLength).length == argLength) : "Array not built with correct length";
        return afloats.get(argLength);
    }

    public final int[] getIntArray(int argLength) {
        if (!aints.containsKey(argLength)) {
            aints.put(argLength, new int[argLength]);
        }

        assert (aints.get(argLength).length == argLength) : "Array not built with correct length";
        return aints.get(argLength);
    }

    public final v2[] getVec2Array(int argLength) {
        if (!avecs.containsKey(argLength)) {
            v2[] ray = IntStream.range(0, argLength).mapToObj(i -> new v2()).toArray(v2[]::new);
            avecs.put(argLength, ray);
        }

        assert (avecs.get(argLength).length == argLength) : "Array not built with correct length";
        return avecs.get(argLength);
    }
}
