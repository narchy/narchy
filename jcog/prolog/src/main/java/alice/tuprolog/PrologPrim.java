/*
 * tuProlog - Copyright (C) 2001-2002  aliCE team at deis.unibo.it
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
package alice.tuprolog;

import jcog.Util;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;


/**
 * Primitive class
 * referring to a builtin predicate or functor
 *
 * @see Struct
 */
public class PrologPrim {

    public static final int DIRECTIVE = 0;
    public static final int PREDICATE = 1;
    public static final int FUNCTOR = 2;
    private static final MethodHandles.Lookup LOOKUP = MethodHandles.lookup();

    public final int type;
    /**
     * lib object where the builtin is defined
     */
    public final PrologLib source;
    public final int arity;
    /**
     * for optimization purposes
     */

    public final String key;

//    private final Method method;
    private final MethodHandle mh;


    public PrologPrim(int type, String key, PrologLib lib, Method m, int arity) throws NoSuchMethodException {
        if (m == null)
            throw new NoSuchMethodException();

        this.type = type;
        this.key = key;
        this.source = lib;
        this.arity = arity;
        try {
            m.setAccessible(true);
            MethodHandle L = LOOKUP.unreflect(m);
            mh = (Modifier.isStatic(m.getModifiers()) ?
                        L : L.bindTo(source))
                    .asSpreader(Object[].class, m.getParameterCount());
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * evaluates the primitive as a directive
     *
     * @throws InvocationTargetException
     * @throws IllegalAccessException
     * @throws Exception                 if invocation directive failure
     */
    void evalAsDirective(Struct g) throws Throwable {
        try {
            mh.invokeExact(argsArray(g));
        } catch (RuntimeException throwable) {
            throw throwable.getCause();
        }
    }

    private Object[] argsArray(Struct g) {
        return Util.arrayOf(
            //g::subResolve
            g::sub
        , new Object[arity]);
    }

    /**
     * evaluates the primitive as a predicate
     *
     * @throws Exception if invocation primitive failure
     */
    boolean evalAsPredicate(Struct g) throws Throwable {
        return (boolean) mh.invokeExact(argsArray(g));
    }


    /**
     * evaluates the primitive as a functor
     *
     * @throws Throwable
     */
    Term evalAsFunctor(Struct g) throws Throwable {
        return (Term) mh.invokeExact(argsArray(g));
    }

//
//    public String toString() {
//        return "[ primitive: method " + method.getName() + " - "
//                + source.getClass().getName() + " ]\n";
//    }

}