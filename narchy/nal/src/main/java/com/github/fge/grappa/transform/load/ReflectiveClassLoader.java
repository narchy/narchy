package com.github.fge.grappa.transform.load;

import com.github.fge.grappa.transform.ParserTransformException;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Wrapper to provide reflective access to a {@link ClassLoader}
 *
 * <p>Necessary in order to grant correct access to package private fields of
 * parser classes.</p>
 */
public final class ReflectiveClassLoader implements AutoCloseable
{
    private static final String CLASSLOADER = "java.lang.ClassLoader";
    private static final String FIND_LOADED_CLASS = "findLoadedClass";
    private static final String DEFINE_CLASS = "defineClass";

    private final ClassLoader loader;
    private final MethodHandle findClass;
    private final Method loadClass;

    /**
     * Constructor
     *
     * @param loader the classloader to use
     */
    public ReflectiveClassLoader(ClassLoader loader)
    {
        this.loader = Objects.requireNonNull(loader);

        try {
            Class<?> loaderClass = Class.forName(CLASSLOADER);
            loadClass = loaderClass.getDeclaredMethod(DEFINE_CLASS,
                    String.class, byte[].class, int.class, int.class);

            Method fcm = loaderClass.getDeclaredMethod(FIND_LOADED_CLASS, String.class);
            fcm.trySetAccessible();
            findClass = MethodHandles.privateLookupIn(loaderClass, MethodHandles.lookup())
                    .unreflect(fcm).bindTo(loader);


        } catch (ClassNotFoundException | NoSuchMethodException | IllegalAccessException e) {
            throw new ParserTransformException(
                "unable to find the needed methods", e);
        }

        ParserTransformException exception = null;

//        try {
//            //findClass.trySetAccessible();//setAccessible(true);
//
//        } catch (SecurityException e) {
//            exception.addSuppressed(e);
//        }

        try {
            loadClass.trySetAccessible();//setAccessible(true);
        } catch (SecurityException e) {
            if (exception == null)
                exception = new ParserTransformException(
                        "could not change the necessary access modifiers");
            exception.addSuppressed(e);
        }

        if (exception!=null)
            throw exception;
    }

    /**
     * Returns the class with the given name if it has already been loaded by
     * the given class loader
     *
     * <p>If the class has not been loaded yet, this method returns {@code
     * null}.</p>
     *
     * @param className the full name of the class to be loaded
     * @return the class instance, if found
     */
    @Nullable
    public Class<?> findClass(String className)
    {
//        Objects.requireNonNull(className);

        try {
            return (Class<?>) findClass.invokeWithArguments(className);
        } catch (Throwable e) {
            throw new ParserTransformException("unable to find class by name ("
                + className + ')', e);
        }
    }

    /**
     * Loads the class defined with the given name and bytecode using the given
     * class loader
     *
     * <p>Since package and class idendity includes the ClassLoader instance
     * used to load a class, we use reflection on the given class loader to
     * define generated classes.</p>
     *
     * <p>If we used our own class loader (in order to be able to access the
     * protected "defineClass" method), we would likely still be able to load
     * generated classes; however, they would not have access to package-private
     * classes and members of their super classes.</p>
     *
     * @param className the full name of the class to be loaded
     * @param bytecode the bytecode of the class to load
     * @return the class instance
     */
    public Class<?> loadClass(String className, byte[] bytecode)
    {
//        Objects.requireNonNull(className);
//        Objects.requireNonNull(bytecode);

        try {
            //TEMPORARY
            Files.write(Path.of("/tmp", className.substring(className.lastIndexOf('.')+1) + ".class"), bytecode);
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            return (Class<?>) loadClass.invoke(loader, className, bytecode, 0,
                bytecode.length);
        } catch (InvocationTargetException | IllegalAccessException e) {
            throw new ParserTransformException("unable to load class by name",
                e);
        }
    }

    @Override
    public void close()
    {
        ParserTransformException exception = null;
//        try {
//            findClass.setAccessible(false);
//        } catch (SecurityException e) {
//            exception.addSuppressed(e);
//        }

//        try {
//            //loadClass.setAccessible(false);
//            loadClass.trySetAccessible();
//        } catch (SecurityException e) {
//            if (exception == null)
//                exception = new ParserTransformException(
//                    "could not close classloader properly");
//            exception.addSuppressed(e);
//        }

        if (exception!=null)
            throw exception;
    }
}
