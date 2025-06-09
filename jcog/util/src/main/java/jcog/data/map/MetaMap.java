package jcog.data.map;

import org.jetbrains.annotations.Nullable;

import java.lang.ref.Reference;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.util.function.Function;
import java.util.function.Supplier;

/** minimal metadata map interface */
public interface MetaMap {
    /** Map.get */
    @Nullable <X> X meta(String key);

    /** Map.put */
    <X> X meta(String key, Object value);

    /** Map.computeIfAbsent */
    <X> X meta(String key, Function<String, X> valueIfAbsent);

    <X> X meta(String key, Supplier<X> valueIfAbsent);

    private static <X> Reference<X> refSoftOrWeak(X v, boolean softOrWeak) {
        return softOrWeak ? new SoftReference<>(v) : new WeakReference<>(v);
    }

    @Nullable default <X> X metaWeak(String key, Supplier<X> ifAbsent) {
        return (X)meta(key, () -> refSoftOrWeakOrNull(ifAbsent.get(), false));
    }

    @Nullable default <X> X metaSoft(String key, Supplier<X> ifAbsent) {
        return (X)meta(key, () -> refSoftOrNull(ifAbsent));
    }

    @Nullable
    private static <X> Reference<X> refSoftOrNull(Supplier<X> V) {
        var v = V.get();
        return v != null ? new SoftReference<>(v) : null;
    }

    @Nullable
    private static <X> Reference<X> refSoftOrWeakOrNull(X v, boolean softOrWeak) {
        return v != null ? refSoftOrWeak(v, softOrWeak) : null;
    }

//    @Nullable private static <X> X unwrap(@Nullable Reference<X> r) {
//        return r == null ? null : r.get();
//    }
//    @Nullable default <X> X metaWeak(String key, boolean softOrWeak, Function<String,X> valueIfAbsent) {
//        Reference<X> r = meta(key, k ->
//            refSoftOrWeakOrNull(valueIfAbsent.apply(k), softOrWeak));
//        return r == null ? null : r.get();
//    }


}
