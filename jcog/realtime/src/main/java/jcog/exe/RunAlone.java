package jcog.exe;

import jcog.Util;
import jcog.data.map.CustomConcurrentHashMap;

import java.util.concurrent.atomic.AtomicBoolean;

import static jcog.data.map.CustomConcurrentHashMap.*;

public class RunAlone {
    private static final CustomConcurrentHashMap<Object, AtomicBoolean> uniqueRuns =
            new CustomConcurrentHashMap<>(WEAK, IDENTITY, STRONG, IDENTITY, 128);

    public static boolean runAlone(Object key, Runnable r) {
        AtomicBoolean b = uniqueRuns.computeIfAbsent(key, z -> new AtomicBoolean());
        if (Util.enterAlone(b)) {
            Exe.run(() -> {
                try {
                    r.run();
                } finally {
                    Util.exitAlone(b);
                }
            });
            return true;
        }
        return false;
    }
}