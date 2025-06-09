package nars.web;

import jcog.Str;
import jcog.event.Off;
import jcog.event.RunThese;
import jcog.exe.Exe;
import jcog.net.http.EvalSocket;
import jcog.net.http.HttpServer;
import jcog.net.http.WebSocketConnection;
import jcog.pri.bag.impl.PriArrayBag;
import jcog.pri.op.PriMerge;
import nars.NAR;
import nars.Task;
import org.java_websocket.WebSocket;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.jetbrains.annotations.Nullable;

import java.net.URI;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Supplier;

public abstract class NARWeb extends EvalSocket<NAR> {

    static final int DEFAULT_PORT = 60606;

    protected NARWeb(Supplier target) {
        super(target);
    }


    @Override
    public boolean wssConnect(WebSocketConnection conn) {
        NAR n = nar(conn, conn.url().getPath());
        if (n != null) {

            //logger.info("..
            //System.out.println("NAR " + System.identityHashCode(n) + " for " + url);
//
//            try {
//                n.input("x. |");
//                n.input("y. |");
//                n.input("z. |");
//            } catch (Narsese.NarseseException e) {
//                e.printStackTrace();
//            }
            conn.setAttachment(
                    new NARConnection(n,
                            n.main().eventTask.on(new WebSocketLogger(conn, n))
                            //...
                    )
            );

            return true;
        } else {
            return false;
        }
    }

    protected abstract @Nullable NAR nar(WebSocketConnection conn, String url);


//    @Override
//    public void wssMessage(WebSocket ws, String message) {
//        try {
//            NAR n = ((NARWeb.NARConnection) ws.getAttachment()).nar;
//            n.input(message);
////            System.out.println(n.loop + " " + n.loop.isRunning());
////            System.out.println(Iterables.toString(n.attn.active));
////            System.out.println(Iterators.toString(n.services().iterator()));
////            System.out.println(n.exe);
//
//        } catch (Narsese.NarseseException e) {
//            ws.send(e.toString()); //e.printStackTrace();
//        }
//    }

    static class NARConnection extends RunThese {
        public final NAR nar;

        NARConnection(NAR n, Off... ons) {
            this.nar = n;
            for (Off o : ons)
                add(o);
        }
    }

    protected void stopping(NAR n) {

    }

    protected void starting(NAR n) {

    }


    /**
     * Web Interface for 1 NAR
     */
    public static class Single extends NARWeb {

        private final NAR nar;

        public Single(NAR nar) {
            super(()->nar);
            this.nar = nar;
        }

        public static void main(String[] args) {

//            ClientBuilder.rebuildAsync(NARWebClient.class, false);

            int port = args.length > 0 ? Str.i(args[0]) : DEFAULT_PORT;

            NAR nar = null;//NARchy.core(1);
            //nar.log();

            HttpServer h = new HttpServer(port, new Single(nar));
            h.fps(10f);

            nar.startFPS(1f);
            nar.cpuThrottle.set(0.01f);
        }

        @Override
        protected @Nullable NAR nar(WebSocketConnection conn, String url) {
            return "/".equals(url) ? nar : null;
        }
    }

//    /**
//     * Shared Multi-NAR Server
//     * TODO
//     */
//    public static class Multi extends NARWeb {
//        private final NAR nar;
//        /**
//         * adapter
//         */
//        private final Memory sharedIndex;
//
//        public Multi() {
//            super(null/*TODO*/);
//            this.nar = NARchy.core();
//            //this.nar.loop.setFPS(10);
//            this.sharedIndex = new ProxyMemory(nar.memory);
//        }
//
//        @Override
//        protected NAR nar(WebSocketConnection conn, String url) {
//            if ("/".equals(url)) {
//                return null;
//            }
//
//            return reasoners.computeIfAbsent(url, (Function<String, NAR>) this::nar);
//        }
//
//        //TODO <URI,NAR>
//        final CustomConcurrentHashMap<String, NAR> reasoners = new CustomConcurrentHashMap<>(
//                STRONG, EQUALS, WEAK, IDENTITY, 64) {
//            @Override
//            protected void reclaim(NAR n) {
//                Multi.this.remove(n);
//            }
//        };
//
//        private void remove(NAR n) {
//            stopping(n);
//            n.reset();
//        }
//
//        /**
//         * create a NAR
//         */
//        private NAR nar(String path) {
//            Exec exe = nar.exe;
//            Exec sharedExec = new UniExec(8) {
//
////                @Override
////                public boolean concurrent() {
////                    return false;
////                }
//
//                @Override
//                public void execute(Runnable r) {
//                    exe.execute(r);
//                }
//
//            };
//
//            NAR n = new NARS().withNAL(1, 8)
//                    .time(new RealTime.MS()).exe(sharedExec).index(sharedIndex).get();
//
//
//            assert (path.charAt(0) == '/');
//            path = path.substring(1);
//
//
//            n.log(); //temporary
//
//            int initialFPS = 5;
//            n.startFPS(initialFPS);
//
//            starting(n);
//
//            return n;
//        }
//
//    }
//
////        Util.sleep(100);
////
////        WebClient c1 = new WebClient(URI.create("ws://localhost:60606/a"));
////        WebClient c2 = new WebClient(URI.create("ws://localhost:60606/b"));
////
////        Util.sleep(500);
////        c1.closeBlocking();
////        c2.closeBlocking();


    /**
     * client access for use in java
     */
    public static class WebClient extends WebSocketClient {

        public WebClient(URI serverUri) {
            super(serverUri);
            try {
                connectBlocking();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

        }

        @Override
        public void onOpen(ServerHandshake handshakedata) {

        }

        @Override
        public void onMessage(String message) {
            System.out.println(message);
        }

        @Override
        public void onClose(int code, String reason, boolean remote) {

        }

        @Override
        public void onError(Exception ex) {

        }
    }

    static class WebSocketLogger implements Consumer<Task> {

        volatile WebSocket w;
        final PriArrayBag<Task> out = new PriArrayBag<>(PriMerge.max, 4);
        final AtomicBoolean busy = new AtomicBoolean();
        WebSocketLogger(WebSocket ws, NAR n) {
            this.w = ws;

        }

        @Override
        public void accept(Task t) {
            if (out.put(t)!=null) {
                if (busy.compareAndSet(false, true)) {
                    Exe.run(this::drain);
                }
            }
        }


        protected void drain() {
            if (w.isOpen()) {
                busy.set(false);

//                final StringBuilder buf = new StringBuilder(2*1024);
//
//                buf.append('[');
//                out.clear(t -> buf.append('\"').append(t.toString(true)).append("\",")); //tmp
//                if (buf.length() > 0) {
//                    buf.setLength(buf.length() - 1);
//                }
//                buf.append(']');
//
//                String s = buf.toString();
//                w.send(s);

//                ArrayNode a = Serials.cborMapper.createArrayNode();
//                out.clear(t -> taskify(t, a.addArray()));
//                try {
//                    w.send(Serials.jsonMapper.writeValueAsString(a));
//                } catch (JsonProcessingException e) {
//                    e.printStackTrace();
//                }



//                out.clear(t -> {
//                    taskify(t, a.addArray());
//                });
//
//                if (a.size() > 0) {
//                    try {
//                        w.send(cborMapper.writeValueAsBytes(a));
//                    } catch (JsonProcessingException e) {
//                        //logger.error("")
//                        e.printStackTrace();
//                    }
//                }

            } else {
                //closed, dont un-busy
            }
        }



    }


}