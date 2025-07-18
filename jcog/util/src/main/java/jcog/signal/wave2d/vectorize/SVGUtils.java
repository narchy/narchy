package jcog.signal.wave2d.vectorize;

import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;


class SVGUtils {

    ////////////////////////////////////////////////////////////
    //
    //  SVG Drawing functions
    //
    ////////////////////////////////////////////////////////////

    private static float roundtodec(float val, float places) {
        return (float) (Math.round(val * Math.pow(10, places)) / Math.pow(10, places));
    }

    // Getting SVG path element string from a traced path
    private static void svgpathstring(
            StringBuilder sb,
            String desc,
            List<Double[]> segments,
            String colorstr,
            HashMap<String, Float> options) {
        float scale = options.get("scale"),
                lcpr = options.get("lcpr"),
                qcpr = options.get("qcpr"),
                roundcoords = (float) Math.floor(options.get("roundcoords"));
        // Path
        sb.append("<path ")
                .append(desc)
                .append(colorstr)
                .append("d=\"")
                .append("M ")
                .append(segments.get(0)[1] * scale)
                .append(" ")
                .append(segments.get(0)[2] * scale)
                .append(" ");

        if (roundcoords == -1) {
            for (Double[] segment : segments) {
                if (segment[0] == 1.0) {
                    sb.append("L ")
                            .append(segment[3] * scale)
                            .append(" ")
                            .append(segment[4] * scale)
                            .append(" ");
                } else {
                    sb.append("Q ")
                            .append(segment[3] * scale)
                            .append(" ")
                            .append(segment[4] * scale)
                            .append(" ")
                            .append(segment[5] * scale)
                            .append(" ")
                            .append(segment[6] * scale)
                            .append(" ");
                }
            }
        } else {
            for (Double[] segment : segments) {
                if (segment[0] == 1.0) {
                    sb.append("L ")
                            .append(roundtodec((float) (segment[3] * scale), roundcoords))
                            .append(" ")
                            .append(roundtodec((float) (segment[4] * scale), roundcoords))
                            .append(" ");
                } else {
                    sb.append("Q ")
                            .append(roundtodec((float) (segment[3] * scale), roundcoords))
                            .append(" ")
                            .append(roundtodec((float) (segment[4] * scale), roundcoords))
                            .append(" ")
                            .append(roundtodec((float) (segment[5] * scale), roundcoords))
                            .append(" ")
                            .append(roundtodec((float) (segment[6] * scale), roundcoords))
                            .append(" ");
                }
            }
        } // End of roundcoords check

        sb.append("Z\" />");

        // Rendering control points
        for (Double[] segment : segments) {
            if ((lcpr > 0) && (segment[0] == 1.0)) {
                sb.append("<circle cx=\"")
                        .append(segment[3] * scale)
                        .append("\" cy=\"")
                        .append(segment[4] * scale)
                        .append("\" r=\"")
                        .append(lcpr)
                        .append("\" fill=\"white\" stroke-width=\"")
                        .append(lcpr * 0.2)
                        .append("\" stroke=\"black\" />");
            }
            if ((qcpr > 0) && (segment[0] == 2.0)) {
                sb.append("<circle cx=\"")
                        .append(segment[3] * scale)
                        .append("\" cy=\"")
                        .append(segment[4] * scale)
                        .append("\" r=\"")
                        .append(qcpr)
                        .append("\" fill=\"cyan\" stroke-width=\"")
                        .append(qcpr * 0.2)
                        .append("\" stroke=\"black\" />");
                sb.append("<circle cx=\"")
                        .append(segment[5] * scale)
                        .append("\" cy=\"")
                        .append(segment[6] * scale)
                        .append("\" r=\"")
                        .append(qcpr)
                        .append("\" fill=\"white\" stroke-width=\"")
                        .append(qcpr * 0.2)
                        .append("\" stroke=\"black\" />");
                sb.append("<line x1=\"")
                        .append(segment[1] * scale)
                        .append("\" y1=\"")
                        .append(segment[2] * scale)
                        .append("\" x2=\"")
                        .append(segment[3] * scale)
                        .append("\" y2=\"")
                        .append(segment[4] * scale)
                        .append("\" stroke-width=\"")
                        .append(qcpr * 0.2)
                        .append("\" stroke=\"cyan\" />");
                sb.append("<line x1=\"")
                        .append(segment[3] * scale)
                        .append("\" y1=\"")
                        .append(segment[4] * scale)
                        .append("\" x2=\"")
                        .append(segment[5] * scale)
                        .append("\" y2=\"")
                        .append(segment[6] * scale)
                        .append("\" stroke-width=\"")
                        .append(qcpr * 0.2)
                        .append("\" stroke=\"cyan\" />");
            } // End of quadratic control points
        }
    } // End of svgpathstring()

    // Converting tracedata to an SVG string, paths are drawn according to a Z-index
    // the optional lcpr and qcpr are linear and quadratic control point radiuses
    public static String getsvgstring(ImageTracer.IndexedImage ii, HashMap<String, Float> options) {
        // SVG start
        int w = (int) (ii.width * options.get("scale")), h = (int) (ii.height * options.get("scale"));
        String viewboxorviewport =
                options.get("viewbox") != 0
                        ? "viewBox=\"0 0 " + w + " " + h + "\" "
                        : "width=\"" + w + "\" height=\"" + h + "\" ";
        StringBuilder svgstr =
                new StringBuilder(
                        "<svg " + viewboxorviewport + "version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" ");
        if (options.get("desc") != 0) {
            svgstr.append("desc=\"Created with ImageTracer.java\" ");
        }
        svgstr.append(">");

        // creating Z-index
        SortedMap<Double, Integer[]> zindex = new TreeMap<>();
        double label;
        // Layer loop
        for (int k = 0; k < ii.layers.size(); k++) {

            // Path loop
            int s = ii.layers.get(k).size();
            for (int pcnt = 0; pcnt < s; pcnt++) {

                // Label (Z-index key) is the startpoint of the path, linearized
                label = (ii.layers.get(k).get(pcnt).get(0)[2] * w) + ii.layers.get(k).get(pcnt).get(0)[1];
                // Creating new list if required
                if (!zindex.containsKey(label)) {
                    zindex.put(label, new Integer[2]);
                }
                // Adding layer and path number to list
                zindex.get(label)[0] = k;
                zindex.get(label)[1] = pcnt;
            } // End of path loop
        } // End of layer loop

        // Sorting Z-index is not required, TreeMap is sorted automatically

        // Drawing
        // Z-index loop
        String thisdesc;
        for (Entry<Double, Integer[]> entry : zindex.entrySet()) {
            if (options.get("desc") != 0) {
                thisdesc = "desc=\"l " + entry.getValue()[0] + " p " + entry.getValue()[1] + "\" ";
            } else {
                thisdesc = "";
            }
            svgpathstring(
                    svgstr,
                    thisdesc,
                    ii.layers.get(entry.getValue()[0]).get(entry.getValue()[1]),
                    tosvgcolorstr(ii.palette[entry.getValue()[0]]),
                    options);
        }

        // SVG End
        svgstr.append("</svg>");

        return svgstr.toString();
    } // End of getsvgstring()

    private static String tosvgcolorstr(byte[] c) {
        return "fill=\"rgb("
                + (c[0] + 128)
                + ","
                + (c[1] + 128)
                + ","
                + (c[2] + 128)
                + ")\" stroke=\"rgb("
                + (c[0] + 128)
                + ","
                + (c[1] + 128)
                + ","
                + (c[2] + 128)
                + ")\" stroke-width=\"1\" opacity=\""
                + ((c[3] + 128) / 255.0)
                + "\" ";
    }
}
