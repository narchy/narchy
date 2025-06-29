/*
 *   __               .__       .__  ._____.           
 * _/  |_  _______  __|__| ____ |  | |__\_ |__   ______
 * \   __\/  _ \  \/  /  |/ ___\|  | |  || __ \ /  ___/
 *  |  | (  <_> >    <|  \  \___|  |_|  || \_\ \\___ \ 
 *  |__|  \____/__/\_ \__|\___  >____/__||___  /____  >
 *                   \/       \/             \/     \/ 
 *
 * Copyright (c) 2006-2011 Karsten Schmidt
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * http://creativecommons.org/licenses/LGPL/2.1/
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

package toxi.geom;

import toxi.math.MathUtils;

import java.util.List;
import java.util.stream.IntStream;

import static java.lang.Float.floatToIntBits;

public class Rect implements Shape2D {

    /**
     * Factory method, constructs a new rectangle from a center point and extent
     * vector.
     * 
     * @param center
     * @param extent
     * @return new rect
     */
    public static Rect fromCenterExtent(ReadonlyVec2D center, Vec2D extent) {
        return new Rect(center.sub(extent), center.add(extent));
    }

    /**
     * Factory method, computes & returns the bounding rect for the given list
     * of points.
     * 
     * @param points
     * @return bounding rect
     * @since 0021
     */
    public static Rect getBoundingRect(List<? extends Vec2D> points) {
        Vec2D first = points.get(0);
        Rect bounds = new Rect(first.x, first.y, 0, 0);
        for (int i = 1, num = points.size(); i < num; i++) {
            bounds.growToContainPoint(points.get(i));
        }
        return bounds;
    }

    public float x;
    public float y;
    public float width;
    public float height;

    /**
     * Constructs an empty rectangle at point 0,0 with no dimensions.
     */
    public Rect() {

    }

    /**
     * Constructs a new rectangle using a point and dimensions
     * 
     * @param x
     *            x of top left
     * @param y
     *            y of top left
     * @param width
     * @param height
     */
    public Rect(float x, float y, float width, float height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    /**
     * Constructs a new rectangle defined by the two points given.
     * 
     * @param p1
     * @param p2
     */
    public Rect(ReadonlyVec2D p1, ReadonlyVec2D p2) {
        Vec2D tl = Vec2D.min(p1, p2);
        Vec2D br = Vec2D.max(p1, p2);
        x = tl.x;
        y = tl.y;
        width = br.x - x;
        height = br.y - y;
    }

    /**
     * Checks if the given point is within the rectangle's bounds.
     * 
     * @param p
     *            point to check
     * @return true, if point is contained
     */
    @Override
    public boolean containsPoint(ReadonlyVec2D p) {
        float px = p.x();
        if (px < x || px > x + width)
            return false;

        float py = p.y();
        return (py >= y || py <= y + height);
    }

//    public boolean stretchPoint(ReadonlyVec2D p, float margin) {
//        boolean change = false;
//        float px = p.x();
//        float x1 = x;
//        float x2 = x1 + width;
//        if (px < x1)  { x1 = px - margin; change = true; }
//        if (px > x2) { x2 = px + margin; change = true; }
//
//        float py = p.y();
//        float y1 = y;
//        float y2 = y1 + height;
//        if (py < y1)  { y1 = py - margin; change = true; }
//        if (py > y2) { y2 = py + margin; change = true; }
//
//        if (change) {
//            x = x1;
//            y = y1;
//            width = x2-x1; height = y2-y1;
//            return true;
//        }
//        return false;
//    }

    /**
     * Creates a copy of this rectangle
     * 
     * @return new instance
     */
    public Rect copy() {
        return new Rect(x, y, width, height);
    }

    /**
     * Returns true if the Object o is of type Rect and all of the data members
     * of o are equal to the corresponding data members in this rectangle.
     * 
     * @param o
     *            the Object with which the comparison is made
     * @return true or false
     */
    @Override
    public boolean equals(Object o) {
        if (o instanceof Rect r) {
            return (x == r.x && y == r.y && width == r.width && height == r.height);
        }
        return false;
    }

    @Override
    public final float getArea() {
        return width * height;
    }

    /**
     * Computes the aspect ratio of the rect as width over height.
     * 
     * @return aspect ratio
     */
    public final float getAspect() {
        return width / height;
    }

    public float getBottom() {
        return y + height;
    }

    public Vec2D getBottomLeft() {
        return new Vec2D(x, y + height);
    }

    public final Vec2D getBottomRight() {
        return new Vec2D(x + width, y + height);
    }

    @Override
    public Circle getBoundingCircle() {
        return new Circle(getCentroid(),
                new Vec2D(width, height).magnitude() / 2);
    }

    /**
     * Only provided because the Rect class implements {@link Shape2D}. The
     * bounding rect of a rectangle is itself.
     * 
     * @return itself
     * 
     * @see Shape2D#getBounds()
     */
    @Override
    public Rect getBounds() {
        return this;
    }

    /**
     * Returns the centroid of the rectangle.
     * 
     * @return centroid vector
     */
    public final Vec2D getCentroid() {
        return new Vec2D(x + width * 0.5f, y + height * 0.5f);
    }

    @Override
    public final float getCircumference() {
        return 2 * width + 2 * height;
    }

    /**
     * Returns a vector containing the width and height of the rectangle.
     * 
     * @return dimension vector
     */
    public final Vec2D getDimensions() {
        return new Vec2D(width, height);
    }

    /**
     * Returns one of the rectangles edges as {@link Line2D}. The edge IDs are:
     * <ul>
     * <li>0 - top</li>
     * <li>1 - right</li>
     * <li>2 - bottom</li>
     * <li>3 - left</li>
     * </ul>
     * 
     * @param id
     *            edge ID
     * @return edge as Line2D
     */
    public Line2D getEdge(int id) {
        // top
// right
// bottom
// left
        Line2D edge = switch (id) {
            case 0 -> new Line2D(x, y, x + width, y);
            case 1 -> new Line2D(x + width, y, x + width, y + height);
            case 2 -> new Line2D(x, y + height, x + width, y + height);
            case 3 -> new Line2D(x, y, x, y + height);
            default -> throw new IllegalArgumentException("edge ID needs to be 0...3");
        };
        return edge;
    }

    @Override
    public List<Line2D> getEdges() {
        List<Line2D> edges = IntStream.range(0, 4).mapToObj(this::getEdge).toList();
        return edges;
    }

    public float getLeft() {
        return x;
    }

    /**
     * Computes the normalized position of the given point within this
     * rectangle, so that a point at the top-left corner becomes {0,0} and
     * bottom-right {1,1}. The original point is not modified. Together with
     * {@link #getUnmappedPointInRect(Vec2D)} this function can be used to map a
     * point from one rectangle to another.
     * 
     * @param p
     *            point to be mapped
     * @return mapped Vec2D
     */
    public Vec2D getMappedPointInRect(Vec2D p) {
        return new Vec2D((p.x - x) / width, (p.y - y) / height);
    }

    /**
     * Creates a random point within the rectangle.
     * 
     * @return Vec2D
     */
    @Override
    public Vec2D getRandomPoint() {
        return new Vec2D(MathUtils.random(x, x + width),
                MathUtils.random(y, y + height));
    }

    public float getRight() {
        return x + width;
    }

    public float getTop() {
        return y;
    }

    public final Vec2D getTopLeft() {
        return new Vec2D(x, y);
    }

    public Vec2D getTopRight() {
        return new Vec2D(x + width, y);
    }

    /**
     * Inverse operation of {@link #getMappedPointInRect(Vec2D)}. Given a
     * normalized point it computes the position within this rectangle, so that
     * a point at {0,0} becomes the top-left corner and {1,1} bottom-right. The
     * original point is not modified. Together with
     * {@link #getUnmappedPointInRect(Vec2D)} this function can be used to map a
     * point from one rectangle to another.
     *
     * @param p
     *            point to be mapped
     * @return mapped Vec2D
     */
    public Vec2D getUnmappedPointInRect(Vec2D p) {
        return new Vec2D(p.x * width + x, p.y * height + y);
    }

    public Rect growToContainPoint(ReadonlyVec2D p) {
        if (!containsPoint(p)) {
            if (p.x() < x) {
                width = getRight() - p.x();
                x = p.x();
            } else if (p.x() > getRight()) {
                width = p.x() - x;
            }
            if (p.y() < y) {
                height = getBottom() - p.y();
                y = p.y();
            } else if (p.y() > getBottom()) {
                height = p.y() - y;
            }
        }
        return this;
    }

    /**
     * Returns a hash code value based on the data values in this object. Two
     * different Rect objects with identical data values (i.e., Rect.equals
     * returns true) will return the same hash code value. Two objects with
     * different data members may return the same hash value, although this is
     * not likely.
     * 
     * @return the integer hash code value
     */
    @Override
    public int hashCode() {
        long bits = 1L;
        bits = 31L * bits + floatToIntBits(x);
        bits = 31L * bits + floatToIntBits(y);
        bits = 31L * bits + floatToIntBits(width);
        bits = 31L * bits + floatToIntBits(height);
        return (int) (bits ^ (bits >> 32));
    }

    /**
     * Creates a new rectangle by forming the intersection of this rectangle and
     * the given other rect. The resulting bounds will be the rectangle of the
     * overlay area or null if the rects do not intersect.
     * 
     * @param r
     *            intersection partner rect
     * @return new Rect or null
     */
    public final Rect intersectionRectWith(Rect r) {
        Rect isec = null;
        if (intersectsRect(r)) {
            float x1 = Math.max(x, r.x);
            float y1 = Math.max(y, r.y);
            float x2 = Math.min(getRight(), r.getRight());
            float y2 = Math.min(getBottom(), r.getBottom());
            isec = new Rect(x1, y1, x2 - x1, y2 - y1);
        }
        return isec;
    }

    public boolean intersectsCircle(Vec2D c, float r) {
        float s, d = 0;

        float thisX = this.x;
        float x2 = thisX + width;
        float thisY = this.y;
        float y2 = thisY + height;
        float cx = c.x;
        if (cx < thisX) {
            s = cx - thisX;
            d = s * s;
        } else if (cx > x2) {
            s = cx - x2;
            d += s * s;
        }
        float rr = r * r;
        if (d > rr)
            return false;

        float cy = c.y;
        if (cy < thisY) {
            s = cy - thisY;
            d += s * s;
        } else if (cy > y2) {
            s = cy - y2;
            d += s * s;
        }
        return d <= rr;
    }

    /**
     * Checks if the rectangle intersects with the given ray and if so computes
     * the first intersection point. The method takes a min/max distance
     * interval along the ray in which the intersection must occur.
     * 
     * @param ray
     *            intersection ray
     * @param minDist
     *            minimum distance
     * @param maxDist
     *            max distance
     * @return intersection point or null if no intersection in the given
     *         interval
     */
    public ReadonlyVec2D intersectsRay(Ray2D ray, float minDist, float maxDist) {
        Vec2D invDir = ray.getDirection().reciprocal();
        boolean signDirX = invDir.x < 0;
        boolean signDirY = invDir.y < 0;
        Vec2D min = getTopLeft();
        Vec2D max = getBottomRight();
        Vec2D bbox = signDirX ? max : min;
        float tmin = (bbox.x - ray.x) * invDir.x;
        bbox = signDirX ? min : max;
        float tmax = (bbox.x - ray.x) * invDir.x;
        bbox = signDirY ? max : min;
        float tymin = (bbox.y - ray.y) * invDir.y;
        bbox = signDirY ? min : max;
        float tymax = (bbox.y - ray.y) * invDir.y;
        if ((tmin > tymax) || (tymin > tmax)) {
            return null;
        }
        if (tymin > tmin) {
            tmin = tymin;
        }
        if (tymax < tmax) {
            tmax = tymax;
        }
        if ((tmin < maxDist) && (tmax > minDist)) {
            return ray.getPointAtDistance(tmin);
        }
        return null;
    }

    /**
     * Checks if this rectangle intersects/overlaps the given one.
     * 
     * @param r
     *            another rect
     * @return true, if intersecting
     */
    public boolean intersectsRect(Rect r) {
        return !(x > r.x + r.width || x + width < r.x || y > r.y + r.height || y
                + height < r.y);
    }

    public Rect scale(float s) {
        Vec2D c = getCentroid();
        width *= s;
        height *= s;
        x = c.x - width * 0.5f;
        y = c.y - height * 0.5f;
        return this;
    }

    /**
     * Sets new bounds for this rectangle.
     * 
     * @param x
     *            x of top left
     * @param y
     *            y of top right
     * @param w
     *            width
     * @param h
     *            height
     * @return itself
     */
    public final Rect set(float x, float y, float w, float h) {
        this.x = x;
        this.y = y;
        this.width = w;
        this.height = h;
        return this;
    }

    public final Rect set(Rect r) {
        x = r.x;
        y = r.y;
        width = r.width;
        height = r.height;
        return this;
    }

    public final Rect setDimension(Vec2D dim) {
        width = dim.x;
        height = dim.y;
        return this;
    }

    public final Rect setPosition(Vec2D pos) {
        x = pos.x;
        y = pos.y;
        return this;
    }

    /**
     * Adds corner vertices for rounded rectangles polygon construction.
     * 
     * @param poly
     * @param o
     * @param radius
     * @param theta
     * @param res
     */
    private static void toPolyArc(Polygon2D poly, Vec2D o, float radius, float theta,
                                  int res) {
        for (int i = 0; i <= res; i++) {
            poly.add(o.add(Vec2D.fromTheta(theta + i * MathUtils.HALF_PI / res)
                    .scaleSelf(radius)));
        }
    }

    /**
     * Creates a {@link Polygon2D} instance of the rect.
     * 
     * @return rect as polygon
     */
    @Override
    public Polygon2D toPolygon2D() {
        Polygon2D poly = new Polygon2D();
        poly.add(new Vec2D(x, y));
        poly.add(new Vec2D(x + width, y));
        poly.add(new Vec2D(x + width, y + height));
        poly.add(new Vec2D(x, y + height));
        return poly;
    }

    /**
     * Turns this rectangle into a rounded rectangle shaped {@link Polygon2D}
     * instance with the given corner radius. The number of corner vertices to
     * be used, can be specified as well.
     * 
     * @param radius
     *            corner radius
     * @param res
     *            number of vertices per corner
     * @return rounded rect as polygon
     */
    public Polygon2D toPolygon2D(float radius, int res) {
        Polygon2D poly = new Polygon2D();
        toPolyArc(poly, new Vec2D(x + width - radius, y + radius), radius,
                -MathUtils.HALF_PI, res);
        toPolyArc(poly, new Vec2D(x + width - radius, y + height - radius),
                radius, 0, res);
        toPolyArc(poly, new Vec2D(x + radius, y + height - radius), radius,
                MathUtils.HALF_PI, res);
        toPolyArc(poly, new Vec2D(x + radius, y + radius), radius,
                MathUtils.PI, res);
        return poly;
    }

    @Override
    public String toString() {
        return "rect: {x:" + x + ", y:" + y + ", width:" + width + ", height:"
                + height + '}';
    }

    public Rect translate(float dx, float dy) {
        x += dx;
        y += dy;
        return this;
    }

    public Rect translate(ReadonlyVec2D offset) {
        x += offset.x();
        y += offset.y();
        return this;
    }

    /**
     * @deprecated use {@link #unionRectWith(Rect)} instead. Also note, that
     *             {@link #unionRectWith(Rect)} does NOT modify the original
     *             Rect anymore, but produces a new Rect instance.
     * @param r
     * @return new Rect
     */
    @Deprecated
    public final Rect union(Rect r) {
        return unionRectWith(r);
    }

    /**
     * Creates a new rectangle by forming an union of this rectangle and the
     * given other Rect. The resulting bounds will be inclusive of both.
     * 
     * @param r
     * @return new Rect
     */
    public final Rect unionRectWith(Rect r) {
        float x1 = Math.min(x, r.x);
        float x2 = Math.max(x + width, r.x + r.width);
        float w = x2 - x1;
        float y1 = Math.min(y, r.y);
        float y2 = Math.max(y + height, r.y + r.height);
        float h = y2 - y1;
        return new Rect(x1, y1, w, h);
    }

}