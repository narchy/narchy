package nars.experiment;


import jcog.Util;
import jcog.data.list.Lst;
import jcog.random.XoRoShiRo128PlusRandom;
import nars.game.Game;
import spacegraph.space2d.phys.callbacks.ContactListener;
import spacegraph.space2d.phys.collision.shapes.EdgeShape;
import spacegraph.space2d.phys.collision.shapes.PolygonShape;
import spacegraph.space2d.phys.dynamics.Body2D;
import spacegraph.space2d.phys.dynamics.BodyDef;
import spacegraph.space2d.phys.dynamics.BodyType;
import spacegraph.space2d.phys.dynamics.Dynamics2D;
import spacegraph.space2d.phys.dynamics.contacts.Contact;

import java.util.Random;

/**
 * LuNAR Lander
 * adapted from https://raw.githubusercontent.com/openai/gym/master/gym/envs/box2d/lunar_lander.py
 * IN PROGRESS
 * <p>
 * Rocket trajectory optimization is a classic topic in Optimal Control.
 * <p>
 * According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
 * turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
 * <p>
 * The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
 * Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
 * If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
 * comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
 * Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
 * Solved is 200 points.
 * <p>
 * Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
 * on its first attempt. Please see the source code for details.
 * <p>
 * Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
 */
public class LuNAR extends Game {

    static final float[] LANDER_POLY = {
            -14, +17, -17, 0, -17, -10,
            +17, -10, +17, 0, +14, +17
    };
    static float FPS = 50;
    static float SCALE = 30.0f;   //affects how fast-paced the game is, forces should be adjusted as well
    static float MAIN_ENGINE_POWER = 13.0f;
    static float SIDE_ENGINE_POWER = 0.6f;
    static float INITIAL_RANDOM = 1000.0f;   //Set 1500 to make game harder
    static float LEG_AWAY = 20;
    static float LEG_DOWN = 18;
    static float LEG_W = 2, LEG_H = 8;
    static float LEG_SPRING_TORQUE = 40;

    static float SIDE_ENGINE_HEIGHT = 14.0f;
    static float SIDE_ENGINE_AWAY = 12.0f;

    static float VIEWPORT_W = 600;
    static float VIEWPORT_H = 400;
    final Dynamics2D world = new Dynamics2D();
    final Lst particles = new Lst();
    float W, H;
    private ContactDetector contactListener;
    private float[] height;
    private float[] chunk_x;
    private float helipad_x1;
    private float helipad_x2;
    private float helipad_y;
    private Body2D lander;

    public LuNAR(String id) {
        super(id);
    }

    @Override
    protected void init() {
        super.init();

//
//        //useful range is -1 .. +1, but spikes can be higher
//    this.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
//
//            if this.continuous:
//            //Action is two floats [main engine, left-right engines].
//            //Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
//            //Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
//    this.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
//            else:
//            //Nop, fire left engine, main engine, right engine
//    this.action_space = spaces.Discrete(4)
//
//            this.reset()

        W = VIEWPORT_W / SCALE;
        H = VIEWPORT_H / SCALE;

        this.contactListener = new ContactDetector();

//        //terrain
        int CHUNKS = 11;
        Random rng = new XoRoShiRo128PlusRandom(1);
        this.height = Util.arrayOf(i -> rng.nextFloat() * H / 2, new float[CHUNKS]); //    height = this.np_random.uniform(0, H/2, size=(CHUNKS+1,))
        this.chunk_x = Util.arrayOf(i -> W / (CHUNKS - 1) * i, new float[CHUNKS]);
        this.helipad_x1 = chunk_x[CHUNKS / 2 - 1];
        this.helipad_x2 = chunk_x[CHUNKS / 2 + 1];
        this.helipad_y = H / 4;
        height[CHUNKS / 2 - 2] = this.helipad_y;
        height[CHUNKS / 2 - 1] = this.helipad_y;
        height[CHUNKS / 2 + 0] = this.helipad_y;
        height[CHUNKS / 2 + 1] = this.helipad_y;
        height[CHUNKS / 2 + 2] = this.helipad_y;
        float[] smooth_y = Util.arrayOf(i -> 0.33f * (height[i - 1] + height[i + 0] + height[i + 1]), new float[CHUNKS]);

        BodyDef moonDef = new BodyDef(BodyType.STATIC);
        Body2D moon = world.addBody(moonDef);
        moon.addFixture(new EdgeShape(0, 0, W, 0), 0);
//
//    this.sky_polys = []
//            for i in range(CHUNKS-1):
//    p1 = (chunk_x[i], smooth_y[i])
//    p2 = (chunk_x[i+1], smooth_y[i+1])
//            this.moon.CreateEdgeFixture(
//    vertices=[p1,p2],
//    density=0,
//    friction=0.1)
//            this.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
//
//    this.moon.color1 = (0.0, 0.0, 0.0)
//    this.moon.color2 = (0.0, 0.0, 0.0)
//
        float initial_y = VIEWPORT_H / SCALE;
        float[] p = LANDER_POLY.clone();
        Util.mul(p, 1f / SCALE);
        this.lander = this.world.newDynamicBody(new PolygonShape(p), 5, 0.1f);
        lander.pos.set(VIEWPORT_W / SCALE / 2, initial_y);
        lander.angle(0);

//    categoryBits=0x0010,
//    maskBits=0x001,   //collide only with ground
//    restitution=0.0)  //0.99 bouncy
//                )

//    this.lander.color1 = (0.5, 0.4, 0.9)
//    this.lander.color2 = (0.3, 0.3, 0.5)

//            this.lander.ApplyForceToCenter( (
//            this.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
//            this.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
//            ), True)
//
//    this.legs = []
//            for i in [-1, +1]:
//    leg = this.world.CreateDynamicBody(
//    position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
//    angle=(i * 0.05),
//    fixtures=fixtureDef(
//            shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
//    density=1.0,
//    restitution=0.0,
//    categoryBits=0x0020,
//    maskBits=0x001)
//            )
//    leg.ground_contact = False
//    leg.color1 = (0.5, 0.4, 0.9)
//    leg.color2 = (0.3, 0.3, 0.5)
//    rjd = revoluteJointDef(
//            bodyA=this.lander,
//            bodyB=leg,
//            localAnchorA=(0, 0),
//    localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
//    enableMotor=True,
//    enableLimit=True,
//    maxMotorTorque=LEG_SPRING_TORQUE,
//    motorSpeed=+0.3 * i  //low enough not to jump back into the sky
//                )
//                        if i == -1:
//    rjd.lowerAngle = +0.9 - 0.5  //The most esoteric numbers here, angled legs have freedom to travel within
//    rjd.upperAngle = +0.9
//            else:
//    rjd.lowerAngle = -0.9
//    rjd.upperAngle = -0.9 + 0.5
//    leg.joint = this.world.CreateJoint(rjd)
//            this.legs.append(leg)
//
//    this.drawlist = [this.lander] + this.legs
//
//        return this.step(np.array([0, 0]) if this.continuous else 0)[0]

    }

    //
//    def _create_particle(self, mass, x, y, ttl):
//    p = this.world.CreateDynamicBody(
//    position = (x, y),
//    angle=0.0,
//    fixtures = fixtureDef(
//            shape=circleShape(radius=2/SCALE, pos=(0, 0)),
//    density=mass,
//    friction=0.1,
//    categoryBits=0x0100,
//    maskBits=0x001,  //collide only with ground
//    restitution=0.3)
//            )
//    p.ttl = ttl
//        this.particles.append(p)
//            this._clean_particles(False)
//            return p
//
//    def _clean_particles(self, all):
//            while this.particles and (all or this.particles[0].ttl < 0):
//            this.world.DestroyBody(this.particles.pop(0))
//
    void update() {
//    def step(self, action):
//            if this.continuous:
//    action = np.clip(action, -1, +1).astype(np.float32)
//        else:
//                assert this.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
//
//            //Engines
//    tip  = (math.sin(this.lander.angle), math.cos(this.lander.angle))
//    side = (-tip[1], tip[0])
//    dispersion = [this.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]
//
//    m_power = 0.0
//            if (this.continuous and action[0] > 0.0) or (not this.continuous and action == 2):
//            //Main engine
//            if this.continuous:
//    m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   //0.5..1.0
//            assert m_power >= 0.5 and m_power <= 1.0
//            else:
//    m_power = 1.0
//    ox = (tip[0] * (4/SCALE + 2 * dispersion[0]) +
//    side[0] * dispersion[1])  //4 is move a bit downwards, +-2 for randomness
//            oy = -tip[1] * (4/SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
//    impulse_pos = (this.lander.position[0] + ox, this.lander.position[1] + oy)
//    p = this._create_particle(3.5,  //3.5 is here to make particle speed adequate
//    impulse_pos[0],
//    impulse_pos[1],
//    m_power)  //particles are just a decoration
//            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
//    impulse_pos,
//    True)
//            this.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
//    impulse_pos,
//    True)
//
//    s_power = 0.0
//            if (this.continuous and np.abs(action[1]) > 0.5) or (not this.continuous and action in [1, 3]):
//            //Orientation engines
//            if this.continuous:
//    direction = np.sign(action[1])
//    s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
//            assert s_power >= 0.5 and s_power <= 1.0
//            else:
//    direction = action-2
//    s_power = 1.0
//    ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
//    oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
//    impulse_pos = (this.lander.position[0] + ox - tip[0] * 17/SCALE,
//    this.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
//    p = this._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
//            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
//    impulse_pos
//                                 , True)
//            this.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
//    impulse_pos,
//    True)
//
//            this.world.Step(1.0/FPS, 6*30, 2*30)
//
//    pos = this.lander.position
//            vel = this.lander.linearVelocity
//    state = [
//            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
//            (pos.y - (this.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
//    vel.x*(VIEWPORT_W/SCALE/2)/FPS,
//    vel.y*(VIEWPORT_H/SCALE/2)/FPS,
//    this.lander.angle,
//            20.0*this.lander.angularVelocity/FPS,
//            1.0 if this.legs[0].ground_contact else 0.0,
//            1.0 if this.legs[1].ground_contact else 0.0
//            ]
//            assert len(state) == 8
//
//    reward = 0
//    shaping = \
//            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
//            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
//            - 100*abs(state[4]) + 10*state[6] + 10*state[7]  //And ten points for legs contact, the idea is if you
//                                                             //lose contact again after landing, you get negative reward
//        if this.prev_shaping is not None:
//    reward = shaping - this.prev_shaping
//    this.prev_shaping = shaping
//
//    reward -= m_power*0.30  //less fuel spent is better, about -30 for heuristic landing
//    reward -= s_power*0.03
//
//    done = False
//        if this.game_over or abs(state[0]) >= 1.0:
//    done = True
//            reward = -100
//        if not this.lander.awake:
//    done = True
//            reward = +100
//        return np.array(state, dtype=np.float32), reward, done, {}
    }


//    def _destroy(self):
//            if not this.moon: return
//    this.world.contactListener = None
//        this._clean_particles(True)
//            this.world.DestroyBody(this.moon)
//    this.moon = None
//        this.world.DestroyBody(this.lander)
//    this.lander = None
//        this.world.DestroyBody(this.legs[0])
//            this.world.DestroyBody(this.legs[1])
//
//    def reset(self):
//            this._destroy()
//    this.game_over = False
//    this.prev_shaping = None
//

    void render() {
//
//    def render(self, mode='human'):
//    from gym.envs.classic_control import rendering
//        if this.viewer is None:
//    this.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
//            this.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
//
//            for obj in this.particles:
//    obj.ttl -= 0.15
//    obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
//    obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
//
//            this._clean_particles(False)
//
//            for p in this.sky_polys:
//            this.viewer.draw_polygon(p, color=(0, 0, 0))
//
//            for obj in this.particles + this.drawlist:
//            for f in obj.fixtures:
//    trans = f.body.transform
//                if type(f.shape) is circleShape:
//    t = rendering.Transform(translation=trans*f.shape.pos)
//            this.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
//                    this.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
//                else:
//    path = [trans*v for v in f.shape.vertices]
//            this.viewer.draw_polygon(path, color=obj.color1)
//            path.append(path[0])
//            this.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
//
//            for x in [this.helipad_x1, this.helipad_x2]:
//    flagy1 = this.helipad_y
//            flagy2 = flagy1 + 50/SCALE
//            this.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
//            this.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/SCALE), (x + 25/SCALE, flagy2 - 5/SCALE)],
//    color=(0.8, 0.8, 0))
//
//            return this.viewer.render(return_rgb_array=mode == 'rgb_array')
    }

    private static class ContactDetector implements ContactListener {

        @Override
        public boolean beginContact(Contact contact) {
            //            if this.env.lander == contact.fixtureA.body or this.env.lander == contact.fixtureB.body:
//    this.env.game_over = True
//        for i in range(2):
//            if this.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
//    this.env.legs[i].ground_contact = True
            return false;
        }

        @Override
        public void endContact(Contact contact) {
//            for i in range(2):
//            if this.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
//    this.env.legs[i].ground_contact = False
        }

    }
//
//    def close(self):
//            if this.viewer is not None:
//            this.viewer.close()
//    this.viewer = None
//
//
//    class LunarLanderContinuous(LunarLander):
//    continuous = True
//
//    def heuristic(env, s):
//            """
//    The heuristic for
//    1. Testing
//    2. Demonstration rollout.
//
//    Args:
//        env: The environment
//        s (list): The state. Attributes:
//                  s[0] is the horizontal coordinate
//                  s[1] is the vertical coordinate
//                  s[2] is the horizontal speed
//                  s[3] is the vertical speed
//                  s[4] is the angle
//                  s[5] is the angular speed
//                  s[6] 1 if first leg has contact, else 0
//                  s[7] 1 if second leg has contact, else 0
//    returns:
//         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
//    """
//
//    angle_targ = s[0]*0.5 + s[2]*1.0         //angle should point towards center
//    if angle_targ > 0.4: angle_targ = 0.4    //more than 0.4 radians (22 degrees) is bad
//    if angle_targ < -0.4: angle_targ = -0.4
//    hover_targ = 0.55*np.abs(s[0])           //target y should be proportional to horizontal offset
//
//    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5])*1.0
//    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
//
//            if s[6] or s[7]:  //legs have contact
//            angle_todo = 0
//    hover_todo = -(s[3])*0.5  //override to reduce fall speed, that's all we need after contact
//
//            if env.continuous:
//    a = np.array([hover_todo*20 - 1, -angle_todo*20])
//    a = np.clip(a, -1, +1)
//            else:
//    a = 0
//            if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
//    elif angle_todo < -0.05: a = 3
//    elif angle_todo > +0.05: a = 1
//            return a
//
//    def demo_heuristic_lander(env, seed=None, render=False):
//            env.seed(seed)
//    total_reward = 0
//    steps = 0
//    s = env.reset()
//            while True:
//    a = heuristic(env, s)
//    s, r, done, info = env.step(a)
//    total_reward += r
//
//        if render:
//    still_open = env.render()
//            if still_open == False: break
//
//            if steps % 20 == 0 or done:
//    print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
//    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
//    steps += 1
//            if done: break
//            return total_reward
//
//
//if __name__ == '__main__':
//    demo_heuristic_lander(LunarLander(), render=True)

}