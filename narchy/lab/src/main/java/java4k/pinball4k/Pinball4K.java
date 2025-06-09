package java4k.pinball4k;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferStrategy;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.util.Base64;

/**
 * table width / ball diameter = 20
 * <p>
 * 4238 - no mouse, optimized score and background
 * 4238 - 4189 OPTIMIZE_YIELD
 * 4189 - 4093 ignore beziers
 * 4093 - 4169 No score, background or yield optimizations
 * 4093 - 4160 Pretesselate beziers
 * 4160 - 4141 reduced pre tesselation to 8
 * 4141 - 4132 reduced pre tesselation to 7
 * 4132 - 4118 Skip reading beziers
 * 4118 - 4124 Changed background color from black to blue
 * 4124 - 4115 Changed game over text from: "Press Enter To Play Again" to "Enter to Play"
 * 4115 - 4105 presort lines and sircles according to state
 * 4105 - 4079 removed thick lines
 * 4079 - 4087 tilt fix
 * 4087 - 4085 made flipper speed faster (from 75 to 50)
 * 4085 - 4078 changed bounce values and redues short bezier to 6
 * 4078 - 4027 line strips
 * 4027 - 4086 turned off score and line width optimization
 * 4086 - 4092 made left bumper wall bezier and fixed right flipper position
 * 4092 - 4072 switched fillArc with fillOval
 * 4072 - 4050 optimize shadow
 * 4050 - 4075 thick lines and different color on bumpers
 * 4075 - 4070 changed bounce from a byte to a flag in the file
 * 4071 - 4054 replaced all readByte() with readUnsignedByte()
 * 4054 - 4047 hardcoded flipperLength
 * 4047 - 3922 inlined getGroup(int) and setGroup(...)
 * 3922 - 3908 inlined text rendering
 * 3908 - 3915 hardcoded text centering on static text
 * 3908 - 3876 freeze flippers when tilting + some tilt changes
 * 3876 - 3854 rewrote level format
 * 3854 - 3844 changes to the get and set group code
 * 3844 - 3830 removed setResisable(false)
 * 3830 - 3823 rearanged variables in constructor
 * 3823 - 3811 changed multiplier to 1, 2, 3, 4, 5, 6, 7, 8 + collision optimization
 * 3811 - 3806 blinking bumpers optimization
 * 3806 - 3821 Added rollover group in the bumper lane
 * 3821 - 3920 group colors
 * 3920 - 3972 replaced one bumper with a dropdown group
 * 3972 - 4054 added arrows (kzip not smallest?)
 * 4054 - 4099 lots of changes
 * 4099 - 4105 Fixed score on hidden drop target bug
 * 4105 - 4091 main target handler
 * 4091 - 4084 misc
 * 4084 - 4182 added outer lanes kickbacks
 * 4182 - 4195 use frame.getHeight() to calculate levely
 * 4195 - 4188 optimize group colors
 * 4188 - 4097 no group colors
 * 4097 - 4128 generated group colors using pseudo random generator
 * 4128 - 4097 removed ball jitter
 * 4097 - 4109 Added random launch speed
 * 4109 - 4121 Fixed multiple push bug
 * 4121 - 4117 replaced BALL_RADIUS * 0.7 with constant 16 and added friction to y (1byte)
 * 4117 - 4124 Fixed bug where bonus text shown was not multiplied with multiplier
 * 4124 - 1107 replaced for with while in loading and objCountOff
 * 1107 - 4092 rewrote group to not use two dimensional array
 * 4092 - 4094 disable resize and removed some lines to get under limit
 * 4092 - 4084 optimized level
 * 4084 - 4089 switched to 16 iterations
 * 4089 - 4087 made text color white
 * 4087 - 4063 Changed grouping in level format + ?
 * 4063 - 4090 Added upper lane groups
 * 4090 - 4080 rewrote flipper normal calculation (although introduced possible div by zero bug) *
 * 4091 - 4138 extra ball
 * 4087 - 3981 don't center text
 * 3981 - 4015 center manually
 * 4015 - 4037 ???
 * 4037 - 3998 use frameIdx in push time checks
 * 3998 - 4023 animatede score
 * 4023 - 4034 rewrote blinking to only blink when rollover group is complete
 * 4034 - 4042 added lines. Removed g.dispose() and and keyCode.
 * 4042 - 3991 removed concatination of strings (StringBuilder), static k[]
 * 3991 - 3984 multiply score by 1000 at end
 * 3984 - 3974 replace Integer.toString() with String.valueOf()
 * 3974 - 3968 Saved objCount in level
 * 3968 - 3967 made length() private static final
 * 3967 - 4000 gradient background
 * 4000 - 4029 scrolling blinking background on bonus
 * 4029 - 4026 changed game over text to Game Over - Press Enter
 * 4026 - 3992 ???
 * 3992 - 3925 inline data file
 * 3925 - 3920 removed redundant setColor calls
 * 3922 - 3893 rewrote level format where x,y is written first then flags, score and objId
 * 3993 - 3888 use MULTIPLIER_COLOR as seed for group colors
 * 3888 - 3882 changed order of position and properties in level
 * 3879 - 3911 Added multiplier spheres to level (32 bytes, 4 bytes per sphere)
 * 3911 - 3906 switched from simple multiplier code
 * 3906 - 3903 changed sorting of sircles when saveing level
 * 3903 - 2897 optimized new multiplier count display code
 * 3897 - 3934 Added flashing
 * 3934 - 3944 Increase sphere size on when flashing
 * 3944 - 3896 Use 4KJO to obfuscate
 * 3896 - 3906 store group activation time and activation count
 * 3906 - 3902 removed a redundant setColor
 * 3922 - 3888 disabled USE_SHADED_BALL, OUTLINE_SIRCLES, FLASH_SIRCLE_SIZE, USE_GROUP_BONUS, DRAW_SHADOWS
 * 3888 - 4071 enable USE_SHADED_BALL, OUTLINE_SIRCLES, FLASH_SIRCLE_SIZE, USE_GROUP_BONUS, DRAW_SHADOWS
 * 3883 - 3837 desiabled score and gradient
 * 3986 - 3966 store bonus time in groupData
 * <p>
 * 23.12.2007:
 * 4360 - enable all
 * 4193 - disable text
 * 4185 - inline isBumper
 * 4184 - inline ballShadowIdx
 * 4168 - optimisations, blink bumpers by making them darker
 * 4162 - rewrote draw while to for loop
 * 4158 - changed a |= to =
 * 4156
 * 4137 - darken by shifting and masking instead of useing Color.darken()
 * 4135 - pushTime optimisation
 * 4133 - collision nDot check against 0 instead of -1
 * 4134 - fix megabonus scoring
 * 4132 - commented out infotext
 * 4129 - changed a != to = when reseting gate
 * 4127 - inlined radius calculation
 * 4124 - collision normal calculation
 * 4123 - changed group magic number from 0xff to 0x7f
 * 4120 - blink target arrows for 30 seconds instead if 15
 * 4096 - optimised bumper shadow code
 * 4114 - no bounce or score after tilt
 * 4091 - don't store bounce in floatData
 * 4088 - changed FLIPPER_XXX constants
 * 4075 - made bonusx and bonusy ints
 * 4073 - made bonus vars part of intData
 * 4072 - reordered ID_XXX constants
 * 4063 - removed all + ID_FLAGS, since it is 0
 * 4056 - removed FD_FLIPPER_ANGLE_VEL since it is 0
 * 4058 - fixed bug caused by making bonus vars part of intData
 * 4054 - hardcoded som behaviourObjMaps
 * 4053 - do not subtract 25 from bonus y
 * <p>
 * 4106
 * <p>
 * bullseye text - 31 bytes
 * <p>
 * sound - 500 bytes
 * multiball - 300 bytes
 * megabonus -15 bytes
 * <p>
 * (3992 - 3896 inline level file as attribute and run it threw 4KJO
 * (4085 - 4086 changed switches to ifs)
 * (3854 - 3871 runs physics at 16 times rendering speed instead of 2)
 * (3922 - 3447 removed all text rendering)
 * (3876 - 3872 changed background fill from fillRect to fillOval, but is too slow)
 * <p>
 * 3842 - 3963 added descriptive text
 * 3963 - 4319 added bonuses
 * <p>
 * <p>
 * Rules
 * -----
 * Bumpers and slingshots gives 1000 points. Bumpers and rollovers 4000 points.
 * <p>
 * 5000 bonus points for completeing a dropdown or rollover group.
 * <p>
 * Completeing the kicker dropdowns will light the coresponding kicker.
 * Triggering a kicker will kick the ball back and award 15000 bonus points.
 * <p>
 * Completeing one of the three upper dropdown groups will ligth a bullseye
 * arrow. The bullseye bonus depends on the number of arrow lit:
 * 0:  25 000
 * 1:  50 000
 * 2: 100 000
 * 3: 200 000
 * <p>
 * Completeing the upper rollovers will increase the multiplier. All bonuses
 * are multiplied by the multiplier.
 * <p>
 * The inlane rollovers will lite the left lane arrow for 15 seconds. The left
 * lane awards 10000 bonus point when its arrow is lit.
 * <p>
 * The bumper rolloverswill lite the right lane arrow for 15 seconds. The right
 * lane awards 10000 bonus points when its arrow is lit.
 * <p>
 * Hitting the bumper rollovers 3 times will enable the bumper bonus for 30
 * seconds. Hitting a bumper when the bonus is on awards 5000 bonus points.
 * <p>
 * Hitting the left lane 3 time will enable the rollover bonus for 30 seconds.
 * Hitting a rollover when the bonus is one awards 10000 bonus points.
 * <p>
 * Hitting the right lane 3 time will enable the dropdown bonus for 30 seconds.
 * Hitting a dropdown when the bonus is one awards 10000 bonus points.
 * <p>
 * Extraball is awarded when reaching the following scores:
 * 1 000 000
 * 2 000 000
 * 4 000 000
 * 8 000 000
 * 16 000 000
 * 32 000 000
 * 64 000 000
 * etc
 * <p>
 * Extraballs do not stack.
 *
 * @author tombr
 */
public class Pinball4K extends JFrame {

    private static final boolean USE_ANIMATED_SCORE = true;
    private static final boolean USE_THICK_LINES = true;
    private static final boolean SHOW_BONUS_TEXT = true;
    private static final boolean USE_BLINK = true;
    private static final boolean DISABLE_RESIZE = false;
    private static final boolean USE_GET_HEIGHT = true;
    private static final boolean USE_GETES = true;
    private static final boolean USE_EXTRABALL = true;
    private static final boolean BACKGROUND_GRADIENT = true;
    private static final boolean USE_FLASH = true;
    private static final boolean USE_SHADED_BALL = true;
    private static final boolean OUTLINE_SIRCLES = true;
    private static final boolean FLASH_SIRCLE_SIZE = true;
    private static final boolean USE_GROUP_BONUS = true;
    private static final boolean DRAW_BALL_SHADOW = true;
    private static final boolean DRAW_BUMPER_SHADOWS = true;
    private static final boolean USE_EXTRABALL_TEXT = false;
    private static final boolean USE_BULLSEYE_TEXT = true;
    private static final boolean USE_MULTIPLIER_TEXT = true;

    private static final float ANGLE_SCALE = (2 * (float) Math.PI) / 127;

    private static final int FONT_ITALIC_BOLD = 3;
    private static final int LINE_COLOR = 0xaaaa66;
    private static final int BACKGROUND_COLOR = 0xff2f174f;
    private static final int MULTIPLIER_COLOR = 0x1f6faf;


    private static final int BUMPER_COLOR = 0x0a0d09;

    private static final int FLASH_FRAME_IDX = ((512 * 3 + 100) / 24);

    private static final int VK_LEFT = 37;
    private static final int VK_RIGHT = 39;
    private static final int VK_NEW_GAME = 10;
    private static final int VK_ESCAPE = 27;
    private static final int VK_TILT = 32;

    static final int BEHAVIOUR_GROUP1_ARROW = 1;
    static final int BEHAVIOUR_GROUP2_ARROW = 2;
    static final int BEHAVIOUR_GROUP3_ARROW = 3;
    private static final int BEHAVIOUR_LEFT_OUTER_LANE = 4;
    private static final int BEHAVIOUR_RIGHT_OUTER_LANE = 5;


    private static final int BEHAVIOUR_UPPER_LEFT = 8;
    private static final int BEHAVIOUR_UPPER_RIGHT = 9;
    private static final int BEHAVIOUR_START = 10;
    private static final int BEHAVIOUR_GAME_OVER = 11;
    private static final int BEHAVIOUR_BULLSEYE = 12;
    static final int BEHAVIOUR_BLINKERS = 13;
    static final int BEHAVIOUR_MULTIPLIER = 14;

    private static final int GROUP_MULTIPLIER = 0;
    static final int GROUP_DROP1 = 1;
    static final int GROUP_DROP2 = 2;
    static final int GROUP_DROP3 = 3;
    static final int GROUP_DROP4 = 4;
    static final int GROUP_DROP5 = 5;
    static final int GROUP_INLANE = 6;
    private static final int GROUP_BUMPER = 7;
    private static final int GROUP_UPPER_LEFT = 8;
    private static final int GROUP_UPPER_RIGHT = 9;

    private static final int VISIBLE_MASK = (1 << 0);
    private static final int COLLIDABLE_MASK = (1 << 1);
    private static final int ROLL_OVER_MASK = (1 << 2);
    private static final int DROP_DOWN_MASK = (1 << 3);
    private static final int GATE_MASK = (1 << 4);
    private static final int BUMPER_MASK = (1 << 5);


    private static final int FRAME_WIDTH = 1024;
    private static final int FRAME_HALF_WIDTH = FRAME_WIDTH / 2;
    private static final int FRAME_HEIGHT = 768;
    private static final int LEVEL_HEIGHT = 256 * 6 + 48;


    private static final int LOADING = 0;
    private static final int PLAYING = 1;
//    private static final int GAME_OVER = 2;


    private static final float PI = (float) Math.PI;
    private static final float BOUNCE_NORMAL = 1.5f;
    private static final float BOUNCE_BUMPER = 2.2f;


    private static final int BALL_RADIUS = 24;


    private static final int flipperLength = 134;


    private static final int MAX_SPEED = 3;
    private static final float GRAVITY = 0.00077f;
    private static final float FRICTION = 0.999985f;
    private static final float FLIPPER_SPEED = (PI * 2 / 400f);
    private static final int LAUNCH_SPEED = -2;
    private static final int LAUNCH_DIV = 512;
    private static final float BUMPER_ADD = 0.25f;
    private static final int PUSH_DIV_X = 4;
    private static final float PUSH_DIV_Y = 1.7f;
    private static final int ITERATIONS = 4;
    private static final int KICKER_VEL = -2;


    private static final int MAX_OBJ_COUNT = 0x10000;


    private static final int ID_SCORE = 1;
    private static final int ID_TYPE = 2;
    private static final int ID_BEHAVIOUR = 3;
    private static final int ID_X = 4;
    private static final int ID_Y = 5;
    private static final int ID_X2 = 6;
    private static final int ID_Y2 = 7;
    private static final int ID_COLLISION_TIME = 8;
    private static final int ID_IS_BALL_OVER = 9;
    private static final int ID_COLOR = 10;
    static final int ID_SPECIAL = 11;

    private static final int ID_DISPLAY_SCORE = 12;
    private static final int ID_BONUS_TIME = 13;
    private static final int ID_BONUS_X = 14;
    private static final int ID_BONUS_Y = 15;
    private static final int ID_BONUS_TEXT = 16;
    private static final int ID_BULLSEYE_TIME = 17;
    private static final int ID_EXTRABALL_TIME = 18;
    private static final int ID_MULTIPLIER_TIME = 19;
    static final int ID_INFO = 17;


    private static final int FD_FLIPPER_ANGLE = 1;
    static final int FD_FLIPPER_LENGTH = 2;
    private static final int FD_FLIPPER_MIN_ANGLE = 3;
    private static final int FD_FLIPPER_MAX_ANGLE = 4;
    private static final int ID_FLIPPER_ANGLE_DIR = 20;

    static final int GRP_COUNT = 0;
    private static final int GRP_FIRST_INDEX = 1;
    private static final int GRP_ACTIVATE_CNT = 22;
    private static final int GRP_ACTIVATE_FRAME_IDX = 23;
    private static final int GRP_BONUS_TIME = 24;


    private static final int LINE = 0;
    private static final int FLIPPER = 1;
    private static final int SIRCLE = 2;
    private static final int ARROW = 3;

    private static final int STRIDE = 0x20;

    private static final int BONUS_ROLLOVER = 5;
    private static final int BONUS_DROPDOWN = 5;
    private static final int BONUS_UPPER_LEFT = 25;
    private static final int BONUS_UPPER_RIGHT = 25;
    private static final int BONUS_KICKER = 15;
    private static final int BUMPER_TIME_BONUS = 5;
    private static final int DROPDOWN_TIME_BONUS = 10;
    private static final int ROLLOVER_TIME_BONUS = 10;

    private static final int BUMPER_ARROW_IDX = 32;
    private static final int INLINE_ARROW_IDX = 36;
    private static final int START_IDX = 5;

	private static final Color WHITE = new Color(0xffffff);
	private static final Color BLACK = new Color(0);

    private static final boolean[] k = new boolean[0x10000];

    /**
     * Constructor where the game loop is in.
     */
	private Pinball4K() throws Exception {
        super("Pinball 4K");


        setIgnoreRepaint(true);
        setSize(FRAME_WIDTH, FRAME_HEIGHT);


        if (DISABLE_RESIZE) {
            setResizable(false);
        }


        show();


        createBufferStrategy(2);
        BufferStrategy b = getBufferStrategy();

        int flashFrameIdx = 0;
        Font font = new Font("", FONT_ITALIC_BOLD, 32);
        int extraBallTarget = 0;
        int pushTime = 0;
        boolean wasTiltKeyPressed = false;
        boolean pushedBall = false;
        boolean pushed = false;
        boolean tilt = false;
        int state = LOADING;
        int levely = 0;
        int multiplier = 0;
        int score = 0;
        int frameIdx = 0;
        int objCount = 0;
        int objy = 0;
        int objx = 0;
        int[] blinkData = null;
        int[] behaviourObjMap = new int[MAX_OBJ_COUNT];
        int[] intData = new int[MAX_OBJ_COUNT];
        int numGroups = 0;
        int[] groupData = null;
        float[] floatData = new float[MAX_OBJ_COUNT];
        float flipperAngleVel = 0;
        int flipperUpDelta = 0;
        float flipperAngle = 0;
        float ballVelx = 0;
        float ballVely = 0;
        float ballx = 0;
        float bally = 0;
        long lastFrame = 0;
        do {

            int bonus = 0;


            for (int updateIdx = 0; updateIdx < ITERATIONS; updateIdx++) {


                if (state == LOADING) {

                    blinkData = new int[MAX_OBJ_COUNT];
                    groupData = new int[MAX_OBJ_COUNT];


                    //URL s = getClass().getResource("./a");

                    DataInputStream dataIn = new DataInputStream(new ByteArrayInputStream(Base64.getDecoder().decode(
					"fHwDIAUfRyMBACMBACMBAAEKBAEKBQAACgUZDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQCAAQC" +
						"AAQCAAQCAAQCCQQECAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAIABwAAAQAAAgAA" +
						"AwAABgMAAAMAAAMAAAMACwMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAABMAABMA" +
						"ABMAACMBACMBAAsEAAsEAAsEAAsEAAsAAAsAAAsAAAsEAAsEAAsEAAsEAAsEAAsEACaGYTqNRRjb" +
						"2Nv1+tU6eNB4wni0eKZ4mHiKeHx4bvNE7TPiJc8ZuBE5GicnIDgjShh4DoYYlCjMyMyzKZkggSPZ" +
						"b8c7v0e3UyJTUOug69WSlf96IJId/FD8UIhdPiugGNC9ILzAusC6M7vudK0UOxG9u0ja3a/QqyCr" +
						"E69oV3lcV1KqT7FEuDlKI0AsQDg8PDwUFBQUICAgICAgICAUFBQUFBQUFBQUFBQUFBQUFGxubm6M" +
						"cogANkwCKkACW/96J5Ik/P/6QFZOSyCgJ9DbINu9u8DUMLr8WLMLRRuo2jO71KzHqCmoHKx1W4Zg" +
						"ZFalV6xMs0FDKUA0QEALBClwJ3AgdR56IAQqiCSIHY0bkh0HK+7/7nTVksWm4K/g5pX/Hyz2M+8n" +
						"5B7XFccPtAueCIYHbgdZCUcNOBErFyEfGicWMRU9GUsrbBpvDXUGfASGBpANmBqeK6IrphCvEOZb" +
						"/wMtVlCIX4hdCC5FGzUkKy8pOy5JNlQ+Qz4rFS+mKrMivSmyMbs3n2OrZ7Ng0TPfOMJlwG3Ydd9u" +
						"7lPpP+Ev1CLCGa0UoBgBMKDrATFQ6wIzqNyo2gM0MNRI3EjaCwMdA0QDQQM+AjwCOgIbAxgEFAUP" +
						"CAc=")));


                    while (!(dataIn.readUnsignedByte() == 124 && dataIn.readUnsignedByte() == 124)) {
                    }
                    int flippers = dataIn.readUnsignedByte();
                    int sircles = dataIn.readUnsignedByte();
                    int arrows = dataIn.readUnsignedByte();
                    int lines = dataIn.readUnsignedByte();
                    objCount = dataIn.readUnsignedByte();


                    for (int i = 0; i < objCount; i++) {
                        intData[i * STRIDE] = dataIn.readUnsignedByte();
                        intData[i * STRIDE + ID_SCORE] = dataIn.readUnsignedByte();
                        intData[i * STRIDE + ID_BEHAVIOUR] = dataIn.readUnsignedByte();
                        intData[i * STRIDE + ID_COLOR] = LINE_COLOR;
                    }


                    for (int i = 0; i < objCount; i++) {

                        behaviourObjMap[intData[i * STRIDE + ID_BEHAVIOUR]] = i;
                        intData[i * STRIDE + ID_X] = dataIn.readUnsignedByte() * 4;
                        intData[i * STRIDE + ID_Y] = dataIn.readUnsignedByte() * 6;
                    }

                    int c = 0;


                    while (sircles-- > 0) {
                        intData[c + ID_TYPE] = SIRCLE;
                        intData[c + ID_X2] = dataIn.readUnsignedByte();
                        c += STRIDE;
                    }


                    while (arrows-- > 0) {
                        intData[c + ID_TYPE] = ARROW;
                        intData[c + ID_X2] = dataIn.readUnsignedByte();
                        c += STRIDE;
                    }


                    while (flippers-- > 0) {
                        intData[c + ID_TYPE] = FLIPPER;
                        floatData[c + FD_FLIPPER_MIN_ANGLE] = dataIn.readUnsignedByte() * ANGLE_SCALE;
                        floatData[c + FD_FLIPPER_MAX_ANGLE] = dataIn.readUnsignedByte() * ANGLE_SCALE;
                        intData[c + ID_FLIPPER_ANGLE_DIR] = (dataIn.readUnsignedByte() - 1);
                        c += STRIDE;
                    }


                    while (lines-- > 0) {
                        intData[c + ID_X2] = dataIn.readUnsignedByte() * 4;
                        intData[c + ID_Y2] = dataIn.readUnsignedByte() * 6;
                        c += STRIDE;
                    }


                    int strips = dataIn.readUnsignedByte();
                    while (strips-- > 0) {

                        int stripLines = dataIn.readUnsignedByte();


                        int prevIdx = dataIn.readUnsignedByte();

                        while (stripLines-- > 0) {

                            for (int i = 0; i < STRIDE; i++) {
                                intData[c + i] = intData[prevIdx * STRIDE + i];
                                floatData[c + i] = floatData[prevIdx * STRIDE + i];
                            }


                            intData[c + ID_X] = intData[c + ID_X2];
                            intData[c + ID_Y] = intData[c + ID_Y2];


                            intData[c + ID_X2] = dataIn.readUnsignedByte() * 4;
                            intData[c + ID_Y2] = dataIn.readUnsignedByte() * 6;


                            prevIdx = objCount++;
                            c += STRIDE;
                        }
                    }


                    numGroups = dataIn.readUnsignedByte();
                    for (int i = 0; i < numGroups * 2; i++) {
                        groupData[i] = dataIn.readUnsignedByte();
                    }


                    multiplier = 1;
                    score = 0;
                    state = PLAYING;
                    tilt = false;
                    extraBallTarget = 1000;
                    ballx = intData[behaviourObjMap[BEHAVIOUR_START] * STRIDE + ID_X];
                    bally = intData[behaviourObjMap[BEHAVIOUR_START] * STRIDE + ID_Y];
                } else if (state == PLAYING) {

                    pushed = !tilt && k[VK_TILT];
                    if (!wasTiltKeyPressed && pushed) {
                        if (frameIdx < pushTime) {
                            tilt = true;
                        }
                        pushTime = frameIdx + 31;
                        pushedBall = false;
                    }
                    wasTiltKeyPressed = pushed;


                    ballVelx = Math.min(MAX_SPEED, ballVelx * FRICTION);
                    ballVely = Math.min(MAX_SPEED, ballVely * FRICTION + GRAVITY);
                    ballVelx = Math.max(-MAX_SPEED, ballVelx);
                    ballVely = Math.max(-MAX_SPEED, ballVely);

                    ballx += ballVelx;
                    bally += ballVely;


                    float closestx = 0;
                    float closesty = 0;


                    boolean foundCollision = false;


                    int collisionObjIdx = 0;


                    float closestDistance = 0;


                    for (int objIdx = 0; objIdx < objCount; objIdx++) {
                        int objFlags = intData[objIdx * STRIDE];
                        int objBehaviour = intData[objIdx * STRIDE + ID_BEHAVIOUR];
                        objx = intData[objIdx * STRIDE + ID_X];
                        objy = intData[objIdx * STRIDE + ID_Y];
                        int linex2 = intData[objIdx * STRIDE + ID_X2];
                        int liney2 = intData[objIdx * STRIDE + ID_Y2];


                        float tempProjectedx = 0;
                        float tempProjectedy = 0;
                        float dist = 0;
                        boolean intersected = false;

                        switch (intData[objIdx * STRIDE + ID_TYPE]) {
                            case FLIPPER:

                                flipperUpDelta = intData[objIdx * STRIDE + ID_FLIPPER_ANGLE_DIR];
                                flipperAngle = floatData[objIdx * STRIDE + FD_FLIPPER_ANGLE];
                                flipperAngleVel = floatData[objIdx * STRIDE];
                                float newAngle = flipperAngle - (!tilt && k[flipperUpDelta < 0 ? VK_LEFT : VK_RIGHT]
                                        ? -flipperUpDelta
                                        : flipperUpDelta) * FLIPPER_SPEED;
                                newAngle = Math.max(floatData[objIdx * STRIDE + FD_FLIPPER_MIN_ANGLE],
                                        Math.min(floatData[objIdx * STRIDE + FD_FLIPPER_MAX_ANGLE],
                                                newAngle));
                                floatData[objIdx * STRIDE] = newAngle - flipperAngle;
                                linex2 = (int) (objx + Math.cos(newAngle) * flipperLength);
                                liney2 = (int) (objy + Math.sin(newAngle) * flipperLength);

                                intData[objIdx * STRIDE + ID_X2] = linex2;
                                intData[objIdx * STRIDE + ID_Y2] = liney2;
                                floatData[objIdx * STRIDE + FD_FLIPPER_ANGLE] = newAngle;


                            case LINE:

                                float rrr = (ballx - objx) * (linex2 - objx) + (bally - objy) * (liney2 - objy);
                                float len = length(linex2 - objx, liney2 - objy);
                                float t = rrr / len / len;
                                if (t >= 0 && t <= 1) {
                                    tempProjectedx = objx + (t * (linex2 - objx));
                                    tempProjectedy = objy + (t * (liney2 - objy));

                                    dist = length(ballx - tempProjectedx, bally - tempProjectedy);
                                    intersected = (dist <= BALL_RADIUS);
                                } else {

                                    dist = length(ballx - objx, bally - objy);
                                    float distance2 = length(ballx - linex2, bally - liney2);
                                    if (dist < BALL_RADIUS) {
                                        intersected = true;
                                        tempProjectedx = objx;
                                        tempProjectedy = objy;
                                    }
                                    if (distance2 < BALL_RADIUS && distance2 < dist) {
                                        intersected = true;
                                        tempProjectedx = linex2;
                                        tempProjectedy = liney2;
                                        dist = distance2;
                                    }
                                }
                                break;

                            case SIRCLE:

                                float dx = ballx - objx;
                                float dy = bally - objy;
                                dist = length(dx, dy) - linex2;
                                if (dist < BALL_RADIUS) {
                                    intersected = true;
                                    tempProjectedx = objx + (dx / length(dx, dy) * linex2);
                                    tempProjectedy = objy + (dy / length(dx, dy) * linex2);
                                }
                                break;
                        }

                        if (intersected) {
                            float nDotBall = 0;


                            if (USE_GETES) {
                                if ((objFlags & GATE_MASK) != 0) {


                                    nDotBall = (ballx - objx) * -(liney2 - objy) + (bally - objy) * (linex2 - objx);
                                    if (nDotBall > 0) {

                                        intData[objIdx * STRIDE] &= (0xff ^ COLLIDABLE_MASK);
                                    }
                                }
                            }


                            if ((nDotBall <= 0)
                                    && (objFlags & COLLIDABLE_MASK) != 0
                                    && (!foundCollision || dist < closestDistance)) {
                                closestDistance = dist;
                                foundCollision = intersected;
                                collisionObjIdx = objIdx;
                                closestx = tempProjectedx;
                                closesty = tempProjectedy;
                            }


                            if (intData[objIdx * STRIDE + ID_IS_BALL_OVER] == 0) {
                                intData[objIdx * STRIDE + ID_IS_BALL_OVER] = 1;


                                if (!tilt &&
                                        ((intData[objIdx * STRIDE] & DROP_DOWN_MASK) == 0
                                                || (intData[objIdx * STRIDE] & VISIBLE_MASK) != 0)) {
                                    score += intData[objIdx * STRIDE + ID_SCORE];
                                }

                                intData[objIdx * STRIDE + ID_COLLISION_TIME] = frameIdx;
                                if (objBehaviour == BEHAVIOUR_GAME_OVER) {
                                    if (USE_EXTRABALL) {


                                        if (blinkData[START_IDX] != 0) {
                                            blinkData[START_IDX] = 0;
                                            ballx = intData[behaviourObjMap[BEHAVIOUR_START] * STRIDE + ID_X];
                                            bally = intData[behaviourObjMap[BEHAVIOUR_START] * STRIDE + ID_Y];
                                            foundCollision = false;
                                            tilt = false;
                                        } else {
                                            state = LOADING;
                                        }
                                    } else {
                                        state = LOADING;
                                    }

                                }
                                if (objBehaviour == BEHAVIOUR_START) {

                                    ballVely = LAUNCH_SPEED - ((frameIdx & 0xff) / (float) LAUNCH_DIV);


                                    flashFrameIdx = frameIdx + FLASH_FRAME_IDX;
                                }
                                if (USE_GROUP_BONUS) {
                                    if ((objFlags & BUMPER_MASK) != 0
                                            && groupData[GROUP_BUMPER * STRIDE + GRP_BONUS_TIME] > frameIdx) {
                                        bonus += BUMPER_TIME_BONUS;
                                    }
                                }
                                if ((objFlags & DROP_DOWN_MASK) != 0) {
                                    if (USE_GROUP_BONUS) {
                                        if (frameIdx < groupData[GROUP_UPPER_RIGHT * STRIDE + GRP_BONUS_TIME]
                                                && (intData[objIdx * STRIDE] & VISIBLE_MASK) != 0) {
                                            bonus += DROPDOWN_TIME_BONUS;
                                        }
                                    }

                                    intData[objIdx * STRIDE] = DROP_DOWN_MASK;
                                }
                                if ((objFlags & ROLL_OVER_MASK) != 0) {
                                    intData[objIdx * STRIDE] |= VISIBLE_MASK;
                                    if (USE_GROUP_BONUS) {
                                        if (frameIdx < groupData[GROUP_UPPER_LEFT * STRIDE + GRP_BONUS_TIME]) {
                                            bonus += ROLLOVER_TIME_BONUS;
                                        }
                                    }
                                }
                                if (objBehaviour == BEHAVIOUR_UPPER_LEFT


                                        && frameIdx < blinkData[INLINE_ARROW_IDX]) {
                                    blinkData[INLINE_ARROW_IDX] = 0;
                                    bonus += BONUS_UPPER_LEFT;
                                }
                                if (objBehaviour == BEHAVIOUR_UPPER_RIGHT


                                        && frameIdx < blinkData[BUMPER_ARROW_IDX]) {
                                    blinkData[BUMPER_ARROW_IDX] = 0;
                                    bonus += BONUS_UPPER_RIGHT;
                                }
                                if (objBehaviour == BEHAVIOUR_BULLSEYE) {
                                    int bonusShift = 0;
                                    for (int i = 0; i < 3; i++) {


                                        if (frameIdx < blinkData[33 + i]) {
                                            blinkData[33 + i] = 0;
                                            bonusShift++;
                                        }
                                    }

                                    bonus += 25 << bonusShift;
                                    if (USE_BULLSEYE_TEXT) {
                                        intData[ID_BULLSEYE_TIME] = frameIdx + 60 * 3;
                                    }
                                }
                                if ((objBehaviour == BEHAVIOUR_RIGHT_OUTER_LANE
                                        || objBehaviour == BEHAVIOUR_LEFT_OUTER_LANE)
                                        && objFlags != 0) {
                                    intData[objIdx * STRIDE] = 0;
                                    ballVely = KICKER_VEL;
                                    bonus += BONUS_KICKER;
                                }
                            }
                        } else if (intData[objIdx * STRIDE + ID_IS_BALL_OVER] == 1) {

                            intData[objIdx * STRIDE + ID_IS_BALL_OVER] = 0;

                            if (USE_GETES) {
                                if ((objFlags & GATE_MASK) != 0) {

                                    intData[objIdx * STRIDE] = COLLIDABLE_MASK | GATE_MASK | VISIBLE_MASK;
                                }
                            }
                        }
                    }


                    if (foundCollision) {
                        float objVelx = 0;
                        float objVely = 0;


                        if (intData[collisionObjIdx * STRIDE + ID_TYPE] == FLIPPER) {
                            float dx = closestx - intData[collisionObjIdx * STRIDE + ID_X];
                            float dy = closesty - intData[collisionObjIdx * STRIDE + ID_Y];
                            float absVel = floatData[collisionObjIdx * STRIDE] * length(dx, dy);

                            if (length(dx, dy) != 0) {
                                objVely = absVel * dx / length(dx, dy);
                                objVelx = absVel * -dy / length(dx, dy);
                            }
                        }


                        float normalx = (ballx - closestx) / length(ballx - closestx, bally - closesty);
                        float normaly = (bally - closesty) / length(ballx - closestx, bally - closesty);


                        ballx = closestx + normalx * BALL_RADIUS;
                        bally = closesty + normaly * BALL_RADIUS;


                        float impactSpeed = ((intData[collisionObjIdx * STRIDE] & BUMPER_MASK) == 0 || tilt)
                                ? ((objVelx - ballVelx) * normalx + (objVely - ballVely) * normaly) * BOUNCE_NORMAL
                                : ((objVelx - ballVelx) * normalx + (objVely - ballVely) * normaly) * BOUNCE_BUMPER + BUMPER_ADD;

                        ballVelx += normalx * impactSpeed;
                        ballVely += normaly * impactSpeed;


                        if (!pushedBall && pushed && frameIdx < pushTime) {
                            pushedBall = true;
                            ballVelx += normalx / PUSH_DIV_X;
                            ballVely += normaly / PUSH_DIV_Y;
                        }
                    }


                    int c = MULTIPLIER_COLOR;


                    for (int groupIdx = 0; groupIdx < numGroups; groupIdx++) {

                        int groupOr = 0;
                        int groupAnd = 0x7f;
                        int blinkTime = 0;
                        for (int i = 0; i < groupData[groupIdx * 2]; i++) {
                            int objIdx = groupData[groupIdx * 2 + GRP_FIRST_INDEX] + i;
                            groupOr |= intData[objIdx * STRIDE];
                            groupAnd &= intData[objIdx * STRIDE];
                            if (USE_BLINK) {
                                blinkTime = blinkData[objIdx];
                            }
                        }


                        int or = 0;
                        if ((groupOr & VISIBLE_MASK) == 0 && (groupOr & DROP_DOWN_MASK) != 0) {

                            bonus += BONUS_DROPDOWN;


                            if (intData[behaviourObjMap[groupIdx] * STRIDE + ID_TYPE] == ARROW) {
                                blinkData[behaviourObjMap[groupIdx]] = 0xffffff;
                            } else {
                                intData[behaviourObjMap[groupIdx] * STRIDE] |= VISIBLE_MASK;
                            }


                            or = VISIBLE_MASK | COLLIDABLE_MASK;
                        }


                        int and = 0x7f;
                        if ((groupAnd & VISIBLE_MASK) != 0 && (groupAnd & ROLL_OVER_MASK) != 0) {

                            bonus += BONUS_ROLLOVER;


                            and = 0xff ^ (VISIBLE_MASK | COLLIDABLE_MASK);

                            if (USE_GROUP_BONUS) {

                                groupData[groupIdx * STRIDE + GRP_ACTIVATE_CNT]++;
                                groupData[groupIdx * STRIDE + GRP_ACTIVATE_FRAME_IDX] = frameIdx;
                            }

                            if (groupIdx == GROUP_MULTIPLIER) {

                                multiplier = Math.min(8, multiplier + 1);
                                if (USE_MULTIPLIER_TEXT) {
                                    intData[ID_MULTIPLIER_TIME] = frameIdx + 60 * 3;
                                }
                            } else {
                                if (USE_GROUP_BONUS) {
                                    if (groupData[groupIdx * STRIDE + GRP_ACTIVATE_CNT] % 3 == 0) {
                                        bonus += BONUS_DROPDOWN;
                                        groupData[groupIdx * STRIDE + GRP_BONUS_TIME] = frameIdx + 60 * 30;
                                    }
                                }

                                blinkData[behaviourObjMap[groupIdx]] = frameIdx + 60 * 30;
                            }


                            blinkTime = frameIdx + 90;
                        }


                        for (int i = 0; i < groupData[groupIdx * 2]; i++) {
                            int objIdx = groupData[groupIdx * 2 + GRP_FIRST_INDEX] + i;
                            intData[objIdx * STRIDE] |= or;
                            intData[objIdx * STRIDE] &= and;


                            intData[objIdx * STRIDE + ID_COLOR] = c;
                            if (USE_BLINK) {
                                blinkData[objIdx] = blinkTime;
                            }
                        }


                        if (groupIdx > 0) {
                            intData[behaviourObjMap[groupIdx] * STRIDE + ID_COLOR] = c;
                        }


                        c += c * c;
                    }
                }
            }


            if (bally + levely < 200) {
                levely = Math.min(0, 200 - (int) bally);
            }
            if (bally + levely > 400) {
                levely = -(int) bally + 400;
            }


			levely = Math.max((USE_GET_HEIGHT ? getHeight() : FRAME_HEIGHT) - LEVEL_HEIGHT, levely);


            Graphics2D g = (Graphics2D) b.getDrawGraphics();

            if (!BACKGROUND_GRADIENT) {

                g.setColor(new Color(BACKGROUND_COLOR));
                g.fillRect(0, 0, 1024 * 2, 1024 * 2);
            }


            int levely2 = levely + (pushed ? -4 : 0);


            g.translate(0, levely2);

            if (BACKGROUND_GRADIENT) {

                int backColor = BACKGROUND_COLOR;
                for (int i = 0; i < 16; i++) {
                    g.setColor(new Color(backColor));
                    backColor += 0x20300;
                    g.fillRect(0, i * 0x7f, 1024 * 2, 0x7f);
                }
            }

            g.setFont(font);
            Rectangle2D bounds = null;

            final int SHADOW_COLOR = 0x2f2f2f;

			Color shadow = DRAW_BALL_SHADOW ? new Color(SHADOW_COLOR) : null;

			for (int objIdx = 0; objIdx < objCount; objIdx++) {

				if (DRAW_BALL_SHADOW) {

                    if (objIdx == 37) {
                        g.setColor(shadow);
                        g.fillArc((int) ballx - BALL_RADIUS + 8, (int) bally - BALL_RADIUS + 8, BALL_RADIUS * 2, BALL_RADIUS * 2, 0, 360);
                    }
                }

                int c = intData[objIdx * STRIDE + ID_COLOR];
                objx = intData[objIdx * STRIDE + ID_X];
                objy = intData[objIdx * STRIDE + ID_Y];
                int linex2 = intData[objIdx * STRIDE + ID_X2];
                int liney2 = intData[objIdx * STRIDE + ID_Y2];


                if ((intData[objIdx * STRIDE] & BUMPER_MASK) != 0) {
                    int color = Math.min(0xf, (frameIdx - intData[objIdx * STRIDE + ID_COLLISION_TIME]));
                    c = color * BUMPER_COLOR;
                }

                if (USE_BLINK) {
                    if (frameIdx < blinkData[objIdx]) {

                        if ((frameIdx & 31) < 15) {

                            c = (c >> 1) & 0x7f7f7f;
                        }

                    } else if ((intData[objIdx * STRIDE] & VISIBLE_MASK) == 0) {

                        c = (c >> 1) & 0x7f7f7f;
                    }
                } else {

                    if ((intData[objIdx * STRIDE] & VISIBLE_MASK) == 0) {
                        c = (c >> 1) & 0x7f7f7f;
                    }
                }

                if (USE_GROUP_BONUS) {

                    if ((intData[objIdx * STRIDE] & BUMPER_MASK) != 0
                            && groupData[GROUP_BUMPER * STRIDE + GRP_BONUS_TIME] > frameIdx
                            && (frameIdx & 31) < 15) {
                        c = (c >> 1) & 0x7f7f7f;
                    }
                }

                if (DRAW_BUMPER_SHADOWS) {
                    if ((intData[objIdx * STRIDE] & BUMPER_MASK) != 0
                            && intData[objIdx * STRIDE + ID_TYPE] == SIRCLE) {
                        g.setColor(shadow);
                        g.fillArc(objx - linex2 + 12, objy - linex2 + 12, linex2 + linex2, linex2 + linex2, 0, 360);
                    }
                }

                g.setColor(new Color(c));


                int flashy = (flashFrameIdx - frameIdx) * 24;
                boolean pan = objy > flashy && objy < flashy + 200;
                if (pan) {
                    g.setColor(WHITE);
                }

                switch (intData[objIdx * STRIDE + ID_TYPE]) {
                    case FLIPPER:

                        g.setColor(WHITE);

                    case LINE:
                        if (USE_GROUP_BONUS) {
                            if ((frameIdx & 31) < 15
                                    && groupData[GROUP_UPPER_RIGHT * STRIDE + GRP_BONUS_TIME] > frameIdx
                                    && (intData[objIdx * STRIDE] & DROP_DOWN_MASK) != 0
                                    && (intData[objIdx * STRIDE] & VISIBLE_MASK) != 0) {
                                g.setColor(WHITE);
                            }
                        }

                        g.drawLine(objx, objy, linex2, liney2);


                        if (USE_THICK_LINES) {
                            g.drawLine(objx - 1, objy, linex2 - 1, liney2);
                            g.drawLine(objx + 1, objy, linex2 + 1, liney2);


                            g.drawLine(objx, objy + 1, linex2, liney2 + 1);
                        }

                        break;
                    case SIRCLE:

                        int r = linex2;

                        if (FLASH_SIRCLE_SIZE) {
                            if (pan) {
                                r += 5;
                            }
                        }

                        g.fillArc(objx - r, objy - r, r + r, r + r, 0, 360);
                        if (OUTLINE_SIRCLES) {
                            g.setColor(BLACK);
                            if (USE_GROUP_BONUS) {
                                if ((frameIdx & 31) < 15
                                        && frameIdx < groupData[GROUP_UPPER_LEFT * STRIDE + GRP_BONUS_TIME]
                                        && (intData[objIdx * STRIDE] & ROLL_OVER_MASK) != 0) {
                                    g.setColor(WHITE);
                                }
                            }
                            g.drawArc(objx - r, objy - r, r + r, r + r, 0, 360);
                        }
                        break;
                    case ARROW:
                        g.fillArc(objx, objy, 80, 80, linex2 * 2, 45);

                }
            }


            g.setColor(WHITE);
            String text;
            for (int i = 0; i < 8; i++) {
                text = String.valueOf((i + 1));
                g.drawString(text, 484 - 14, 1260 - i * 84);
                intData[(groupData[21] + i) * STRIDE] = (i < multiplier ? VISIBLE_MASK : 0);
            }


            if (USE_SHADED_BALL) {
                int c = 0x5f5f5f;
                int add = 0;
				Color C = new Color(c);
                for (int i = 0; i < 16; i++) {
					g.setColor(C);
                    g.fillArc((int) ballx - BALL_RADIUS + i, (int) bally - BALL_RADIUS + i, BALL_RADIUS * 2 - i * 3, BALL_RADIUS * 2 - i * 3, 0, 360);
                    c += add;
                    add += 0x10101;
                }
                g.setColor(BLACK);
                g.drawArc((int) ballx - BALL_RADIUS, (int) bally - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2, 0, 360);
            } else {
                g.fillArc((int) ballx - BALL_RADIUS, (int) bally - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2, 0, 360);
            }


            g.setColor(WHITE);
            if (tilt) {
                bonus = 0;
            }
            bonus *= multiplier;
            score += bonus;

            if (SHOW_BONUS_TEXT) {
                if (bonus > 0) {
                    intData[ID_BONUS_TEXT] = bonus;
                    intData[ID_BONUS_X] = (int) ballx;
                    intData[ID_BONUS_Y] = (int) bally;
                    intData[ID_BONUS_TIME] = frameIdx + 100;
                }
                if (frameIdx < intData[ID_BONUS_TIME]) {
                    text = String.valueOf(intData[ID_BONUS_TEXT] * 1000);
                    g.drawString(text, intData[ID_BONUS_X] - text.length() * 8, intData[ID_BONUS_Y]--);
                }
            }


            g.translate(0, -levely2);
            if (USE_FLASH) {
                if (bonus > 0) {

                    flashFrameIdx = frameIdx + FLASH_FRAME_IDX;
                }
            }

            if (USE_EXTRABALL) {

                if (score > extraBallTarget) {


                    blinkData[START_IDX] = 0xffffff;


                    extraBallTarget *= 2;

                    if (USE_EXTRABALL_TEXT) {
                        intData[ID_EXTRABALL_TIME] = frameIdx + 60 * 3;
                    }
                }
            }


            if (tilt) {
                text = "TILT";
                g.drawString(text, 479, 120);
            }
            if (USE_EXTRABALL_TEXT) {
                if (frameIdx < intData[ID_EXTRABALL_TIME]) {
                    text = "Extraball!";
                    g.drawString(text, 441, 160);
                }
            }
            if (USE_BULLSEYE_TEXT) {
                if (frameIdx < intData[ID_BULLSEYE_TIME]) {
                    text = "Bullseye!";
                    g.drawString(text, 441, 200);
                }
            }
            if (USE_MULTIPLIER_TEXT) {
                if (frameIdx < intData[ID_MULTIPLIER_TIME]) {
                    text = "Multiplier!";
                    g.drawString(text, 441, 240);
                }
            }


//            if (state == GAME_OVER) {
//                text = "Game Over - Press Enter";
//                g.drawString(text, FRAME_HALF_WIDTH - 188, 320);
//            }

            if (USE_ANIMATED_SCORE) {

                intData[ID_DISPLAY_SCORE] += (score * 1000 - intData[ID_DISPLAY_SCORE]) < 8 ? (score * 1000 - intData[ID_DISPLAY_SCORE]) : (score * 1000 - intData[ID_DISPLAY_SCORE]) / 8;
                text = String.valueOf(intData[ID_DISPLAY_SCORE]);
            } else {
                text = String.valueOf(score * 1000);
            }


            g.drawString(text, FRAME_HALF_WIDTH - text.length() * 8, 80);


            b.show();

            frameIdx++;

            //Util.sleepNS(lastFrame - 16000000);

			long now;
            while ((now = System.nanoTime()) < lastFrame + 16000000) {
                Thread.yield();
            }
            lastFrame = now;


        } while (!k[VK_ESCAPE] && isVisible());

        System.exit(0);
    }


    /**
     * Calculates the length of the (x, y) vector.
     */
    private static float length(float x, float y) {
        return (float) Math.sqrt(y * y + x * x);
    }

    public static void main(String[] args) throws Exception {
        new Pinball4K();
    }

    /**
     * Sets the k member with the key states.
     */
    @Override
    public void processKeyEvent(KeyEvent e) {
        k[e.getKeyCode()] = (e.getID() == 401);
    }
}