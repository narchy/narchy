
//-------------------------------------------------------------------
// shader:    SixteenSegmentDisplay.glsl                   2015-11-05
//            16 Segment Display Example v4.6              2017-10-18
//            http://glslsandbox.com/e#43106.1
// see also:  http://glslsandbox.com/e#38293.0
// hint:      change neon colors by moving around with your mouse!
//-------------------------------------------------------------------
// Question:  Extremely high memory usage at startup - why?

#ifdef GL_ES
precision highp float;
#endif

//uniform float time;
//uniform vec2 mouse;
//uniform vec2 resolution;

const float text_size = 26.0;
const vec2 ch_size  = vec2(1.0, 1.0);               // character size
const vec2 ch_space = ch_size + vec2(0.4, 0.6);     // character distance
const vec2 ch_start = vec2 (ch_space.x * -12.6,9.); // start position
      vec2 ch_pos   = vec2 (0.0, 0.0);              // character position
      vec3 ch_color = vec3 (1, 1, 1);         // character color
      vec3 bg_color = vec3 (0.0, 0.0, 0.0);         // background color

vec2 uv;    // current position

/*========== 16 segment display ==============    bin. hex
                                                  0000 0
Segment bit positions:                            0001 1
                                                  0010 2
  __2__ __1__         any bit adds one segment    0011 3
 |\    |    /|                                    0100 4
 | \   |   / |     bit:   15 12 11 8 7654 3210    0101 5
 3  11 10 9  0             |  | |  | |||| ||||    0110 6
 |   \ | /   |    binary:  0000 0000 0000 0000    0111 7
 |    \|/    |                                    1000 8
  _12__ __8__         example: letter A           1001 9
 |           |                                    1010 A
 |    /|\    |            15 12 11 8 7654 3210    1011 B
 4   / | \   7             |  | |  | |||| ||||    1100 C
 | 13 14  15 |             0001 0001 1001 1111    1101 D
 | /   |   \ |                                    1110 E
  __5__|__6__          binary to hex -> 0x119F    1111 F
*/

#define n0 ddigit(0x22FF);
#define n1 ddigit(0x0281);
#define n2 ddigit(0x1177);
#define n3 ddigit(0x11E7);
#define n4 ddigit(0x5508);
#define n5 ddigit(0x11EE);
#define n6 ddigit(0x11FE);
#define n7 ddigit(0x2206);
#define n8 ddigit(0x11FF);
#define n9 ddigit(0x11EF);

#define A ddigit(0x119F);
#define B ddigit(0x927E);
#define C ddigit(0x007E);
#define D ddigit(0x44E7);
#define E ddigit(0x107E);
#define F ddigit(0x101E);
#define G ddigit(0x807E);
#define H ddigit(0x1199);
#define I ddigit(0x4466);
#define J ddigit(0x4436);
#define K ddigit(0x9218);
#define L ddigit(0x0078);
#define M ddigit(0x0A99);
#define N ddigit(0x8899);
#define O ddigit(0x00FF);
#define P ddigit(0x111F);
#define Q ddigit(0x80FF);
#define R ddigit(0x911F);
#define S ddigit(0x8866);
#define T ddigit(0x4406);
#define U ddigit(0x00F9);
#define V ddigit(0x2218);
#define W ddigit(0xA099);
#define X ddigit(0xAA00);
#define Y ddigit(0x4A00);
#define Z ddigit(0x2266);

#define s_dot     ddots(0);
#define s_ddot    ddots(1);
#define s_excl    ddots(2);
#define s_apo1    ddigit(0x0400);
#define s_apo2    ddigit(0x0401);
#define s_minus   ddigit(0x1100);
#define s_plus    ddigit(0x5500);
#define s_mult    ddigit(0xBB00);
#define s_div     ddigit(0x2200);
#define s_greater ddigit(0x2800);
#define s_less    ddigit(0x8200);
#define s_open    ddigit(0x003C);
#define s_close   ddigit(0x00C3);
#define s_sqrt    ddigit(0x0C02);
#define s_uline   ddigit(0x0060);
#define _  ch_pos.x += ch_space.x;  // blanc
#define nl ch_pos.x = ch_start.x;  ch_pos.y -= 3.0;

//-------------------------------------------------------------------
float dseg(vec2 p0, vec2 p1)    // draw segment
{
  p0 *= ch_size;
  p1 *= ch_size;
  vec2 dir = normalize(p1 - p0);
  vec2 cp = (uv - ch_pos - p0) * mat2(dir.x, dir.y,-dir.y, dir.x);
  return 2.0*distance(cp, clamp(cp, vec2(0), vec2(distance(p0, p1), 0)));
}

//--- return true if bit b of n is set ---
bool bit(int n, int b)
{
  return mod(floor(float(n) / exp2(floor(float(b)))), 2.0) != 0.0;
}

float d = 1.0;

//--- draw segment lines ---
void ddigit(int n)
{
  float v = 1.0;
  if (bit(n,  0)) v = min(v, dseg(vec2( 0.500,  0.063), vec2( 0.500,  0.937)));
  if (bit(n,  1)) v = min(v, dseg(vec2( 0.438,  1.000), vec2( 0.063,  1.000)));
  if (bit(n,  2)) v = min(v, dseg(vec2(-0.063,  1.000), vec2(-0.438,  1.000)));
  if (bit(n,  3)) v = min(v, dseg(vec2(-0.500,  0.937), vec2(-0.500,  0.062)));
  if (bit(n,  4)) v = min(v, dseg(vec2(-0.500, -0.063), vec2(-0.500, -0.938)));
  if (bit(n,  5)) v = min(v, dseg(vec2(-0.438, -1.000), vec2(-0.063, -1.000)));
  if (bit(n,  6)) v = min(v, dseg(vec2( 0.063, -1.000), vec2( 0.438, -1.000)));
  if (bit(n,  7)) v = min(v, dseg(vec2( 0.500, -0.938), vec2( 0.500, -0.063)));
  if (bit(n,  8)) v = min(v, dseg(vec2( 0.063,  0.000), vec2( 0.438, -0.000)));
  if (bit(n,  9)) v = min(v, dseg(vec2( 0.063,  0.063), vec2( 0.438,  0.938)));
  if (bit(n, 10)) v = min(v, dseg(vec2( 0.000,  0.063), vec2( 0.000,  0.937)));
  if (bit(n, 11)) v = min(v, dseg(vec2(-0.063,  0.063), vec2(-0.438,  0.938)));
  if (bit(n, 12)) v = min(v, dseg(vec2(-0.438,  0.000), vec2(-0.063, -0.000)));
  if (bit(n, 13)) v = min(v, dseg(vec2(-0.063, -0.063), vec2(-0.438, -0.938)));
  if (bit(n, 14)) v = min(v, dseg(vec2( 0.000, -0.938), vec2( 0.000, -0.063)));
  if (bit(n, 15)) v = min(v, dseg(vec2( 0.063, -0.063), vec2( 0.438, -0.938)));
  ch_pos.x += ch_space.x;
  d = min(d, v);
}

//--- draw dot characters ---
void ddots(int n)
{
  float v = 1.0;
  v = min(v, dseg(vec2(-0.005, -1.000), vec2( 0.000, -1.000)));
  if (n == 1) v = min(v, dseg(vec2( 0.005,  0.000), vec2( 0.000,  0.000)));
  if (n == 2) ddigit(0x0400);

  ch_pos.x += ch_space.x;
  d = min(d, v);
}

//-------------------------------------------------------------------
//--- show one digit ---
void showDigit (float dd)
{
  if      (dd < 0.5) n0
  else if (dd < 1.5) n1
  else if (dd < 2.5) n2
  else if (dd < 3.5) n3
  else if (dd < 4.5) n4
  else if (dd < 5.5) n5
  else if (dd < 6.5) n6
  else if (dd < 7.5) n7
  else if (dd < 8.5) n8
  else if (dd < 9.5) n9
}
//-------------------------------------------------------------------
//--- show float value with 6.2 format and leading 0 ---
void showFloatN (float value, int digits)
{
  for(int ni = 5; ni > -3; ni--)
  {
    if (ni == -1) s_dot;   // add dot
    float dd = (value / pow(10.0,float(ni)));
    dd = mod(floor(dd), 10.0);
    showDigit (dd);
  }
}
//--- show float value with n.2 char format without leading 0 ---
void showFloat (float value)
{
  bool startDisplay = false;
  for(int ni = 6; ni > -3; ni--)
  {
    if (ni == -1) s_dot;   // add dot
    float dd = (value / pow(10.0,float(ni)));
    dd = mod(floor(dd), 10.0);
    if ((dd > 0.) || startDisplay || (ni <= 0))
    {
	showDigit (dd);
	startDisplay = true;
    }
  }
}//-------------------------------------------------------------------
//--- show integer value with n digits and leading zero digits ---
void showIntegerN (int value, in int digits)
{
  float fv = float(value);
  for(int ni = digits; ni >= 0; ni--)
  {
    float dd = fv / pow(10.0,float(ni));
    dd = mod(floor(dd), 10.0);
    showDigit (dd);
  }
}

//--- show integer value without leading zero digits ---
void showInteger (int value)
{
  bool startDisplay = false;
  float fv = float(value) / 10000.;
  for(int ni = 5; ni > 0; ni--)
  {
    float dd = mod(floor(fv), 10.0);
    if ((dd > 1.0)||(ni==1)) startDisplay = true;
    if (startDisplay)
      showDigit (dd);
    fv = fv * 10.0;
  }
}
//-------------------------------------------------------------------
vec3 HSVtoRGB(vec3 hsv)     // Hue-Saturation-Value to Red-Green-Blue
{
  return((clamp(abs(fract(hsv.x +vec3(0.,2./3.,1./3.))*2.-1.)*3.-1.,0.,1.)-1.)*hsv.y+1.)*hsv.z;
}
//-------------------------------------------------------------------
void main( void )
{
  vec2 res = vec2(100,100);
  //vec2 mp = mouse * res;
  vec2 aspect = res / res.y;
  uv = ( gl_FragCoord.xy / res.y ) - aspect / 2.0;
  uv *= text_size; // + sin(time);     //  set zoom size
  ch_pos = ch_start;  // set start position

  // display text
  _ _ _ n1 n6 s_minus S E G M E N T s_minus D I S P L A Y s_ddot  nl
  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z nl

  s_plus s_minus n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 s_dot s_ddot s_excl s_uline s_less s_greater s_open s_close s_apo1 s_apo2 s_sqrt nl
  //V I E W P O R T _ S I Z E s_ddot  _ showInteger(int(res.x)); _ s_mult _ showInteger(int(res.y)); nl
  //M O U S E _ P O S I T I O N s_ddot _ showInteger(int(mp.x)); _ s_mult _ showInteger(int (mp.y));  nl
  R U N T I M E s_ddot _ /*showFloat(time);*/ _ S E C O N D S nl
  //K E Y P R E S S E D s_ddot _ showInteger(key);


  ch_color = vec3(1,1,1); //HSVtoRGB (vec3 (mouse.x, 1.0-mouse.y, 1.0));
//  ch_color = vec3 (2.6 - 4.*mouse.x, 1.2-mouse.x*mouse.y, 0.5+mouse.y);

  vec3 color = mix(ch_color, bg_color, 1.0- (0.08 / d));  // shading
  gl_FragColor = vec4(color, 1.0);
}