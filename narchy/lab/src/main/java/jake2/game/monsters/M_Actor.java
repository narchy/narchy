/*
 * Copyright (C) 1997-2001 Id Software, Inc.
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.
 * 
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA 02111-1307, USA.
 *  
 */



package jake2.game.monsters;

import jake2.Defines;
import jake2.game.*;
import jake2.util.Lib;
import jake2.util.Math3D;

public class M_Actor {
    

    private static final int FRAME_attak01 = 0;

    public static final int FRAME_attak02 = 1;

    public static final int FRAME_attak03 = 2;

    private static final int FRAME_attak04 = 3;

    private static final int FRAME_death101 = 4;

    public static final int FRAME_death102 = 5;

    public static final int FRAME_death103 = 6;

    public static final int FRAME_death104 = 7;

    public static final int FRAME_death105 = 8;

    public static final int FRAME_death106 = 9;

    private static final int FRAME_death107 = 10;

    private static final int FRAME_death201 = 11;

    public static final int FRAME_death202 = 12;

    public static final int FRAME_death203 = 13;

    public static final int FRAME_death204 = 14;

    public static final int FRAME_death205 = 15;

    public static final int FRAME_death206 = 16;

    public static final int FRAME_death207 = 17;

    public static final int FRAME_death208 = 18;

    public static final int FRAME_death209 = 19;

    public static final int FRAME_death210 = 20;

    public static final int FRAME_death211 = 21;

    public static final int FRAME_death212 = 22;

    private static final int FRAME_death213 = 23;

    public static final int FRAME_death301 = 24;

    public static final int FRAME_death302 = 25;

    public static final int FRAME_death303 = 26;

    public static final int FRAME_death304 = 27;

    public static final int FRAME_death305 = 28;

    public static final int FRAME_death306 = 29;

    public static final int FRAME_death307 = 30;

    public static final int FRAME_death308 = 31;

    public static final int FRAME_death309 = 32;

    public static final int FRAME_death310 = 33;

    public static final int FRAME_death311 = 34;

    public static final int FRAME_death312 = 35;

    public static final int FRAME_death313 = 36;

    public static final int FRAME_death314 = 37;

    public static final int FRAME_death315 = 38;

    private static final int FRAME_flip01 = 39;

    public static final int FRAME_flip02 = 40;

    public static final int FRAME_flip03 = 41;

    public static final int FRAME_flip04 = 42;

    public static final int FRAME_flip05 = 43;

    public static final int FRAME_flip06 = 44;

    public static final int FRAME_flip07 = 45;

    public static final int FRAME_flip08 = 46;

    public static final int FRAME_flip09 = 47;

    public static final int FRAME_flip10 = 48;

    public static final int FRAME_flip11 = 49;

    public static final int FRAME_flip12 = 50;

    public static final int FRAME_flip13 = 51;

    private static final int FRAME_flip14 = 52;

    public static final int FRAME_grenad01 = 53;

    public static final int FRAME_grenad02 = 54;

    public static final int FRAME_grenad03 = 55;

    public static final int FRAME_grenad04 = 56;

    public static final int FRAME_grenad05 = 57;

    public static final int FRAME_grenad06 = 58;

    public static final int FRAME_grenad07 = 59;

    public static final int FRAME_grenad08 = 60;

    public static final int FRAME_grenad09 = 61;

    public static final int FRAME_grenad10 = 62;

    public static final int FRAME_grenad11 = 63;

    public static final int FRAME_grenad12 = 64;

    public static final int FRAME_grenad13 = 65;

    public static final int FRAME_grenad14 = 66;

    public static final int FRAME_grenad15 = 67;

    public static final int FRAME_jump01 = 68;

    public static final int FRAME_jump02 = 69;

    public static final int FRAME_jump03 = 70;

    public static final int FRAME_jump04 = 71;

    public static final int FRAME_jump05 = 72;

    public static final int FRAME_jump06 = 73;

    private static final int FRAME_pain101 = 74;

    public static final int FRAME_pain102 = 75;

    private static final int FRAME_pain103 = 76;

    private static final int FRAME_pain201 = 77;

    public static final int FRAME_pain202 = 78;

    private static final int FRAME_pain203 = 79;

    private static final int FRAME_pain301 = 80;

    public static final int FRAME_pain302 = 81;

    private static final int FRAME_pain303 = 82;

    public static final int FRAME_push01 = 83;

    public static final int FRAME_push02 = 84;

    public static final int FRAME_push03 = 85;

    public static final int FRAME_push04 = 86;

    public static final int FRAME_push05 = 87;

    public static final int FRAME_push06 = 88;

    public static final int FRAME_push07 = 89;

    public static final int FRAME_push08 = 90;

    public static final int FRAME_push09 = 91;

    public static final int FRAME_run01 = 92;

    private static final int FRAME_run02 = 93;

    public static final int FRAME_run03 = 94;

    public static final int FRAME_run04 = 95;

    public static final int FRAME_run05 = 96;

    public static final int FRAME_run06 = 97;

    private static final int FRAME_run07 = 98;

    public static final int FRAME_run08 = 99;

    public static final int FRAME_run09 = 100;

    public static final int FRAME_run10 = 101;

    public static final int FRAME_run11 = 102;

    public static final int FRAME_run12 = 103;

    public static final int FRAME_runs01 = 104;

    public static final int FRAME_runs02 = 105;

    public static final int FRAME_runs03 = 106;

    public static final int FRAME_runs04 = 107;

    public static final int FRAME_runs05 = 108;

    public static final int FRAME_runs06 = 109;

    public static final int FRAME_runs07 = 110;

    public static final int FRAME_runs08 = 111;

    public static final int FRAME_runs09 = 112;

    public static final int FRAME_runs10 = 113;

    public static final int FRAME_runs11 = 114;

    public static final int FRAME_runs12 = 115;

    public static final int FRAME_salute01 = 116;

    public static final int FRAME_salute02 = 117;

    public static final int FRAME_salute03 = 118;

    public static final int FRAME_salute04 = 119;

    public static final int FRAME_salute05 = 120;

    public static final int FRAME_salute06 = 121;

    public static final int FRAME_salute07 = 122;

    public static final int FRAME_salute08 = 123;

    public static final int FRAME_salute09 = 124;

    public static final int FRAME_salute10 = 125;

    public static final int FRAME_salute11 = 126;

    public static final int FRAME_salute12 = 127;

    private static final int FRAME_stand101 = 128;

    public static final int FRAME_stand102 = 129;

    public static final int FRAME_stand103 = 130;

    public static final int FRAME_stand104 = 131;

    public static final int FRAME_stand105 = 132;

    public static final int FRAME_stand106 = 133;

    public static final int FRAME_stand107 = 134;

    public static final int FRAME_stand108 = 135;

    public static final int FRAME_stand109 = 136;

    public static final int FRAME_stand110 = 137;

    public static final int FRAME_stand111 = 138;

    public static final int FRAME_stand112 = 139;

    public static final int FRAME_stand113 = 140;

    public static final int FRAME_stand114 = 141;

    public static final int FRAME_stand115 = 142;

    public static final int FRAME_stand116 = 143;

    public static final int FRAME_stand117 = 144;

    public static final int FRAME_stand118 = 145;

    public static final int FRAME_stand119 = 146;

    public static final int FRAME_stand120 = 147;

    public static final int FRAME_stand121 = 148;

    public static final int FRAME_stand122 = 149;

    public static final int FRAME_stand123 = 150;

    public static final int FRAME_stand124 = 151;

    public static final int FRAME_stand125 = 152;

    public static final int FRAME_stand126 = 153;

    public static final int FRAME_stand127 = 154;

    public static final int FRAME_stand128 = 155;

    public static final int FRAME_stand129 = 156;

    public static final int FRAME_stand130 = 157;

    public static final int FRAME_stand131 = 158;

    public static final int FRAME_stand132 = 159;

    public static final int FRAME_stand133 = 160;

    public static final int FRAME_stand134 = 161;

    public static final int FRAME_stand135 = 162;

    public static final int FRAME_stand136 = 163;

    public static final int FRAME_stand137 = 164;

    public static final int FRAME_stand138 = 165;

    public static final int FRAME_stand139 = 166;

    private static final int FRAME_stand140 = 167;

    public static final int FRAME_stand201 = 168;

    public static final int FRAME_stand202 = 169;

    public static final int FRAME_stand203 = 170;

    public static final int FRAME_stand204 = 171;

    public static final int FRAME_stand205 = 172;

    public static final int FRAME_stand206 = 173;

    public static final int FRAME_stand207 = 174;

    public static final int FRAME_stand208 = 175;

    public static final int FRAME_stand209 = 176;

    public static final int FRAME_stand210 = 177;

    public static final int FRAME_stand211 = 178;

    public static final int FRAME_stand212 = 179;

    public static final int FRAME_stand213 = 180;

    public static final int FRAME_stand214 = 181;

    public static final int FRAME_stand215 = 182;

    public static final int FRAME_stand216 = 183;

    public static final int FRAME_stand217 = 184;

    public static final int FRAME_stand218 = 185;

    public static final int FRAME_stand219 = 186;

    public static final int FRAME_stand220 = 187;

    public static final int FRAME_stand221 = 188;

    public static final int FRAME_stand222 = 189;

    public static final int FRAME_stand223 = 190;

    public static final int FRAME_swim01 = 191;

    public static final int FRAME_swim02 = 192;

    public static final int FRAME_swim03 = 193;

    public static final int FRAME_swim04 = 194;

    public static final int FRAME_swim05 = 195;

    public static final int FRAME_swim06 = 196;

    public static final int FRAME_swim07 = 197;

    public static final int FRAME_swim08 = 198;

    public static final int FRAME_swim09 = 199;

    public static final int FRAME_swim10 = 200;

    public static final int FRAME_swim11 = 201;

    public static final int FRAME_swim12 = 202;

    public static final int FRAME_sw_atk01 = 203;

    public static final int FRAME_sw_atk02 = 204;

    public static final int FRAME_sw_atk03 = 205;

    public static final int FRAME_sw_atk04 = 206;

    public static final int FRAME_sw_atk05 = 207;

    public static final int FRAME_sw_atk06 = 208;

    public static final int FRAME_sw_pan01 = 209;

    public static final int FRAME_sw_pan02 = 210;

    public static final int FRAME_sw_pan03 = 211;

    public static final int FRAME_sw_pan04 = 212;

    public static final int FRAME_sw_pan05 = 213;

    public static final int FRAME_sw_std01 = 214;

    public static final int FRAME_sw_std02 = 215;

    public static final int FRAME_sw_std03 = 216;

    public static final int FRAME_sw_std04 = 217;

    public static final int FRAME_sw_std05 = 218;

    public static final int FRAME_sw_std06 = 219;

    public static final int FRAME_sw_std07 = 220;

    public static final int FRAME_sw_std08 = 221;

    public static final int FRAME_sw_std09 = 222;

    public static final int FRAME_sw_std10 = 223;

    public static final int FRAME_sw_std11 = 224;

    public static final int FRAME_sw_std12 = 225;

    public static final int FRAME_sw_std13 = 226;

    public static final int FRAME_sw_std14 = 227;

    public static final int FRAME_sw_std15 = 228;

    public static final int FRAME_sw_std16 = 229;

    public static final int FRAME_sw_std17 = 230;

    public static final int FRAME_sw_std18 = 231;

    public static final int FRAME_sw_std19 = 232;

    public static final int FRAME_sw_std20 = 233;

    private static final int FRAME_taunt01 = 234;

    public static final int FRAME_taunt02 = 235;

    public static final int FRAME_taunt03 = 236;

    public static final int FRAME_taunt04 = 237;

    public static final int FRAME_taunt05 = 238;

    public static final int FRAME_taunt06 = 239;

    public static final int FRAME_taunt07 = 240;

    public static final int FRAME_taunt08 = 241;

    public static final int FRAME_taunt09 = 242;

    public static final int FRAME_taunt10 = 243;

    public static final int FRAME_taunt11 = 244;

    public static final int FRAME_taunt12 = 245;

    public static final int FRAME_taunt13 = 246;

    public static final int FRAME_taunt14 = 247;

    public static final int FRAME_taunt15 = 248;

    public static final int FRAME_taunt16 = 249;

    private static final int FRAME_taunt17 = 250;

    private static final int FRAME_walk01 = 251;

    public static final int FRAME_walk02 = 252;

    public static final int FRAME_walk03 = 253;

    public static final int FRAME_walk04 = 254;

    public static final int FRAME_walk05 = 255;

    public static final int FRAME_walk06 = 256;

    public static final int FRAME_walk07 = 257;

    private static final int FRAME_walk08 = 258;

    public static final int FRAME_walk09 = 259;

    public static final int FRAME_walk10 = 260;

    public static final int FRAME_walk11 = 261;

    public static final int FRAME_wave01 = 262;

    public static final int FRAME_wave02 = 263;

    public static final int FRAME_wave03 = 264;

    public static final int FRAME_wave04 = 265;

    public static final int FRAME_wave05 = 266;

    public static final int FRAME_wave06 = 267;

    public static final int FRAME_wave07 = 268;

    public static final int FRAME_wave08 = 269;

    public static final int FRAME_wave09 = 270;

    public static final int FRAME_wave10 = 271;

    public static final int FRAME_wave11 = 272;

    public static final int FRAME_wave12 = 273;

    public static final int FRAME_wave13 = 274;

    public static final int FRAME_wave14 = 275;

    public static final int FRAME_wave15 = 276;

    public static final int FRAME_wave16 = 277;

    public static final int FRAME_wave17 = 278;

    public static final int FRAME_wave18 = 279;

    public static final int FRAME_wave19 = 280;

    public static final int FRAME_wave20 = 281;

    public static final int FRAME_wave21 = 282;

    public static final int FRAME_bl_atk01 = 283;

    public static final int FRAME_bl_atk02 = 284;

    public static final int FRAME_bl_atk03 = 285;

    public static final int FRAME_bl_atk04 = 286;

    public static final int FRAME_bl_atk05 = 287;

    public static final int FRAME_bl_atk06 = 288;

    public static final int FRAME_bl_flp01 = 289;

    public static final int FRAME_bl_flp02 = 290;

    public static final int FRAME_bl_flp13 = 291;

    public static final int FRAME_bl_flp14 = 292;

    public static final int FRAME_bl_flp15 = 293;

    public static final int FRAME_bl_jmp01 = 294;

    public static final int FRAME_bl_jmp02 = 295;

    public static final int FRAME_bl_jmp03 = 296;

    public static final int FRAME_bl_jmp04 = 297;

    public static final int FRAME_bl_jmp05 = 298;

    public static final int FRAME_bl_jmp06 = 299;

    public static final int FRAME_bl_pn101 = 300;

    public static final int FRAME_bl_pn102 = 301;

    public static final int FRAME_bl_pn103 = 302;

    public static final int FRAME_bl_pn201 = 303;

    public static final int FRAME_bl_pn202 = 304;

    public static final int FRAME_bl_pn203 = 305;

    public static final int FRAME_bl_pn301 = 306;

    public static final int FRAME_bl_pn302 = 307;

    public static final int FRAME_bl_pn303 = 308;

    public static final int FRAME_bl_psh08 = 309;

    public static final int FRAME_bl_psh09 = 310;

    public static final int FRAME_bl_run01 = 311;

    public static final int FRAME_bl_run02 = 312;

    public static final int FRAME_bl_run03 = 313;

    public static final int FRAME_bl_run04 = 314;

    public static final int FRAME_bl_run05 = 315;

    public static final int FRAME_bl_run06 = 316;

    public static final int FRAME_bl_run07 = 317;

    public static final int FRAME_bl_run08 = 318;

    public static final int FRAME_bl_run09 = 319;

    public static final int FRAME_bl_run10 = 320;

    public static final int FRAME_bl_run11 = 321;

    public static final int FRAME_bl_run12 = 322;

    public static final int FRAME_bl_rns03 = 323;

    public static final int FRAME_bl_rns04 = 324;

    public static final int FRAME_bl_rns05 = 325;

    public static final int FRAME_bl_rns06 = 326;

    public static final int FRAME_bl_rns07 = 327;

    public static final int FRAME_bl_rns08 = 328;

    public static final int FRAME_bl_rns09 = 329;

    public static final int FRAME_bl_sal10 = 330;

    public static final int FRAME_bl_sal11 = 331;

    public static final int FRAME_bl_sal12 = 332;

    public static final int FRAME_bl_std01 = 333;

    public static final int FRAME_bl_std02 = 334;

    public static final int FRAME_bl_std03 = 335;

    public static final int FRAME_bl_std04 = 336;

    public static final int FRAME_bl_std05 = 337;

    public static final int FRAME_bl_std06 = 338;

    public static final int FRAME_bl_std07 = 339;

    public static final int FRAME_bl_std08 = 340;

    public static final int FRAME_bl_std09 = 341;

    public static final int FRAME_bl_std10 = 342;

    public static final int FRAME_bl_std11 = 343;

    public static final int FRAME_bl_std12 = 344;

    public static final int FRAME_bl_std13 = 345;

    public static final int FRAME_bl_std14 = 346;

    public static final int FRAME_bl_std15 = 347;

    public static final int FRAME_bl_std16 = 348;

    public static final int FRAME_bl_std17 = 349;

    public static final int FRAME_bl_std18 = 350;

    public static final int FRAME_bl_std19 = 351;

    public static final int FRAME_bl_std20 = 352;

    public static final int FRAME_bl_std21 = 353;

    public static final int FRAME_bl_std22 = 354;

    public static final int FRAME_bl_std23 = 355;

    public static final int FRAME_bl_std24 = 356;

    public static final int FRAME_bl_std25 = 357;

    public static final int FRAME_bl_std26 = 358;

    public static final int FRAME_bl_std27 = 359;

    public static final int FRAME_bl_std28 = 360;

    public static final int FRAME_bl_std29 = 361;

    public static final int FRAME_bl_std30 = 362;

    public static final int FRAME_bl_std31 = 363;

    public static final int FRAME_bl_std32 = 364;

    public static final int FRAME_bl_std33 = 365;

    public static final int FRAME_bl_std34 = 366;

    public static final int FRAME_bl_std35 = 367;

    public static final int FRAME_bl_std36 = 368;

    public static final int FRAME_bl_std37 = 369;

    public static final int FRAME_bl_std38 = 370;

    public static final int FRAME_bl_std39 = 371;

    public static final int FRAME_bl_std40 = 372;

    public static final int FRAME_bl_swm01 = 373;

    public static final int FRAME_bl_swm02 = 374;

    public static final int FRAME_bl_swm03 = 375;

    public static final int FRAME_bl_swm04 = 376;

    public static final int FRAME_bl_swm05 = 377;

    public static final int FRAME_bl_swm06 = 378;

    public static final int FRAME_bl_swm07 = 379;

    public static final int FRAME_bl_swm08 = 380;

    public static final int FRAME_bl_swm09 = 381;

    public static final int FRAME_bl_swm10 = 382;

    public static final int FRAME_bl_swm11 = 383;

    public static final int FRAME_bl_swm12 = 384;

    public static final int FRAME_bl_swk01 = 385;

    public static final int FRAME_bl_swk02 = 386;

    public static final int FRAME_bl_swk03 = 387;

    public static final int FRAME_bl_swk04 = 388;

    public static final int FRAME_bl_swk05 = 389;

    public static final int FRAME_bl_swk06 = 390;

    public static final int FRAME_bl_swp01 = 391;

    public static final int FRAME_bl_swp02 = 392;

    public static final int FRAME_bl_swp03 = 393;

    public static final int FRAME_bl_swp04 = 394;

    public static final int FRAME_bl_swp05 = 395;

    public static final int FRAME_bl_sws01 = 396;

    public static final int FRAME_bl_sws02 = 397;

    public static final int FRAME_bl_sws03 = 398;

    public static final int FRAME_bl_sws04 = 399;

    public static final int FRAME_bl_sws05 = 400;

    public static final int FRAME_bl_sws06 = 401;

    public static final int FRAME_bl_sws07 = 402;

    public static final int FRAME_bl_sws08 = 403;

    public static final int FRAME_bl_sws09 = 404;

    public static final int FRAME_bl_sws10 = 405;

    public static final int FRAME_bl_sws11 = 406;

    public static final int FRAME_bl_sws12 = 407;

    public static final int FRAME_bl_sws13 = 408;

    public static final int FRAME_bl_sws14 = 409;

    public static final int FRAME_bl_tau14 = 410;

    public static final int FRAME_bl_tau15 = 411;

    public static final int FRAME_bl_tau16 = 412;

    public static final int FRAME_bl_tau17 = 413;

    public static final int FRAME_bl_wlk01 = 414;

    public static final int FRAME_bl_wlk02 = 415;

    public static final int FRAME_bl_wlk03 = 416;

    public static final int FRAME_bl_wlk04 = 417;

    public static final int FRAME_bl_wlk05 = 418;

    public static final int FRAME_bl_wlk06 = 419;

    public static final int FRAME_bl_wlk07 = 420;

    public static final int FRAME_bl_wlk08 = 421;

    public static final int FRAME_bl_wlk09 = 422;

    public static final int FRAME_bl_wlk10 = 423;

    public static final int FRAME_bl_wlk11 = 424;

    public static final int FRAME_bl_wav19 = 425;

    public static final int FRAME_bl_wav20 = 426;

    public static final int FRAME_bl_wav21 = 427;

    public static final int FRAME_cr_atk01 = 428;

    public static final int FRAME_cr_atk02 = 429;

    public static final int FRAME_cr_atk03 = 430;

    public static final int FRAME_cr_atk04 = 431;

    public static final int FRAME_cr_atk05 = 432;

    public static final int FRAME_cr_atk06 = 433;

    public static final int FRAME_cr_atk07 = 434;

    public static final int FRAME_cr_atk08 = 435;

    public static final int FRAME_cr_pan01 = 436;

    public static final int FRAME_cr_pan02 = 437;

    public static final int FRAME_cr_pan03 = 438;

    public static final int FRAME_cr_pan04 = 439;

    public static final int FRAME_cr_std01 = 440;

    public static final int FRAME_cr_std02 = 441;

    public static final int FRAME_cr_std03 = 442;

    public static final int FRAME_cr_std04 = 443;

    public static final int FRAME_cr_std05 = 444;

    public static final int FRAME_cr_std06 = 445;

    public static final int FRAME_cr_std07 = 446;

    public static final int FRAME_cr_std08 = 447;

    public static final int FRAME_cr_wlk01 = 448;

    public static final int FRAME_cr_wlk02 = 449;

    public static final int FRAME_cr_wlk03 = 450;

    public static final int FRAME_cr_wlk04 = 451;

    public static final int FRAME_cr_wlk05 = 452;

    public static final int FRAME_cr_wlk06 = 453;

    public static final int FRAME_cr_wlk07 = 454;

    public static final int FRAME_crbl_a01 = 455;

    public static final int FRAME_crbl_a02 = 456;

    public static final int FRAME_crbl_a03 = 457;

    public static final int FRAME_crbl_a04 = 458;

    public static final int FRAME_crbl_a05 = 459;

    public static final int FRAME_crbl_a06 = 460;

    public static final int FRAME_crbl_a07 = 461;

    public static final int FRAME_crbl_p01 = 462;

    public static final int FRAME_crbl_p02 = 463;

    public static final int FRAME_crbl_p03 = 464;

    public static final int FRAME_crbl_p04 = 465;

    public static final int FRAME_crbl_s01 = 466;

    public static final int FRAME_crbl_s02 = 467;

    public static final int FRAME_crbl_s03 = 468;

    public static final int FRAME_crbl_s04 = 469;

    public static final int FRAME_crbl_s05 = 470;

    public static final int FRAME_crbl_s06 = 471;

    public static final int FRAME_crbl_s07 = 472;

    public static final int FRAME_crbl_s08 = 473;

    public static final int FRAME_crbl_w01 = 474;

    public static final int FRAME_crbl_w02 = 475;

    public static final int FRAME_crbl_w03 = 476;

    public static final int FRAME_crbl_w04 = 477;

    public static final int FRAME_crbl_w05 = 478;

    public static final int FRAME_crbl_w06 = 479;

    public static final int FRAME_crbl_w07 = 480;

    private static final float MODEL_SCALE = 1.000000f;

    private static final int MAX_ACTOR_NAMES = 8;

    private static final String[] actor_names = { "Hellrot", "Tokay", "Killme", "Disruptor",
            "Adrianator", "Rambear", "Titus", "Bitterman" };

    private static final EntThinkAdapter actor_stand = new EntThinkAdapter() {
        @Override
        public String getID() { return "actor_stand";}
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = actor_move_stand;

            
            if (GameBase.level.time < 1.0)
                self.s.frame = self.monsterinfo.currentmove.firstframe
                        + (Lib.rand() % (self.monsterinfo.currentmove.lastframe
                                - self.monsterinfo.currentmove.firstframe + 1));
            return true;
        }
    };

    private static final mframe_t[] actor_frames_stand = {
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null),
            new mframe_t(GameAI.ai_stand, 0f, null) };

    private static final mmove_t actor_move_stand = new mmove_t(FRAME_stand101,
            FRAME_stand140, actor_frames_stand, null);

    private static final mframe_t[] actor_frames_walk = {
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 6, null),
            new mframe_t(GameAI.ai_walk, 10, null),
            new mframe_t(GameAI.ai_walk, 3, null),
            new mframe_t(GameAI.ai_walk, 2, null),
            new mframe_t(GameAI.ai_walk, 7, null),
            new mframe_t(GameAI.ai_walk, 10, null),
            new mframe_t(GameAI.ai_walk, 1, null),
            new mframe_t(GameAI.ai_walk, 4, null),
            new mframe_t(GameAI.ai_walk, 0, null),
            new mframe_t(GameAI.ai_walk, 0, null) };

    private static final mmove_t actor_move_walk = new mmove_t(FRAME_walk01, FRAME_walk08,
            actor_frames_walk, null);

    private static final EntThinkAdapter actor_walk = new EntThinkAdapter() {
        @Override
        public String getID() { return "actor_walk";}
        @Override
        public boolean think(edict_t self) {
            self.monsterinfo.currentmove = actor_move_walk;
            return true;
        }
    };

    private static final mframe_t[] actor_frames_run = {
            new mframe_t(GameAI.ai_run, 4, null),
            new mframe_t(GameAI.ai_run, 15, null),
            new mframe_t(GameAI.ai_run, 15, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 20, null),
            new mframe_t(GameAI.ai_run, 15, null),
            new mframe_t(GameAI.ai_run, 8, null),
            new mframe_t(GameAI.ai_run, 17, null),
            new mframe_t(GameAI.ai_run, 12, null),
            new mframe_t(GameAI.ai_run, -2, null),
            new mframe_t(GameAI.ai_run, -2, null),
            new mframe_t(GameAI.ai_run, -1, null) };

    private static final mmove_t actor_move_run = new mmove_t(FRAME_run02, FRAME_run07,
            actor_frames_run, null);

    private static final EntThinkAdapter actor_run = new EntThinkAdapter() {
        @Override
        public String getID() { return "actor_run";}
        @Override
        public boolean think(edict_t self) {
            if ((GameBase.level.time < self.pain_debounce_time)
                    && (self.enemy == null)) {
                if (self.movetarget != null)
                    actor_walk.think(self);
                else
                    actor_stand.think(self);
                return true;
            }

            if ((self.monsterinfo.aiflags & Defines.AI_STAND_GROUND) != 0) {
                actor_stand.think(self);
                return true;
            }

            self.monsterinfo.currentmove = actor_move_run;
            return true;

        }
    };

    private static final mframe_t[] actor_frames_pain1 = {
            new mframe_t(GameAI.ai_move, -5, null),
            new mframe_t(GameAI.ai_move, 4, null),
            new mframe_t(GameAI.ai_move, 1, null) };

    private static final mmove_t actor_move_pain1 = new mmove_t(FRAME_pain101, FRAME_pain103,
            actor_frames_pain1, actor_run);

    private static final mframe_t[] actor_frames_pain2 = {
            new mframe_t(GameAI.ai_move, -4, null),
            new mframe_t(GameAI.ai_move, 4, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t actor_move_pain2 = new mmove_t(FRAME_pain201, FRAME_pain203,
            actor_frames_pain2, actor_run);

    private static final mframe_t[] actor_frames_pain3 = {
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t actor_move_pain3 = new mmove_t(FRAME_pain301, FRAME_pain303,
            actor_frames_pain3, actor_run);

    private static final mframe_t[] actor_frames_flipoff = {
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null) };

    private static final mmove_t actor_move_flipoff = new mmove_t(FRAME_flip01, FRAME_flip14,
            actor_frames_flipoff, actor_run);

    private static final mframe_t[] actor_frames_taunt = {
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null),
            new mframe_t(GameAI.ai_turn, 0, null) };

    private static final mmove_t actor_move_taunt = new mmove_t(FRAME_taunt01, FRAME_taunt17,
            actor_frames_taunt, actor_run);

    private static final String[] messages = { "Watch it", "#$@*&", "Idiot",
            "Check your targets" };

    private static final EntPainAdapter actor_pain = new EntPainAdapter() {
        @Override
        public String getID() { return "actor_pain";}
        @Override
        public void pain(edict_t self, edict_t other, float kick, int damage) {

            if (self.health < (self.max_health / 2))
                self.s.skinnum = 1;

            if (GameBase.level.time < self.pain_debounce_time)
                return;

            self.pain_debounce_time = GameBase.level.time + 3;
            
            

            if ((other.client != null) && (Lib.random() < 0.4)) {
                float[] v = {0, 0, 0};

                Math3D.VectorSubtract(other.s.origin, self.s.origin, v);
                self.ideal_yaw = Math3D.vectoyaw(v);
                if (Lib.random() < 0.5f)
                    self.monsterinfo.currentmove = actor_move_flipoff;
                else
                    self.monsterinfo.currentmove = actor_move_taunt;


                String name = actor_names[(self.index) % MAX_ACTOR_NAMES];

                game_import_t.cprintf(other, Defines.PRINT_CHAT, name + ": "
                        + messages[Lib.rand() % 3] + "!\n");
                return;
            }

            int n = Lib.rand() % 3;
            self.monsterinfo.currentmove = switch (n) {
                case 0 -> actor_move_pain1;
                case 1 -> actor_move_pain2;
                default -> actor_move_pain3;
            };
        }
    };

    private static final EntThinkAdapter actor_dead = new EntThinkAdapter() {
        @Override
        public String getID() { return "actor_dead";}
        @Override
        public boolean think(edict_t self) {
            Math3D.VectorSet(self.mins, -16, -16, -24);
            Math3D.VectorSet(self.maxs, 16, 16, -8);
            self.movetype = Defines.MOVETYPE_TOSS;
            self.svflags |= Defines.SVF_DEADMONSTER;
            self.nextthink = 0;
            game_import_t.linkentity(self);
            return true;
        }
    };

    private static final mframe_t[] actor_frames_death1 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -13, null),
            new mframe_t(GameAI.ai_move, 14, null),
            new mframe_t(GameAI.ai_move, 3, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, 1, null) };

    private static final mmove_t actor_move_death1 = new mmove_t(FRAME_death101,
            FRAME_death107, actor_frames_death1, actor_dead);

    private static final mframe_t[] actor_frames_death2 = {
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, 7, null),
            new mframe_t(GameAI.ai_move, -6, null),
            new mframe_t(GameAI.ai_move, -5, null),
            new mframe_t(GameAI.ai_move, 1, null),
            new mframe_t(GameAI.ai_move, 0, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -2, null),
            new mframe_t(GameAI.ai_move, -1, null),
            new mframe_t(GameAI.ai_move, -9, null),
            new mframe_t(GameAI.ai_move, -13, null),
            new mframe_t(GameAI.ai_move, -13, null),
            new mframe_t(GameAI.ai_move, 0, null) };

    private static final mmove_t actor_move_death2 = new mmove_t(FRAME_death201,
            FRAME_death213, actor_frames_death2, actor_dead);

    private static final EntDieAdapter actor_die = new EntDieAdapter() {
        @Override
        public String getID() { return "actor_die";}
        @Override
        public void die(edict_t self, edict_t inflictor, edict_t attacker,
                        int damage, float[] point) {
            int n;

            
            if (self.health <= -80) {
                
                
                for (n = 0; n < 2; n++)
                    GameMisc.ThrowGib(self, "models/objects/gibs/bone/tris.md2",
                            damage, Defines.GIB_ORGANIC);
                for (n = 0; n < 4; n++)
                    GameMisc.ThrowGib(self,
                            "models/objects/gibs/sm_meat/tris.md2", damage,
                            Defines.GIB_ORGANIC);
                GameMisc.ThrowHead(self, "models/objects/gibs/head2/tris.md2",
                        damage, Defines.GIB_ORGANIC);
                self.deadflag = Defines.DEAD_DEAD;
                return;
            }

            if (self.deadflag == Defines.DEAD_DEAD)
                return;

            
            
            self.deadflag = Defines.DEAD_DEAD;
            self.takedamage = Defines.DAMAGE_YES;

            n = Lib.rand() % 2;
            if (n == 0)
                self.monsterinfo.currentmove = actor_move_death1;
            else
                self.monsterinfo.currentmove = actor_move_death2;
        }
    };

    private static final EntThinkAdapter actor_fire = new EntThinkAdapter() {
        @Override
        public String getID() { return "actor_fire";}
        @Override
        public boolean think(edict_t self) {
            actorMachineGun(self);

            if (GameBase.level.time >= self.monsterinfo.pausetime)
                self.monsterinfo.aiflags &= ~Defines.AI_HOLD_FRAME;
            else
                self.monsterinfo.aiflags |= Defines.AI_HOLD_FRAME;

            return true;
        }
    };

    private static final mframe_t[] actor_frames_attack = {
            new mframe_t(GameAI.ai_charge, -2, actor_fire),
            new mframe_t(GameAI.ai_charge, -2, null),
            new mframe_t(GameAI.ai_charge, 3, null),
            new mframe_t(GameAI.ai_charge, 2, null) };

    private static final mmove_t actor_move_attack = new mmove_t(FRAME_attak01,
            FRAME_attak04, actor_frames_attack, actor_run);

    private static final EntThinkAdapter actor_attack = new EntThinkAdapter() {
        @Override
        public String getID() { return "actor_attack";}
        @Override
        public boolean think(edict_t self) {

            self.monsterinfo.currentmove = actor_move_attack;
            int n = (Lib.rand() & 15) + 3 + 7;
            self.monsterinfo.pausetime = GameBase.level.time + n
                    * Defines.FRAMETIME;

            return true;
        }
    };

    private static final EntUseAdapter actor_use = new EntUseAdapter() {
        @Override
        public String getID() { return "actor_use";}
        @Override
        public void use(edict_t self, edict_t other, edict_t activator) {

            self.goalentity = self.movetarget = GameBase
                    .G_PickTarget(self.target);
            if ((null == self.movetarget)
                    || (Lib.strcmp(self.movetarget.classname, "target_actor") != 0)) {
                game_import_t
                        .dprintf(self.classname + " has bad target "
                                + self.target + " at "
                                + Lib.vtos(self.s.origin) + '\n');
                self.target = null;
                self.monsterinfo.pausetime = 100000000;
                self.monsterinfo.stand.think(self);
                return;
            }

            float[] v = {0, 0, 0};
            Math3D.VectorSubtract(self.goalentity.s.origin, self.s.origin, v);
            self.ideal_yaw = self.s.angles[Defines.YAW] = Math3D.vectoyaw(v);
            self.monsterinfo.walk.think(self);
            self.target = null;
        }
    };

    /*
     * QUAKED target_actor (.5 .3 0) (-8 -8 -8) (8 8 8) JUMP SHOOT ATTACK x HOLD
     * BRUTAL JUMP jump in set direction upon reaching this target SHOOT take a
     * single shot at the pathtarget ATTACK attack pathtarget until it or actor
     * is dead
     * 
     * "target" next target_actor "pathtarget" target of any action to be taken
     * at this point "wait" amount of time actor should pause at this point
     * "message" actor will "say" this to the player
     * 
     * for JUMP only: "speed" speed thrown forward (default 200) "height" speed
     * thrown upwards (default 200)
     */

    private static final EntTouchAdapter target_actor_touch = new EntTouchAdapter() {
        @Override
        public String getID() { return "target_actor_touch";}
        @Override
        public void touch(edict_t self, edict_t other, cplane_t plane,
                          csurface_t surf) {

            if (other.movetarget != self)
                return;

            if (other.enemy != null)
                return;

            other.goalentity = other.movetarget = null;

            if (self.message != null) {

                for (int n = 1; n <= GameBase.game.maxclients; n++) {
                    edict_t ent = GameBase.g_edicts[n];
                    if (!ent.inuse)
                        continue;
                    game_import_t.cprintf(ent, Defines.PRINT_CHAT,
                            actor_names[(other.index) % MAX_ACTOR_NAMES] + ": "
                                    + self.message + '\n');
                }
            }

            if ((self.spawnflags & 1) != 0) 
            {
                other.velocity[0] = self.movedir[0] * self.speed;
                other.velocity[1] = self.movedir[1] * self.speed;

                if (other.groundentity != null) {
                    other.groundentity = null;
                    other.velocity[2] = self.movedir[2];
                    game_import_t.sound(other, Defines.CHAN_VOICE, game_import_t
                            .soundindex("player/male/jump1.wav"), 1,
                            Defines.ATTN_NORM, 0);
                }
            }

            if ((self.spawnflags & 2) != 0) 
            {
            } else if ((self.spawnflags & 4) != 0) 
            {
                other.enemy = GameBase.G_PickTarget(self.pathtarget);
                if (other.enemy != null) {
                    other.goalentity = other.enemy;
                    if ((self.spawnflags & 32) != 0)
                        other.monsterinfo.aiflags |= Defines.AI_BRUTAL;
                    if ((self.spawnflags & 16) != 0) {
                        other.monsterinfo.aiflags |= Defines.AI_STAND_GROUND;
                        actor_stand.think(other);
                    } else {
                        actor_run.think(other);
                    }
                }
            }

            if (0 != (self.spawnflags & 6) && (self.pathtarget != null)) {

                String savetarget = self.target;
                self.target = self.pathtarget;
                GameUtil.G_UseTargets(self, other);
                self.target = savetarget;
            }

            other.movetarget = GameBase.G_PickTarget(self.target);

            if (other.goalentity == null)
                other.goalentity = other.movetarget;

            if (null == other.movetarget && null == other.enemy) {
                other.monsterinfo.pausetime = GameBase.level.time + 100000000;
                other.monsterinfo.stand.think(other);
            } else if (other.movetarget == other.goalentity) {
                float[] v = {0, 0, 0};
                Math3D.VectorSubtract(other.movetarget.s.origin,
                        other.s.origin, v);
                other.ideal_yaw = Math3D.vectoyaw(v);
            }
        }
    };

    private static void actorMachineGun(edict_t self) {

        float[] forward = {0, 0, 0};
        float[] right = {0, 0, 0};

        Math3D.AngleVectors(self.s.angles, forward, right, null);

        float[] start = {0, 0, 0};
        Math3D.G_ProjectSource(self.s.origin,
                M_Flash.monster_flash_offset[Defines.MZ2_ACTOR_MACHINEGUN_1],
                forward, right, start);

        if (self.enemy != null) {
            float[] target = {0, 0, 0};
            if (self.enemy.health > 0) {
                Math3D.VectorMA(self.enemy.s.origin, -0.2f,
                        self.enemy.velocity, target);
                target[2] += self.enemy.viewheight;
            } else {
                Math3D.VectorCopy(self.enemy.absmin, target);
                target[2] += (self.enemy.size[2] / 2);
            }
            Math3D.VectorSubtract(target, start, forward);
            Math3D.VectorNormalize(forward);
        } else {
            Math3D.AngleVectors(self.s.angles, forward, null, null);
        }
        Monster.monster_fire_bullet(self, start, forward, 3, 4,
                Defines.DEFAULT_BULLET_HSPREAD, Defines.DEFAULT_BULLET_VSPREAD,
                Defines.MZ2_ACTOR_MACHINEGUN_1);
    }

    /**
     * QUAKED misc_actor (1 .5 0) (-16 -16 -24) (16 16 32)
     */

    public static void SP_misc_actor(edict_t self) {
        if (GameBase.deathmatch.value != 0) {
            GameUtil.G_FreeEdict(self);
            return;
        }

        if (self.targetname != null) {
            game_import_t.dprintf("untargeted " + self.classname + " at "
                    + Lib.vtos(self.s.origin) + '\n');
            GameUtil.G_FreeEdict(self);
            return;
        }

        if (self.target != null) {
            game_import_t.dprintf(self.classname + " with no target at "
                    + Lib.vtos(self.s.origin) + '\n');
            GameUtil.G_FreeEdict(self);
            return;
        }

        self.movetype = Defines.MOVETYPE_STEP;
        self.solid = Defines.SOLID_BBOX;
        self.s.modelindex = game_import_t.modelindex("players/male/tris.md2");
        Math3D.VectorSet(self.mins, -16, -16, -24);
        Math3D.VectorSet(self.maxs, 16, 16, 32);

        if (0 == self.health)
            self.health = 100;
        self.mass = 200;

        self.pain = actor_pain;
        self.die = actor_die;

        self.monsterinfo.stand = actor_stand;
        self.monsterinfo.walk = actor_walk;
        self.monsterinfo.run = actor_run;
        self.monsterinfo.attack = actor_attack;
        self.monsterinfo.melee = null;
        self.monsterinfo.sight = null;

        self.monsterinfo.aiflags |= Defines.AI_GOOD_GUY;

        game_import_t.linkentity(self);

        self.monsterinfo.currentmove = actor_move_stand;
        self.monsterinfo.scale = MODEL_SCALE;

        GameAI.walkmonster_start.think(self);

        
        
        self.use = actor_use;
    }

    public static void SP_target_actor(edict_t self) {
        if (self.targetname != null)
            game_import_t.dprintf(self.classname + " with no targetname at "
                    + Lib.vtos(self.s.origin) + " \n");

        self.solid = Defines.SOLID_TRIGGER;
        self.touch = target_actor_touch;
        Math3D.VectorSet(self.mins, -8, -8, -8);
        Math3D.VectorSet(self.maxs, 8, 8, 8);
        self.svflags = Defines.SVF_NOCLIENT;

        if ((self.spawnflags & 1) != 0) {
            if (0 == self.speed)
                self.speed = 200;
            if (0 == GameBase.st.height)
                GameBase.st.height = 200;
            if (self.s.angles[Defines.YAW] == 0)
                self.s.angles[Defines.YAW] = 360;
            GameBase.G_SetMovedir(self.s.angles, self.movedir);
            self.movedir[2] = GameBase.st.height;
        }

        game_import_t.linkentity(self);
    }
}