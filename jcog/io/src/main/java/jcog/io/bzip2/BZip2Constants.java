/*
 * The Apache Software License, Version 1.1
 *
 * Copyright (c) 2001 The Apache Software Foundation.  All rights
 * reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. The end-user documentation included with the redistribution, if
 *    any, must include the following acknowlegement:
 *       "This product includes software developed by the
 *        Apache Software Foundation (http:
 *    Alternately, this acknowlegement may appear in the software itself,
 *    if and wherever such third-party acknowlegements normally appear.
 *
 * 4. The names "Ant" and "Apache Software
 *    Foundation" must not be used to endorse or promote products derived
 *    from this software without prior written permission. For written
 *    permission, please contact apache@apache.org.
 *
 * 5. Products derived from this software may not be called "Apache"
 *    nor may "Apache" appear in their names without prior written
 *    permission of the Apache Group.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * ====================================================================
 *
 * This software consists of voluntary contributions made by many
 * individuals on behalf of the Apache Software Foundation.  For more
 * information on the Apache Software Foundation, please see
 * <http:
 */

/*
 * This package is based on the work done by Keiron Liddle, Aftex Software
 * <keiron@aftexsw.com> to whom the Ant project is very grateful for his
 * great code.
 */

package jcog.io.bzip2;

/**
 * Base class for both the compress and decompress classes.
 * Holds common arrays, and static data.
 *
 * @author <a href="mailto:keiron@aftexsw.com">Keiron Liddle</a>
 */
interface BZip2Constants {

    int baseBlockSize = 100000;
    int MAX_ALPHA_SIZE = 258;
    int MAX_CODE_LEN = 23;
    int RUNA = 0;
    int RUNB = 1;
    int N_GROUPS = 6;
    int G_SIZE = 50;
    int N_ITERS = 4;
    int MAX_SELECTORS = (2 + (900000 / G_SIZE));
    int NUM_OVERSHOOT_BYTES = 20;

    int[] rNums = {
        619, 720, 127, 481, 931, 816, 813, 233, 566, 247,
        985, 724, 205, 454, 863, 491, 741, 242, 949, 214,
        733, 859, 335, 708, 621, 574, 73, 654, 730, 472,
        419, 436, 278, 496, 867, 210, 399, 680, 480, 51,
        878, 465, 811, 169, 869, 675, 611, 697, 867, 561,
        862, 687, 507, 283, 482, 129, 807, 591, 733, 623,
        150, 238, 59, 379, 684, 877, 625, 169, 643, 105,
        170, 607, 520, 932, 727, 476, 693, 425, 174, 647,
        73, 122, 335, 530, 442, 853, 695, 249, 445, 515,
        909, 545, 703, 919, 874, 474, 882, 500, 594, 612,
        641, 801, 220, 162, 819, 984, 589, 513, 495, 799,
        161, 604, 958, 533, 221, 400, 386, 867, 600, 782,
        382, 596, 414, 171, 516, 375, 682, 485, 911, 276,
        98, 553, 163, 354, 666, 933, 424, 341, 533, 870,
        227, 730, 475, 186, 263, 647, 537, 686, 600, 224,
        469, 68, 770, 919, 190, 373, 294, 822, 808, 206,
        184, 943, 795, 384, 383, 461, 404, 758, 839, 887,
        715, 67, 618, 276, 204, 918, 873, 777, 604, 560,
        951, 160, 578, 722, 79, 804, 96, 409, 713, 940,
        652, 934, 970, 447, 318, 353, 859, 672, 112, 785,
        645, 863, 803, 350, 139, 93, 354, 99, 820, 908,
        609, 772, 154, 274, 580, 184, 79, 626, 630, 742,
        653, 282, 762, 623, 680, 81, 927, 626, 789, 125,
        411, 521, 938, 300, 821, 78, 343, 175, 128, 250,
        170, 774, 972, 275, 999, 639, 495, 78, 352, 126,
        857, 956, 358, 619, 580, 124, 737, 594, 701, 612,
        669, 112, 134, 694, 363, 992, 809, 743, 168, 974,
        944, 375, 748, 52, 600, 747, 642, 182, 862, 81,
        344, 805, 988, 739, 511, 655, 814, 334, 249, 515,
        897, 955, 664, 981, 649, 113, 974, 459, 893, 228,
        433, 837, 553, 268, 926, 240, 102, 654, 459, 51,
        686, 754, 806, 760, 493, 403, 415, 394, 687, 700,
        946, 670, 656, 610, 738, 392, 760, 799, 887, 653,
        978, 321, 576, 617, 626, 502, 894, 679, 243, 440,
        680, 879, 194, 572, 640, 724, 926, 56, 204, 700,
        707, 151, 457, 449, 797, 195, 791, 558, 945, 679,
        297, 59, 87, 824, 713, 663, 412, 693, 342, 606,
        134, 108, 571, 364, 631, 212, 174, 643, 304, 329,
        343, 97, 430, 751, 497, 314, 983, 374, 822, 928,
        140, 206, 73, 263, 980, 736, 876, 478, 430, 305,
        170, 514, 364, 692, 829, 82, 855, 953, 676, 246,
        369, 970, 294, 750, 807, 827, 150, 790, 288, 923,
        804, 378, 215, 828, 592, 281, 565, 555, 710, 82,
        896, 831, 547, 261, 524, 462, 293, 465, 502, 56,
        661, 821, 976, 991, 658, 869, 905, 758, 745, 193,
        768, 550, 608, 933, 378, 286, 215, 979, 792, 961,
        61, 688, 793, 644, 986, 403, 106, 366, 905, 644,
        372, 567, 466, 434, 645, 210, 389, 550, 919, 135,
        780, 773, 635, 389, 707, 100, 626, 958, 165, 504,
        920, 176, 193, 713, 857, 265, 203, 50, 668, 108,
        645, 990, 626, 197, 510, 357, 358, 850, 858, 364,
        936, 638
    };

    /**
     * A simple class the hold and calculate the CRC for sanity checking
     * of the data.
     *
     * @author <a href="mailto:keiron@aftexsw.com">Keiron Liddle</a>
     */
    class CRC {
        private static final int[] crc32Table = {
            0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9,
            0x130476dc, 0x17c56b6b, 0x1a864db2, 0x1e475005,
            0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61,
            0x350c9b64, 0x31cd86d3, 0x3c8ea00a, 0x384fbdbd,
            0x4c11db70, 0x48d0c6c7, 0x4593e01e, 0x4152fda9,
            0x5f15adac, 0x5bd4b01b, 0x569796c2, 0x52568b75,
            0x6a1936c8, 0x6ed82b7f, 0x639b0da6, 0x675a1011,
            0x791d4014, 0x7ddc5da3, 0x709f7b7a, 0x745e66cd,
            0x9823b6e0, 0x9ce2ab57, 0x91a18d8e, 0x95609039,
            0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5,
            0xbe2b5b58, 0xbaea46ef, 0xb7a96036, 0xb3687d81,
            0xad2f2d84, 0xa9ee3033, 0xa4ad16ea, 0xa06c0b5d,
            0xd4326d90, 0xd0f37027, 0xddb056fe, 0xd9714b49,
            0xc7361b4c, 0xc3f706fb, 0xceb42022, 0xca753d95,
            0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1,
            0xe13ef6f4, 0xe5ffeb43, 0xe8bccd9a, 0xec7dd02d,
            0x34867077, 0x30476dc0, 0x3d044b19, 0x39c556ae,
            0x278206ab, 0x23431b1c, 0x2e003dc5, 0x2ac12072,
            0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16,
            0x018aeb13, 0x054bf6a4, 0x0808d07d, 0x0cc9cdca,
            0x7897ab07, 0x7c56b6b0, 0x71159069, 0x75d48dde,
            0x6b93dddb, 0x6f52c06c, 0x6211e6b5, 0x66d0fb02,
            0x5e9f46bf, 0x5a5e5b08, 0x571d7dd1, 0x53dc6066,
            0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba,
            0xaca5c697, 0xa864db20, 0xa527fdf9, 0xa1e6e04e,
            0xbfa1b04b, 0xbb60adfc, 0xb6238b25, 0xb2e29692,
            0x8aad2b2f, 0x8e6c3698, 0x832f1041, 0x87ee0df6,
            0x99a95df3, 0x9d684044, 0x902b669d, 0x94ea7b2a,
            0xe0b41de7, 0xe4750050, 0xe9362689, 0xedf73b3e,
            0xf3b06b3b, 0xf771768c, 0xfa325055, 0xfef34de2,
            0xc6bcf05f, 0xc27dede8, 0xcf3ecb31, 0xcbffd686,
            0xd5b88683, 0xd1799b34, 0xdc3abded, 0xd8fba05a,
            0x690ce0ee, 0x6dcdfd59, 0x608edb80, 0x644fc637,
            0x7a089632, 0x7ec98b85, 0x738aad5c, 0x774bb0eb,
            0x4f040d56, 0x4bc510e1, 0x46863638, 0x42472b8f,
            0x5c007b8a, 0x58c1663d, 0x558240e4, 0x51435d53,
            0x251d3b9e, 0x21dc2629, 0x2c9f00f0, 0x285e1d47,
            0x36194d42, 0x32d850f5, 0x3f9b762c, 0x3b5a6b9b,
            0x0315d626, 0x07d4cb91, 0x0a97ed48, 0x0e56f0ff,
            0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623,
            0xf12f560e, 0xf5ee4bb9, 0xf8ad6d60, 0xfc6c70d7,
            0xe22b20d2, 0xe6ea3d65, 0xeba91bbc, 0xef68060b,
            0xd727bbb6, 0xd3e6a601, 0xdea580d8, 0xda649d6f,
            0xc423cd6a, 0xc0e2d0dd, 0xcda1f604, 0xc960ebb3,
            0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7,
            0xae3afba2, 0xaafbe615, 0xa7b8c0cc, 0xa379dd7b,
            0x9b3660c6, 0x9ff77d71, 0x92b45ba8, 0x9675461f,
            0x8832161a, 0x8cf30bad, 0x81b02d74, 0x857130c3,
            0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640,
            0x4e8ee645, 0x4a4ffbf2, 0x470cdd2b, 0x43cdc09c,
            0x7b827d21, 0x7f436096, 0x7200464f, 0x76c15bf8,
            0x68860bfd, 0x6c47164a, 0x61043093, 0x65c52d24,
            0x119b4be9, 0x155a565e, 0x18197087, 0x1cd86d30,
            0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec,
            0x3793a651, 0x3352bbe6, 0x3e119d3f, 0x3ad08088,
            0x2497d08d, 0x2056cd3a, 0x2d15ebe3, 0x29d4f654,
            0xc5a92679, 0xc1683bce, 0xcc2b1d17, 0xc8ea00a0,
            0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb, 0xdbee767c,
            0xe3a1cbc1, 0xe760d676, 0xea23f0af, 0xeee2ed18,
            0xf0a5bd1d, 0xf464a0aa, 0xf9278673, 0xfde69bc4,
            0x89b8fd09, 0x8d79e0be, 0x803ac667, 0x84fbdbd0,
            0x9abc8bd5, 0x9e7d9662, 0x933eb0bb, 0x97ffad0c,
            0xafb010b1, 0xab710d06, 0xa6322bdf, 0xa2f33668,
            0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4
        };

        private int globalCrc;

        public CRC() {
            init();
        }

        void init() {
            globalCrc = 0xffffffff;
        }

        /** final CRC */
        int get() {
            return ~globalCrc;
        }

        void update(int inCh) {
            int temp = (globalCrc >> 24) ^ inCh;
            if (temp < 0) {
                temp = 256 + temp;
            }
            globalCrc = (globalCrc << 8) ^ crc32Table[temp];
        }
        void update(int inCh, int n) {
            int globalCrc = this.globalCrc;
            for (int i = 0; i < n; i++) {
                int temp = (globalCrc >> 24) ^ inCh;
                if (temp < 0)
                    temp = 256 + temp;
                globalCrc = (globalCrc << 8) ^ crc32Table[temp];
            }
            this.globalCrc = globalCrc;
        }

    }
}
