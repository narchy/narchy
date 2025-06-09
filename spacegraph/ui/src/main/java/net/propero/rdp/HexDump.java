/* HexDump.java
 * Component: ProperJavaRDP
 *
 * Revision: $Revision: #2 $
 * Author: $Author: tvkelley $
 * Date: $Date: 2009/09/15 $
 *
 * Copyright (c) 2005 Propero Limited
 *
 * Purpose: Manages debug information for all data
 *          sent and received, outputting in hex format
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 * USA
 *
 * (See gpl.txt for details of the GNU General Public License.)
 *
 */
package net.propero.rdp;


enum HexDump { ;

    /**
     * Encode data as hex and output as debug messages along with supplied
     * custom message
     *
     * @param data Array of byte data to be encoded
     * @param msg  Message to include with outputted hex debug messages
     */
    public static void encode(byte[] data, String msg/* PrintStream out */) {
        int count = 0;

        while (count < data.length) {
            String index = Integer.toHexString(count);
            switch (index.length()) {
                case (1):
                    index = "0000000" + index;
                    break;
                case (2):
                    index = "000000" + index;
                    break;
                case (3):
                    index = "00000" + index;
                    break;
                case (4):
                    index = "0000" + index;
                    break;
                case (5):
                    index = "000" + index;
                    break;
                case (6):
                    index = "00" + index;
                    break;
                case (7):
                    index = "0" + index;
                    break;
                case (8):
                    break;
                default:
                    return;
            }

            index += ": ";
            
            for (int i = 0; i < 16; i++) {
                if (count >= data.length) {
                    break;
                }
                String number = Integer.toHexString((data[count] & 0x000000ff));
                switch (number.length()) {
                    case (1):
                        number = "0" + number;
                        break;
                    case (2):
                        break;
                    default:

                        
                        return;
                }
                index += (number + ' ');
                
                count++;
            }

            
        }

    }
}
