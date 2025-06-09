/* Secure.java
 * Component: ProperJavaRDP
 *
 * Revision: $Revision: #2 $
 * Author: $Author: tvkelley $
 * Date: $Date: 2009/09/15 $
 *
 * Copyright (c) 2005 Propero Limited
 * Copyright (c) 2008 IsmAvatar <cmagicj@nni.com>
 *
 * Purpose: Secure layer of communication
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

import net.propero.rdp.crypto.CryptoException;
import net.propero.rdp.crypto.RC4;
import net.propero.rdp.rdp5.VChannels;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.math.BigInteger;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.StringTokenizer;

public class Secure {
    /* constants for the secure layer */
    public static final int SEC_ENCRYPT = 0x0008;
    public static final int SEC_LOGON_INFO = 0x0040;
    private static final Logger logger = LoggerFactory.getLogger(Secure.class);
    static final int SEC_RANDOM_SIZE = 32;
    static final int SEC_MODULUS_SIZE = 64;
    private static final int SEC_MAX_MODULUS_SIZE = 256;
    static final int SEC_PADDING_SIZE = 8;
    static final int SEC_LICENCE_NEG = 0x0080;
    private static final int SEC_EXPONENT_SIZE = 4;
    private static final int SEC_CLIENT_RANDOM = 0x0001;
    private static final int SEC_TAG_SRV_INFO = 0x0c01;
    private static final int SEC_TAG_SRV_CRYPT = 0x0c02;
    private static final int SEC_TAG_SRV_3 = 0x0c03;
    private static final int SEC_TAG_SRV_CHANNELS = 0x0c03;
    private static final int SEC_TAG_CLI_INFO = 0xc001;
    private static final int SEC_TAG_CLI_CRYPT = 0xc002;
    private static final int SEC_TAG_CLI_CHANNELS = 0xc003;
    private static final int SEC_TAG_CLI_4 = 0xc004;
    private static final int SEC_TAG_PUBKEY = 0x0006;
    private static final int SEC_TAG_KEYSIG = 0x0008;
    private static final int SEC_RSA_MAGIC = 0x31415352; /* RSA1 */
    private static final int SEC_CC_REDIRECTION_SUPPORTED = 0x00000001;
    private static final int SEC_CC_REDIRECT_SESSIONID_FIELD_VALID = 0x00000002;
    private static final int SEC_CC_REDIRECTED_SMARTCARD = 0x00000040;
    private static final int SEC_CC_REDIRECT_VERSION_MASK = 0x0000003c;
    private static final int SEC_CC_REDIRECT_VERSION_3 = 0x02;
    private static final int SEC_CC_REDIRECT_VERSION_4 = 0x03;
    private static final int SEC_CC_REDIRECT_VERSION_5 = 0x04;
    private static final int SEC_CC_REDIRECT_VERSION_6 = 0x05;
    private static final byte[] pad_54 = {54, 54, 54, 54, 54, 54, 54, 54, 54,
            54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
            54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54};
    private static final byte[] pad_92 = {92, 92, 92, 92, 92, 92, 92, 92, 92,
            92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
            92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
            92, 92, 92, 92, 92};
    private final Licence licence = new Licence(this);
    private final MCS McsLayer;
    private final RC4 rc4_enc;

    private final Object rc4_enc_lock = new Object();

    private final RC4 rc4_dec;

    private final Object rc4_dec_lock = new Object();

    private final RC4 rc4_update;




    private final Object digestLock = new Object();
    private final byte[] sec_sign_key;
    private final byte[] sec_decrypt_update_key;
    private final byte[] sec_encrypt_update_key;
    private final byte[] client_random = new byte[SEC_RANDOM_SIZE];
    private final VChannels channels;
    private boolean readCert;
    
    
    boolean licenceIssued;
    private MessageDigest sha1;
    private MessageDigest md5;
    private int keylength;
    private int enc_count;
    private int dec_count;
    private int server_public_key_len;
    private byte[] sec_decrypt_key;
    private byte[] sec_encrypt_key;
    private byte[] sec_crypted_random;
    private byte[] exponent;
    private byte[] modulus;
    private byte[] server_random;

    /**
     * Initialise Secure layer of communications
     *
     * @param channels Virtual channels for this connection
     */
    public Secure(VChannels channels) {
        this.channels = channels;
        McsLayer = new MCS(channels);
        Common.mcs = McsLayer;
        rc4_dec = new RC4();
        rc4_enc = new RC4();
        rc4_update = new RC4();



        try {
            sha1 = MessageDigest.getInstance("SHA-1");
            md5 = MessageDigest.getInstance("MD5");
        } catch (Exception e) {
            e.printStackTrace();
        }

        sec_sign_key = new byte[16]; 
        sec_decrypt_key = new byte[16];
        sec_encrypt_key = new byte[16];
        sec_decrypt_update_key = new byte[16]; 
        
        sec_encrypt_update_key = new byte[16]; 
        
        sec_crypted_random = new byte[64];

    }

    /**
     * Read server info from packet, specifically the RDP version of the server
     *
     * @param mcs_data Packet to read
     */
    private static void processSrvInfo(RdpPacket_Localised mcs_data) {
        Options.server_rdp_version = mcs_data.getLittleEndian16(); 
        
        logger.debug("Server RDP version {}", Options.server_rdp_version);
        if (1 == Options.server_rdp_version)
            Options.use_rdp5 = false;
    }

    /**
     * Reverse the values in the provided array
     *
     * @param data Array as passed reversed on return
     */
    private static void reverse(byte[] data) {
        int i = 0, j = 0;
        byte temp = 0;

        for (i = 0, j = data.length - 1; i < j; i++, j--) {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    public static void reverse(byte[] data, int length) {
        int i = 0, j = 0;
        byte temp = 0;

        for (i = 0, j = length - 1; i < j; i++, j--) {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    /**
     * Generate a 40-bit key and store in the parameter key.
     *
     * @param key
     */
    private static void make40bit(byte[] key) {
        key[0] = (byte) 0xd1;
        key[1] = 0x26;
        key[2] = (byte) 0x9e;
    }

    /**
     * Write a 32-bit integer value to an array of bytes, length 4
     *
     * @param data  Modified by method to be a 4-byte array representing the
     *              parameter value
     * @param value Integer value to return as a little-endian 32-bit value
     */
    public static void setLittleEndian32(byte[] data, int value) {

        data[3] = (byte) ((value >>> 24) & 0xff);
        data[2] = (byte) ((value >>> 16) & 0xff);
        data[1] = (byte) ((value >>> 8) & 0xff);
        data[0] = (byte) (value & 0xff);
    }

    /**
     * Connect to server
     *
     * @param host Address of server to connect to
     * @param port Port to connect to
     * @throws UnknownHostException
     * @throws IOException
     * @throws RdesktopException
     * @throws SocketException
     * @throws CryptoException
     * @throws OrderException
     */
    private void connect(InetAddress host, int port)
            throws IOException, RdesktopException,
            CryptoException, OrderException, UnknownHostException {
        if (Options.hostname.isEmpty()) {
            InetAddress localhost = InetAddress.getLocalHost();
            String name = localhost.getHostName();
            StringTokenizer tok = new StringTokenizer(name, ".");
            Options.hostname = tok.nextToken();
            Options.hostname.trim();
        }

        RdpPacket_Localised mcs_data = this.sendMcsData();
        McsLayer.connect(host, port, mcs_data);

        this.processMcsData(mcs_data);
        if (Constants.encryption) {
            this.establishKey();
        }
    }

    /**
     * Connect to server on default port
     *
     * @param host Server to connect to
     * @throws IOException
     * @throws RdesktopException
     * @throws OrderException
     * @throws CryptoException
     */
    public void connect(InetAddress host) throws IOException,
            RdesktopException, OrderException, CryptoException {
        this.connect(host, Options.port);
    }

    /**
     * Close connection
     */
    public void disconnect() {
        McsLayer.disconnect();
    }

    /**
     * Construct MCS data, including channel, encryption and display options
     *
     * @return Packet populated with MCS data
     */
    private RdpPacket_Localised sendMcsData() {
        logger.debug("Secure.sendMcsData");

        RdpPacket_Localised buffer = new RdpPacket_Localised(512);

        int hostlen = 2 * (Options.hostname == null ? 0 : Options.hostname
                .length());

        if (hostlen > 30) {
            hostlen = 30;
        }

        int length = 162 + 76 + 12 + 4;

        if (Options.use_rdp5 && (channels.num_channels() > 0))
            length += channels.num_channels() * 12 + 8;

        buffer.setBigEndian16(5); /* unknown */
        buffer.setBigEndian16(0x14);
        buffer.set8(0x7c);
        buffer.setBigEndian16(1);

        buffer.setBigEndian16(length | 0x8000); 

        buffer.setBigEndian16(8); 
        buffer.setBigEndian16(16);
        buffer.set8(0);
        buffer.setLittleEndian16(0xc001);
        buffer.set8(0);

        buffer.setLittleEndian32(0x61637544); 
        buffer.setBigEndian16(length - 14 | 0x8000); 

        
        buffer.setLittleEndian16(SEC_TAG_CLI_INFO);
        buffer.setLittleEndian16(216); 
        buffer.setLittleEndian16(Options.use_rdp5 ? 4 : 1);
        buffer.setLittleEndian16(8);
        buffer.setLittleEndian16(Options.width);
        buffer.setLittleEndian16(Options.height);
        buffer.setLittleEndian16(0xca01);
        buffer.setLittleEndian16(0xaa03);
        buffer.setLittleEndian32(Options.keylayout);
        buffer.setLittleEndian32(2600); 
        
        
        
        
        

        /* Unicode name of client, padded to 32 bytes */
        buffer.outUnicodeString(Options.hostname, hostlen);
        buffer.incrementPosition(30 - hostlen);

        buffer.setLittleEndian32(4);
        buffer.setLittleEndian32(0);
        buffer.setLittleEndian32(12);
        buffer.incrementPosition(64); /* reserved? 4 + 12 doublewords */

        buffer.setLittleEndian16(0xca01); 
        buffer.setLittleEndian16(1);

        buffer.setLittleEndian32(0); 
        buffer.set8(Options.server_bpp); 
        buffer.setLittleEndian16(0x0700); 
        buffer.set8(0); 
        buffer.setLittleEndian32(1); 
        buffer.incrementPosition(64);
        buffer.setLittleEndian32(0);

        buffer.setLittleEndian16(SEC_TAG_CLI_4); 
        buffer.setLittleEndian16(12); 

        int cluster_flags = 0;
        cluster_flags |= SEC_CC_REDIRECTION_SUPPORTED;
        cluster_flags |= (SEC_CC_REDIRECT_VERSION_3 << 2);
        if (Options.console_session) {
            cluster_flags |= SEC_CC_REDIRECT_SESSIONID_FIELD_VALID;
        }
        buffer.setLittleEndian32(cluster_flags); 
        buffer.setLittleEndian32(0); 

        
        buffer.setLittleEndian16(SEC_TAG_CLI_CRYPT);
        buffer.setLittleEndian16(12); 
        buffer.setLittleEndian32(Options.encryption ? 0x3 : 0);
        buffer.setLittleEndian32(0); 

        if (Options.use_rdp5 && (channels.num_channels() > 0)) {
            logger.debug("num_channels {}",channels.num_channels());
            buffer.setLittleEndian16(SEC_TAG_CLI_CHANNELS); 
            
            buffer.setLittleEndian16(channels.num_channels() * 12 + 8); 
            buffer.setLittleEndian32(channels.num_channels()); 
            
            
            
            
            for (int i = 0; i < channels.num_channels(); i++) {
                logger.debug("Requesting channel {}", channels.channel(i).name());
                buffer.out_uint8p(channels.channel(i).name(), 8); 
                
                
                buffer.setBigEndian32(channels.channel(i).flags()); 
                
            }
        }

        buffer.markEnd();
        return buffer;
    }

    /**
     * Handle MCS info from server (server info, encryption info and channel
     * information)
     *
     * @param mcs_data Data received from server
     */
    public void processMcsData(RdpPacket_Localised mcs_data)
            throws RdesktopException, CryptoException {
        logger.debug("Secure.processMcsData");

        mcs_data.incrementPosition(21);
        int len = mcs_data.get8();

        if ((len & 0x00000080) != 0) {
            len = mcs_data.get8();
        }

        int nexttag = 0;
        int length = 0;
        int tag = 0;
        while (mcs_data.getPosition() < mcs_data.getEnd()) {
            tag = mcs_data.getLittleEndian16();
            length = mcs_data.getLittleEndian16();

            if (length <= 4)
                return;

            nexttag = mcs_data.getPosition() + length - 4;

            switch (tag) {
                case (Secure.SEC_TAG_SRV_INFO):
                    processSrvInfo(mcs_data);
                    break;
                case (Secure.SEC_TAG_SRV_CRYPT):
                    this.processCryptInfo(mcs_data);
                    break;
                case (Secure.SEC_TAG_SRV_CHANNELS):
                    /*
                     * FIXME: We should parse this information and use it to map
                     * RDP5 channels to MCS channels
                     */
                    break;

                default:
                    throw new RdesktopException("Not implemented! Tag:" + tag
                            + "not recognized!");
            }

            mcs_data.setPosition(nexttag);
        }
    }

    private void establishKey() throws RdesktopException, IOException,
            CryptoException {
        RdpPacket_Localised buffer;
        int flags = SEC_CLIENT_RANDOM;
        if (readCert) {

            buffer = this.init(flags, 76);

            int length = SEC_MODULUS_SIZE + SEC_PADDING_SIZE;
            buffer.setLittleEndian32(length);

            buffer.copyFromByteArray(this.sec_crypted_random, 0, buffer.getPosition(), SEC_MODULUS_SIZE);
            buffer.incrementPosition(SEC_MODULUS_SIZE);
        } else {
            int length = server_public_key_len + SEC_PADDING_SIZE;
            buffer = this.init(flags, length + 4);

            buffer.setLittleEndian32(length);

            buffer.copyFromByteArray(this.sec_crypted_random, 0, buffer
                    .getPosition(), server_public_key_len);
            buffer.incrementPosition(server_public_key_len);
        }
        buffer.incrementPosition(SEC_PADDING_SIZE);
        buffer.markEnd();
        this.send(buffer, flags);
    }

    private void processCryptInfo(RdpPacket_Localised data)
            throws RdesktopException, CryptoException {

        int rc4_key_size = this.parseCryptInfo(data);
        if (rc4_key_size == 0) {
            return;
        }

        
        logger.debug("readCert = {}", readCert);
        if (readCert) { /*
         * Which means we should use RDP5-style encryption
         */

            
            

            
            
            /*
             * this.exponent =
             * this.server_public_key.getPublicExponent().toByteArray();
             * this.modulus = this.server_public_key.getModulus().toByteArray();
             *
             * System.out.println("Exponent: " +
             * server_public_key.getPublicExponent());
             * System.out.println("Modulus: " + server_public_key.getModulus());
             */

            
            
            
            
            
            
            
        } else {
            this.generateRandom();
            this.RSAEncrypt(SEC_RANDOM_SIZE, server_public_key_len);
        }
        this.generate_keys(rc4_key_size);
    }

    /**
     * Intialise a packet at the Secure layer
     *
     * @param flags  Encryption flags
     * @param length Length of packet
     * @return Intialised packet
     * @throws RdesktopException
     */
    public RdpPacket_Localised init(int flags, int length) throws RdesktopException {
        int headerlength = 0;

        if (!this.licenceIssued)
            headerlength = ((flags & SEC_ENCRYPT) != 0) ? 12 : 4;
        else
            headerlength = ((flags & SEC_ENCRYPT) != 0) ? 12 : 0;

        RdpPacket_Localised buffer = MCS.init(length + headerlength);
        buffer.pushLayer(RdpPacket.SECURE_HEADER, headerlength);
        
        
        
        return buffer;
    }

    /**
     * Send secure data on the global channel
     *
     * @param sec_data Data to send
     * @param flags    Encryption flags
     * @throws RdesktopException
     * @throws IOException
     * @throws CryptoException
     */
    public void send(RdpPacket_Localised sec_data, int flags)
            throws RdesktopException, IOException, CryptoException {
        send_to_channel(sec_data, flags, MCS.MCS_GLOBAL_CHANNEL);
    }

    /**
     * Prepare data as a Secure PDU and pass down to the MCS layer
     *
     * @param sec_data Data to send
     * @param flags    Encryption flags
     * @param channel  Channel over which to send data
     * @throws RdesktopException
     * @throws IOException
     * @throws CryptoException
     */
    public void send_to_channel(RdpPacket_Localised sec_data, int flags,
                                int channel) throws RdesktopException, IOException, CryptoException {

        sec_data.setPosition(sec_data.getHeader(RdpPacket.SECURE_HEADER));

        if (!this.licenceIssued || (flags & SEC_ENCRYPT) != 0) {
            sec_data.setLittleEndian32(flags);
        }

        if (Options.debug_hexdump) {
            int length = sec_data.getEnd() - sec_data.getPosition();
            byte[] packet = new byte[length];
            sec_data.copyToByteArray(packet, 0, sec_data.getPosition(), sec_data.getEnd() - sec_data.getPosition());
            System.out.println("Sending packet:");
            System.out.println(net.propero.rdp.tools.HexDump.dumpHexString(packet));
        }

        if ((flags & SEC_ENCRYPT) != 0) {
            flags &= ~SEC_ENCRYPT;
            int datalength = sec_data.getEnd() - sec_data.getPosition() - 8;
            byte[] data = new byte[datalength];
            sec_data.copyToByteArray(data, 0, sec_data.getPosition() + 8,
                    datalength);
            byte[] signature = this.sign(this.sec_sign_key, 8, this.keylength, data,
                    datalength);

            byte[] buffer = this.encrypt(data, datalength);

            sec_data.copyFromByteArray(signature, 0, sec_data.getPosition(), 8);
            sec_data.copyFromByteArray(buffer, 0, sec_data.getPosition() + 8,
                    datalength);

        }
        
        McsLayer.send_to_channel(sec_data, channel);
    }

    /**
     * Generate MD5 signature
     *
     * @param session_key Key with which to sign data
     * @param length      Length of signature
     * @param keylen      Length of key
     * @param data        Data to sign
     * @param datalength  Length of data to sign
     * @return Signature for data
     * @throws CryptoException
     */
    public byte[] sign(byte[] session_key, int length, int keylen, byte[] data,
                       int datalength) {
        byte[] lenhdr = new byte[4];

        Secure.setLittleEndian32(lenhdr, datalength);

        byte[] signature = new byte[length];
        synchronized (digestLock) {
            sha1.reset();
            sha1.update(session_key, 0, keylen);
            sha1.update(pad_54, 0, 40);
            sha1.update(lenhdr, 0, 4);
            sha1.update(data, 0, datalength);
            byte[] shasig = sha1.digest();
            sha1.reset();

            md5.reset();
            md5.update(session_key, 0, keylen/* length */);
            md5.update(pad_92, 0, 48);
            md5.update(shasig, 0, 20);
            byte[] md5sig = md5.digest();
            md5.reset();

            System.arraycopy(md5sig, 0, signature, 0, length);
        }
        return signature;
    }

    /**
     * Encrypt specified number of bytes from provided data using RC4 algorithm
     *
     * @param data   Data to encrypt
     * @param length Number of bytes to encrypt (from start of array)
     * @return Encrypted data
     * @throws CryptoException
     */
    private byte[] encrypt(byte[] data, int length) throws CryptoException {
        synchronized (rc4_enc_lock) {
            if (this.enc_count == 4096) {
                sec_encrypt_key = this.update(this.sec_encrypt_key,
                        this.sec_encrypt_update_key);
                byte[] key = new byte[this.keylength];
                System.arraycopy(this.sec_encrypt_key, 0, key, 0, this.keylength);
                this.rc4_enc.engineInitEncrypt(key);
                
                this.enc_count = 0;
            }

            byte[] buffer = this.rc4_enc.crypt(data, 0, length);
            this.enc_count++;
            return buffer;
        }
    }

    /**
     * Encrypt provided data using the RC4 algorithm
     *
     * @param data Data to encrypt
     * @return Encrypted data
     * @throws CryptoException
     */
    public byte[] encrypt(byte[] data) throws CryptoException {
        synchronized (rc4_enc_lock) {
            if (this.enc_count == 4096) {
                sec_encrypt_key = this.update(this.sec_encrypt_key,
                        this.sec_encrypt_update_key);
                byte[] key = new byte[this.keylength];
                System.arraycopy(this.sec_encrypt_key, 0, key, 0, this.keylength);
                this.rc4_enc.engineInitEncrypt(key);
                
                this.enc_count = 0;
            }


            byte[] buffer = this.rc4_enc.crypt(data);
            this.enc_count++;
            return buffer;
        }
    }

    /**
     * Decrypt specified number of bytes from provided data using RC4 algorithm
     *
     * @param data   Data to decrypt
     * @param length Number of bytes to decrypt (from start of array)
     * @return Decrypted data
     * @throws CryptoException
     */
    public byte[] decrypt(byte[] data, int length) throws CryptoException {
        synchronized (rc4_dec_lock) {
            if (this.dec_count == 4096) {
                sec_decrypt_key = this.update(this.sec_decrypt_key,
                        this.sec_decrypt_update_key);
                byte[] key = new byte[this.keylength];
                System.arraycopy(this.sec_decrypt_key, 0, key, 0, this.keylength);
                this.rc4_dec.engineInitDecrypt(key);
                
                this.dec_count = 0;
            }

            byte[] buffer = this.rc4_dec.crypt(data, 0, length);
            this.dec_count++;
            return buffer;
        }
    }

    /**
     * Decrypt provided data using RC4 algorithm
     *
     * @param data Data to decrypt
     * @return Decrypted data
     * @throws CryptoException
     */
    public byte[] decrypt(byte[] data) throws CryptoException {
        synchronized (rc4_dec_lock) {
            if (this.dec_count == 4096) {
                sec_decrypt_key = this.update(this.sec_decrypt_key,
                        this.sec_decrypt_update_key);
                byte[] key = new byte[this.keylength];
                System.arraycopy(this.sec_decrypt_key, 0, key, 0, this.keylength);
                this.rc4_dec.engineInitDecrypt(key);
                
                this.dec_count = 0;
            }


            byte[] buffer = this.rc4_dec.crypt(data);
            this.dec_count++;
            return buffer;
        }
    }

    /**
     * Read encryption information from a Secure layer PDU, obtaining and
     * storing level of encryption and any keys received
     *
     * @param data Packet to read encryption information from
     * @return Size of RC4 key
     * @throws RdesktopException
     */
    private int parseCryptInfo(RdpPacket_Localised data)
            throws RdesktopException {
        logger.debug("Secure.parseCryptInfo");

        int rc4_key_size = data.getLittleEndian32();
        int encryption_level = data.getLittleEndian32();
        
        if (encryption_level == 0) { 
            return 0;
        }
        int random_length = data.getLittleEndian32();
        int RSA_info_length = data.getLittleEndian32();

        if (random_length != SEC_RANDOM_SIZE) {
            throw new RdesktopException("Wrong Size of Random! Got"
                    + random_length + "expected" + SEC_RANDOM_SIZE);
        }
        this.server_random = new byte[random_length];
        data.copyToByteArray(this.server_random, 0, data.getPosition(),
                random_length);
        data.incrementPosition(random_length);

        int end = data.getPosition() + RSA_info_length;

        if (end > data.getEnd()) {
            logger.debug("Reached end of crypt info prematurely ");
            return 0;
        }

        
        int flags = data.getLittleEndian32(); 
        
        
        logger.debug("Flags = 0x{}", Integer.toHexString(flags));
        if ((flags & 1) != 0) {
            logger.debug(("We're going for the RDP4-style encryption"));
            data.incrementPosition(8);

            int next_tag = 0;
            int length = 0;
            int tag = 0;
            while (data.getPosition() < data.getEnd()) {
                tag = data.getLittleEndian16();
                length = data.getLittleEndian16();

                next_tag = data.getPosition() + length;

                switch (tag) {

                    case (Secure.SEC_TAG_PUBKEY):

                        if (!parsePublicKey(data)) {
                            return 0;
                        }

                        break;
                    case (Secure.SEC_TAG_KEYSIG):
                        
                        break;

                    default:
                        throw new RdesktopException("Unimplemented decrypt tag "
                                + tag);
                }
                data.setPosition(next_tag);
            }

            if (data.getPosition() == data.getEnd()) {
                return rc4_key_size;
            } else {
                logger.warn("End not reached!");
                return 0;
            }

        } else {
            logger.debug(("We're going for the RDP5-style encryption"));
            
            int num_certs = data.getLittleEndian32();

            int cacert_len = data.getLittleEndian32();
            data.incrementPosition(cacert_len);
            int cert_len = data.getLittleEndian32();
            data.incrementPosition(cert_len);

            readCert = true;

            return rc4_key_size;
        }

    }

    /*
     * public X509Certificate readCert(int length, RdpPacket_Localised data){
     * byte[] buf = new byte[length];
     *
     * data.copyToByteArray(buf,0,data.getPosition(),buf.length);
     * data.incrementPosition(length);
     *
     * for(int i = 0; i < buf.length; i++){ buf[i] = (byte) (buf[i] & 0xFF); }
     *
     * ByteArrayInputStream bIn = new ByteArrayInputStream(buf); X509Certificate
     * cert = null; CertificateFactory cf = null; try { cf =
     * CertificateFactory.getInstance("X.509"); cert =
     * (X509Certificate)cf.generateCertificate(bIn); } catch
     * (CertificateException e) { 
     * e.printStackTrace(); }
     *
     * bIn.reset(); return cert; }
     */
    private void generateRandom() {

        try {
            SecureRandom random = SecureRandom.getInstance("SHA1PRNG");
            random.nextBytes(this.sec_crypted_random);
            random.nextBytes(this.client_random);
        } catch (NoSuchAlgorithmException e) {
            logger.warn("No Such Random Algorithm");
        }

    }

    private void RSAEncrypt(int length, int modulus_size)
            throws RdesktopException {

        Secure.reverse(this.exponent);
        Secure.reverse(this.modulus);
        byte[] inr = new byte[length];
        System.arraycopy(this.client_random, 0, inr, 0, length);
        Secure.reverse(inr);

        BigInteger mod = null;
        if ((this.modulus[0] & 0x80) != 0) {
            byte[] temp = new byte[this.modulus.length + 1];
            System.arraycopy(this.modulus, 0, temp, 1, this.modulus.length);
            temp[0] = 0;
            mod = new BigInteger(temp);
        } else {
            mod = new BigInteger(this.modulus);
        }
        BigInteger exp = null;
        if ((this.exponent[0] & 0x80) != 0) {
            byte[] temp = new byte[this.exponent.length + 1];
            System.arraycopy(this.exponent, 0, temp, 1, this.exponent.length);
            temp[0] = 0;
            exp = new BigInteger(temp);
        } else {
            exp = new BigInteger(this.exponent);
        }
        BigInteger x = null;
        if ((inr[0] & 0x80) != 0) {
            byte[] temp = new byte[inr.length + 1];
            System.arraycopy(inr, 0, temp, 1, inr.length);
            temp[0] = 0;
            x = new BigInteger(temp);
        } else {
            x = new BigInteger(inr);
        }

        BigInteger y = x.modPow(exp, mod);
        this.sec_crypted_random = y.toByteArray();

        if ((this.sec_crypted_random[0] & 0x80) != 0) {
            throw new RdesktopException(
                    "Wrong Sign! Expected positive Integer!");
        }

        if (this.sec_crypted_random.length > SEC_MAX_MODULUS_SIZE) {
            logger.warn("sec_crypted_random too big!"); /* FIXME */
        }
        Secure.reverse(this.sec_crypted_random);

        if (this.sec_crypted_random.length < modulus_size) {
            byte[] temp = new byte[SEC_MAX_MODULUS_SIZE];
            System.arraycopy(this.sec_crypted_random, 0, temp, 0,
                    this.sec_crypted_random.length);
            for (int i = this.sec_crypted_random.length; i < temp.length; i++) {
                temp[i] = 0;
            }
            this.sec_crypted_random = temp;

        }

    }

    /**
     * Read in a public key from a provided Secure layer PDU, and store in
     * this.exponent and this.modulus
     *
     * @param data Secure layer PDU containing key data
     * @return True if key successfully read
     * @throws RdesktopException
     */
    private boolean parsePublicKey(RdpPacket_Localised data)
            throws RdesktopException {

        int magic = data.getLittleEndian32();

        if (magic != SEC_RSA_MAGIC) {
            throw new RdesktopException("Wrong magic! Expected" + SEC_RSA_MAGIC
                    + "got:" + magic);
        }

        int modulus_length = data.getLittleEndian32() - SEC_PADDING_SIZE;

        if (modulus_length < 64 || modulus_length > SEC_MAX_MODULUS_SIZE) {
            throw new RdesktopException("Bad server public key size ("
                    + (modulus_length * 8) + " bites)");
        }

        data.incrementPosition(8); 
        this.exponent = new byte[SEC_EXPONENT_SIZE];
        data.copyToByteArray(this.exponent, 0, data.getPosition(),
                SEC_EXPONENT_SIZE);
        data.incrementPosition(SEC_EXPONENT_SIZE);
        this.modulus = new byte[modulus_length];
        data.copyToByteArray(this.modulus, 0, data.getPosition(),
                modulus_length);
        data.incrementPosition(modulus_length);
        data.incrementPosition(SEC_PADDING_SIZE);
        this.server_public_key_len = modulus_length;

        return data.getPosition() <= data.getEnd();
    }

    public byte[] hash48(byte[] in, byte[] salt1, byte[] salt2, int salt) {
        byte[] out = new byte[48];

        synchronized (digestLock) {
            sha1.reset();
            md5.reset();
            int i = 0;
            byte[] pad = new byte[4];
            byte[] shasig = new byte[20];
            for (i = 0; i < 3; i++) {
                for (int j = 0; j <= i; j++) {
                    pad[j] = (byte) (salt + i);
                }
                sha1.update(pad, 0, i + 1);
                sha1.update(in, 0, 48);
                sha1.update(salt1, 0, 32);
                sha1.update(salt2, 0, 32);
                shasig = sha1.digest();
                sha1.reset();

                md5.update(in, 0, 48);
                md5.update(shasig, 0, 20);
                System.arraycopy(md5.digest(), 0, out, i * 16, 16);
            }
        }

        return out;
    }

    public byte[] hash16(byte[] in, byte[] salt1, byte[] salt2, int in_position) {
        synchronized (digestLock) {
            md5.reset();
            md5.update(in, in_position, 16);
            md5.update(salt1, 0, 32);
            md5.update(salt2, 0, 32);
            return md5.digest();
        }
    }

    /**
     * @param key
     * @param update_key
     * @return
     * @throws CryptoException
     */
    private byte[] update(byte[] key, byte[] update_key) throws CryptoException {
        byte[] update = new byte[this.keylength];
        
        byte[] thekey = new byte[key.length];

        synchronized (digestLock) {
            sha1.reset();
            sha1.update(update_key, 0, keylength);
            sha1.update(pad_54, 0, 40);
            sha1.update(key, 0, keylength);

            byte[] shasig = sha1.digest();
            sha1.reset();

            md5.reset();
            md5.update(update_key, 0, keylength); 
            
            md5.update(pad_92, 0, 48);
            md5.update(shasig, 0, 20);
            thekey = md5.digest();
            md5.reset();

            System.arraycopy(thekey, 0, update, 0, this.keylength);
            rc4_update.engineInitDecrypt(update);
            
            thekey = rc4_update.crypt(thekey, 0, this.keylength);

            if (this.keylength == 8) {
                Secure.make40bit(thekey);
            }
        }


        return thekey;
    }

    /**
     * Receive a Secure layer PDU from the MCS layer
     *
     * @return Packet representing received Secure PDU
     * @throws RdesktopException
     * @throws IOException
     * @throws CryptoException
     * @throws OrderException
     */
    public RdpPacket_Localised receive() throws RdesktopException, IOException,
            CryptoException, OrderException {
        int sec_flags = 0;
        RdpPacket_Localised buffer = null;
        while (true) {
            int[] channel = new int[1];
            buffer = McsLayer.receive(channel);
            if (buffer == null)
                return null;
            buffer.setHeader(RdpPacket.SECURE_HEADER);
            if (Constants.encryption || (!this.licenceIssued)) {

                sec_flags = buffer.getLittleEndian32();

                if ((sec_flags & SEC_LICENCE_NEG) != 0) {
                    licence.process(buffer);
                    continue;
                }
                if ((sec_flags & SEC_ENCRYPT) != 0) {
                    buffer.incrementPosition(8); 
                    byte[] data = new byte[buffer.size() - buffer.getPosition()];
                    buffer.copyToByteArray(data, 0, buffer.getPosition(),
                            data.length);
                    byte[] packet = this.decrypt(data);

                    buffer.copyFromByteArray(packet, 0, buffer.getPosition(),
                            packet.length);

                    
                    
                }
            }

            if (channel[0] != MCS.MCS_GLOBAL_CHANNEL) {
                channels.channel_process(buffer, channel[0]);
                continue;
            }

            buffer.setStart(buffer.getPosition());
            return buffer;
        }
    }

    /**
     * Generate encryption keys of applicable size for connection
     *
     * @param rc4_key_size Size of keys to generate (1 if 40-bit encryption, otherwise
     *                     128-bit)
     * @throws CryptoException
     */
    private void generate_keys(int rc4_key_size) throws CryptoException {
        byte[] input = new byte[48];

        System.arraycopy(this.client_random, 0, input, 0, 24);
        System.arraycopy(this.server_random, 0, input, 24, 24);

        byte[] temp_hash = this.hash48(input, this.client_random, this.server_random,
                65);
        byte[] session_key = this.hash48(temp_hash, this.client_random,
                this.server_random, 88);

        System.arraycopy(session_key, 0, this.sec_sign_key, 0, 16);
        

        this.sec_decrypt_key = this.hash16(session_key, this.client_random,
                this.server_random, 16);
        this.sec_encrypt_key = this.hash16(session_key, this.client_random,
                this.server_random, 32);

        if (rc4_key_size == 1) {
            logger.info("40 Bit Encryption enabled");
            Secure.make40bit(this.sec_sign_key);
            Secure.make40bit(this.sec_decrypt_key);
            Secure.make40bit(this.sec_encrypt_key);
            this.keylength = 8;
        } else {
            logger.info("128 Bit Encryption enabled");
            this.keylength = 16;
        }

        System.arraycopy(this.sec_decrypt_key, 0, this.sec_decrypt_update_key,
                0, 16); 
        System.arraycopy(this.sec_encrypt_key, 0, this.sec_encrypt_update_key,
                0, 16); 

        byte[] key = new byte[this.keylength];
        System.arraycopy(this.sec_encrypt_key, 0, key, 0, this.keylength);
        rc4_enc.engineInitEncrypt(key);
        System.arraycopy(this.sec_decrypt_key, 0, key, 0, this.keylength);
        rc4_dec.engineInitDecrypt(key);
    }

    /**
     * @return MCS user ID
     */
    public int getUserID() {
        return McsLayer.getUserID();
    }
}