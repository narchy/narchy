/** This code is copyright Articulate Software (c) 2003.  Some
portions copyright Teknowledge (c) 2003 and reused under the terms of
the GNU license.  This software is released under the GNU Public
License <http:
also consent, by use of this code, to credit Articulate Software and
Teknowledge in any writings, briefings, publications, presentations,
or other representations of any software which incorporates, builds
on, or uses this code.  Please cite the following article in any
publication with references:

Pease, A., (2003). The Sigma Ontology Development Environment, in
Working Notes of the IJCAI-2003 Workshop on Ontology and Distributed
Systems, August 9, Acapulco, Mexico.
See also http:

Authors:
Adam Pease
Infosys LTD.
*/

package nars.func.kif;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.SimpleTimeZone;
import java.util.stream.Collectors;

/** ***************************************************************
 * A utility class that defines static methods for common string
 * manipulation operations.
 */
public enum StringUtil {
    ;

    private static String CHARSET = "UTF-8";

    /** ************************************************************
     * Sets the default value of CHARSET to charEncoding.
     * charEncoding should be a String denoting a valid W3C character
     * encoding scheme, such as &quot;UTF-8&quot;.
     *
     * @param charEncoding A String denoting a character encoding scheme
     */
    public static void setCharset(String charEncoding) {
        CHARSET = charEncoding;
    }

    /**************************************************************
     * Returns a String denoting a character encoding scheme.  The
     * default value is &quot;UTF-8&quot;.
     *
     * @return String
     */
    public static String getCharset() {
        return CHARSET;
    }

    /**************************************************************
     * Returns a URL encoded String obtained from input, which is
     * assumed to be a String composed of characters in the default
     * charset.  In most cases, the charset will be UTF-8.
     *
     * @param input A String which has not yet been URL encoded
     * @return A URL encoded String
     */
    public static String encode(String input) {

        String encoded = input;
        try {
            encoded = URLEncoder.encode(input, getCharset());
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return encoded;
    }

    /** ************************************************************
     * Returns a URL decoded String obtained from input, which is
     * assumed to be a URL encoded String composed of characters in
     * the default charset.  In most cases, the charset will be UTF-8.
     *
     * @param input A URL encoded String
     * @return A String that has been URL decoded
     */
    public static String decode(String input) {

        String decoded = input;
        try {
            decoded = URLDecoder.decode(input, getCharset());
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return decoded;
    }

    /*************************************************************
     * Returns the default line separator token for the current
     * runtime platform.
     *
     * @return String
     */
    public static String getLineSeparator() {
        return System.getProperty("line.separator");
    }

    /**************************************************************
     * Sets the default line separator token for the current runtime
     * platform.
     *
     * @param separator The String to use as the line separator
     */
    public static void setLineSeparator(String separator) {
        System.setProperty("line.separator", separator);
    }

    /**************************************************************
     * A String token that separates a qualified KIF target name
     * from the namespace abbreviation prefix that qualifies it.
     */
    private static String KIF_NAMESPACE_DELIMITER = ":";

    /*************************************************************
     * Returns the string used in SUO-KIF to separate a namespace
     * prefix from the target it qualifies.
     */
    public static String getKifNamespaceDelimiter() {
        return KIF_NAMESPACE_DELIMITER;
    }

    /**************************************************************
     * Sets to str the String used in SUO-KIF to separate a namespace
     * prefix from the target it qualifies.
     */
    public static String setKifNamespaceDelimiter(String str) {
        KIF_NAMESPACE_DELIMITER = str;
        return KIF_NAMESPACE_DELIMITER;
    }

    /**************************************************************
     * A String token that separates a qualified target name from
     * the W3C namespace abbreviation prefix that qualifies it.
     */
    private static String W3C_NAMESPACE_DELIMITER = ":";

    /***************************************************************
     * Returns the string preferred by W3C to separate a namespace
     * prefix from the target it qualifies.
     */
    public static String getW3cNamespaceDelimiter() {
        return W3C_NAMESPACE_DELIMITER;
    }

    /****************************************************************
     * Sets to str the String token preferred by W3C to separate a
     * namespace prefix from the target it qualifies.
     */
    public static String setW3cNamespaceDelimiter(String str) {
        W3C_NAMESPACE_DELIMITER = str;
        return W3C_NAMESPACE_DELIMITER;
    }

    /***************************************************************
     * A "safe" alphanumeric ASCII string that can be substituted for
     * the W3C or SUO-KIF string delimiting a namespace prefix from an
     * unqualified target name.  The safe delimiter is used to produce
     * input formulae or files that can be loaded by Vampire and other
     * provers unable to handle target names containing non-alphanumeric
     * characters.
     */
    private static String SAFE_NAMESPACE_DELIMITER = "0xx1";

    /***************************************************************
     * Returns a "safe" alphanumeric ASCII string that can be
     * substituted for the W3C or SUO-KIF string delimiting a
     * namespace prefix from an unqualified target name.  The safe
     * delimiter is used to produce input formulae or files that can
     * be loaded by Vampire and other provers unable to handle target
     * names containing non-alphanumeric characters.
     */
    public static String getSafeNamespaceDelimiter() {
        return SAFE_NAMESPACE_DELIMITER;
    }

    /****************************************************************
     * Sets to str the "safe" alphanumeric ASCII String value that can
     * be substituted for the W3C or SUO-KIF string delimiting a
     * namespace prefix from an unqualified target name.  The safe
     * delimiter is used to produce input formulae or files that can
     * be loaded by Vampire and other provers unable to handle target
     * names containing non-alphanumeric characters.
     */
    public static String setSafeNamespaceDelimiter(String str) {

        SAFE_NAMESPACE_DELIMITER = str;
        return SAFE_NAMESPACE_DELIMITER;
    }





































































    /****************************************************************
     *
     * @param obj Any object
     * @return true if obj is a non-empty String, else false.
     */
    public static boolean isNonEmptyString(Object obj) {

        return ((obj instanceof String) && !obj.equals(""));
    }

    /***************************************************************
     *
     * @param s An input Object, expected to be a String.
     * @return true if s == null or s is an empty String, else false.
     */
    public static boolean emptyString(Object s) {
        return ((s == null)
                || ((s instanceof String)
                && ((String) s).isEmpty()));
    }

    /****************************************************************
     *
     * @return just the first n characters of the string.  If returning
     * less than the full string, append "..."
     */
    public static String getFirstNChars(String s, int n) {

        if (s.isEmpty()) return "";
        if (s.length() > n) return s.substring(0, n) + "...";
        return s;
    }
//
//    /****************************************************************
//     */
//    public static String spacesToUnderlines(String input) {
//
//        String[] s = input.split("\\s+");
//        StringBuilder sb = new StringBuilder();
//        for (int i = 0; i < s.length; i++) {
//            sb.append(s[i]);
//            if ((i+1) < s.length) {
//                sb.append('_');
//            }
//        }
//        return sb.toString();
//    }

    /****************************************************************
     */
    public static String camelCaseToUnderlines(String input) {

        StringBuilder sb = new StringBuilder();
        char lastChar = ' ';
        for (char c : input.toCharArray()) {
            if (Character.isUpperCase(c) && Character.isLowerCase(lastChar))
                sb.append("_").append(c);
            else
                sb.append(c);
            lastChar = c;
        }
        return sb.toString();
    }

    /****************************************************************
     * Removes all balanced ASCII double-quote characters from each
     * end of the String s, if any are present.
     */
    public static String removeEnclosingQuotes(String s) {

        return removeEnclosingChar(s, Integer.MAX_VALUE, '"');
    }

    /***************************************************************
     * Removes n layers of balanced ASCII double-quote characters from each
     * end of the String s, if any are present.
     */
    public static String removeEnclosingChar(String s, int n, char c) {

        StringBuilder sb = new StringBuilder();
        if (isNonEmptyString(s)) {
            sb.append(s);
            int lasti = (sb.length() - 1);
            for (int count = 0; ((count < n)
                    && (lasti > 0)
                    && (sb.charAt(0) == c)
                    && (sb.charAt(lasti) == c)); count++) {
                sb.deleteCharAt(lasti);
                sb.deleteCharAt(0);
                lasti = (sb.length() - 1);
            }
        }
        return sb.toString();
    }

    /****************************************************************
     * Removes n layers of balanced characters from each
     * end of the String s, if any are present.
     */
    public static String removeEnclosingChars(String s, int n, char c) {

        StringBuilder sb = new StringBuilder();
        if (isNonEmptyString(s)) {
            sb.append(s);
            int lasti = (sb.length() - 1);
            for (int count = 0; ((count < n)
                    && (lasti > 0)
                    && (sb.charAt(0) == c)
                    && (sb.charAt(lasti) == c)); count++) {
                sb.deleteCharAt(lasti);
                sb.deleteCharAt(0);
                lasti = (sb.length() - 1);
            }
        }
        return sb.toString();
    }

    /****************************************************************
     * Removes n layers of balanced characters from each
     * end of the String s, if any are present.
     */
    public static String removeEnclosingCharPair(String s, int n, char c1, char c2) {

        StringBuilder sb = new StringBuilder();
        if (isNonEmptyString(s)) {
            sb.append(s);
            int lasti = (sb.length() - 1);
            for (int count = 0; ((count < n)
                    && (lasti > 0)
                    && (sb.charAt(0) == c1)
                    && (sb.charAt(lasti) == c2)); count++) {
                sb.deleteCharAt(lasti);
                sb.deleteCharAt(0);
                lasti = (sb.length() - 1);
            }
        }
        return sb.toString();
    }

//    /****************************************************************
//     * Remove punctuation and contractions from a sentence.
//     */
//    public static String removePunctuation(String sentence) {
//
//        if (StringUtil.emptyString(sentence))
//            return sentence;
//        Matcher m = Pattern.compile("(\\w)'re").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        m = Pattern.compile("(\\w)'m").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        m = Pattern.compile("(\\w)n't").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        m = Pattern.compile("(\\w)'ll").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        m = Pattern.compile("(\\w)'s").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        m = Pattern.compile("(\\w)'d").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        m = Pattern.compile("(\\w)'ve").matcher(sentence);
//
//        while (m.find()) {
//
//            String group = m.group(1);
//            sentence = m.replaceFirst(group);
//            m.reset(sentence);
//        }
//        sentence = sentence.replaceAll("'", "");
//        sentence = sentence.replaceAll("\"", "");
//        sentence = sentence.replaceAll("\\.", "");
//        sentence = sentence.replaceAll(";", "");
//        sentence = sentence.replaceAll(":", "");
//        sentence = sentence.replaceAll("\\?", "");
//        sentence = sentence.replaceAll("!", "");
//        sentence = sentence.replaceAll(", ", " ");
//        sentence = sentence.replaceAll(",[^ ]", ", ");
//        sentence = sentence.replaceAll(" {2}", " ");
//        return sentence;
//    }

//    /****************************************************************
//     * Remove HTML markup from a sentence.
//     */
//    public static String removeHTML(String sentence) {
//
//        return sentence.replaceAll("<[^>]+>", "");
//    }

    /****************************************************************
     *
     * @param str A String
     * @return A String with space characters normalized to match the
     * conventions for written English text.  All linefeeds and
     * carriage returns are replaced with spaces.
     */
    public static String normalizeSpaceChars(String str) {

        String ans = str;
        if (isNonEmptyString(ans)) {
            
            ans = ans.replaceAll("\\s+", " ");
            ans = ans.replaceAll("\\(\\s+", "(");
            
            
            
            
        }
        return ans;
    }
//
//    /***************************************************************
//     * Convert an arbitrary string to a legal KIF identifier by
//     * substituting dashes for illegal characters. TODO:
//     * isJavaIdentifierPart() isn't sufficient, since it allows
//     * characters KIF doesn't
//     */
//    public static String arrayListToSpacedString(ArrayList<String> al) {
//
//        if (al == null || al.size() < 1)
//            return "";
//        String sb = String.join(" ", al);
//        return sb;
//    }

//    /***************************************************************
//     * Convert an arbitrary string to a legal KIF identifier by
//     * substituting dashes for illegal characters. TODO:
//     * isJavaIdentifierPart() isn't sufficient, since it allows
//     * characters KIF doesn't
//     */
//    public static String StringToKIFid(String s) {
//
//        if (s == null)
//            return s;
//        s = s.trim();
//        if (s.length() < 1)
//            return s;
//        if (s.length() > 1 && s.charAt(0) == '"' && s.charAt(s.length() - 1) == '"')
//            s = s.substring(1, s.length() - 1);
//        if (s.charAt(0) != '?' &&
//                (!Character.isJavaIdentifierStart(s.charAt(0)) ||
//                        s.charAt(0) > 122))
//            s = 'S' + s.substring(1);
//        int i = 1;
//        while (i < s.length()) {
//            if (!Character.isJavaIdentifierPart(s.charAt(i)) ||
//                    s.charAt(i) > 122)
//                s = s.substring(0, i) + '-' + s.substring(i + 1);
//            i++;
//        }
//        return s;
//    }
//
//    /****************************************************************
//     * Convert an arbitrary string to a legal Prolog identifier by
//     * substituting dashes for illegal characters.
//     */
//    public static String StringToPrologID(String s) {
//
//        if (s == null)
//            return s;
//        s = s.trim();
//        if (s.length() < 1)
//            return s;
//        if (Character.isUpperCase(s.charAt(0)))
//            s = Character.toLowerCase(s.charAt(0)) + s.substring(1);
//        if (!Character.isLetter(s.charAt(0)))
//            s = 's' + s.substring(1);
//        int i = 1;
//        while (i < s.length()) {
//            if (!Character.isLetter(s.charAt(i)) &&
//                    !Character.isDigit(s.charAt(i)) &&
//                    s.charAt(i) != '_')
//                s = s.substring(0, i) + '_' + s.substring(i + 1);
//            i++;
//        }
//        return s;
//    }

    /****************************************************************
     *
     * @param str A String
     * @return A String with all double quote characters properly
     * escaped with a left slash character.
     */
    public static String escapeQuoteChars(String str) {

        String ans = str;
        if (isNonEmptyString(str)) {
            int l = str.length();
            StringBuilder sb = new StringBuilder(l);
            char prevCh = 'x';
            char ch = 'x';
            for (int i = 0; i < l; i++) {
                ch = str.charAt(i);
                if ((ch == '"') && (prevCh != '\\'))
                    sb.append('\\');
                sb.append(prevCh = ch);
            }
            return sb.toString();
        }
        return ans;
    }

//    /****************************************************************
//     *
//     * @param str A String
//     * @return A String with all escape characters properly
//     * escaped with a left slash character.
//     */
//    public static String escapeEscapeChars(String str) {
//
//        String ans = str;
//        if (isNonEmptyString(str)) {
//            StringBuilder sb = new StringBuilder();
//            char prevCh = 'x';
//            char ch = 'x';
//            for (int i = 0; i < str.length(); i++) {
//                ch = str.charAt(i);
//                if ((ch == '\\') && (prevCh != '\\')) {
//                    sb.append('\\');
//                }
//                sb.append(ch);
//                prevCh = ch;
//            }
//            ans = sb.toString();
//        }
//        return ans;
//    }
//
//    /****************************************************************
//     *
//     * @param str A String
//     * @return A String with all left slash characters removed
//     */
//    public static String removeQuoteEscapes(String str) {
//
//        String ans = str;
//        if (isNonEmptyString(str)) {
//            StringBuilder sb = new StringBuilder();
//            char prevCh = 'x';
//            char ch = 'x';
//            int strlen = str.length();
//            for (int i = 0; i < strlen; i++) {
//                ch = str.charAt(i);
//                if ((ch == '"') && (prevCh == '\\')) {
//                    sb.deleteCharAt(sb.length() - 1);
//                }
//                sb.append(ch);
//                prevCh = ch;
//            }
//            ans = sb.toString();
//        }
//        return ans;
//    }
//
//    /***************************************************************
//     *
//     * @param str A String
//     * @return A String in which each sequence of two or more
//     * slash-escape chars is replaced by one slash-escape char.
//     */
//    public static String removeEscapedEscapes(String str) {
//
//        String ans = str;
//        if (isNonEmptyString(str)) {
//            StringBuilder sb = new StringBuilder();
//            char prevCh = 'x';
//            char ch = 'x';
//            int strlen = str.length();
//            for (int i = 0; i < strlen; i++) {
//                ch = str.charAt(i);
//                if ((ch == '\\') && (prevCh == '\\')) {
//                    sb.deleteCharAt(sb.length() - 1);
//                }
//                sb.append(ch);
//                prevCh = ch;
//            }
//            ans = sb.toString();
//        }
//        return ans;
//    }
//
//    /****************************************************************
//     *
//     * @param str A String
//     * @return A String with all internal escaped double quote
//     * characters removed
//     */
//    public static String removeEscapedDoubleQuotes(String str) {
//
//        String ans = str;
//        if (isNonEmptyString(str)) {
//            StringBuilder sb = new StringBuilder();
//            char prevCh = 'x';
//            char ch = 'x';
//            int strlen = str.length();
//            for (int i = 0; i < strlen; i++) {
//                ch = str.charAt(i);
//                if ((ch == '"') && (prevCh == '\\')) {
//                    sb.deleteCharAt(sb.length() - 1);
//                    int prevI = (i - 2);
//                    if (prevI > -1) {
//                        prevCh = str.charAt(prevI);
//                    }
//                    else {
//                        prevCh = 'x';
//                    }
//                    continue;
//                }
//                sb.append(ch);
//                prevCh = ch;
//            }
//            ans = sb.toString();
//        }
//        return ans;
//    }
//
//    /****************************************************************
//     * @param str A String
//     * @return A String with all internal double quote characters
//     * removed
//     */
//    public static String removeInternalDoubleQuotes(String str) {
//
//        String ans = str;
//        if (isNonEmptyString(str)) {
//            String newstr = str.trim();
//            StringBuilder sb = new StringBuilder();
//            char ch = 'x';
//            int strlen = newstr.length();
//            for (int i = 0; i < strlen; i++) {
//                ch = newstr.charAt(i);
//                if (ch != '"') {
//                    sb.append(ch);
//                }
//            }
//            ans = sb.toString();
//        }
//        return ans;
//    }

    /****************************************************************
     * @param str A String
     * @return A String with all sequences of two double quote
     * characters have been replaced by a left slash character
     * followed by a double quote character.
     */
    public static String replaceRepeatedDoubleQuotes(String str) {

        String ans = str;
        if (isNonEmptyString(str)) {
            StringBuilder sb = new StringBuilder();
            char prevCh = 'x';
            char ch = 'x';
            for (int i = 0; i < str.length(); i++) {
                ch = str.charAt(i);
                if ((ch == '"') && (prevCh == '"')) {
                    sb.setCharAt(sb.length() - 1, '\\');
                }
                sb.append(ch);
                prevCh = ch;
            }
            ans = sb.toString();
        }
        return ans;
    }

    /****************************************************************
     */
    public static String removeDoubleSpaces(String s) {

        if (emptyString(s))
            return s;
        return s.replaceAll(" [ ]+"," ");
    }

    /****************************************************************
     */
    public static String allCapsToSUMOID(CharSequence str) {

        if (emptyString(str)) {
            System.out.println("Error in StringUtil.allCapsToSUMOID(): str is null");
            return "";
        }
        StringBuilder sb = new StringBuilder();
        boolean under = false;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '_')
                under = true;
            else {
                if (under || i == 0)
                    sb.append(str.charAt(i));
                else
                    sb.append(Character.toLowerCase(str.charAt(i)));
                under = false;
            }
        }
        return sb.toString();
    }

    /****************************************************************
     */
    public static String asSUMORelationID(String str) {

        if (emptyString(str)) {
            System.out.println("Error in StringUtil.asSUMORelationID(): str is null");
            return "";
        }
        return Character.toLowerCase(str.charAt(0)) + str.substring(1);
    }

    /****************************************************************
     *
     * @param str A String
     * @return true if str contains any non-ASCII characters, else
     * false.
     */
    public static boolean containsNonAsciiChars(String str) {

        return isNonEmptyString(str) && str.matches(".*[^\\p{ASCII}].*");
    }

    /****************************************************************
     *
     * @param str A String
     * @return A String with all non-ASCII characters replaced by "x".
     */
    public static String replaceNonAsciiChars(String str) {

        String ans = str;
        if (isNonEmptyString(ans)) {
            ans = ans.replaceAll("[^\\p{ASCII}]", "x");
        }
        return ans;
    }

    /***************************************************************
     * Replace any character that isn't a valid KIF identifier
     * character with a lower-case x.
     */
    public static String replaceNonIdChars(String st) {

        String ans = st;
        if (isNonEmptyString(ans)) {
            ans = ans.replaceAll("[\\W.]", "x");
            while (ans.matches(".+[^\\p{Alnum}]$")) {
                ans = ans.substring(0, ans.length() - 1);
            }
        }
        return ans;
    }

    /***************************************************************
     * Returns a date/time string corresponding to pattern.  The
     * date/time returned is the date/time of the method call.  The
     * locale is UTC (Greenwich).
     *
     * @param pattern Examples: yyyy, yyyy-MM-dd.
     */
    public static String getDateTime(String pattern) {

        String dateTime = "";
        try {
            if (isNonEmptyString(pattern)) {
                SimpleDateFormat sdf = new SimpleDateFormat(pattern);
                sdf.setTimeZone(new SimpleTimeZone(0, "Greenwich"));
                dateTime = sdf.format(new Date());
            }
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return dateTime;
    }

    /****************************************************************
     * If the input String contains the sequence {date}pattern{date},
     * replaces the first occurrence of this sequence with a UTC
     * date/time string formatted according to pattern.  If the input
     * String does not contain the sequence, it is returned unaltered.
     *
     * @param input The input String into which a formatted date/time
     *              will be inserted
     * @return String
     */
    public static String replaceDateTime(String input) {

        String output = input;
        try {
            String token = "{date}";
            if (isNonEmptyString(output) && output.contains(token)) {
                int tlen = token.length();
                StringBuilder sb = new StringBuilder(output);
                int p1f = sb.indexOf(token);
                while (p1f > -1) {
                    int p1b = (p1f + tlen);
                    if (p1b < sb.length()) {
                        int p2f = sb.indexOf(token, p1b);
                        if (p2f > -1) {
                            String pattern = sb.substring(p1b, p2f);
                            int p2b = (p2f + tlen);
                            sb.replace(p1f, p2b, getDateTime(pattern));
                            p1f = sb.indexOf(token);
                        }
                    }
                }
                output = sb.toString();
            }
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return output;
    }

    /****************************************************************
     * Returns true if input appears to be a URI string, else returns
     * false.
     *
     * @param input A String
     */
    public static boolean isUri(String input) {

        boolean ans = false;
        try {
            ans = (isNonEmptyString(input)
                    && (input.matches("^.?http://.+")
                    || input.matches("^.?file://.+")));
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return ans;
    }

    /****************************************************************
     * Returns true if input is a String containing some whitespace
     * chars, else returns false.
     *
     * @param input A String
     * @return true or false
     */
    public static boolean isStringWithSpaces(String input) {

        return (isNonEmptyString(input) && input.matches(".*\\s+.*"));
    }

    /****************************************************************
     * Returns true if input is a String is an integer (of any size)
     *
     * @param s A String
     * @return true or false
     */
    public static boolean isInteger(String s) {

        boolean isValidInteger = false;
        try {
            Integer.parseInt(s.trim()); 
            isValidInteger = true;
        }
        catch (NumberFormatException ex) {
            
        }
        return isValidInteger;
    }

    /****************************************************************
     * Returns true if input appears to be a quoted String, else
     * returns false.
     *
     * @param input A String
     */
    public static boolean isQuotedString(CharSequence input) {

        boolean ans = false;
        try {
            if (isNonEmptyString(input)) {
                int ilen = input.length();
                if (ilen > 2) {
                    char fc = input.charAt(0);
                    char lc = input.charAt(ilen - 1);
                    ans = (((fc == '"') && (lc == '"'))
                            || ((fc == '\'') && (lc == '\'')) || (fc == '`'));
                }
            }
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return ans;
    }

    /****************************************************************
     * Returns a new String formed by adding quoteChar to each end of
     * input.
     *
     * @param input A String
     */
    public static String makeQuotedString(String input, char quoteChar) {

        String ans = input;
        try {
            if (isNonEmptyString(input)
                    && !isQuotedString(input)
                    && (input.charAt(0) != quoteChar)) {
                ans = quoteChar + input + quoteChar;
            }
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return ans;
    }

    /****************************************************************
     * Returns true if every char in input is a digit char, else
     * returns false.
     *
     * @param input A String
     * @return true or false
     */
    public static boolean isDigitString(String input) {

        return isNonEmptyString(input) && !input.matches(".*\\D+.*");
    }

    /****************************************************************
     */
    public static boolean isNumeric(String input) {

        try {
            Integer.parseInt(input);
            return true;
        }
        catch (NumberFormatException e) {
            
            return false;
        }
    }

    /****************************************************************
     * Returns a String formed from n concatenations of input.
     *
     * @param input A String
     * @param n     A non-negative int
     * @return true or false
     */
    public static String concatN(String input, int n) {

        String ans = "";
        try {
            ans = String.valueOf(input).repeat(Math.max(0, n));
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return ans;
    }

    /****************************************************************
     * Performs a depth-first search of tree, replacing all terms
     * matching oldPattern with newTerm.
     *
     * @param oldPattern A regular expression pattern to be matched
     *                   against terms in tree
     * @param newTerm    A String to replace terms matching oldPattern
     * @param tree       A String representing a SUO-KIF Formula (list)
     * @return A new tree (String), with all occurrences of terms
     * matching oldPattern replaced by newTerm
     */
    public static String treeReplace(String oldPattern, String newTerm, String tree) {

        String result = tree;
        try {
            StringBuilder sb = new StringBuilder();
            if (tree.matches(oldPattern))
                sb.append(newTerm);
            else if (Formula.listP(tree)) {
                if (Formula.empty(tree)) {
                    sb.append(tree);
                }
                else {
                    Formula f = new Formula();
                    f.read(tree);
                    List tuple = f.literalToArrayList();
                    sb.append('(');
                    int i = 0;
                    for (Iterator it = tuple.iterator(); it.hasNext(); i++) {
                        if (i > 0) sb.append(' ');
                        sb.append(treeReplace(oldPattern,
                                newTerm,
                                (String) it.next()));
                    }
                    sb.append(')');
                }
            } else {
                sb.append(tree);
            }
            result = sb.toString();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return result;
    }

//    /****************************************************************
//     * Returns a new ArrayList formed by extracting in order the
//     * top-level members of kifListAsString, which is assumed to be
//     * the String representation of a SUO-KIF (LISP) list.
//     *
//     * @param kifListAsString A SUO-KIF list represented as a String
//     * @return ArrayList
//     */
//    public static List kifListToArrayList(String kifListAsString) {
//
//        List ans = new ArrayList();
//        try {
//            if (isNonEmptyString(kifListAsString)) {
//                Formula f = new Formula();
//                f.read(kifListAsString);
//                ans = f.literalToArrayList();
//            }
//        }
//        catch (Exception ex) {
//            ex.printStackTrace();
//        }
//        return ans;
//    }

//    /****************************************************************
//     * Replaces non-alphanumeric namespace delimiters in input with an
//     * alphanumeric form that can be handled by Vampire and other
//     * provers.
//     *
//     * @param input The String representation of a SUO-KIF Formula or
//     *              other expression
//     * @return String The input string with unsafe namespace
//     * delimiters replaced by a safe alphanumeric form
//     */
//    public static String replaceUnsafeNamespaceDelimiters(String input) {
//
//        String output = input;
//        try {
//            if (isNonEmptyString(output)) {
//                String safe = ("$1" + getSafeNamespaceDelimiter() + "$2");
//                List<String> unsafe = Arrays.asList(getKifNamespaceDelimiter(),
//                        getW3cNamespaceDelimiter());
//                for (String delim : unsafe) {
//                    output = output.replaceAll("(\\w)" + delim + "(\\w)", safe);
//                }
//            }
//        }
//        catch (Exception ex) {
//            ex.printStackTrace();
//        }
//        return output;
//    }
//
//    /***************************************************************
//     * Replaces all occurrences of the "safe" namespace delimiter in
//     * input with the default KIF namespace delimiter, which might
//     * contain characters that are not acceptable to some provers.
//     *
//     * @param input The String representation of a SUO-KIF Formula or
//     *              other expression, such as a TPTP Formula
//     * @return String The input string with occurrences of the safe
//     * namespace delimiters replaced by the default KIF namespace
//     * delimiter
//     */
//    public static String safeToKifNamespaceDelimiters(String input) {
//
//        String output = input;
//        try {
//            if (isNonEmptyString(output)) {
//                String safedelim = getSafeNamespaceDelimiter();
//                String kifdelim = getKifNamespaceDelimiter();
//                output = output.replace(safedelim, kifdelim);
//            }
//        }
//        catch (Exception ex) {
//            ex.printStackTrace();
//        }
//        return output;
//    }

    /***************************************************************
     * Replaces the namespace delimiter in target with a string that is
     * safe for inference, and for all common file systems.
     */
    public static String toSafeNamespaceDelimiter(String term) {

        String ans = term;
        try {
            if (isNonEmptyString(term) && !isUri(term)) {
                String safe = getSafeNamespaceDelimiter();
                String kif = getKifNamespaceDelimiter();
                String w3c = getW3cNamespaceDelimiter();
                ans = term.replaceFirst(kif, safe);
                if (!kif.equals(w3c)) {
                    ans = ans.replaceFirst(w3c, safe);
                }
            }
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return ans;
    }

    /***************************************************************
     * Replaces the namespace delimiter in target with a string that is
     * safe for inference and for all common file systems, but only if
     * kbHref is an empty string or == null.  If kbHref is not empty,
     * target is probably being prepared for display in the Sigma
     * Browser and does not have to be converted to a "safe" form.
     */
    public static String toSafeNamespaceDelimiter(String kbHref, String term) {


        String ans = term;
        try {
            if (StringUtil.emptyString(kbHref))
                ans = toSafeNamespaceDelimiter(term);
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return ans;
    }

    /***************************************************************
     */
    public static String w3cToKif(String term) {

        String ans = term;
        if (isNonEmptyString(term) && !StringUtil.isUri(term)) {
            ans = term.replaceFirst(getW3cNamespaceDelimiter(),
                    getKifNamespaceDelimiter());
        }
        return ans;
    }

    /****************************************************************
     */
    public static String kifToW3c(String term) {

        String ans = term;
        if (isNonEmptyString(term) && !StringUtil.isUri(term)) {
            ans = term.replaceFirst(getKifNamespaceDelimiter(),
                    getW3cNamespaceDelimiter());
        }
        return ans;
    }

    /****************************************************************
     */
    public static boolean quoted(String input) {

        String trimmed = input.trim();
        if (trimmed.charAt(0) == '\'' && trimmed.charAt(trimmed.length()-1) == '\'')
            return true;
        return trimmed.charAt(0) == '\"' && trimmed.charAt(trimmed.length() - 1) == '\"';
    }

    /****************************************************************
     */
    public static String quote(String input) {

        String str = StringUtil.removeEnclosingQuotes(input);
        str = StringUtil.escapeQuoteChars(str);
        return StringUtil.makeQuotedString(str, '"');
    }

//    /***************************************************************
//     */
//    public static String unquote(String input) {
//
//        String ans = input;
//        ans = StringUtil.removeEnclosingQuotes(ans);
//        return StringUtil.replaceRepeatedDoubleQuotes(ans);
//    }
//
//    /****************************************************************
//     */
//    public static boolean isLocalTermReference(String term) {
//
//        boolean ans = false;
//        if (isNonEmptyString(term)) {
//            List<String> blankNodeTokens = Arrays.asList("#", "~", getLocalReferenceBaseName());
//            for (String bnt : blankNodeTokens) {
//                ans = (term.startsWith(bnt) && !term.matches(".*\\s+.*"));
//                if (ans) break;
//            }
//        }
//        return ans;
//    }

    /***************************************************************
     * The base String used to create the names of local composite
     * members.
     */
//    private static String LOCAL_REF_BASE_NAME = "LocalRef";

    /***************************************************************
//     */
//    public static String getLocalReferenceBaseName() {
//        return LOCAL_REF_BASE_NAME;
//    }
//
//    /****************************************************************
//     */
//    public static void setLocalReferenceBaseName(String basename) {
//        LOCAL_REF_BASE_NAME = basename;
//    }
//
//    /********************************************************************
//     * If the file f already exists, this method returns a new File
//     * object with a unique name formed by appending an integer.  If
//     * the filename has a file type suffix, the integer is inserted
//     * between the base name and the suffix.  If f does not exist, it
//     * is simply returned.
//     *
//     * @param f A File
//     * @return A File with a name different from f's name, if f
//     * exists, else f.
//     */
//    public static File renameFileIfExists(File f) {
//
//        File result = f;
//        try {
//            String canonicalPath = result.getCanonicalPath();
//            int lidx = canonicalPath.lastIndexOf('.');
//            String suff = "";
//            String base = canonicalPath;
//            if (lidx != -1) {
//                suff = canonicalPath.substring(lidx);
//                base = canonicalPath.substring(0, lidx);
//            }
//            int fc = 0;
//            while (result.exists()) {
//                fc++;
//                result = new File(base + '-' + fc + suff);
//            }
//        }
//        catch (Exception ex) {
//            ex.printStackTrace();
//        }
//        return result;
//    }

//    /*******************************************************************
//     */
//    public static String wordWrap(String input, int length) {
//
//        String result = input;
//        try {
//            if (StringUtil.isNonEmptyString(input)
//                    && (length > 0)
//                    && (input.length() > length)) {
//                StringBuilder sb = new StringBuilder(input);
//                String ls = System.getProperty("line.separator");
//                int lslen = ls.length();
//                int j = length;
//                int i = 0;
//                while (sb.length() > j) {
//                    while ((j > i) && !Character.isWhitespace(sb.charAt(j)))
//                        j--;
//                    if (j > i) {
//                        sb.deleteCharAt(j);
//                        sb.insert(j, ls);
//                        i = (j + lslen);
//                        j = (i + length);
//                    }
//                    else {
//                        j += length;
//                        while ((j < sb.length()) && !Character.isWhitespace(sb.charAt(j)))
//                            j++;
//                        if (j < sb.length()) {
//                            sb.deleteCharAt(j);
//                            sb.insert(j, ls);
//                            i = (j + lslen);
//                            j = (i + length);
//                        }
//                    }
//                }
//                result = sb.toString();
//            }
//        }
//        catch (Exception ex) {
//            ex.printStackTrace();
//        }
//        return result;
//    }
//
//    /********************************************************************
//     * Convenience method with default line length of 70
//     */
//    public static String wordWrap(String input) {
//        return StringUtil.wordWrap(input, 70);
//    }
//
//    /********************************************************************
//     * @return a string version of the integer padded with 0's to a length of 8
//     */
//    public static String integerToPaddedString(int intval) {
//
//        String ID = String.valueOf(intval);
//        if (ID.length() > 8) {
//            System.out.println("Error in StringUtil.integerToPaddedString(): max number exceeded: " +
//                ID);
//            return null;
//        }
//        if (ID.length() < 8)
//            return StringUtil.fillString(ID,'0',8,true);
//        else
//            return ID;
//    }
//
//    /** *******************************************************************
//     *  Convert any arbitrary string to a valid KIF id.  There is no guarantee that
//     *  it is unique however, since the current KB isn't inspected.
//     */
//    public static String stringToKIF(CharSequence input, boolean upcaseFirst) {
//
//        if (StringUtil.emptyString(input))
//            return null;
//        StringBuilder result = new StringBuilder();
//        if (Character.isJavaIdentifierStart(input.charAt(0)))
//            if (upcaseFirst)
//                result.append(Character.toUpperCase(input.charAt(0)));
//            else
//                result.append(Character.toLowerCase(input.charAt(0)));
//        else
//            if (upcaseFirst)
//                result.append("I_");
//            else
//                result.append("r_");
//        for (int i = 1; i < input.length(); i++) {
//            if (input.charAt(i) != ' ') {
//                if (Character.isJavaIdentifierPart(input.charAt(i)) && input.charAt(i) != '$')
//                    result.append(input.charAt(i));
//                else
//                    result.append('x');
//            }
//
//        }
//        return result.toString();
//    }
//
//    /** *****************************************************************
//     */
//    public static String indent(int num, String indentChars) {
//
//        return String.valueOf(indentChars).repeat(Math.max(0, num));
//    }
//
//
//    /** *****************************************************************
//     * Find the parenthesis that balances the one in st at character pIndex
//     * @return -1 if not found
//     */
//    public static int findBalancedParen(int pIndex, CharSequence st) {
//
//        int parenLevel = 1;
//        for (int i = pIndex + 1; i < st.length(); i++) {
//            if (st.charAt(i) == '(')
//                parenLevel++;
//            if (st.charAt(i) == ')') {
//                parenLevel--;
//                if (parenLevel == 0)
//                    return i;
//            }
//        }
//        return -1;
//    }
//
    /** *****************************************************************
     * Fill a string with the desired character up to the totalLength.
     * If string is null return a completely filled string.
     */
    public static String fillString(String st, char fillchar, int totalLength, boolean prepend) {
        
        StringBuilder result = null;
        if (st != null)
            result = new StringBuilder(st);
        else
            result = new StringBuilder();
        for (int i = 0; i < totalLength - st.length(); i++) {
            if (prepend)
                result.insert(0, fillchar);
            else
                result.append(fillchar);
        }
        return result.toString();
    }

//    /** *****************************************************************
//     */
//    public static boolean urlExists(String URLName) {
//
//        boolean result = false;
//        try {
//            URL url = new URL("ftp://ftp1.freebsd.org/pub/FreeBSD/");
//            InputStream input = url.openStream();
//            return true;
//        }
//        catch (Exception ex) {
//            System.out.println("error in StringUtil.urlExists()");
//        }
//        return false;
//    }

//    /** *****************************************************************
//     */
//    public static void main(String[] args) {
//
//        System.out.println(StringUtil.fillString("111",'0',8,true));
//
//    }

    /** *****************************************************************
     * Fetch the entire contents of a text file, and return it in a String.
     * This style of implementation does not throw Exceptions to the caller.
     *
     * @param aFile is a file which already exists and can be read.
     */
     public static String getContents(File aFile) {
        
        String contents = "";

        try {


            try (BufferedReader input = new BufferedReader(new FileReader(aFile))) {
                /*
                 * readLine is a bit quirky :
                 * it returns the content of a line MINUS the newline.
                 * it returns null only for the END of the stream.
                 * it returns an empty String if two newlines appear in a row.
                 */
                contents = input.lines().map(line -> line + System.getProperty("line.separator")).collect(Collectors.joining());
            }
        }
        catch (IOException ex){
            ex.printStackTrace();
        }

        return contents;
    }

    /** ***************************************************************
     * Remove HTML from input string.
     * @param input
     * @return
     */
    public static String filterHtml(String input)  {
        
        String out = input.replaceAll("<.*?>", "");

        
        out = out.replaceAll(" +", " ");
        
        out = out.replaceAll(",(\\S)", ", $1");

        return out;
    }

} 