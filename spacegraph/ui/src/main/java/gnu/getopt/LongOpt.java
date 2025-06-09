package gnu.getopt;

import java.text.MessageFormat;
import java.util.Locale;
import java.util.ResourceBundle;

/**
 * This object represents the definition of a long option in the Java port
 * of GNU getopt.  An array of LongOpt objects is passed to the Getopt
 * object to define the list of valid long options for a given parsing
 * session.  Refer to the getopt documentation for details on the
 * format of long options.
 *
 * @author Aaron M. Renn (arenn@urbanophile.com)
 * @version 1.0.5
 * @see Getopt
 */
public class LongOpt  {

/**************************************************************************/

    /*
     * Class Variables
     */

    /**
     * Constant value used for the "has_arg" constructor argument.  This
     * value indicates that the option takes no argument.
     */
    public static final int NO_ARGUMENT = 0;

    /**
     * Constant value used for the "has_arg" constructor argument.  This
     * value indicates that the option takes an argument that is required.
     */
    public static final int REQUIRED_ARGUMENT = 1;

    /**
     * Constant value used for the "has_arg" constructor argument.  This
     * value indicates that the option takes an argument that is optional.
     */
    private static final int OPTIONAL_ARGUMENT = 2;

/**************************************************************************/

    /*
     * Instance Variables
     */

    /**
     * The name of the long option
     */
    String name;

    /**
     * Indicates whether the option has no argument, a required argument, or
     * an optional argument.
     */
    int has_arg;

    /**
     * If this variable is not null, then the value stored in "val" is stored
     * here when this long option is encountered.  If this is null, the value
     * stored in "val" is treated as the name of an equivalent short option.
     */
    StringBuffer flag;

    /**
     * The value to store in "flag" if flag is not null, otherwise the
     * equivalent short option character for this long option.
     */
    int val;

    /**************************************************************************/

    /*
     * Constructors
     */

    /**
     * Create a new LongOpt object with the given parameter values.  If the
     * value passed as has_arg is not valid, then an exception is thrown.
     *
     * @param name    The long option String.
     * @param has_arg Indicates whether the option has no argument (NO_ARGUMENT), a required argument (REQUIRED_ARGUMENT) or an optional argument (OPTIONAL_ARGUMENT).
     * @param flag    If non-null, this is a location to store the value of "val" when this option is encountered, otherwise "val" is treated as the equivalent short option character.
     * @param val     The value to return for this long option, or the equivalent single letter option to emulate if flag is null.
     * @throws IllegalArgumentException If the has_arg param is not one of NO_ARGUMENT, REQUIRED_ARGUMENT or OPTIONAL_ARGUMENT.
     */
    public LongOpt(String name, int has_arg,
                   StringBuffer flag, int val) throws IllegalArgumentException {
        
        if ((has_arg != NO_ARGUMENT) && (has_arg != REQUIRED_ARGUMENT)
                && (has_arg != OPTIONAL_ARGUMENT)) {
            Object[] msgArgs = {Integer.valueOf(has_arg).toString()};
            /**
             * Localized strings for error messages
             */
            ResourceBundle _messages = ResourceBundle.getBundle(
                    "gnu/getopt/MessagesBundle", Locale.getDefault());
            throw new IllegalArgumentException(MessageFormat.format(
                    _messages.getString("getopt.invalidValue"), msgArgs));
        }

        
        this.name = name;
        this.has_arg = has_arg;
        this.flag = flag;
        this.val = val;
    }

/**************************************************************************/

    /**
     * Returns the name of this LongOpt as a String
     *
     * @return Then name of the long option
     */
    public String
    getName() {
        return (name);
    }

/**************************************************************************/

    /**
     * Returns the value set for the 'has_arg' field for this long option
     *
     * @return The value of 'has_arg'
     */
    public int
    getHasArg() {
        return (has_arg);
    }

/**************************************************************************/

    /**
     * Returns the value of the 'flag' field for this long option
     *
     * @return The value of 'flag'
     */
    public StringBuffer
    getFlag() {
        return (flag);
    }

    /**
     * Returns the value of the 'val' field for this long option
     *
     * @return The value of 'val'
     */
    public int
    getVal() {
        return (val);
    }

/**************************************************************************/

} 
