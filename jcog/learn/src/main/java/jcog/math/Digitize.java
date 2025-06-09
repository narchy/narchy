package jcog.math;

import jcog.Is;
import jcog.Util;

import static jcog.Util.sqr;
import static jcog.Util.unitizeSafe;

/**
 * decides the truth value of a 'digit'. returns frequency float
 *
 * @param conceptIndex the 'digit' concept
 * @param x            the value being input
 * @maxDigits the total size of the set of digits being calculated
 */
@FunctionalInterface
@Is({"Digitization",
    "Analog-to-digital_converter",
    "Quantization_(signal_processing)",
    "Data_binning", "Fuzzy_logic"
})
public interface Digitize {
    /**
     * "HARD" - analogous to a filled volume of liquid
     * <p>
     * [ ] [ ] [ ] [ ] 0.00
     * [x] [ ] [ ] [ ] 0.25
     * [x] [x] [ ] [ ] 0.50
     * [x] [x] [x] [ ] 0.75
     * [x] [x] [x] [x] 1.00
     * <p>
     * key:
     * [ ] = freq 0
     * [x] = freq 1,
     * TODO
     */
    Digitize Fluid = (v, iDiscrete, n) -> {

        int vDiscrete = Util.bin(v, n);

        float f;
        if (iDiscrete < vDiscrete) {
            //below
            f = 1;
        }else if (iDiscrete == vDiscrete) {
            //partial boundary
            float i = ((float)iDiscrete) / n;
            float x = (v - i) * n;
            f = unitizeSafe(x);
        } else {
            //above
            f = 0;
        }

        return f;

    };

    Digitize Proportional = (v, iDiscrete, n) -> {
        int vDiscrete = Util.bin(v, n);
        int dist = Math.abs(iDiscrete - vDiscrete);
        float distF = dist /
            (n/2f)
            //(n-1f)
            //n*2
            //n
        ;
        return Math.max(0, 1 - distF);
    };

    Digitize ProportionalSqr = (v, iDiscrete, n) -> sqr(Proportional.digit(v, iDiscrete, n));

    /**
     * hard
     */
    Digitize BinaryNeedle = (v, i, indices) -> {
        float vv = v * indices;
        int which = (int) vv;
        return i == which ? 1 : 0;
    };

    /**
     * analogous to a needle on a guage, the needle being the triangle spanning several of the 'digits'
     * /          |       \
     * /         / \        \
     * /        /   \         \
     * + + +    + + +     + + +
     * TODO need to analyze the interaction of the produced frequency values being reported by all concepts.
     */
    Digitize FuzzyNeedle = (v, i, indices) -> {
        //assertUnitized(v);
        float dr = 1f / (indices - 1);
        return Math.max(0, 1 - Math.abs((i * dr) - v) / dr);
    };

    /**
     * http://www.personal.reading.ac.uk/~sis01xh/teaching/ComputerControl/fcslide3.pdf
     *
     * */
    @Is("Radial_basis_function")
    Digitize FuzzyGaussian = (v, i, indices) -> {
       double dx = v - (((float)i)/(indices-1));
       double width = 0.5f/indices;
       return (float) Math.exp(-sqr(dx)/(2*sqr(width)));
    };
    Digitize FuzzyNeedleCurve = (v, i, indices) -> {
        float dr = 1f / (indices - 1);
        return Math.max(0, 1f - sqr(Math.abs((i * dr) - v) / dr));
        //return  Util.sqrt(Math.max(0,1f - Math.abs((i * dr) - v) / dr));
    };
    /**
     * TODO not quite working yet. it is supposed to recursively subdivide like a binary number, and each concept represents the balance corresponding to each radix's progressively increasing sensitivity
     */
    Digitize FuzzyBinary = (v, i, indices) -> {
        float b = v, dv = 1;
        for (int j = 0; j < i; j++) {
            dv /= 2f;
            b = Math.max(0, b - dv);
        }
        return b / (dv);
    };
    Digitize TriState_2ary = (v, i, indices) -> {
        //assert(indices==2);
        return (2 * Math.max(0, i == 0 ? v - 0.5f : 0.5f - v));
    };

    /**
     * Explanation:
     *
     * We first check for the same edge cases as before and throw appropriate exceptions.
     * We calculate the multiplier by raising 2 to the power of maxDigits. This helps us scale the input x to the desired number of binary digits.
     * We multiply x by the multiplier and cast the result to an integer. This gives us the encoded value of x with the specified number of binary digits.
     * To extract the value of a specific binary digit (bit), we use the bitwise right shift operator >> to shift the desired bit to the least significant position, and then use the bitwise AND operator & with 1 to get the bit value (0 or 1).
     * Finally, we return the bit value as a float, which will be either 0.0 or 1.0.
     *
     * Note: If maxDigits is very large (e.g., more than 30), the float data type may not have enough precision to represent the resolution accurately. In such cases, you might need to use a higher-precision data type like double.
     */
    Digitize Binary = (x, digit, maxDigits) -> {
        int multiplier = (int) Math.pow(2, maxDigits);
        int encodedValue = (int) (x * multiplier);
        int digitValue = (encodedValue >> (maxDigits - 1 - digit)) & 1;

        return digitValue;
    };

    /**
     We first check for the same edge cases as before and throw appropriate exceptions.
     We calculate the resolution of the specified digit (bit) by raising 2 to the power of -digit. This represents the range of values that the digit can represent.
     We calculate the offset by dividing the resolution by 2. This offset is used to shift the output value to the middle of the range represented by the digit.
     We divide the input x by the resolution to scale it to the range [0, 2^(maxDigits - digit)].
     We calculate the output value by flooring the scaled value (to get the discrete value represented by the digit), multiplying it by the resolution (to scale it back to the original range), and adding the offset (to shift it to the middle of the range).
     Finally, we use Math.min() to ensure that the output value is capped at 1.0, since the last digit may have a range that extends beyond 1.0.
     Now, the method provides a continuous output that represents the range of values that the specified digit (bit) can represent. The output value will be in the range [0, 1], and it will be centered within the range represented by the digit.

     Note: If maxDigits is very large (e.g., more than 30), the float data type may not have enough precision to represent the resolution accurately. In such cases, you might need to use a higher-precision data type like double.
     */
    Digitize BinaryContinuous = (x, digit, maxDigits) -> {
        float resolution = (float) Math.pow(2, -digit);
        float offset = resolution / 2;
        float scaledValue = x / resolution;
        float digitValue = (float) (Math.floor(scaledValue) * resolution + offset);

        return Math.min(digitValue, 1);
    };

    float digit(float x, int digit, int maxDigits);

    default float defaultTruth() {
        return Float.NaN;
    }

    default float[] digits(float x, int digits) {
        return digits(x, new float[digits]);
    }

    default float[] digits(float x, float[] digits) {
        int n = digits.length;
        for (int i = 0; i < n; i++)
            digits[i] = digit(x, i, n);
        return digits;
    }

}