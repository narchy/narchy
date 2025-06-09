package jcog.pri;

import jcog.Is;
import jcog.pri.bag.Bag;
import jcog.pri.op.PriAdd;
import jcog.pri.op.PriMult;
import jcog.random.RandomBits;
import jcog.random.XoRoShiRo128PlusRandom;
import org.jetbrains.annotations.Nullable;

import java.util.function.Consumer;

import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static jcog.Util.*;
import static jcog.pri.Prioritized.EPSILON;

@Is("Forgetting_curve") public enum Forgetting {
    ;


    public static @Nullable <Y extends Prioritizable> Consumer<Y> forget(Bag<?, Y> b, float strength) {
        return SimpleForgetMultiply.forget(b, strength);

        //return Homeostatic.forget(b, strength, false);
        //return Homeostatic.forget(b, strength, true);


        //return SimpleForgetMultiplySigmoid.forget(b, strength);

        //return SimpleForgetSubtract.forgetSubtract(b, strength);

        //return InfoForgetting.forgetMultiply(b, strength);

        //return FieldForgetting.forget(b, strength);

    }


    private static double idealExcess(double m, int n) {
        return Math.max(idealLimitRate * (m - (n * idealMeanPri)), 0);
    }

    public static final double idealMeanPri =
        //1/2.0  //LINEAR DISTRIBUTION (definite Integral of f=x on 0..1)
        1/3.0  //X^2    DISTRIBUTION (definite Integral of f=x^2 on 0..1)
        //1/4f
        //1/8f
        ;

    public static final double idealLimitRate =
        //0; //DISABLED
        //1;
        0.5f;
        //0.1f;
        //0.01f;

    private static final boolean idealLimit = idealLimitRate > 0;

    /**
     * Implements forgetting based on instantaneous system properties.
     * 1. Conservation of total priority mass
     * 2. Pressure-based decay
     * 3. Homeostatic equilibrium
     *
     * Multiplicative decay preserves the relative ranking of items, with higher-priority items being less affected.
     */
    public enum SimpleForgetMultiply { ;

//        @Deprecated private static final boolean multMethod1Or2 = false;

        @Nullable
        public static <Y extends Prioritizable> PriMult<Y> forget(Bag<?, Y> b, float temperature) {
            assertFinite(temperature);
            var s = b.size();
            if (s > 0) {
                var c = b.capacity();
                if (c > 0) {
                    double m = b.mass();
                    if (m > EPSILON) {

                        var pressure = b.depressurizePct(1);
                        var excess = temperature * pressure;
                        if (idealLimit) excess += idealExcess(m, s);

                        if (excess > EPSILON) {
                            var factor = (float)(//!multMethod1Or2 ?
                                /* asymptotic: */ m / (excess + m)
                                // /* linear: */ Util.max(1 - excess / m, 0) :
                            );
                            if (factor < 1 - EPSILON)
                                return new PriMult<>(factor);
                        }

                        //b.pressurize(pressure); //return unused pressure
                    }
                }
            }
            return null;
        }

    }

    /** provide a smooth, bidirectional homeostatic adjustment based on the total mass relative to the ideal target mass. */
    public enum Homeostatic {
        ;

        // --- Configuration Constants ---

        /** Target average priority for items in the bag. */
        public static final float IDEAL_MEAN_PRI =
            0;
            //1f / 3.0; // e.g., For x^2 distribution integral 0..1

        /**
         * Controls the strength of the homeostatic adjustment.
         * Higher values mean stronger push towards the IDEAL_MEAN_PRI.
         * Set to 0 to disable this homeostatic mechanism.
         */
        public static final float HOMEOSTATIC_RATE =
            //0.5f;
            //0.1f;
            0.01f;

        /**
         * Minimum denominator factor relative to mass 'm' in factor calculation.
         * Prevents division by zero/negative and limits the maximum growth factor
         * when effective excess is very negative. E.g., 1e-6 implies max growth factor ~1,000,000.
         */
        private static final float MIN_DENOMINATOR_FACTOR = Prioritized.EPSILON;

        /**
         * Calculates the homeostatic adjustment term based on the deviation
         * from the ideal total mass (n * IDEAL_MEAN_PRI).
         * This term is added to the pressure-based excess.
         * - Positive if current mass > ideal mass (increases forgetting).
         * - Negative if current mass < ideal mass (decreases forgetting / causes growth).
         *
         * @param currentTotalMass (m) Current sum of priorities.
         * @param itemCount (n) Number of items.
         * @param idealMeanPriority Target average priority.
         * @return The calculated adjustment value (can be positive or negative).
         */
        private static float homeostaticAdjustment(float currentTotalMass, int itemCount, float idealMeanPriority) {
            // Cannot calculate target mass or adjustment is meaningless if n=0 or ideal=0
            if (itemCount <= 0 || idealMeanPriority <= Prioritized.EPSILON)
                return 0;

            // Target total mass if all items had the ideal average priority
            float idealTotalMass = itemCount * idealMeanPriority;
            // Deviation from the ideal total mass
            return currentTotalMass - idealTotalMass;
        }

        /**
         * Implements forgetting/adjustment based on instantaneous system properties.
         * 1. Multiplicative factor preserves relative ranking.
         * 2. Includes pressure-based decay (via Bag's pressure mechanism).
         * 3. Includes homeostatic equilibrium seeking:
         *    - Increases forgetting if total mass > ideal target (factor < 1).
         *    - Decreases forgetting (or causes growth) if total mass < ideal target (factor > 1).
         *    - Smoothly guides the average priority towards IDEAL_MEAN_PRI over time.
         *
         * @param b The bag of prioritizable items.
         * @param temperature Scaling factor for pressure-based decay.
         * @return A PriMult object with the calculated factor if adjustment is needed, otherwise null.
         */
        @Nullable
        public static <Y extends Prioritizable> PriMult<Y> forget(Bag<?, Y> b, float temperature, boolean allowIncrease) {
            //assertFinite(temperature); // Ensure temperature is valid

            int s = b.size();
            if (s <= 0 || b.capacity() <= 0)
                return null; // No items or bag cannot hold items, nothing to adjust

            float m = b.mass();
            if (m <= Prioritized.EPSILON)
                return null; // Mass is negligible, no meaningful adjustment possible


            // 1. Calculate base excess from pressure mechanism
            // Assumes depressurizePct returns a non-negative value representing pressure potential removed.
            float pressure = b.depressurizePct(1.0f); // Attempt to use all available pressure potential
//            double pressureExcess = temperature * pressure;
//            // Sanity check pressure contribution
//            if (!Double.isFinite(pressureExcess) || pressureExcess < 0) {
//                // Log warning or handle error? For now, reset and maybe return pressure.
//                //System.err.println("Warning: Invalid pressureExcess calculated: " + pressureExcess);
//                pressureExcess = 0;
//                // If pressure was taken but deemed invalid, should we return it? Depends on Bag logic.
//                // if (pressure > 0) b.pressurize(pressure);
//                // Let's proceed without pressureExcess for now.
//            }


            // 2. Calculate homeostatic adjustment (can be positive or negative)
            float excess = (pressure + homeostaticAdjustment(m, s, IDEAL_MEAN_PRI) * HOMEOSTATIC_RATE) * temperature;
            // Sanity check adjustment
//                if (!Double.isFinite(homeostaticAdjustment)) {
////                    System.err.println("Warning: Invalid homeostaticAdjustment calculated: " + homeostaticAdjustment);
//                    homeostaticAdjustment = 0; // Ignore if calculation failed
//                }
//            // 3. Combine sources of excess/deficit
//            double excess = homeostaticAdjustment/* + pressureExcess*/;

            // 4. Calculate the multiplicative factor using the extended asymptotic formula
            // Formula: factor = m / (m + totalEffectiveExcess)
            // Protect the denominator: ensure it's positive and not too close to zero.
            // Clamp denominator to max( m * MIN_DENOMINATOR_FACTOR, EPSILON )
            // Using relative factor m * MIN_DENOMINATOR_FACTOR is generally robust.
            // Adding absolute EPSILON provides safety if m itself is pathologically small but > EPSILON.
            float minDenominator = Math.max(m * MIN_DENOMINATOR_FACTOR, Prioritized.EPSILON);
            float denominator = Math.max(m + excess, minDenominator);

            float factor = m / denominator;
            if (factor < 0)
                throw new UnsupportedOperationException();

            // 5. Final checks and decision to apply
            // Ensure factor is valid (finite, positive)
            if (factor==factor) {// Check if the factor is significantly different from 1.0
                boolean apply;
                if (allowIncrease) {
                    apply = Math.abs(1 - factor) > Prioritized.EPSILON;
                } else {
                    apply = 1 - factor > Prioritized.EPSILON;
                }
                if (apply) {
                    // Factor indicates meaningful decay (factor < 1) or growth (factor > 1).
                    // Pressure taken is considered "used" to achieve this adjustment.
                    return new PriMult<>(factor);
                }// else {
                    // Factor is too close to 1.0, no significant change needed.
                //}
            //} else {
                //System.err.println("Warning: Invalid factor calculated (" + factor + "). m=" + m
                //  + ", totalEffectiveExcess=" + totalEffectiveExcess + ", denominator=" + denominator +". No adjustment applied.");
            }

            //if (pressure > 0) b.pressurize(pressure); // Return unused pressure if factor calculation failed
            return null;
        }
    }
    /**
     * decay_factor=1−(base_k+sigmoid((current_mass−ideal_mass)×scaling_factor))×temperature
     *   base_k: A constant representing the base decay rate.
     *   sigmoid(x): A sigmoid function to smoothly map the mass difference to the adjustment term.
     *   scaling_factor: Determines the sensitivity of the sigmoid function to the mass difference
     *   temperature: A parameter modulating the overall decay rate.
     */
    public enum SimpleForgetMultiplySigmoid {
        ;

        @Deprecated private static final boolean multMethod1Or2 = false;

        @Nullable
        public static <Y extends Prioritizable> PriMult<Y> forget(Bag<?, Y> b, float temperature) {
            assertFinite(temperature);
            var s = b.size();
            if (s > 0) {
                var c = b.capacity();
                if (c > 0) {
                    double m = b.mass();
                    if (m > EPSILON) {

                        var pressure = b.depressurizePct(1);

                        float base_k = 0;
                        float scalingFactor = pressure/2;
                        float factor = (float)(
                            1 - (base_k + sigmoid(m/c - idealMeanPri) * scalingFactor) * temperature
                        );

                        if (factor < 1 - EPSILON)
                            return new PriMult<>(factor);

                        b.pressurize(pressure); //return unused pressure
                    }
                }
            }
            return null;
        }

    }
    public enum SimpleForgetSubtract {
        ;


        @Nullable
        public static <Y extends Prioritizable> PriAdd<Y> forgetSubtract(Bag<?, Y> b, float temperature) {
            assertFinite(temperature);

            var s = b.size();
            if (s > 0) {
                var c = b.capacity();
                if (c > 0) {
                    double m = b.mass();
                    if (m > EPSILON) {

                        float pressure = b.depressurizePct(1);

                        double excess = ((double)temperature) * pressure;
                        if (idealLimit) excess += idealExcess(m, s);

                        var excessEach = (float) (excess / s);
                        if (excessEach > EPSILON)
                            return new PriAdd<>(-excessEach);

                        b.pressurize(pressure); //return unused pressure

                    }

                }
            }

            return null;
        }


//        /**
//         * allows strength values >1 for active forgetting
//         */
//        public static <X extends Prioritizable> Consumer<X> forgetSubtractSustain(Bag<?, X> b, float strength) {
//            assertValidForgetStrength(strength);
//            var s = b.size();
//            if (s <= 0) return null;
//
//            var p = forgetSubtract(b, Util.min(1, strength));
//            if (p == null) {
//                if (strength <= 1)
//                    return null;
//                p = new PriAdd<>(0);
//            }
//            if (strength > 1) {
//                double priMean = b.mass() / s;
//                p.x -= priMean * (strength - 1) * idealMeanPri;
//                //p.x -= priMean * (strength-1);
//            }
//
//            return p.x > -EPSILON ? null : p;
//        }
//
//        public static <X extends Prioritizable> Consumer<X> forgetMultiplySustain(Bag<?, X> b, float strength) {
//            assertValidForgetStrength(strength);
//            return forgetMultiply(b, strength);
//        }
//
//        public static void assertValidForgetStrength(float strength) {
//            if (strength < 0 || strength > 2)
//                throw new UnsupportedOperationException();
//        }
    }


    /**
     * Implements priority forgetting using field theory principles.
     * Priorities behave as charges in a field with quantum properties,
     * creating natural forgetting through field interactions.
     */
    public enum FieldForgetting { ;

        private static final double EPSILON = 1e-10;
        private static final double φ = (1 + sqrt(5)) / 2;  // Golden ratio

        /**
         * Field configuration for the cognitive space.
         * - entropyWeight: Balance between field and entropy effects [0,1]
         * - fieldStrength: Base interaction strength
         * - quantization: Minimum distinguishable priority difference
         */
        public record FieldConfig(
                double entropyWeight,
                double fieldStrength,
                double quantization
        ) {
            public FieldConfig {
                if (entropyWeight < 0 || entropyWeight > 1 ||
                        fieldStrength <= 0 || quantization <= 0) {
                    throw new IllegalArgumentException("Invalid field parameters");
                }
            }

        }

        public static final FieldConfig DEFAULT = new FieldConfig(0.5, 1.0, 0.01);

        /**
         * Creates a forgetting operation based on field interactions.
         * @param temperature Controls overall decay rate
         * @return Priority adjustment function or null if no adjustment needed
         */
        public static <X extends Prioritizable> Consumer<X> forget(Bag<?,X> b, float temperature/*, FieldConfig cfg*/) {

            var cfg = DEFAULT;

            if (temperature < 0 || !Double.isFinite(temperature))
                throw new IllegalArgumentException("Invalid temperature");

            var s = b.size();
            if (s == 0) return null;

            double mass = b.mass();
            if (mass <= EPSILON) return null;

            // Field metrics
            var density = mass / s;
            double pressure = b.pressure();

            // Field potential combining local density and global pressure
            var fieldPotential = sqrt(
                    pressure * density *
                            mass * cfg.fieldStrength / pow(s, φ)
            );

            // Entropy factor using q-exponential
            var entropyFactor = qExp(1 - density/mass, 1 + temperature);

            var entropyWeight = cfg.entropyWeight;

            // Combined decay strength
            var decayStrength = temperature * (
                    (1 - entropyWeight) * fieldPotential +
                            entropyWeight * entropyFactor
            );

            return decayStrength <= EPSILON ? null :
                decay((float) decayStrength, cfg);
        }

        /**
         * Creates the priority decay function based on field strength
         */
        private static <X extends Prioritizable> Consumer<X> decay(float strength, FieldConfig cfg) {

            var rng = new RandomBits(new XoRoShiRo128PlusRandom());
            return x -> {
                double pri = x.pri();

                float nextPri;

                // Quantum tunneling for small priorities
                if (pri < cfg.quantization && rng.nextBooleanFast8(strength)) {
                    nextPri = 0;
                } else {

                    // Field-based priority update with quantization
                    var phase = pri * φ;
                    var fieldEffect = strength * Math.sin(phase);
                    var continuous = pri - fieldEffect;

                    // Quantum effects near quantization boundaries
                    var quantized = round(continuous, cfg.quantization);

                    // Interpolate based on field strength
                    nextPri = (float) Math.max(0, lerp(strength, continuous, quantized));
                }
                x.pri(nextPri);
            };
        }

        /**
         * Calculates q-exponential for non-extensive entropy
         */
        private static double qExp(double x, double q) {
            if (Math.abs(q - 1) < EPSILON) return Math.exp(x);
            var base = 1 + (1 - q) * x;
            return base > 0 ? pow(base, 1/(1-q)) : 0;
        }

    }

    /** Information-theoretic forgetting mechanism based on entropy maximization principles
     *
     *    Combines physical constraints (bag capacity) with information-theoretic principles (entropy) to create a self-regulating system that:
     *
     *    Maintains optimal cognitive load
     *    Fairly distributes resources
     *    Preserves important priorities
     *    Automatically adapts to system state
     *
     *    Homeostasis:
     *      System maintains balance around IDEAL_ENTROPY
     *      Automatic adjustment based on current state
     *      System increases forgetting when entropy deviates from ideal
     *    Scale invariant (works with any priority range)
     *    Numerically stable (handles small values)
     *    Continuous and smooth transitions     *
     *
     *    a) Multiplicative
     *      Asymptotic decay that preserves relative priorities
     *      Entropy-weighted to maintain distribution balance
     *      More stable for long-term memory simulation
     *    b) Subtractive:
     *      Linear decay that reduces absolute priorities
     *      Distributes forgetting pressure evenly
     *      Better for immediate resource management

     * */
    public enum InfoForgetting {
        ;

        /** ln(3) == ~1.098: theoretical optimum for cognitive load */
        private static final double ENTROPY_IDEAL = Math.log(3);

        private static final double EPSILON = Prioritized.EPSILON;

        public static <Y extends Prioritizable> PriMult<Y> forgetMultiply(Bag<?,Y> b, float temperature) {
            var s = b.size();
            var m = b.mass();
            if (!isValid(temperature, s, m)) return null;

            var H = entropy(b, m);
            var excess = temperature * pressure(b, H, s);
            if (idealLimit) excess += idealExcess(m, s /*b.capacity()*/);

            if (excess <= EPSILON) return null;

            var factor = (float)(m / (excess + m) * sqrt(H/Math.log(s)));
            return (factor < 1 - EPSILON) ? new PriMult<>(factor) : null;
        }

        private static <Y extends Prioritizable> double pressure(Bag<?, Y> b, double H, int n) {
            return b.depressurizePct(1) *
                    (1 + Math.max(0, (ENTROPY_IDEAL - H)));
                    //(1 + Util.max(0, (ENTROPY_IDEAL - H) / Math.log(n)));
                   //(1 + Util.max(0, IDEAL_ENTROPY - H / Math.log(n)));
        }

        private static <Y extends Prioritizable> boolean isValid(float temperature, int n, float mass) {
            return temperature > 0 && n > 0 && mass > EPSILON;
        }

        private static <Y extends Prioritizable> double entropy(Bag<?,Y> b, double mass) {
            double H = 0;
            for (var x : b) {
                var p = x.priElseZero() / mass;
                if (p > EPSILON) H -= p * Math.log(p);
            }
            return H;
        }
//        private static <Y extends Prioritizable> double entropyB(Bag<?,Y> b, double mass) {
//            double BOLTZMANN = 1;  // Normalized constant
//            double Z = 0;  // Partition function
//            double E = 0;  // Energy
//            double T = b.pressure();  // Temperature from pressure
//
//            for (var p : b) {
//                double e = -Math.log(p.priElseZero() + EPSILON);  // Energy level
//                var dz = Math.exp(-e / (BOLTZMANN * T));
//                Z += dz;
//                E += e * dz;
//            }
//            return Math.log(Z) + E / (T * BOLTZMANN);
//        }

    }


}
