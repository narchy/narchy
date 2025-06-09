package nars.util;

import jcog.Util;
import jcog.data.set.MetalLongSet;
import jcog.table.ARFF;
import jcog.table.DataTable;
import nars.NAR;
import nars.NARS;
import nars.Narsese;
import nars.task.NALTask;
import nars.truth.Truth;
import org.jetbrains.annotations.Nullable;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static nars.Op.*;

/** TODO */
@Disabled class NALDataTest {

    private final NAR n = NARS.tmp();
    {
        n.complexMax.set(64);
    }

    @Test
    void validatePredictionXOR() throws Exception {
        validatePrediction(n, xor2ARFF);
    }

    @Test
    void validatePredictionShuttle() throws Exception {
        //n.log();
        validatePrediction(n, shuttleARFF);
    }

    @Test
    void validatePredictionIris() throws Exception {
//        n.log();
        validatePrediction(n, irisARFF);
    }


    static NAR validatePrediction(NAR n, String arffData, String... hints) throws IOException, ARFF.ARFFParseError {
//        ArrayHashSet<Row> data = new ArrayHashSet<>();

        DataTable dataset = new ARFF(arffData) {
            @Override
            public boolean add(Object... point) {
                
                if (arffData == irisARFF) {
                    
                    for (int i = 0, pointLength = point.length; i < pointLength; i++) {
                        Object x = point[i];
                        if (x instanceof Number) {
                            point[i] = Math.round(((Number) x).floatValue() * 10);
                        }
                    }
                }
                return super.add(point);
            }
        };


//        int originalDataSetSize = dataset.rowCount();
//
//        Random rng = new XoRoShiRo128PlusRandom(1);
//
//        int validationSetSize = 1;
////        Collection<Row> validationPointsActual = new Lst(validationSetSize);
//        Collection<Row> validationPoints = new Lst(validationSetSize);
//        for (int i= 0; i < validationSetSize; i++) {
//            int r = rng.nextInt(originalDataSetSize);
//            Row randomPoint = dataset.row(r);
//            validationPoints.add(randomPoint);
//
//
////            MutableList randomPointErased = randomPoint.row(0).toList();
////            randomPointErased.set(randomPointErased.size()-1, "?class");
////            validationPoints.add(randomPointErased.toImmutable());
//        }
////
//        Table validation = dataset.emptyCopy();
//        validationPoints.forEach(validation::addRow);
//        System.out.println(validation.print());
//
//        assertEquals(originalDataSetSize, validation.rowCount() + dataset.rowCount());






        MetalLongSet questions = new MetalLongSet(4);
        n.main().onTask(t2 ->{
            NALTask t12 = (NALTask) t2;
            if (t12.isInput())
                questions.add(t12.stamp()[0]);
        }, QUESTION, QUEST);

        //if (t.isInput())
        n.main().onTask(t11 ->{
            NALTask t1 = (NALTask) t11;
            if (overlapsAny(questions, t1.stamp())) {
                //if (t.isInput())
                    System.out.println("ANSWER: " + t11);
            }
        }, BELIEF, GOAL);


        NALData.believe(dataset, NALData.predictsLast, n);



//        Task[] questions1 = NALSchema.data(n, validation, QUESTION, NALSchema.predictsLast).toArray(Task[]::new);
//        new DialogTask(n, questions1) {
//            @Override
//            protected boolean onTask(Task x, Term unifiedWith) {
//                System.out.println(unifiedWith + ": " + x);
//                return super.onTask(x, unifiedWith);
//            }
//            //            @Override
////            protected boolean onTask(Task x) {
////                if (!x.isInput())
////                    System.out.println(x);
////                return true;
////            }
//        };

        try {
            n.input(hints);
        } catch (Narsese.NarseseException e) {
            e.printStackTrace();
        }

        n.run(5000);

        //check beliefs for contradictions or drift

        float freqRes = n.freqRes.floatValue();
        NALData.beliefs(dataset, NALData.predictsLast, n).forEach(t -> {
            @Nullable Truth actual = n.beliefTruth(t.term(), ETERNAL);
            if (actual == null) {
                //lost belief
                System.err.println("forgot: " + t);
            } else {
                Truth ideal = t.truth();
                if (ideal.POSITIVE()!=actual.POSITIVE()) {
                    //polarity reversal
                    System.err.println("polarity reversal: " + t  + " -> " + actual);
                } else {
                    if (!Util.equals(ideal.freq(), actual.freq(), freqRes)) {
                        System.err.println("freq change: " + t  + " -> " + actual);
                    }
                }
            }
        });

        return n;
    }

    @Deprecated static boolean overlapsAny(MetalLongSet aa,  long[] b) {
        for (long x : b)
            if (aa.contains(x))
                return true;
        return false;
    }


    /** https://github.com/renatopp/arff-datasets/blob/master/boolean/xor.arff */
    private static final String xor2ARFF = """
            %
            % XOR
            %

            @RELATION XOR

            @ATTRIBUTE input1 REAL
            @ATTRIBUTE input2 REAL
            @ATTRIBUTE y REAL

            @DATA
            0.0,0.0,0.0
            0.0,1.0,1.0
            1.0,0.0,1.0
            1.0,1.0,0.0
            """;

    private static final String shuttleARFF = """
            % 1. Title: Space Shuttle Autolanding Domain
            %\s
            % 2. Sources:
            %     (a) Original source: unknown
            %         -- NASA: Mr. Roger Burke's autolander design team
            %     (b) Donor: Bojan Cestnik
            %                 Jozef Stefan Institute
            %                 Jamova 39
            %                 61000 Ljubljana
            %                 Yugoslavia (tel.: (38)(+61) 214-399 ext.287)\s
            %     (c) Date: November 1988
            %\s
            % 3. Past Usage: (several, it appears)
            %      Example: Michie,D. (1988).  The Fifth Generation's Unbridged Gap.
            %               In Rolf Herken (Ed.) The Universal Turing Machine: A
            %               Half-Century Survey, 466-489, Oxford University Press.
            %\s
            % 4. Relevant Information:
            %      This is a tiny database.  Michie reports that Burke's group used
            %      RULEMASTER to generate comprehendable rules for determining
            %      the conditions under which an autolanding would be preferable to
            %      manual control of the spacecraft.
            %\s
            % 5. Number of Instances: 15
            %\s
            % 6. Number of Attributes: 7 (including the class attribute)
            %\s
            % 7. Attribute Information:
            %     1. STABILITY: stab, xstab
            %     2. ERROR: XL, LX, MM, SS
            %     3. SIGN: pp, nn
            %     4. WIND: head, tail
            %     5. MAGNITUDE: Low, Medium, Strong, OutOfRange
            %     6. VISIBILITY: yes, no
            %     7. Class: noauto, auto
            %        -- that is, advise using manual/automatic control
            %\s
            % 8. Missing Attribute Values:
            %    -- none
            %    -- but several "don't care" values: (denoted by "_")
            %          Attribute Number:   Number of Don't Care Values:
            %                         2:   2
            %                         3:   3
            %                         4:   8
            %                         5:   8
            %                         6:   5
            %                         7:   0
            %\s
            % 9. Class Distribution:
            %     1. Use noauto control: 6
            %     2. Use automatic control: 9%
            % Information about the dataset
            % CLASSTYPE: nominal
            % CLASSINDEX: first
            %
            @relation shuttle-landing-control
            @attribute STABILITY {1,2}
            @attribute ERROR {1,2,3,4}
            @attribute SIGN {1,2}
            @attribute WIND {1,2}
            @attribute MAGNITUDE {1,2,3,4}
            @attribute VISIBILITY {vis,invis}
            @attribute Class {manual,auto}
            @data
            _,_,_,_,_,invis,auto
            2,_,_,_,_,vis,manual
            1,2,_,_,_,vis,manual
            1,1,_,_,_,vis,manual
            1,3,2,2,_,vis,manual
            _,_,_,_,4,vis,manual
            1,4,_,_,1,vis,auto
            1,4,_,_,2,vis,auto
            1,4,_,_,3,vis,auto
            1,3,1,1,1,vis,auto
            1,3,1,1,2,vis,auto
            1,3,1,2,1,vis,auto
            1,3,1,2,2,vis,auto
            1,3,1,1,3,vis,manual
            1,3,1,2,3,vis,auto
            """;

    private static final String irisARFF = """
            % 1. Title: Iris Plants Database
            %\s
            % 2. Sources:
            %      (a) Creator: R.A. Fisher
            %      (b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
            %      (c) Date: July, 1988
            %\s
            % 3. Past Usage:
            %    - Publications: too many to mention!!!  Here are a few.
            %    1. Fisher,R.A. "The use of multiple measurements in taxonomic problems"
            %       Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions
            %       to Mathematical Statistics" (John Wiley, NY, 1950).
            %    2. Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
            %       (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
            %    3. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
            %       Structure and Classification Rule for Recognition in Partially Exposed
            %       Environments".  IEEE Transactions on Pattern Analysis and Machine
            %       Intelligence, Vol. PAMI-2, No. 1, 67-71.
            %       -- Results:
            %          -- very low misclassification rates (0% for the setosa class)
            %    4. Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE\s
            %       Transactions on Information Theory, May 1972, 431-433.
            %       -- Results:
            %          -- very low misclassification rates again
            %    5. See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al's AUTOCLASS II
            %       conceptual clustering system finds 3 classes in the data.
            %\s
            % 4. Relevant Information:
            %    --- This is perhaps the best known database to be found in the pattern
            %        recognition literature.  Fisher's paper is a classic in the field
            %        and is referenced frequently to this day.  (See Duda & Hart, for
            %        example.)  The data setAt contains 3 classes of 50 instances each,
            %        where each class refers to a type of iris plant.  One class is
            %        linearly separable from the other 2; the latter are NOT linearly
            %        separable from each other.
            %    --- Predicted attribute: class of iris plant.
            %    --- This is an exceedingly simple domain.
            %\s
            % 5. Number of Instances: 150 (50 in each of three classes)
            %\s
            % 6. Number of Attributes: 4 numeric, predictive attributes and the class
            %\s
            % 7. Attribute Information:
            %    1. sepal length in cm
            %    2. sepal width in cm
            %    3. petal length in cm
            %    4. petal width in cm
            %    5. class:\s
            %       -- Iris Setosa
            %       -- Iris Versicolour
            %       -- Iris Virginica
            %\s
            % 8. Missing Attribute Values: None
            %\s
            % Summary Statistics:
            %  \t           Min  Max   Mean    SD   Class Correlation
            %    sepal length: 4.3  7.9   5.84  0.83    0.7826  \s
            %     sepal width: 2.0  4.4   3.05  0.43   -0.4194
            %    petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
            %     petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
            %\s
            % 9. Class Distribution: 33.3% for each of 3 classes.

            @RELATION iris

            @ATTRIBUTE "(sepal,length)"\tREAL
            @ATTRIBUTE "(sepal,width)"\tREAL
            @ATTRIBUTE "(petal,length)"\tREAL
            @ATTRIBUTE "(petal,width)"\tREAL
            @ATTRIBUTE class \t{Iris_setosa,Iris_versicolor,Iris_virginica}

            @DATA
            5.1,3.5,1.4,0.2,Iris_setosa
            4.9,3.0,1.4,0.2,Iris_setosa
            4.7,3.2,1.3,0.2,Iris_setosa
            4.6,3.1,1.5,0.2,Iris_setosa
            5.0,3.6,1.4,0.2,Iris_setosa
            5.4,3.9,1.7,0.4,Iris_setosa
            4.6,3.4,1.4,0.3,Iris_setosa
            5.0,3.4,1.5,0.2,Iris_setosa
            4.4,2.9,1.4,0.2,Iris_setosa
            4.9,3.1,1.5,0.1,Iris_setosa
            5.4,3.7,1.5,0.2,Iris_setosa
            4.8,3.4,1.6,0.2,Iris_setosa
            4.8,3.0,1.4,0.1,Iris_setosa
            4.3,3.0,1.1,0.1,Iris_setosa
            5.8,4.0,1.2,0.2,Iris_setosa
            5.7,4.4,1.5,0.4,Iris_setosa
            5.4,3.9,1.3,0.4,Iris_setosa
            5.1,3.5,1.4,0.3,Iris_setosa
            5.7,3.8,1.7,0.3,Iris_setosa
            5.1,3.8,1.5,0.3,Iris_setosa
            5.4,3.4,1.7,0.2,Iris_setosa
            5.1,3.7,1.5,0.4,Iris_setosa
            4.6,3.6,1.0,0.2,Iris_setosa
            5.1,3.3,1.7,0.5,Iris_setosa
            4.8,3.4,1.9,0.2,Iris_setosa
            5.0,3.0,1.6,0.2,Iris_setosa
            5.0,3.4,1.6,0.4,Iris_setosa
            5.2,3.5,1.5,0.2,Iris_setosa
            5.2,3.4,1.4,0.2,Iris_setosa
            4.7,3.2,1.6,0.2,Iris_setosa
            4.8,3.1,1.6,0.2,Iris_setosa
            5.4,3.4,1.5,0.4,Iris_setosa
            5.2,4.1,1.5,0.1,Iris_setosa
            5.5,4.2,1.4,0.2,Iris_setosa
            4.9,3.1,1.5,0.1,Iris_setosa
            5.0,3.2,1.2,0.2,Iris_setosa
            5.5,3.5,1.3,0.2,Iris_setosa
            4.9,3.1,1.5,0.1,Iris_setosa
            4.4,3.0,1.3,0.2,Iris_setosa
            5.1,3.4,1.5,0.2,Iris_setosa
            5.0,3.5,1.3,0.3,Iris_setosa
            4.5,2.3,1.3,0.3,Iris_setosa
            4.4,3.2,1.3,0.2,Iris_setosa
            5.0,3.5,1.6,0.6,Iris_setosa
            5.1,3.8,1.9,0.4,Iris_setosa
            4.8,3.0,1.4,0.3,Iris_setosa
            5.1,3.8,1.6,0.2,Iris_setosa
            4.6,3.2,1.4,0.2,Iris_setosa
            5.3,3.7,1.5,0.2,Iris_setosa
            5.0,3.3,1.4,0.2,Iris_setosa
            7.0,3.2,4.7,1.4,Iris_versicolor
            6.4,3.2,4.5,1.5,Iris_versicolor
            6.9,3.1,4.9,1.5,Iris_versicolor
            5.5,2.3,4.0,1.3,Iris_versicolor
            6.5,2.8,4.6,1.5,Iris_versicolor
            5.7,2.8,4.5,1.3,Iris_versicolor
            6.3,3.3,4.7,1.6,Iris_versicolor
            4.9,2.4,3.3,1.0,Iris_versicolor
            6.6,2.9,4.6,1.3,Iris_versicolor
            5.2,2.7,3.9,1.4,Iris_versicolor
            5.0,2.0,3.5,1.0,Iris_versicolor
            5.9,3.0,4.2,1.5,Iris_versicolor
            6.0,2.2,4.0,1.0,Iris_versicolor
            6.1,2.9,4.7,1.4,Iris_versicolor
            5.6,2.9,3.6,1.3,Iris_versicolor
            6.7,3.1,4.4,1.4,Iris_versicolor
            5.6,3.0,4.5,1.5,Iris_versicolor
            5.8,2.7,4.1,1.0,Iris_versicolor
            6.2,2.2,4.5,1.5,Iris_versicolor
            5.6,2.5,3.9,1.1,Iris_versicolor
            5.9,3.2,4.8,1.8,Iris_versicolor
            6.1,2.8,4.0,1.3,Iris_versicolor
            6.3,2.5,4.9,1.5,Iris_versicolor
            6.1,2.8,4.7,1.2,Iris_versicolor
            6.4,2.9,4.3,1.3,Iris_versicolor
            6.6,3.0,4.4,1.4,Iris_versicolor
            6.8,2.8,4.8,1.4,Iris_versicolor
            6.7,3.0,5.0,1.7,Iris_versicolor
            6.0,2.9,4.5,1.5,Iris_versicolor
            5.7,2.6,3.5,1.0,Iris_versicolor
            5.5,2.4,3.8,1.1,Iris_versicolor
            5.5,2.4,3.7,1.0,Iris_versicolor
            5.8,2.7,3.9,1.2,Iris_versicolor
            6.0,2.7,5.1,1.6,Iris_versicolor
            5.4,3.0,4.5,1.5,Iris_versicolor
            6.0,3.4,4.5,1.6,Iris_versicolor
            6.7,3.1,4.7,1.5,Iris_versicolor
            6.3,2.3,4.4,1.3,Iris_versicolor
            5.6,3.0,4.1,1.3,Iris_versicolor
            5.5,2.5,4.0,1.3,Iris_versicolor
            5.5,2.6,4.4,1.2,Iris_versicolor
            6.1,3.0,4.6,1.4,Iris_versicolor
            5.8,2.6,4.0,1.2,Iris_versicolor
            5.0,2.3,3.3,1.0,Iris_versicolor
            5.6,2.7,4.2,1.3,Iris_versicolor
            5.7,3.0,4.2,1.2,Iris_versicolor
            5.7,2.9,4.2,1.3,Iris_versicolor
            6.2,2.9,4.3,1.3,Iris_versicolor
            5.1,2.5,3.0,1.1,Iris_versicolor
            5.7,2.8,4.1,1.3,Iris_versicolor
            6.3,3.3,6.0,2.5,Iris_virginica
            5.8,2.7,5.1,1.9,Iris_virginica
            7.1,3.0,5.9,2.1,Iris_virginica
            6.3,2.9,5.6,1.8,Iris_virginica
            6.5,3.0,5.8,2.2,Iris_virginica
            7.6,3.0,6.6,2.1,Iris_virginica
            4.9,2.5,4.5,1.7,Iris_virginica
            7.3,2.9,6.3,1.8,Iris_virginica
            6.7,2.5,5.8,1.8,Iris_virginica
            7.2,3.6,6.1,2.5,Iris_virginica
            6.5,3.2,5.1,2.0,Iris_virginica
            6.4,2.7,5.3,1.9,Iris_virginica
            6.8,3.0,5.5,2.1,Iris_virginica
            5.7,2.5,5.0,2.0,Iris_virginica
            5.8,2.8,5.1,2.4,Iris_virginica
            6.4,3.2,5.3,2.3,Iris_virginica
            6.5,3.0,5.5,1.8,Iris_virginica
            7.7,3.8,6.7,2.2,Iris_virginica
            7.7,2.6,6.9,2.3,Iris_virginica
            6.0,2.2,5.0,1.5,Iris_virginica
            6.9,3.2,5.7,2.3,Iris_virginica
            5.6,2.8,4.9,2.0,Iris_virginica
            7.7,2.8,6.7,2.0,Iris_virginica
            6.3,2.7,4.9,1.8,Iris_virginica
            6.7,3.3,5.7,2.1,Iris_virginica
            7.2,3.2,6.0,1.8,Iris_virginica
            6.2,2.8,4.8,1.8,Iris_virginica
            6.1,3.0,4.9,1.8,Iris_virginica
            6.4,2.8,5.6,2.1,Iris_virginica
            7.2,3.0,5.8,1.6,Iris_virginica
            7.4,2.8,6.1,1.9,Iris_virginica
            7.9,3.8,6.4,2.0,Iris_virginica
            6.4,2.8,5.6,2.2,Iris_virginica
            6.3,2.8,5.1,1.5,Iris_virginica
            6.1,2.6,5.6,1.4,Iris_virginica
            7.7,3.0,6.1,2.3,Iris_virginica
            6.3,3.4,5.6,2.4,Iris_virginica
            6.4,3.1,5.5,1.8,Iris_virginica
            6.0,3.0,4.8,1.8,Iris_virginica
            6.9,3.1,5.4,2.1,Iris_virginica
            6.7,3.1,5.6,2.4,Iris_virginica
            6.9,3.1,5.1,2.3,Iris_virginica
            5.8,2.7,5.1,1.9,Iris_virginica
            6.8,3.2,5.9,2.3,Iris_virginica
            6.7,3.3,5.7,2.5,Iris_virginica
            6.7,3.0,5.2,2.3,Iris_virginica
            6.3,2.5,5.0,1.9,Iris_virginica
            6.5,3.0,5.2,2.0,Iris_virginica
            6.2,3.4,5.4,2.3,Iris_virginica
            5.9,3.0,5.1,1.8,Iris_virginica
            """;
}