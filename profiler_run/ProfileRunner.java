import nars.NAR;
import nars.NARS;
import nars.Narsese;
import nars.concept.util.DefaultConceptBuilder;
import nars.io.NarseseParser;
import nars.memory.CaffeineMemory;
import nars.memory.Memory;
import nars.time.Time;
import nars.utils.Profiler;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class ProfileRunner {

    public static void main(String[] args) {
        System.out.println("Initializing NAR...");
        Memory memory = new CaffeineMemory(100000); // Example size
        Time time = new Time();
        NAR nar = new NAR(memory, time, Random::new, new DefaultConceptBuilder(), false);

        System.out.println("Enabling profiling...");
        nar.enableProfiling();
        Profiler.reset(); // Ensure stats are clean before starting

        String[] nalFiles = {
            "../../narchy/nar/src/main/resources/impl.compose.nal",
            "../../narchy/nar/src/main/resources/goal_analogy.nal",
            "../../narchy/nar/src/main/resources/delta.induction.nal"
        };

        for (String nalFile : nalFiles) {
            System.out.println("Loading NAL file: " + nalFile);
            try (BufferedReader reader = new BufferedReader(new FileReader(nalFile))) {
                String line;
                StringBuilder narseseInput = new StringBuilder();
                while ((line = reader.readLine()) != null) {
                    if (!line.trim().isEmpty() && !line.startsWith("//") && !line.startsWith("''")) {
                        narseseInput.append(line).append("\n");
                    }
                }
                if (narseseInput.length() > 0) {
                    //System.out.println("Inputting Narsese: \n" + narseseInput.toString().substring(0, Math.min(100, narseseInput.length())) + "...");
                    nar.input(narseseInput.toString());
                }
            } catch (IOException e) {
                System.err.println("Error reading NAL file " + nalFile + ": " + e.getMessage());
            } catch (Narsese.NarseseException e) {
                System.err.println("Narsese parsing error in file " + nalFile + ": " + e.getMessage());
            } catch (Exception e) {
                System.err.println("General error processing file " + nalFile + ": " + e.getMessage());
                e.printStackTrace();
            }
        }

        int totalCycles = 5000;
        System.out.println("Running NAR for " + totalCycles + " cycles...");
        for (int i = 0; i < totalCycles; i++) {
            if (i % 1000 == 0 && i > 0) {
                System.out.println("  Completed " + i + " cycles...");
            }
            nar.loop.next();
        }
        System.out.println("NAR run completed.");

        System.out.println("\n--- Collected Profiling Stats ---");
        String stats = nar.getProfilingStats();
        System.out.println(stats);
        System.out.println("---------------------------------");

        nar.delete();
        System.out.println("NAR instance deleted.");
    }
}
