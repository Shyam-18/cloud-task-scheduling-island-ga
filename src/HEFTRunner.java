// src/HEFTRunner.java

import java.io.IOException;

public class HEFTRunner {
    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            System.err.println("Usage: HEFTRunner <dagFile> <rackCount> <rackPenalty>");
            System.exit(-1);
        }
        String dagFile     = args[0];
        int    rackCount   = Integer.parseInt(args[1]);
        double rackPenalty = Double.parseDouble(args[2]);

        // 1) Build and execute HEFT
        HEFTScheduler heft = new HEFTScheduler(dagFile, rackCount, rackPenalty);
        // --- replace getPermutation() with your actual method that returns the schedule ---
        int[] schedule = heft.getPermutation();  

        // 2) Use PaperGA’s evaluator to get all fitness components
        PaperGA evaluator = new PaperGA(dagFile, rackCount, rackPenalty);
        // computeFitness returns [fitness, makespan, energy, crossRack]
        double[] fm = evaluator.computeFitness(schedule, 1.0, 0.0);

        // 3) Print all four metrics (variance is 0.0 for a single run)
        System.out.printf("HEFT_makespan  %.2f%n", fm[1]);
        System.out.printf("HEFT_energy    %.2f%n", fm[2]);
        System.out.printf("HEFT_crossRack %.2f%n", fm[3]);
    }
}

