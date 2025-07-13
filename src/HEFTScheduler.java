// HEFTScheduler.java
// A standalone HEFT heuristic that exposes its task→CPU assignment,
// the HEFT task ordering as a permutation, and basic stats.

import java.io.*;
import java.util.*;

public class HEFTScheduler {
    private final int nTasks;
    private final int nCPUs;
    private final double[][] compCost;
    private final double[][] commCost;
    private final List<List<Integer>> succ;
    private final List<List<Integer>> pred;
    private final double[] cpuReady;
    private final double[] taskFinish;
    private final double[] rankUp;
    private final int[] assignment;

    // remembers the HEFT ordering
    private int[] lastPermutation;

    // rack‐awareness parameters (currently unused)
    private final int numRacks;
    private final double rackPenalty;

    /** 
     * Original constructor (no rack penalty): delegates to full overload.
     */
    public HEFTScheduler(String filename) throws IOException {
        this(filename, 1, 1.0);
    }

    /**
     * Overloaded constructor with rackCount & rackPenalty.
     */
    public HEFTScheduler(String filename, int numRacks, double rackPenalty)
            throws IOException {
        this.numRacks    = numRacks;
        this.rackPenalty = rackPenalty;

        // 1) Read all non-blank lines
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String L;
            while ((L = br.readLine()) != null) {
                if (!L.trim().isEmpty()) lines.add(L.trim());
            }
        }

        // 2) Strip optional last line of two numbers
        int sz = lines.size();
        String last = lines.get(sz - 1);
        String[] parts = last.split("\\s+");
        if (parts.length == 2) {
            try {
                Double.parseDouble(parts[0]);
                Double.parseDouble(parts[1]);
                lines.remove(sz - 1);
                sz--;
            } catch (NumberFormatException ex) { /* ignore */ }
        }

        this.nTasks = sz;
        // determine nCPUs from first line
        String first = lines.get(0);
        String[] hl = first.split("\\|");
        String[] lt = hl[0].trim().split("\\s+");
        this.nCPUs = lt.length - 1;

        compCost = new double[nTasks][nCPUs];
        commCost = new double[nTasks][nTasks];
        succ     = new ArrayList<>(nTasks);
        pred     = new ArrayList<>(nTasks);
        for (int i = 0; i < nTasks; i++) {
            succ.add(new ArrayList<>());
            pred.add(new ArrayList<>());
        }

        // 3) Parse each task line
        for (int idx = 0; idx < nTasks; idx++) {
            String line = lines.get(idx);
            String[] halves = line.split("\\|");
            String[] left = halves[0].trim().split("\\s+");
            int tID = Integer.parseInt(left[0]);
            for (int c = 0; c < nCPUs; c++) {
                compCost[tID][c] = Double.parseDouble(left[c + 1]);
            }
            if (halves.length > 1) {
                for (String chunk : halves[1].split("\\),")) {
                    String P = chunk.replaceAll("[()]", "").trim();
                    if (P.isEmpty()) continue;
                    String[] cw = P.split("[,\\s]+");
                    int ch = Integer.parseInt(cw[0]);
                    double w = Double.parseDouble(cw[1]);
                    succ.get(tID).add(ch);
                    pred.get(ch).add(tID);
                    commCost[tID][ch] = w;
                }
            }
        }

        cpuReady   = new double[nCPUs];
        taskFinish = new double[nTasks];
        rankUp     = new double[nTasks];
        Arrays.fill(rankUp, -1.0);
        assignment = new int[nTasks];
    }

    // ----------------------------------------------------------------
    // Memoized computation of the upward rank for task t
    // ----------------------------------------------------------------
    private double computeRankUp(int t) {
        if (rankUp[t] >= 0) return rankUp[t];
        double avg = 0;
        for (int c = 0; c < nCPUs; c++) avg += compCost[t][c];
        avg /= nCPUs;
        if (succ.get(t).isEmpty()) {
            rankUp[t] = avg;
        } else {
            double maxChild = 0;
            for (int ch : succ.get(t)) {
                double val = commCost[t][ch] + computeRankUp(ch);
                if (val > maxChild) maxChild = val;
            }
            rankUp[t] = avg + maxChild;
        }
        return rankUp[t];
    }

    // ----------------------------------------------------------------
    // Runs HEFT, records assignment[t], and returns makespan.
    // Also captures the sorted task order in lastPermutation.
    // ----------------------------------------------------------------
    public double runHEFT() {
        // 1) compute ranks
        for (int t = 0; t < nTasks; t++) computeRankUp(t);

        // 2) sort tasks by descending rankUp
        Integer[] orderObj = new Integer[nTasks];
        for (int i = 0; i < nTasks; i++) orderObj[i] = i;
        Arrays.sort(orderObj,
            Comparator.comparingDouble((Integer t) -> rankUp[t]).reversed());

        // capture as primitive array
        lastPermutation = new int[nTasks];
        for (int i = 0; i < nTasks; i++) {
            lastPermutation[i] = orderObj[i];
        }

        // reset ready times
        Arrays.fill(cpuReady, 0.0);
        Arrays.fill(taskFinish, 0.0);
        double makespan = 0;

        // 3) schedule tasks
        for (int idx = 0; idx < nTasks; idx++) {
            int t = orderObj[idx];
            double bestFT = Double.POSITIVE_INFINITY;
            int bestCPU = 0;
            for (int c = 0; c < nCPUs; c++) {
                double ready = 0;
                for (int par : pred.get(t)) {
                    ready = Math.max(ready, taskFinish[par] + commCost[par][t]);
                }
                double start = Math.max(cpuReady[c], ready);
                double ft    = start + compCost[t][c];
                if (ft < bestFT) {
                    bestFT = ft;
                    bestCPU = c;
                }
            }
            assignment[t]     = bestCPU;
            taskFinish[t]     = bestFT;
            cpuReady[bestCPU] = bestFT;
            if (bestFT > makespan) makespan = bestFT;
        }

        return makespan;
    }

    // ----------------------------------------------------------------
    // Returns the HEFT ordering (permutation) from the last runHEFT().
    // If runHEFT() wasn't called, it calls it internally.
    // ----------------------------------------------------------------
    public int[] getPermutation() {
        if (lastPermutation == null) {
            runHEFT();
        }
        return lastPermutation.clone();
    }

    // ----------------------------------------------------------------
    // Returns the task→CPU assignment from the last runHEFT().
    // ----------------------------------------------------------------
    public int[] getAssignment() {
        return assignment.clone();
    }

    // ----------------------------------------------------------------
    // Runs HEFT and returns basic stats: [makespan, energy, crossRack, var].
    // Energy/crossRack/var are all zero here (no power model in HEFT).
    // ----------------------------------------------------------------
    public double[] runWithStats() {
        double ms = runHEFT();
        return new double[]{ ms, 0.0, 0.0, 0.0 };
    }
}

