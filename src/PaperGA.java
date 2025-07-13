// PaperGA.java
// Implements a GA‐based DAG scheduler with optional energy and rack‐awareness.
import java.io.*;
import java.util.*;

public class PaperGA {
    private final int nTasks;
    private final int nCPUs;
    private final double[][] compCost;    // compCost[task][cpu]
    private final double[][] commCost;    // compCost[parent][child]
    private final List<List<Integer>> succ;
    private final List<List<Integer>> pred;

    // Power model
    private double pIdle = 0.0;
    private double pBusy = 1.0;

    // Rack‐awareness
    private int numRacks    = 1;
    private double rackPenalty = 1.0;

    // For scheduling simulation
    private final double[] cpuReady;
    private final double[] taskFinish;
    private final Map<Integer,Integer> cpuOf;

    // --- for HEFT‐seeding in GA initialization ---
    private int[] seedPerm = null;

    /** 
     * If set, the very first individual in each GA run will be this perm,
     * instead of random.
     */
    public void seedInitialPermutation(int[] perm) {
        this.seedPerm = perm.clone();
    }

    /** Parse DAG file; optional last line “pIdle pBusy”. */
    public PaperGA(String filename) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String L;
            while ((L = br.readLine()) != null) {
                if (!L.trim().isEmpty()) lines.add(L.trim());
            }
        }
        // Detect idle/busy power on last line if it has exactly two numbers
        int sz = lines.size();
        String[] parts = lines.get(sz-1).split("\\s+");
        if (parts.length==2) {
            try {
                pIdle = Double.parseDouble(parts[0]);
                pBusy = Double.parseDouble(parts[1]);
                lines.remove(sz-1);
            } catch (NumberFormatException ex) { /* ignore */ }
        }
        nTasks = lines.size();
        // determine nCPUs from first line
        String[] halves0 = lines.get(0).split("\\|");
        String[] lt0 = halves0[0].trim().split("\\s+");
        nCPUs = lt0.length - 1;

        compCost = new double[nTasks][nCPUs];
        commCost = new double[nTasks][nTasks];
        succ = new ArrayList<>(nTasks);
        pred = new ArrayList<>(nTasks);
        for (int i=0;i<nTasks;i++){
            succ.add(new ArrayList<>());
            pred.add(new ArrayList<>());
        }

        // parse each task line
        for (String line : lines) {
            String[] halves = line.split("\\|");
            String[] lt = halves[0].trim().split("\\s+");
            int tID = Integer.parseInt(lt[0]);
            for (int c=0;c<nCPUs;c++){
                compCost[tID][c] = Double.parseDouble(lt[c+1]);
            }
            if (halves.length>1 && !halves[1].trim().isEmpty()){
                for (String pair : halves[1].trim().split("\\),")) {
                    String pStr = pair.replaceAll("[()]", "").trim();
                    if (pStr.isEmpty()) continue;
                    String[] cw = pStr.split("[,\\s]+");
                    int child = Integer.parseInt(cw[0]);
                    double w   = Double.parseDouble(cw[1]);
                    succ.get(tID).add(child);
                    pred.get(child).add(tID);
                    commCost[tID][child] = w;
                }
            }
        }

        cpuReady   = new double[nCPUs];
        taskFinish = new double[nTasks];
        cpuOf      = new HashMap<>();
    }

    /** Overload to enable rack‐awareness (γ > 1 penalizes cross‐rack comm). */
    public PaperGA(String filename, int numRacks, double rackPenalty) throws IOException {
        this(filename);
        this.numRacks    = numRacks;
        this.rackPenalty = rackPenalty;
    }

    public int getNumTasks() { return nTasks; }
    public int getNumCPUs()  { return nCPUs;  }
    public double getIdlePower() { return pIdle; }
    public double getBusyPower() { return pBusy; }

    /**
     * Given a permutation “order”, returns:
     * [0]=fitness, [1]=makespan, [2]=energyUse, [3]=crossRackCost
     */
    public double[] computeFitness(int[] order, double alpha, double beta) {
        // 1) HEFT‐biased assignment with rack‐aware comm delays
        Arrays.fill(cpuReady, 0.0);
        Arrays.fill(taskFinish, 0.0);
        cpuOf.clear();

        for (int idx=0; idx<nTasks; idx++){
            int t = order[idx];
            double bestScore = Double.POSITIVE_INFINITY;
            int bestCPU = 0;
            for (int c=0; c<nCPUs; c++){
                double parentReady = 0.0;
                for (int par: pred.get(t)){
                    double pf = taskFinish[par];
                    int pc = cpuOf.getOrDefault(par,-1);
                    double base = (pc>=0 ? commCost[par][t] : 0.0);
                    double comm = base;
                    if (pc>=0) {
                        int r1 = pc/(nCPUs/numRacks),
                            r2 = c /(nCPUs/numRacks);
                        if (r1!=r2) comm = base * rackPenalty;
                    }
                    parentReady = Math.max(parentReady, pf+comm);
                }
                double ready = Math.max(cpuReady[c], parentReady);
                double finish= ready + compCost[t][c];
                double score = alpha*finish + (1-alpha)*parentReady;
                if (score < bestScore){
                    bestScore=score; bestCPU=c;
                }
            }
            cpuOf.put(t,bestCPU);
            cpuReady[bestCPU]=bestScore;
            taskFinish[t]   =bestScore;
        }

        // 2) Final schedule pass: track makespan + cross‐rack penalty
        Arrays.fill(cpuReady,0.0);
        Arrays.fill(taskFinish,0.0);
        double makespan=0.0, crossRackCost=0.0;
        for (int t: order){
            int c = cpuOf.get(t);
            double pr=0.0;
            for(int par: pred.get(t)){
                double pf = taskFinish[par];
                int pc = cpuOf.get(par);
                double base = (pc!=c? commCost[par][t]:0.0);
                double comm=base;
                if (pc!=c) {
                    int r1=pc/(nCPUs/numRacks),
                        r2=c /(nCPUs/numRacks);
                    if (r1!=r2){
                        comm = base * rackPenalty;
                        crossRackCost += base * (rackPenalty - 1.0);
                    }
                }
                pr = Math.max(pr, pf+comm);
            }
            double st = Math.max(cpuReady[c], pr);
            double ft = st + compCost[t][c];
            cpuReady[c]=ft;
            taskFinish[t]=ft;
            makespan = Math.max(makespan, ft);
        }

        // 3) Energy use
        double energy=0.0;
        if (beta>0){
            double busySum=0.0;
            for (int t=0;t<nTasks;t++){
                busySum += compCost[t][ cpuOf.get(t) ];
            }
            energy = makespan*pIdle*nCPUs + (pBusy-pIdle)*busySum;
        }

        double fitness = makespan + beta*energy;
        return new double[]{ fitness, makespan, energy, crossRackCost };
    }

    // cycleCrossover preserves permutation cycles:
    public int[] cycleCrossover(int[] p1,int[] p2,Random R){
        int n=nTasks; int[] child=new int[n];
        Arrays.fill(child,-1);
        int start=R.nextInt(n), idx=start, val=p1[idx];
        do {
            child[idx]=p1[idx];
            idx = findIndex(p2,val);
            val = p1[idx];
        } while(idx!=start);
        for(int i=0;i<n;i++) if(child[i]<0) child[i]=p2[i];
        return child;
    }
    private int findIndex(int[] arr,int v){
        for(int i=0;i<arr.length;i++) if(arr[i]==v) return i;
        return -1;
    }

    /** Generate a random permutation of 0…n-1 */
    private int[] randomPerm(int n, Random R){
        int[] a=new int[n];
        for(int i=0;i<n;i++) a[i]=i;
        for(int i=n-1;i>0;i--){
            int j=R.nextInt(i+1), t=a[i];
            a[i]=a[j]; a[j]=t;
        }
        return a;
    }

    /**
     * Run the GA for popFactor×nTasks population, maxGen generations,
     * nRuns independent restarts, mutation rate mu, weights α,β.
     * Returns the best permutation found.
     */
    public int[] run(int popFactor,int maxGen,int nRuns,
                     double mu,double alpha,double beta){
        int popSize = popFactor*nTasks;
        Random R    = new Random();
        int[] bestPerm=null;
        double bestFit=Double.POSITIVE_INFINITY;

        for(int run=0;run<nRuns;run++){
            List<int[]> pop=new ArrayList<>(popSize);
            double[] fit=new double[popSize];

            // init (maybe HEFT‐seed index 0, then random)
            for(int i=0;i<popSize;i++){
                final int[] perm = (i==0 && seedPerm!=null)
                                   ? seedPerm.clone()
                                   : randomPerm(nTasks,R);
                pop.add(perm);
                double f=computeFitness(perm,alpha,beta)[0];
                fit[i]=f;
                if(f<bestFit){
                    bestFit=f; bestPerm=perm.clone();
                }
            }

            // evolve
            for(int g=0;g<maxGen;g++){
                // elitism
                int e=0; for(int i=1;i<popSize;i++) if(fit[i]<fit[e]) e=i;
                int[] elite=pop.get(e).clone();

                List<int[]> next=new ArrayList<>(popSize);
                double[] nfit=new double[popSize];
                next.add(elite); nfit[0]=fit[e];

                for(int i=1;i<popSize;i++){
                    // tournament selection
                    int a1=R.nextInt(popSize), b1=R.nextInt(popSize);
                    int p1=(fit[a1]<fit[b1]?a1:b1);
                    int a2=R.nextInt(popSize), b2=R.nextInt(popSize);
                    int p2=(fit[a2]<fit[b2]?a2:b2);

                    int[] child = cycleCrossover(pop.get(p1),pop.get(p2),R);
                    if(R.nextDouble()<mu){
                        int x=R.nextInt(nTasks), y=R.nextInt(nTasks),
                            tmp=child[x]; child[x]=child[y]; child[y]=tmp;
                    }

                    next.add(child);
                    double f=computeFitness(child,alpha,beta)[0];
                    nfit[i]=f;
                    if(f<bestFit){
                        bestFit=f; bestPerm=child.clone();
                    }
                }
                pop=next; fit=nfit;
            }
        }
        return bestPerm;
    }

    /**
     * Run one GA runWithStats and return:
     * [ makespan, energyUse, crossRackCost, variance ]
     */
    public double[] runWithStats(int popFactor,
                                 int maxGen,
                                 int nRuns,
                                 double mu,
                                 double alpha,
                                 double beta) {
        int[] best = run(popFactor, maxGen, nRuns, mu, alpha, beta);
        double[] fm = computeFitness(best, alpha, beta);
        double makespan      = fm[1];
        double energyUse     = fm[2];
        double crossRackCost = fm[3];
        double variance      = 0.0;
        return new double[]{ makespan, energyUse, crossRackCost, variance };
    }

    /**
     * Given a fixed assignment (assignment[t]=CPU), compute stats:
     * [fitness, makespan, energyUse, crossRackCost].
     */
    public double[] evaluateAssignment(int[] assignment, double beta) {
        cpuOf.clear();
        for (int t = 0; t < nTasks; t++) cpuOf.put(t, assignment[t]);

        // topo sort
        int[] inDeg = new int[nTasks];
        for (int t = 0; t < nTasks; t++)
            for (int ch : succ.get(t))
                inDeg[ch]++;
        Queue<Integer> q = new ArrayDeque<>();
        for (int t = 0; t < nTasks; t++)
            if (inDeg[t] == 0) q.add(t);
        List<Integer> topo = new ArrayList<>(nTasks);
        while (!q.isEmpty()) {
            int u = q.poll();
            topo.add(u);
            for (int ch : succ.get(u)) {
                if (--inDeg[ch] == 0) q.add(ch);
            }
        }

        Arrays.fill(cpuReady, 0); Arrays.fill(taskFinish, 0);
        double makespan = 0, crossRackCost = 0;
        for (int t : topo) {
            int c = assignment[t];
            double ready = 0;
            for (int par : pred.get(t)) {
                double pf = taskFinish[par];
                int pc = assignment[par];
                double base = (pc!=c ? commCost[par][t] : 0);
                double comm = base;
                int r1 = pc/(nCPUs/numRacks), r2 = c/(nCPUs/numRacks);
                if (pc!=c && r1!=r2) {
                    comm = base * rackPenalty;
                    crossRackCost += base*(rackPenalty-1.0);
                }
                ready = Math.max(ready, pf+comm);
            }
            double st = Math.max(cpuReady[c], ready);
            double ft = st + compCost[t][c];
            cpuReady[c] = ft;
            taskFinish[t] = ft;
            makespan = Math.max(makespan, ft);
        }

        double energy = 0;
        if (beta > 0) {
            double busySum = 0;
            for (int t = 0; t < nTasks; t++)
                busySum += compCost[t][ assignment[t] ];
            energy = makespan*pIdle*nCPUs + (pBusy-pIdle)*busySum;
        }

        double fitness = makespan + beta*energy;
        return new double[]{ fitness, makespan, energy, crossRackCost };
    }
}

