import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class GAJobSchedulerMR extends Configured implements Tool {

    // Prevent splitting so each mapper sees a whole DAG file
    public static class WholeFileInputFormat extends FileInputFormat<Text,Text> {
        @Override protected boolean isSplitable(JobContext ctx, Path file) { return false; }
        @Override public RecordReader<Text,Text> createRecordReader(
                InputSplit split, TaskAttemptContext ctx)
                throws IOException, InterruptedException {
            return new WholeFileRecordReader();
        }
    }

    // Read entire file as one (key=path,value=contents)
    public static class WholeFileRecordReader extends RecordReader<Text,Text> {
        private FileSplit split;
        private Configuration conf;
        private boolean done = false;
        private Text key = new Text(), val = new Text();

        @Override public void initialize(InputSplit is, TaskAttemptContext ctx) 
                throws IOException {
            split = (FileSplit)is;
            conf  = ctx.getConfiguration();
        }

        @Override public boolean nextKeyValue() throws IOException {
            if (done) return false;
            Path path = split.getPath();
            FileSystem fs = path.getFileSystem(conf);
            try (FSDataInputStream in = fs.open(path);
                 ByteArrayOutputStream out = new ByteArrayOutputStream()) {
                IOUtils.copyBytes(in, out, 4096, false);
                val.set(out.toString("UTF-8"));
            }
            key.set(path.toString());
            done = true;
            return true;
        }

        @Override public Text getCurrentKey()   { return key; }
        @Override public Text getCurrentValue() { return val; }
        @Override public float getProgress()    { return done ? 1f : 0f; }
        @Override public void close() throws IOException {}
    }

    public static class IslandMapper 
            extends Mapper<Text,Text,Text,DoubleWritable> {

        private static final Pattern HEADER =
            Pattern.compile("^\\s*\\d+\\s+\\d+\\s+\\d+\\s+\\d+\\s*$");

        // GA parameters
        private int popF, maxG, runs,
                    nIslands, migInterval, eliteCount,
                    injectionInterval, injectionSize,
                    adaptDuration, rackCount,
                    globalBroadcastInterval;
        private boolean adaptiveEnabled;
        private double mu, varianceThreshold, mutationMultiplier,
                       alpha, beta, rackPenalty;

        @Override
        protected void setup(Context ctx) {
            String[] p = ctx.getConfiguration().getStrings("islandGA.params");
            popF                    = Integer.parseInt(p[0]);
            maxG                    = Integer.parseInt(p[1]);
            runs                    = Integer.parseInt(p[2]);
            mu                      = Double.parseDouble(p[3]);
            nIslands                = Integer.parseInt(p[4]);
            migInterval             = Integer.parseInt(p[5]);
            eliteCount              = Integer.parseInt(p[6]);
            injectionInterval       = Integer.parseInt(p[7]);
            injectionSize           = Integer.parseInt(p[8]);
            adaptiveEnabled         = Boolean.parseBoolean(p[9]);
            varianceThreshold       = Double.parseDouble(p[10]);
            mutationMultiplier      = Double.parseDouble(p[11]);
            adaptDuration           = Integer.parseInt(p[12]);
            alpha                   = Double.parseDouble(p[13]);
            beta                    = Double.parseDouble(p[14]);
            rackCount               = Integer.parseInt(p[15]);
            rackPenalty             = Double.parseDouble(p[16]);
            globalBroadcastInterval = Integer.parseInt(p[17]);
        }

        @Override
        protected void map(Text key, Text value, Context ctx)
                throws IOException, InterruptedException {
            // 1) Dump DAG locally
            String[] lines = value.toString().split("\\r?\\n");
            File tmp = File.createTempFile("dag_", ".txt");
            tmp.deleteOnExit();
            try (BufferedWriter w = new BufferedWriter(new FileWriter(tmp))) {
                for (String L : lines) {
                    if (L.trim().isEmpty() || HEADER.matcher(L).matches()) 
                        continue;
                    w.write(L);
                    w.newLine();
                }
            }
            String dagFile = tmp.getAbsolutePath();

            // 2) Run 'runs' independent Island-GA executions, collect makespans
            double[] msArr = new double[runs];
            double bestMs  = Double.POSITIVE_INFINITY;
            double bestEn  = 0, bestCr = 0;

            for (int i = 0; i < runs; i++) {
                IslandGA iga = new IslandGA(
                    dagFile,
                    popF, maxG, runs, mu,
                    nIslands, migInterval, eliteCount,
                    injectionInterval, injectionSize,
                    adaptiveEnabled, varianceThreshold,
                    mutationMultiplier, adaptDuration,
                    alpha, beta,
                    rackCount, rackPenalty,
                    globalBroadcastInterval
                );
                double[] stats = iga.runWithStats();
                double ms = stats[0], en = stats[1], cr = stats[2];
                msArr[i] = ms;
                if (ms < bestMs) {
                    bestMs = ms;
                    bestEn = en;
                    bestCr = cr;
                }
            }

            // 3) Compute variance of makespan
            double mean = 0;
            for (double m : msArr) mean += m;
            mean /= runs;
            double vsum = 0;
            for (double m : msArr) {
                double d = m - mean;
                vsum += d * d;
            }
            double var = vsum / runs;

            // 4) Emit the four metrics
            ctx.write(new Text("IS_makespan"),  new DoubleWritable(bestMs));
            ctx.write(new Text("IS_energy"),    new DoubleWritable(bestEn));
            ctx.write(new Text("IS_crossRack"), new DoubleWritable(bestCr));
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 20) {
            System.err.println(
                "Usage: GAJobSchedulerMR <in> <out> "
              + "<popF> <maxGen> <runs> <mu> "
              + "<nIslands> <migInterval> <eliteCnt> "
              + "<injInterval> <injSize> "
              + "<adaptiveEnabled> <varianceThreshold> "
              + "<mutationMultiplier> <adaptDuration> "
              + "<alpha> <beta> <rackCount> <rackPenalty> "
              + "<globalBroadcastInterval>"
            );
            System.exit(-1);
        }
        Configuration conf = getConf();
        conf.setStrings("islandGA.params", Arrays.copyOfRange(args, 2, 20));

        Job job = Job.getInstance(conf, "Island-GA-MR");
        job.setJarByClass(GAJobSchedulerMR.class);
        job.setInputFormatClass(WholeFileInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setMapperClass(IslandMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new GAJobSchedulerMR(), args));
    }
}

// --------------------------------------------------------------------
// IslandGA: island‐model GA with enhancements
// --------------------------------------------------------------------
class IslandGA {
    private final PaperGA[] gaIslands;
    private final int popF, maxGen, runs,
                      nIslands, migInterval, eliteCount,
                      injectionInterval, injectionSize,
                      adaptDuration, tasks, numRacks,
                      globalBroadcastInterval;
    private final double baseMu, alpha, beta, rackPenalty,
                         varianceThreshold, mutationMultiplier;
    private final boolean adaptiveEnabled;

    private double[] muPerIsland;
    private int[]    adaptCounter;
    private List<int[]>[] pops;
    private double[][]    fits;
    private int[]         bestPerm;

    public IslandGA(String dagFile,
                    int popF, int maxGen, int runs, double mu,
                    int nIslands, int migInterval, int eliteCount,
                    int injectionInterval, int injectionSize,
                    boolean adaptiveEnabled, double varianceThreshold,
                    double mutationMultiplier, int adaptDuration,
                    double alpha, double beta,
                    int numRacks, double rackPenalty,
                    int globalBroadcastInterval)
            throws IOException {
        this.popF                   = popF;
        this.maxGen                 = maxGen;
        this.runs                   = runs;
        this.baseMu                 = mu;
        this.nIslands               = nIslands;
        this.migInterval            = migInterval;
        this.eliteCount             = eliteCount;
        this.injectionInterval      = injectionInterval;
        this.injectionSize          = injectionSize;
        this.adaptiveEnabled        = adaptiveEnabled;
        this.varianceThreshold      = varianceThreshold;
        this.mutationMultiplier     = mutationMultiplier;
        this.adaptDuration          = adaptDuration;
        this.alpha                  = alpha;
        this.beta                   = beta;
        this.numRacks               = numRacks;
        this.rackPenalty            = rackPenalty;
        this.globalBroadcastInterval= globalBroadcastInterval;

        muPerIsland  = new double[nIslands];
        adaptCounter = new int[nIslands];
        Arrays.fill(adaptCounter, 0);
        Arrays.fill(muPerIsland, baseMu);

        gaIslands = new PaperGA[nIslands];
        for (int i = 0; i < nIslands; i++)
            gaIslands[i] = new PaperGA(dagFile, numRacks, rackPenalty);

        tasks = gaIslands[0].getNumTasks();
    }

    public double[] runWithStats() {
        double ms = run();
        double[] fm = gaIslands[0].computeFitness(bestPerm, alpha, beta);
        return new double[]{ ms, fm[2], fm[3], 0.0 };
    }

    public double run() {
        int perIsland = popF / nIslands;
        int perRack   = (int)Math.ceil((double)nIslands / numRacks);

        pops = new ArrayList[nIslands];
        fits = new double[nIslands][perIsland];
        Random R = new Random();
        double bestMS = Double.POSITIVE_INFINITY;

        // initialize islands
        for (int i = 0; i < nIslands; i++) {
            pops[i] = new ArrayList<>(perIsland);
            for (int j = 0; j < perIsland; j++) {
                int[] perm = randomPerm(tasks, R);
                pops[i].add(perm);
                double[] fm = gaIslands[i].computeFitness(perm, alpha, beta);
                fits[i][j] = fm[0];
                if (fm[1] < bestMS) {
                    bestMS   = fm[1];
                    bestPerm = perm.clone();
                }
            }
        }

        // generations
        for (int g = 0; g < maxGen; g++) {

            // global-best broadcast
            if ((g+1) % globalBroadcastInterval == 0) {
                int bi=0, bj=0; double bf=fits[0][0];
                for (int i=0;i<nIslands;i++){
                    for(int j=0;j<perIsland;j++){
                        if(fits[i][j]<bf){
                            bf=fits[i][j]; bi=i; bj=j;
                        }
                    }
                }
                int[] gb = pops[bi].get(bj);
                for(int i=0;i<nIslands;i++){
                    int wi=0; double wf=fits[i][0];
                    for(int j=1;j<perIsland;j++){
                        if(fits[i][j]>wf){ wf=fits[i][j]; wi=j; }
                    }
                    pops[i].set(wi, gb.clone());
                    fits[i][wi] = bf;
                }
            }

            // local evolution (tournament + 1-opt)
            for (int i = 0; i < nIslands; i++) {
                List<int[]> pop = pops[i];
                double[] fit = fits[i];
                List<Double> fitList = new ArrayList<>(perIsland);
                for (double v: fit) fitList.add(v);

                int bestIdx = tournamentSelect(fitList, R);
                int[] elite = pop.get(bestIdx).clone();

                List<int[]> next = new ArrayList<>(perIsland);
                next.add(elite);
                fits[i][0] = fitList.get(bestIdx);

                for (int j = 1; j < perIsland; j++) {
                    int p1 = tournamentSelect(fitList, R);
                    int p2 = tournamentSelect(fitList, R);
                    int[] child = gaIslands[i].cycleCrossover(pop.get(p1), pop.get(p2), R);

                    if (R.nextDouble() < muPerIsland[i]) {
                        int x=R.nextInt(tasks), y=R.nextInt(tasks);
                        int t=child[x]; child[x]=child[y]; child[y]=t;
                    }

                    for (int ls=0; ls<3; ls++){
                        int a=R.nextInt(tasks), b=R.nextInt(tasks);
                        int[] tmp = child.clone();
                        tmp[a]=child[b]; tmp[b]=child[a];
                        double oldF = gaIslands[i].computeFitness(child, alpha, beta)[0];
                        double newF = gaIslands[i].computeFitness(tmp,   alpha, beta)[0];
                        if (newF < oldF) child = tmp;
                    }

                    double fc = gaIslands[i].computeFitness(child, alpha, beta)[0];
                    next.add(child);
                    fits[i][j] = fc;
                    if (fits[i][j] < bestMS) {
                        bestMS   = gaIslands[i].computeFitness(child, alpha, beta)[1];
                        bestPerm = child.clone();
                    }
                }

                pops[i] = next;
            }

           // novelty injection: fixed lambda scoping
            if ((g+1) % injectionInterval == 0) {
                for (int ii = 0; ii < nIslands; ii++) {
                    final int island = ii;
                    Integer[] idx = IntStream.range(0, perIsland).boxed().toArray(Integer[]::new);
                    Arrays.sort(idx, Comparator.comparingDouble(j -> fits[island][j]));
                    int cnt = Math.min(injectionSize, perIsland);
                    for (int k = 0; k < cnt; k++) {
                        int w = idx[perIsland - 1 - k];
                        int[] rnd = randomPerm(tasks, R);
                        pops[island].set(w, rnd);
                        fits[island][w] = gaIslands[island]
                            .computeFitness(rnd, alpha, beta)[0];
                    }
                }
            }

            // adaptive mutation
            if (adaptiveEnabled) {
                for (int i=0; i<nIslands; i++){
                    double sum=0, sum2=0;
                    for (double f: fits[i]){ sum+=f; sum2+=f*f; }
                    double m = sum/perIsland, v = sum2/perIsland - m*m;
                    if (v < varianceThreshold && adaptCounter[i]==0){
                        muPerIsland[i] = baseMu * mutationMultiplier;
                        adaptCounter[i] = adaptDuration;
                    } else if (adaptCounter[i] > 0){
                        if (--adaptCounter[i]==0){
                            muPerIsland[i] = baseMu;
                        }
                    }
                }
            }

            // rack-aware migration: fixed lambda scoping
            if ((g+1) % migInterval == 0) {
                boolean intra = (((g+1)/migInterval) % 2 == 1);
                for (int ii = 0; ii < nIslands; ii++) {
                    final int island = ii;
                    int rackId = Math.min(island / perRack, numRacks-1);
                    int offset = island % perRack;
                    int dest;
                    if (intra) {
                        int o2 = (offset + 1) % perRack;
                        dest = rackId * perRack + o2;
                        if (dest >= nIslands) dest = rackId * perRack;
                    } else {
                        int r2 = (rackId + 1) % numRacks;
                        dest = r2 * perRack + offset;
                        if (dest >= nIslands) dest = r2 * perRack;
                    }
                    PriorityQueue<Integer> pq = new PriorityQueue<>(
                        Comparator.comparingDouble(j -> fits[island][j])
                    );
                    for (int j = 0; j < perIsland; j++) pq.offer(j);
                    int cnt = Math.min(eliteCount, perIsland);
                    for (int k = 0; k < cnt; k++) {
                        int idx2 = pq.poll();
                        int[] cand = pops[island].get(idx2);
                        double wf = fits[dest][0];
                        int wi = 0;
                        for (int j = 1; j < perIsland; j++) {
                            if (fits[dest][j] > wf) {
                                wf = fits[dest][j];
                                wi = j;
                            }
                        }
                        pops[dest].set(wi, cand.clone());
                        fits[dest][wi] = gaIslands[dest]
                            .computeFitness(cand, alpha, beta)[0];
                    }
                }
            }
        }

        return bestMS;
    }

    /** Tournament selection (size=4) */
    private int tournamentSelect(List<Double> fit, Random R) {
        int best = R.nextInt(fit.size());
        for (int i = 1; i < 4; i++) {
            int r = R.nextInt(fit.size());
            if (fit.get(r) < fit.get(best)) best = r;
        }
        return best;
    }

    /** random permutation 0…n-1 */
    private int[] randomPerm(int n, Random R) {
        int[] a = new int[n]; for (int i = 0; i < n; i++) a[i]=i;
        for (int i = n-1; i > 0; i--){
            int j = R.nextInt(i+1), t=a[i]; a[i]=a[j]; a[j]=t;
        }
        return a;
    }
}

