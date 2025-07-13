// src/PureGAJobSchedulerMR.java

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class PureGAJobSchedulerMR extends Configured implements Tool {

    public static class WholeFileInputFormat extends FileInputFormat<Text,Text> {
        @Override protected boolean isSplitable(JobContext ctx, Path file) { return false; }
        @Override public RecordReader<Text,Text> createRecordReader(
                InputSplit split, TaskAttemptContext ctx)
                throws IOException, InterruptedException {
            return new WholeFileRecordReader();
        }
    }

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
            Path p = split.getPath();
            FileSystem fs = p.getFileSystem(conf);
            byte[] data = new byte[(int)split.getLength()];
            try (FSDataInputStream in = fs.open(p)) {
                in.readFully(0, data);
            }
            key.set(p.getName());
            val.set(new String(data, "UTF-8"));
            done = true;
            return true;
        }

        @Override public Text getCurrentKey()   { return key; }
        @Override public Text getCurrentValue() { return val; }
        @Override public float getProgress()    { return done ? 1f : 0f; }
        @Override public void close()           {}
    }

    public static class PureGAMapper extends Mapper<Text,Text,Text,DoubleWritable> {
        private static final Pattern HEADER = Pattern.compile("^\\s*\\d+\\s+\\d+\\s+\\d+\\s+\\d+\\s*$");

        private int    popF, maxG, runs;
        private double mu, alpha, beta;
        private int    rackCount;
        private double rackPenalty;

        @Override
        protected void setup(Context ctx) {
            String[] p = ctx.getConfiguration().getStrings("pureGA.params");
            popF       = Integer.parseInt(p[0]);
            maxG       = Integer.parseInt(p[1]);
            runs       = Integer.parseInt(p[2]);
            mu         = Double.parseDouble(p[3]);
            alpha      = Double.parseDouble(p[4]);
            beta       = Double.parseDouble(p[5]);
            rackCount  = Integer.parseInt(p[6]);
            rackPenalty= Double.parseDouble(p[7]);
        }

        @Override
        protected void map(Text key, Text value, Context ctx)
                throws IOException, InterruptedException {
            // write DAG to temp file
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

            // run GA 'runs' times, collect makespans
            double[] msArr = new double[runs];
            double bestMs = Double.POSITIVE_INFINITY, bestEn = 0, bestCr = 0;

            for (int i = 0; i < runs; i++) {
                PaperGA ga = new PaperGA(tmp.getAbsolutePath(), rackCount, rackPenalty);
                // runWithStats should now return [makespan, energy, crossRack, var]
                double[] st = ga.runWithStats(popF, maxG, runs, mu, alpha, beta);
                msArr[i] = st[0];
                if (st[0] < bestMs) {
                    bestMs = st[0];
                    bestEn = st[1];
                    bestCr = st[2];
                }
            }

            // compute variance of makespans
            double mean = 0;
            for (double m : msArr) mean += m;
            mean /= runs;
            double vsum = 0;
            for (double m : msArr) {
                double d = m - mean;
                vsum += d * d;
            }
            double var = vsum / runs;

            // emit all four metrics
            ctx.write(new Text("GA_makespan"),  new DoubleWritable(bestMs));
            ctx.write(new Text("GA_energy"),    new DoubleWritable(bestEn));
            ctx.write(new Text("GA_crossRack"), new DoubleWritable(bestCr));
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 10) {
            System.err.println(
              "Usage: PureGAJobSchedulerMR <in> <out> "
            + "<popF> <maxGen> <runs> <mu> <alpha> <beta> <rackCount> <rackPenalty>");
            System.exit(-1);
        }
        Configuration conf = getConf();
        conf.setStrings("pureGA.params", Arrays.copyOfRange(args, 2, 10));

        Job job = Job.getInstance(conf, "Pure-GA MR");
        job.setJarByClass(PureGAJobSchedulerMR.class);
        job.setInputFormatClass(WholeFileInputFormat.class);
        WholeFileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setMapperClass(PureGAMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new PureGAJobSchedulerMR(), args));
    }
}

