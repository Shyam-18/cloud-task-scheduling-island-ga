#!/usr/bin/env bash
set -e

#-------------------------------------------------------------------
# 1) Compile & package
#-------------------------------------------------------------------
export HADOOP_CLASSPATH="$(hadoop classpath)"
echo "========== Compiling Java sources =========="
rm -rf build && mkdir -p build
javac -classpath "$HADOOP_CLASSPATH" -d build \
    src/PaperGA.java \
    src/HEFTScheduler.java \
    src/HEFTRunner.java \
    src/PureGAJobSchedulerMR.java \
    src/GAJobSchedulerMR.java

echo "========== Creating JARs =========="
jar cvf PureGA.jar           -C build .
jar cvf GAJobSchedulerMR.jar -C build .

#-------------------------------------------------------------------
# 2) Generate random DAGs
#-------------------------------------------------------------------
echo "========== Generating random DAGs =========="
python3 generate_random_dags.py

#-------------------------------------------------------------------
# 3) Experiment settings
#-------------------------------------------------------------------
# Pure‐GA baseline (384 k evals total)
POP_F=128
MAX_GEN=100
RUNS=30
MU=0.2
ALPHA=1.0
BETA=0.01
RACK_COUNT=2
RACK_PENALTY=1.2

# Island‐GA (same total evals) + new global‐best broadcast
N_ISLANDS=2
MIG_INTERVAL=6
ELITE_CNT=16
INJ_INTERVAL=8
INJ_SIZE=24
ADAPTIVE_ENABLED=true
VAR_THRESHOLD=0.15
MUT_MULT=2.0
ADAPT_DURATION=5
GLOBAL_BROADCAST_INTERVAL=8

#-------------------------------------------------------------------
# 4) Summary header (all metrics)
#-------------------------------------------------------------------
printf "%-6s | %8s %8s %8s %8s | %8s %8s %8s %8s | %8s %8s %8s %8s\n" \
  "SIZE" \
  "HEFT_MS" "HEFT_EN" "HEFT_CR" "HEFT_VAR" \
  "GA_MS"   "GA_EN"   "GA_CR"   "GA_VAR"   \
  "IS_MS"   "IS_EN"   "IS_CR"   "IS_VAR"   \
> summary.txt
printf '%0.s-' {1..100} >> summary.txt
echo >> summary.txt

#-------------------------------------------------------------------
# 5) Run HEFT, Pure‐GA MR (parallel), Island‐GA MR
#-------------------------------------------------------------------
for SIZE in 50 150 250; do
  echo "<<<<< Processing DAG size $SIZE >>>>>"
  DAG="dag-input/dag${SIZE}.txt"
  IN_ISL="file://${PWD}/${DAG}"
  OUT_PURE="/tmp/pure_${SIZE}"
  OUT_ISL="/tmp/isl_${SIZE}"

  ## a) HEFT local run
  java -cp build HEFTRunner "$DAG" $RACK_COUNT $RACK_PENALTY > heft.out
  HEFT_MS=$(awk '/^HEFT_makespan/   {print $2}' heft.out)
  HEFT_EN=$(awk '/^HEFT_energy/     {print $2}' heft.out)
  HEFT_CR=$(awk '/^HEFT_crossRack/  {print $2}' heft.out)
  HEFT_VAR=$(awk '/^HEFT_var/       {print $2}' heft.out)

  ## b) Pure‐GA MR (parallel runs)
  HDFS_PURE_IN="/tmp/pure_input_${SIZE}"
  hdfs dfs -rm -r -f "$HDFS_PURE_IN" || true
  hdfs dfs -mkdir -p "$HDFS_PURE_IN"
  hdfs dfs -put "$DAG" "$HDFS_PURE_IN/dag${SIZE}_1.txt"
  for i in $(seq 2 $RUNS); do
    hdfs dfs -cp "$HDFS_PURE_IN/dag${SIZE}_1.txt" \
             "$HDFS_PURE_IN/dag${SIZE}_$i.txt"
  done

  hdfs dfs -rm -r -f "$OUT_PURE" || true
  hadoop jar PureGA.jar PureGAJobSchedulerMR \
    -D mapreduce.task.timeout=1800000 \
    "$HDFS_PURE_IN" "$OUT_PURE" \
    $POP_F $MAX_GEN 1 $MU $ALPHA $BETA $RACK_COUNT $RACK_PENALTY

  read GA_MS GA_EN GA_CR GA_VAR < <(
    hdfs dfs -cat "${OUT_PURE}/part-m-*" | \
    awk -F'\t' '
      $1=="GA_makespan"  {ms[++r]=$2}
      $1=="GA_energy"    {en[r]=$2}
      $1=="GA_crossRack" {cr[r]=$2}
      $1=="GA_var"       {va[r]=$2}
      END {
        best=1
        for(i=2;i<=r;i++) if(ms[i]<ms[best]) best=i
        printf("%.2f %.2f %.2f %.2f\n",
               ms[best], en[best], cr[best], va[best])
      }'
  )
  hdfs dfs -rm -r "$HDFS_PURE_IN"

  ## c) Island‐GA MR (map‐only, single split)
  hdfs dfs -rm -r -f "$OUT_ISL" || true
  hadoop jar GAJobSchedulerMR.jar GAJobSchedulerMR \
    "$IN_ISL" "$OUT_ISL" \
    $POP_F $MAX_GEN $RUNS $MU \
    $N_ISLANDS $MIG_INTERVAL $ELITE_CNT \
    $INJ_INTERVAL $INJ_SIZE \
    $ADAPTIVE_ENABLED $VAR_THRESHOLD $MUT_MULT $ADAPT_DURATION \
    $ALPHA $BETA $RACK_COUNT $RACK_PENALTY \
    $GLOBAL_BROADCAST_INTERVAL

  read IS_MS IS_EN IS_CR IS_VAR < <(
    hdfs dfs -cat "${OUT_ISL}/part-m-00000" | \
    awk -F'\t' '
      $1=="IS_makespan"  {ms=$2}
      $1=="IS_energy"    {en=$2}
      $1=="IS_crossRack" {cr=$2}
      $1=="IS_var"       {va=$2}
      END { printf("%.2f %.2f %.2f %.2f\n", ms, en, cr, va) }
    '
  )

  printf "%-6s | %8.2f %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f %8.2f\n" \
    "$SIZE" \
    "$HEFT_MS" "$HEFT_EN" "$HEFT_CR" "$HEFT_VAR" \
    "$GA_MS"   "$GA_EN"   "$GA_CR"   "$GA_VAR"   \
    "$IS_MS"   "$IS_EN"   "$IS_CR"   "$IS_VAR"   \
  >> summary.txt
done

#-------------------------------------------------------------------
# 6) Show summary
#-------------------------------------------------------------------
echo
echo "========== Final summary =========="
cat summary.txt
