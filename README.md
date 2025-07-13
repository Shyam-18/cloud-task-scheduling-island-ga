# Island‑GA — Cloud Task‑Scheduling Framework
A MapReduce‑accelerated, **island‑model Genetic Algorithm** that schedules DAG‑based cloud workflows.  
It extends a baseline Pure‑GA with:

* **Island evolution** & periodic migration  
* **Global‑best broadcast** for faster convergence  
* **Adaptive mutation** driven by population variance  
* **Novelty injection** to fight premature convergence  
* **Rack‑aware migration** that respects datacenter topology


## Quick Start

### Prerequisites
* **Java 8+**, **Hadoop 3.x** (pseudo‑distributed is fine)  
* Bash, Python 3

# 1 Clone & enter
git clone git@github.com:Shyam-18/cloud-task-scheduling-island-ga.git
cd cloud-task-scheduling-island-ga

# 2 Compile Java files
export HADOOP_CLASSPATH=$(hadoop classpath)
mkdir -p build
javac -classpath "$HADOOP_CLASSPATH" -d build src/*.java

# 3 Create JARs
jar cvf PureGA.jar -C build .
jar cvf GAJobSchedulerMR.jar -C build .

# 4 Run experiment
./scripts/run_experiment.sh


**Parameter Reference**
Flag	Meaning
--popF	Population = popF × tasks
--maxGen	Generations per GA run
--runs	Independent restarts
--mu	Base mutation probability
--racks, --rackPenalty	Datacenter topology & cost
Island‑only	
--islands	# sub‑populations
--migInterval	Gens between migrations
--eliteCnt	Elites copied during migration
--broadcast	Global‑best push frequency
--injInterval, --injSize	Novelty‑injection settings
--adapt	Toggle adaptive mutation
--varThr, --mutMult, --adaptDur	Adapt‑mutation controls



@inproceedings{shyam2025islandGA,
  author    = {K. Shyam Nageshwar and Soma Naga Sai Pranav and M. Aravind Reddy and G. Jeyakumar},
  title     = {Enhanced Island‑Based Genetic Algorithm for Efficient Task Scheduling in Cloud Computing Environments},
  booktitle = {Proc. 9th International I‑SMAC Conf.},
  year      = 2025
}


**Contributors**
       Name	                                           Role
Kolisetty Shyam Nageshwar	              GA/MapReduce implementation, experiments
Malicheti Aravind Reddy	                DAG generator, data analysis
Soma Naga Sai Pranav	                  Visualization, documentation
Dr. Gurusamy Jeyakumar	                Research guidance, paper review
