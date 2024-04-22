# Activate conda environment
conda activate simple_nlp_tasks

for n_icl in 1 2 4 5 8 16 32 64 128 256
  do
    for model in 169m 430m 1b5 3b 7b
        do
            # Job to perform
            PYTHONPATH=$PWD python scripts/experiments/main.py rwkv ${model} False ${n_icl}
    done
done
# Print some Information about the end-time to STDOUT
echo "DONE";
echo "RWKV Finished at $(date)";


for n_icl in 1 2 4 5 8 16 32 64 128 256
  do
    for model in 130m 370m 790m 1b4 2b8 2.8b-slimpj
        do
            # Job to perform
            PYTHONPATH=$PWD python scripts/experiments/main.py mamba ${model} False ${n_icl}
    done
done
# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Mamba Finished at $(date)";