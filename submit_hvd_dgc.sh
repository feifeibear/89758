#export CUDA_VISIBLE_DEVICES=1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export BATCH_SIZE=5
export EMSIZE=1500
export NHID=1500
export NUM_NODE=4

mpirun -np ${NUM_NODE} python main_lstm.py \
  --use_pruning \
  --gpus 0,1,2,3 \
  --batch_size ${BATCH_SIZE} \
  --cuda \
  --emsize ${EMSIZE} \
  --nhid ${NHID} \
  --dropout 0.65 \
  --save ./dgc_${NUM_NODE}_${BATCH_SIZE}_${EMSIZE}_${NHID}_time \
  --save_dir ./savemodel/dgc_${NUM_NODE}_${BATCH_SIZE}_${EMSIZE}_${NHID}_time \
  --epochs 40 2>&1 | tee ./logs/dgc_${NUM_NODE}_${BATCH_SIZE}_${EMSIZE}_${NHID}_time_log.txt \
