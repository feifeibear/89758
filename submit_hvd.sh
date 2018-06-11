#export CUDA_VISIBLE_DEVICES=1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
#export BATCH_SIZE=5
export BATCH_SIZE=20
export EMSIZE=1500
export NHID=1500
export NUM_NODE=1

mpirun -np ${NUM_NODE} python main_lstm.py \
  --batch_size ${BATCH_SIZE} \
  --cuda \
  --gpus 1,2,3,4,5 \
  --emsize ${EMSIZE} \
  --nhid ${NHID} \
  --dropout 0.65 \
  --save ./savemodel/sgd/ \
  --save_dir ./Results/dist-lstm/sgd_${NUM_NODE}_${BATCH_SIZE}_${EMSIZE}_${NHID} \
  --epochs 40 2>&1 | tee ./logs/sgd-ptb.log
