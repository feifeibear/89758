#export CUDA_VISIBLE_DEVICES=1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

mpirun -np 4 python main_lstm.py \
  --batch_size 5 \
  --cuda \
  --emsize 500 \
  --nhid 500 \
  --dropout 0.65 \
  --save ./savemodel/hvd/ \
  --epochs 40 2>&1 | tee ./logs/sgd-ptb.log
