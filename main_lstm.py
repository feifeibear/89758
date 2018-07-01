# coding: utf-8
import argparse
from datetime import datetime
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import horovod.torch as hvd
import torch.optim
import logging
from utils import *
#from hvd_utils.DGCLSTMoptimizer_2D import DGCLSTMDistributedOptimizer

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--results_dir', metavar='RESULTS_DIR',
                    default='./Results/dist-lstm', help='results dir')
parser.add_argument('--save_dir', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--data', type=str, default='./data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--use_pruning', dest='use_pruning', action='store_true',
                            help='use gradient pruning')
parser.add_argument('--no_use_pruning', dest='use_pruning', action='store_false',
                            help='do not use gradient pruning')
parser.set_defaults(use_pruning=False)

parser.add_argument('--use_cluster', dest='use_cluster', action='store_true',
                            help='synchronize all parameters every sync_interval steps')
parser.add_argument('--no_use_cluster', dest='use_cluster', action='store_false',
                            help='synchronize all parameters every sync_interval steps')
parser.set_defaults(use_cluster=False)

parser.add_argument('--pruning_mode', '-pm', default=0, type=int,
                            help='prune mode')
parser.add_argument('--use_warmup', dest='use_warmup', action='store_true',
                    help='use warm up')
parser.add_argument('--no_use_warmup', dest='use_warmup', action='store_false',
                    help='do not use warm up')
parser.set_defaults(use_warmup=False)

args = parser.parse_args()

if args.pruning_mode == 1:
    from hvd_utils.DGCLSTMoptimizer_thd import DGCLSTMDistributedOptimizer
elif args.pruning_mode == 3:
    from hvd_utils.DGCLSTMoptimizer import DGCLSTMDistributedOptimizer
elif args.pruning_mode == 7:
    from hvd_utils.DGCLSTMoptimizer_quant import DGCLSTMDistributedOptimizer
elif args.pruning_mode == 8:
    from hvd_utils.DGCLSTMoptimizer_thd_quant import DGCLSTMDistributedOptimizer

if hvd.rank():
    print("pruning_mode is ", args.pruning_mode)

# Set the random seed manually for reproducibility.
hvd.init()
torch.manual_seed(args.seed + hvd.rank())

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    args.gpus = [int(i) for i in args.gpus.split(',')]
    if args.use_cluster:
        torch.cuda.set_device(0)
    else:
        if(hvd.local_rank() < len(args.gpus)):
            print("rank, ", hvd.local_rank(), " is runing on ", args.gpus[hvd.local_rank()])
            torch.cuda.set_device(args.gpus[hvd.local_rank()])
        else:
            print("rank, ", hvd.local_rank(), " is runing on ", args.gpus[0])
            torch.cuda.set_device(args.gpus[0])
torch.cuda.set_device(hvd.local_rank())

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

# Loop over epochs.
lr = args.lr
best_val_loss = None



def train(optimizer, best_val_loss):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    lr_decay_interval = len(range(0, train_data.size(0) - 1, args.bptt)) // hvd.size()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    T_dim_size = train_data.size(0) - 1
    data_offset = T_dim_size // hvd.size() * hvd.rank()

    batch_time = AverageMeter()
    pruning_time = AverageMeter()
    select_time = AverageMeter()
    comm_time = AverageMeter()

    end = time.time()
    for batch, i in enumerate(range(0, T_dim_size//hvd.size(), args.bptt)):
        data, targets = get_batch(train_data, i + data_offset)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        # local clipping

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if not args.use_pruning:
            optimizer.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if args.use_pruning:
            pruning_time.update(optimizer.pruning_time)
            select_time.update(optimizer.select_time)
            comm_time.update(optimizer.comm_time)
            optimizer.pruning_time = 0.0
            optimizer.select_time = 0.0
            optimizer.comm_time = 0.0

        optimizer.step()
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            if hvd.rank() == 0:
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | '
                        'Time {:.3f} | pruning time {:.3f} | select time {:3f} | '
                        'comm time {:3f}'.format(
                    epoch, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss),
                        batch_time.val, pruning_time.val, select_time.val, comm_time.val))
            total_loss = 0
            start_time = time.time()

    return optimizer, best_val_loss



def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.save_dir is '':
        args.save_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save_dir)
    if not os.path.exists(save_path):
        if hvd.rank() == 0:
            os.makedirs(save_path)

    results_file = os.path.join('./Results/', 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    if hvd.rank() == 0:
        setup_logging(os.path.join(save_path, 'log.txt'))
        results_file = os.path.join(save_path, 'results.%s')
        results = ResultsLog(results_file % 'csv', results_file % 'html')
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)
        logging.info({i: list(w.size())
            for (i, w) in enumerate(list(model.parameters()))})

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Add Horovod Distributed Optimizer
    if args.use_pruning == False:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    else:
        if not args.cuda:
            optimizer = DGCLSTMDistributedOptimizer(optimizer, named_parameters=model.named_parameters(), use_gpu=False, momentum=0.0, weight_decay=0.0)
        else:
            optimizer = DGCLSTMDistributedOptimizer(optimizer, named_parameters=model.named_parameters(), use_gpu=True, momentum=0.0, weight_decay=0.0)
    # Broadcast parameters from rank 0 to all other processes.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    global_begin_time = time.time()
    for epoch in range(1, args.epochs+1):
        if args.use_warmup:
            if epoch == 1:
                optimizer._use_allgather = False
            else:
                optimizer._use_allgather = True
        epoch_start_time = time.time()
        optimizer, best_val_loss = train(optimizer, best_val_loss)
        val_loss = evaluate(val_data)
        if hvd.rank() == 0:
            logging.info('-' * 89)
            logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            model_file = os.path.join(args.save, 'model.pt')
            with open(model_file, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 4.0

        if(hvd.rank() == 0):
            current_time = time.time() - global_begin_time
            results.add(epoch=epoch, val_loss=val_loss, val_ppl=math.exp(val_loss), eslapes=current_time)
            results.save()

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

if hvd.rank() == 0:
    # Load the best saved model.
    model_file = os.path.join(args.save, 'model.pt')
    with open(model_file, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    logging.info('=' * 89)
    logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logging.info('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
