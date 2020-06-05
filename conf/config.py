import os
import json
from argparse import ArgumentParser


def get_data_path(json_path):
    assert os.path.exists(json_path)
    with open(json_path, 'r', encoding='utf-8') as fr:
        data_opts = json.load(fr)

    print(data_opts)
    return data_opts


def args_config():
    parse = ArgumentParser('Sentence Matcher')

    parse.add_argument('--cuda', type=int, default=-1, help='training device, default on cpu')

    parse.add_argument('-lr', '--learning_rate', type=float, default=0.0004, help='learning rate of training')
    parse.add_argument('-bt1', '--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parse.add_argument('-bt2', '--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parse.add_argument('-eps', '--eps', type=float, default=1e-9, help='eps of Adam optimizer')
    parse.add_argument('-warmup', '--warmup_step', type=int, default=10000, help='warm up steps for optimizer')
    parse.add_argument('--decay', type=float, default=0.75, help='lr decay rate for optimizer')
    parse.add_argument('--decay_step', type=int, default=10000, help='lr decay steps for optimizer')
    parse.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for optimizer')
    parse.add_argument('--scheduler', choices=['cosine', 'inv_sqrt', 'exponent', 'linear', 'step', 'const'], default='const', help='the type of lr scheduler')
    parse.add_argument('--grad_clip', type=float, default=1., help='the max norm of gradient clip')
    parse.add_argument('--patient', type=int, default=5, help='patient number in early stopping')

    parse.add_argument('--batch_size', type=int, default=32, help='train batch size')
    parse.add_argument('--update_steps', type=int, default=1, help='gradient accumulation and update per x steps')
    parse.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
    parse.add_argument('--epoch', type=int, default=20, help='iteration of training')

    parse.add_argument('--enc_type', type=int, default=0, help='encoder type')
    parse.add_argument('--embed_dim', type=int, default=200, help='char embedding size')

    # lstm
    parse.add_argument("--rnn_type", default='GRU', choices=['RNN', 'GRU', 'LSTM'], help='rnn type')
    parse.add_argument('--rnn_depth', type=int, default=1, help='the depth of rnn layer')
    parse.add_argument('--hidden_size', type=int, default=100, help='the hidden size of lstm')
    parse.add_argument("--linear_size", type=int, default=400, help='the output size of encoder layer')

    # cnn
    parse.add_argument('--conv_dim', type=int, default=100, help='the input size of conv_1d layer')
    parse.add_argument('--num_channels', type=list, default=[50, 100, 150], help='the filter number of conv_1d layer')
    parse.add_argument('--kernel_size', type=int, default=4, help='the window size of convolution')

    # transformer
    parse.add_argument('-mpe', '--max_pos_embeddings', default=500, help='max sequence position embeddings')
    parse.add_argument("--d_model", type=int, default=200, help='sub-layer feature size')
    parse.add_argument("--d_ff", type=int, default=800, help='pwffn inner-layer feature size')
    parse.add_argument("--nb_heads", type=int, default=8, help='sub-layer feature size')
    parse.add_argument("--encoder_layer", type=int, default=6, help='the number of encoder layer')

    parse.add_argument('--embed_drop', type=float, default=0.2, help='embedding dropout')
    parse.add_argument('--att_drop', type=float, default=0.1, help='attention dropout')
    parse.add_argument('--dropout', type=float, default=0.5, help='common dropout')
    # parse.add_argument('--is_train', action='store_true', help='默认为False, 当执行`python train.py --is_train`时，is_train变成True')
    args = parse.parse_args()
    print(vars(args))
    return args
