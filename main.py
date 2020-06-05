import time
from log.logger import logger
import torch.nn as nn
import torch.nn.functional as F
from datautils.util import *
from modules.model import SentMatcher
from modules.optimizer import Optimizer
from conf.config import get_data_path, args_config


def calc_acc(pred, gold):
    return (pred.data.argmax(dim=-1) == gold).sum().item()


def criterion(pred, gold):
    return F.cross_entropy(pred, gold)


def train(model, train_data, dev_data, test_data, args, word_vocab, extwd_vocab, lbl_vocab):
    args.max_step = args.epoch * ((len(train_data) + args.batch_size - 1) // (args.batch_size * args.update_steps))
    optimizer = Optimizer(filter(lambda p: p.requires_grad, lni_model.parameters()), args)
    best_dev_acc, best_test_acc = 0, 0
    patient = 0
    for ep in range(1, 1+args.epoch):
        model.train()
        train_loss = 0.
        start_time = time.time()
        for i, batch_data in enumerate(batch_iter(train_data, args.batch_size, True)):
            batcher = batch_variable(batch_data, word_vocab, extwd_vocab, lbl_vocab)
            batcher = (x.to(args.device) for x in batcher)
            sent1, sent2, extsent1, extsent2, gold_lbl = batcher
            pred = model((sent1, sent2), (extsent1, extsent2))
            loss = criterion(pred, gold_lbl)
            if args.update_steps > 1:
                loss = loss / args.update_steps

            loss_val = loss.data.item()
            train_loss += loss_val

            loss.backward()

            if (i+1) % args.update_steps == 0 or (i == args.max_step-1):
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=args.grad_clip)
                optimizer.step()
                model.zero_grad()

            train_acc = calc_acc(pred, gold_lbl) / len(batch_data)
            logger.info('Iter%d time cost: %.2fs, lr: %.8f, train loss: %.3f, train acc: %.3f' % (i + 1, (time.time() - start_time), optimizer.get_lr(), loss_val, train_acc))

        train_loss /= len(train_data)
        dev_acc = eval(model, dev_data, args, word_vocab, extwd_vocab, lbl_vocab)
        logger.info('[Epoch %d] train loss: %.3f, lr: %f, DEV ACC: %.3f' % (
        ep, train_loss, optimizer.get_lr(), dev_acc))

        if dev_acc > best_dev_acc:
            patient = 0
            best_dev_acc = dev_acc
            test_acc = eval(model, test_data, args, word_vocab, extwd_vocab, lbl_vocab)
            logger.info('Test ACC: %.3f' % test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        else:
            patient += 1

        if patient > args.patient:
            break

    logger.info('Final Test ACC: %.3f' % best_test_acc)


def eval(model, test_data, args, word_vocab, extwd_vocab, lbl_vocab):
    model.eval()
    nb_correct = 0
    with torch.no_grad():
        for i, batch_data in enumerate(batch_iter(test_data, args.test_batch_size)):
            batcher = batch_variable(batch_data, word_vocab, extwd_vocab, lbl_vocab)
            batcher = (x.to(args.device) for x in batcher)
            sent1, sent2, extsent1, extsent2, gold_lbl = batcher
            pred = model((sent1, sent2), (extsent1, extsent2))
            nb_correct += calc_acc(pred, gold_lbl)
    acc = nb_correct / len(test_data)
    return acc


if __name__ == '__main__':
    np.random.seed(3046)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1344)
    torch.cuda.manual_seed_all(1344)

    print('cuda available:', torch.cuda.is_available())
    print('cuDnn available:', torch.backends.cudnn.enabled)
    print('GPU numbers:', torch.cuda.device_count())

    data_path = get_data_path("./conf/datapath.json")
    word_vocab, lbl_vocab = create_vocab(data_path['data']['train_data'])
    embed_weights, extwd_vocab = build_pretrain_embedding(data_path['pretrained']['word_embedding'])

    train_data = load_dataset(data_path['data']['train_data'])
    print('train data size:', len(train_data))
    dev_data = load_dataset(data_path['data']['dev_data'])
    print('dev data size:', len(dev_data))
    test_data = load_dataset(data_path['data']['test_data'])
    print('test data size:', len(test_data))

    args = args_config()
    args.vocab_size = len(word_vocab)
    args.lbl_size = len(lbl_vocab)

    lni_model = SentMatcher(args, embed_weights)
    if torch.cuda.is_available() and args.cuda >= 0:
        torch.cuda.empty_cache()
        args.device = torch.device('cuda', args.cuda)
        # if torch.cuda.device_count() > 1:
        #     parser_model = nn.DataParallel(parser_model, device_ids=list(range(torch.cuda.device_count() // 2)))
    else:
        args.device = torch.device('cpu')

    # lni_model = lni_model.cuda(args.device)
    lni_model = lni_model.to(args.device)
    print('模型参数量：', sum(p.numel() for p in lni_model.parameters()))
    # print('模型参数量：', sum(p.nelement() for p in lni_model.parameters()))
    print(lni_model)

    train(lni_model, train_data, dev_data, test_data, args, word_vocab, extwd_vocab, lbl_vocab)


