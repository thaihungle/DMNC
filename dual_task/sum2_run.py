import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import nltk
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

from dual_dnc import Dual_DNC
from cached_dnc.cached_dual_dnc import CachedDual_DNC
from dnc import DNC
from cached_dnc.cached_controller import CachedLSTMController
from recurrent_controller import StatelessRecurrentController



def exact_acc(target_batch, predict_batch, pprint=1.0):
    acc=[]
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        for t in target_batch[b]:
            if t > 1:
                trim_target.append(t)
        for t in predict_batch[b]:
            if t > 1:
                trim_predict.append(t)
        if np.random.rand()>pprint:
            print('{} vs {}'.format(trim_target, trim_predict))
        ac=0
        for n1,n2 in zip(trim_predict, trim_target):
            if n1==n2:
                ac+=1
        acc.append(ac/max(len(trim_target), len(trim_predict)))
    return np.mean(acc)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    # print('-----')
    # print(index)
    vec = np.zeros(int(size), dtype=np.float32)
    vec[int(index)] = 1.0
    return vec

def random_length(length_from, length_to):
    if length_from == length_to:
        return length_from
    return np.random.randint(length_from, length_to + 1)

def sum2_sample(vocab_lower, vocab_upper, length_from, length_to, l=-1):

    if l==-1:
        l1=l2=random_length(length_from, length_to)
    elif l==0:
        l1=random_length(length_from, length_to)
        l2=random_length(length_from, length_to)
    else:
        l1=l2=l
    seed = np.random.choice(list(range(int(vocab_lower), int(vocab_upper))),
                            int(l1), replace=False)
    seed2 = np.random.choice(list(range(int(vocab_lower), int(vocab_upper))),
                            int(l2), replace=False)
    i1 = seed.tolist()
    i2 = seed2.tolist()
    o = []
    maxl=max(len(i1),len(i2))
    for i in range(maxl):
        o.append(i1[i%len(i1)]+i2[::-1][i%len(i2)])
    o2 = o
    # for i in range(len(o) // 2 + 1, len(o)):
    #     o2[i] = o2[i - 1] + 2
    return i1, i2, o2, maxl+1+len(o2), maxl, maxl-len(i1), maxl-len(i2)


def sum2_sample_batch(vocab_lower, vocab_upper, length_from, length_to, vocab_size, bs=1):
    all_ins=[]
    all_ins2=[]
    all_ose=[]
    max_inlb=0
    olb=0
    e1b=100000
    e2b=100000
    ll=random_length(length_from, length_to)
    for i in range(bs):
        # ll = random_length(length_from, length_to)
        ins, ins2, ose, seq_len, max_inl, e1, e2=sum2_sample(vocab_lower, vocab_upper,
                                                             length_from, length_to,
                                                             l=ll)
        all_ins.append(ins)
        all_ins2.append(ins2)
        all_ose.append(ose)
        olb=max(len(ose),olb)
        max_inlb=max(max_inl, max_inlb)
        e1b=min(e1,e1b)
        e2b=min(e2,e2b)
    seq_len = olb+1+max_inlb
    input_vec = np.zeros((bs,seq_len))
    input_vec2 = np.zeros((bs,seq_len))
    output_vec = np.zeros((bs,seq_len))
    masks = np.zeros((bs, seq_len), dtype=np.bool)
    decoder_point = max_inlb + 1


    for i in range(bs):
        ins=all_ins[i]
        for iii, token in enumerate(ins):
            input_vec[i,max_inlb-len(ins)+iii] = token
        input_vec[i,max_inlb] = 1

        ins2 = all_ins2[i]
        for iii, token in enumerate(ins2):
            input_vec2[i, max_inlb-len(ins2)+iii] = token
        input_vec2[i, max_inlb] = 1

        ose=all_ose[i]
        for iii, token in enumerate(ose):
            output_vec[i, decoder_point + iii] = token
            masks[i, decoder_point + iii] = True
        # for iii in range(len(ose), olb):
        #     masks[i, decoder_point + iii]=True

        # print(ins)
        # print(ins2)
        # print(ose)
        # print(input_vec)
        # print(input_vec2)
        # print(output_vec)
        # print('====')
    # raise False
    input_vec1 = np.array([[onehot(code, vocab_size) for code in input_vecr] for input_vecr in input_vec])
    input_vec2 = np.array([[onehot(code, vocab_size) for code in input_vecr] for input_vecr in input_vec2])
    output_vec = np.array([[onehot(code, vocab_size) for code in output_vecr] for output_vecr in output_vec])

    return input_vec1, input_vec2, output_vec, seq_len, decoder_point, e1b, e2b, masks, all_ose


def sum2_sample_single(vocab_lower, vocab_upper, length_from, length_to):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)

    seed = np.random.choice(list(range(int(vocab_lower), int(vocab_upper))),
                            int(random_length()), replace=False)
    seed2 = np.random.choice(list(range(int(vocab_lower), int(vocab_upper))),
                            int(random_length()), replace=False)
    i1 = seed.tolist()
    i2 = seed2.tolist()
    o = []
    maxl=max(len(i1),len(i2))
    for i in range(maxl):
        o.append(i1[i%len(i1)]+i2[i%len(i2)])
    o2 = o
    for i in range(len(o) // 2 + 1, len(o)):
        o2[i] = o2[i - 1] + 2
    return i1+[0]+i2, o2

def sum2_sample_single_batch(vocab_lower, vocab_upper, length_from, length_to, vocab_size, bs=1):
    all_ins=[]
    all_ose=[]
    max_inlb=0
    olb=0

    ll=random_length(length_from, length_to)
    for i in range(bs):
        ins, ins2, ose, seq_len, max_inl, e1, e2=sum2_sample(vocab_lower, vocab_upper,
                                                             length_from, length_to,
                                                             l=ll)
        all_ins.append(ins+[0]+ins2)
        max_inl=len(all_ins)
        all_ose.append(ose)
        olb=max(len(ose),olb)
        max_inlb=max(max_inl, max_inlb)

    seq_len = olb+1+max_inlb
    input_vec = np.zeros((bs,seq_len))
    output_vec = np.zeros((bs,seq_len))
    masks = np.zeros((bs, seq_len), dtype=np.bool)
    decoder_point = max_inlb + 1


    for i in range(bs):
        ins=all_ins[i]
        for iii, token in enumerate(ins):
            input_vec[i,max_inlb-len(ins)+iii] = token
        input_vec[i,max_inlb] = 1


        ose=all_ose[i]
        for iii, token in enumerate(ose):
            output_vec[i, decoder_point + iii] = token
            masks[i, decoder_point + iii]=True

        # print(ins)
        # print(ose)
        # print(input_vec)
        # print(output_vec)
        # print('====')
    # raise False
    input_vec1 = np.array([[onehot(code, vocab_size) for code in input_vecr] for input_vecr in input_vec])
    output_vec = np.array([[onehot(code, vocab_size) for code in output_vecr] for output_vecr in output_vec])

    return input_vec1, output_vec, seq_len, decoder_point,  np.asarray(masks), all_ose


def sum2_task(args):
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/save/sum2/{}'.format(args.task+
                                                                                       args.name)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    print(dirname)
    ckpts_dir = dirname
    batch_size = 50

    vocab_lower = 2
    vocab_upper = 150
    length_from = 1
    length_to = args.seq_len

    input_size = vocab_upper
    output_size = vocab_upper


    words_count = 16
    word_size = 64
    read_heads = 1

    if args.mode=='train':
        ntest=10
    else:
        ntest=50

    iterations = args.num_iter
    start_step = 0



    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:
            # tf.set_random_seed(1)
            llprint("Building Computational Graph ... ")
            if args.type=='cache':
                ncomputer = CachedDual_DNC(
                    CachedLSTMController,
                    input_size,
                    output_size,
                    output_size,
                    words_count,
                    word_size,
                    read_heads,
                    batch_size,
                    hidden_controller_dim=args.hidden_dim,
                    use_mem=args.use_mem,
                    decoder_mode=False,
                    write_protect=True,
                    dual_emb = True,
                    share_mem = args.share_mem,
                )
            else:
                ncomputer = Dual_DNC(
                    StatelessRecurrentController,
                    input_size,
                    output_size,
                    output_size,
                    words_count,
                    word_size,
                    read_heads,
                    batch_size,
                    hidden_controller_dim=args.hidden_dim,
                    use_mem=args.use_mem,
                    decoder_mode=False,
                    write_protect=True,
                    dual_emb=True,
                    share_mem=args.share_mem,
                    attend_dim=args.attend
                )
            optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.9)

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            if args.mode == 'test':
                ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
            llprint("Done!\n")

            last_100_losses = []
            if args.mode=='test':
                iterations=1

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1

            start_time_100 = time.time()
            mintloss = 1000
            minl2 =1000
            avg_100_time = 0.
            avg_counter = 0
            if args.mode == 'train':
                train_writer = tf.summary.FileWriter('./data/summary/sum2/{}/'.format(ncomputer.print_config()), session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    # ins, ins2, ose, seq_len, max_inl, e1, e2 = sum2_sample(vocab_lower, vocab_upper / 3, length_from, length_to)
                    # print(ins)
                    # print(ose)
                    # raise False
                    summerize = (i % 100 == 0)
                    if args.mode=='train':
                        input_vec1,input_vec2, output_vec,seq_len, decoder_point, e1, e2, masks, _\
                            = sum2_sample_batch(vocab_lower, vocab_upper / 3, length_from, length_to,
                                                vocab_size=vocab_upper,bs=batch_size)



                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data1: input_vec1,
                            ncomputer.input_data2: input_vec2,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.encode1_point: e1,
                            ncomputer.encode2_point: e2,
                            ncomputer.decoder_point:decoder_point,
                            ncomputer.mask: masks
                        })

                        last_100_losses.append(loss_value)

                    if summerize:

                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        if args.mode=='train':
                            summary = tf.Summary()
                            summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []

                        best_mem_view=None
                        for ii in range(ntest):
                            # ins, ins2, ose, seq_len, max_inl, e1, e2 = sum2_sample(vocab_lower, vocab_upper / 3,
                            #                                                        length_from, length_to)
                            # input_vec = np.zeros(seq_len)
                            # for iii, token in enumerate(ins):
                            #     input_vec[iii] = token
                            # input_vec[len(ins)] = 1
                            # input_vec2 = np.zeros(seq_len)
                            # for iii, token in enumerate(ins2):
                            #     input_vec2[iii] = token
                            # input_vec2[len(ins)] = 1
                            # output_vec = np.zeros(seq_len)
                            # decoder_point = max_inl + 1
                            # for iii, token in enumerate(ose):
                            #     output_vec[decoder_point + iii] = token
                            # input_vec1 = np.array([[onehot(code, vocab_upper) for code in input_vec]])
                            # input_vec2 = np.array([[onehot(code, vocab_upper) for code in input_vec2]])
                            # output_vec = np.array([[onehot(code, vocab_upper) for code in output_vec]])

                            input_vec1, input_vec2, output_vec, seq_len, decoder_point, e1, e2, masks, all_ose \
                                = sum2_sample_batch(vocab_lower, vocab_upper / 3, length_from, length_to,
                                                    vocab_size=vocab_upper,bs=batch_size)

                            tloss,out, mem_view = session.run([loss, prob, ncomputer.packed_memory_view], feed_dict={ ncomputer.input_data1: input_vec1,
                                                                    ncomputer.input_data2: input_vec2,
                                                                  ncomputer.sequence_length: seq_len,
                                                                  ncomputer.encode1_point: e1,
                                                                  ncomputer.encode2_point: e2,
                                                                  ncomputer.decoder_point: decoder_point,
                                                                  ncomputer.target_output: output_vec,
                                                                  ncomputer.mask: masks})
                            if tloss<mintloss:
                                mintloss=tloss
                                best_mem_view=mem_view

                            out = np.reshape(np.asarray(out), [-1, seq_len, vocab_upper])
                            out = np.argmax(out, axis=-1)
                            bout_list = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io]<=1:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            # for io in range(decoder_point, out.shape[1]):
                            #     out_list.append(out[0][io])

                            trscores.append(exact_acc(np.asarray(all_ose), np.asarray(bout_list), 0.9))

                        if args.mode == 'train' and minl2>mintloss:
                            minl2=mintloss
                            print('save model')
                            ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                        try:
                            print(best_mem_view['write_gates1'].shape)
                            print(best_mem_view['write_gates2'].shape)
                            print(np.mean(best_mem_view['write_gates1'],axis=0))
                            print(np.mean(best_mem_view['write_gates2'], axis=0))
                            print('---')
                            rin1=np.argmax(input_vec1[0],axis=-1)
                            print(rin1[:decoder_point])
                            rin2 = np.argmax(input_vec2[0], axis=-1)
                            print(rin2[:decoder_point])
                            print(bout_list[0])
                            print(all_ose[0])


                        except:
                            print('Something wrong...')
                        print('loss {} bleu {}'.format(np.mean(last_100_losses),np.mean(trscores)))
                        if args.mode=='train':
                            summary.value.add(tag='train_bleu', simple_value=np.mean(trscores))

                            train_writer.add_summary(summary, i)
                            train_writer.flush()



                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []



                except KeyboardInterrupt:
                    sys.exit(0)
            # if args.mode=='train':
            #     ncomputer.save(session, ckpts_dir, ncomputer.print_config())


def sum2_task_single(args):
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/data/save/sum2/{}'.format(args.name)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    print(dirname)
    ckpts_dir = dirname
    batch_size = 50

    vocab_lower = 2
    vocab_upper = 150
    length_from = 1
    length_to = args.seq_len

    input_size = vocab_upper
    output_size = vocab_upper

    words_count = 32
    word_size = 64
    read_heads = 1

    iterations = args.num_iter
    start_step = 0

    sequence_max_length=100
    if args.mode=='train':
        ntest=10
    else:
        ntest=50

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")



            ncomputer = DNC(
                StatelessRecurrentController,
                input_size,
                output_size,
                sequence_max_length,
                words_count,
                word_size,
                read_heads,
                batch_size,
                hidden_controller_dim=args.hidden_dim,
                use_mem=args.use_mem,
                dual_emb=True,
                decoder_mode=True,
                dual_controller=True,
                write_protect=True
            )

            output, prob, loss, apply_gradients = ncomputer.build_loss_function_mask()


            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            if args.mode == 'test':
                ncomputer.restore(session, ckpts_dir, ncomputer.print_config())
                iterations=1

            llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            minloss=1000
            start_time_100 = time.time()
            end_time_100 = None
            avg_100_time = 0.
            avg_counter = 0
            if args.mode=='train':
                train_writer = tf.summary.FileWriter('./data/summary/sum2/{}/'.format(ncomputer.print_config()), session.graph)
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    if args.mode == 'train':
                        input_vec, output_vec, seq_len, decoder_point, masks, all_ose \
                            = sum2_sample_single_batch(vocab_lower, vocab_upper / 3, length_from, length_to,
                                                vocab_size=vocab_upper, bs=batch_size)

                        loss_value, _ = session.run([
                            loss,
                            apply_gradients
                        ], feed_dict={
                            ncomputer.input_data: input_vec,
                            ncomputer.target_output: output_vec,
                            ncomputer.sequence_length: seq_len,
                            ncomputer.decoder_point: decoder_point,
                            ncomputer.mask: masks
                        })
                        last_100_losses.append(loss_value)

                    summerize = (i % 100 == 0)




                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        if args.mode == 'train':
                            summary = tf.Summary()
                            summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores = []
                        mloss=1000
                        for ii in range(ntest):
                            input_vec, output_vec, seq_len, decoder_point, masks, all_ose \
                                = sum2_sample_single_batch(vocab_lower, vocab_upper / 3, length_from, length_to,
                                                           vocab_size=vocab_upper, bs=batch_size)
                            tloss, out = session.run([loss,prob], feed_dict={ncomputer.input_data: input_vec,
                                                                 ncomputer.sequence_length: seq_len,
                                                                 ncomputer.decoder_point: decoder_point,
                                                                 ncomputer.target_output:output_vec,
                                                                 ncomputer.mask: masks})
                            out = np.reshape(np.asarray(out), [-1, seq_len,vocab_upper])
                            out = np.argmax(out, axis=-1)
                            bout_list = []
                            # print('{} vs {}'.format(seq_len,out.shape[1]))

                            for b in range(out.shape[0]):
                                out_list = []
                                for io in range(decoder_point, out.shape[1]):
                                    if out[b][io] == 0:
                                        break
                                    out_list.append(out[b][io])
                                bout_list.append(out_list)

                            # for io in range(decoder_point, out.shape[1]):
                            #     out_list.append(out[0][io])
                            if tloss<mloss:
                                mloss=tloss
                            trscores.append(exact_acc(np.asarray(all_ose), np.asarray(bout_list), 0.9))
                        if args.mode == 'train' and mloss<minloss:
                            minloss=mloss
                            print('save model')
                            ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                        print('test bleu {}', format(np.mean(trscores)))
                        print ('test bleu {}',format(np.mean(trscores)))
                        if args.mode == 'train':
                            summary.value.add(tag='train_bleu', simple_value=np.mean(trscores))
                            train_writer.add_summary(summary, i)
                            train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []


                except KeyboardInterrupt:
                    sys.exit(0)
                    llprint("\nSaving Checkpoint ... "),

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="dual")
    parser.add_argument('--mode', default="train")
    parser.add_argument('--task', default="hard")
    parser.add_argument('--seq_len', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--use_mem', default=True, type=str2bool)
    parser.add_argument('--share_mem', default=True, type=str2bool)
    parser.add_argument('--num_iter', default=10000, type=int)
    parser.add_argument('--attend', default=0, type=int)
    parser.add_argument('--type', default="no_cache")
    args = parser.parse_args()
    if args.name=="dual":
        sum2_task(args)
    else:
        sum2_task_single(args)