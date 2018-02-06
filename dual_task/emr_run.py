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
from sklearn.metrics import roc_auc_score, f1_score
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import beam_search
from dual_dnc import Dual_DNC
from cached_dnc.cached_dual_dnc import CachedDual_DNC
from dnc import DNC
from cached_dnc.cached_controller import CachedLSTMController
from recurrent_controller import StatelessRecurrentController

#input_dim assume seq_len x out_dim
def convert_oh2raw(vec_data, decoder_point):
    data=np.argmax(vec_data, axis=-1)
    inp=[]
    for ci,c in enumerate(data):
        if ci<decoder_point:
            inp.append(c)
    return inp


def roc_auc(target_batch, prob_batch):
    all_auc_macro=[]
    all_auc_micro = []
    for b in range(target_batch.shape[0]):
        target = np.zeros(prob_batch.shape[-1])
        for t in target_batch[b]:
            if t>1:
                target[t]=1
        all_auc_macro.append(roc_auc_score(target, prob_batch[b], average='macro'))
        all_auc_micro.append(roc_auc_score(target, prob_batch[b], average='micro'))
    return np.mean(all_auc_macro),np.mean(all_auc_micro)


def fscore(target_batch, predict_batch, nlabel):
    all_auc_macro=[]
    all_auc_micro = []
    for b in range(target_batch.shape[0]):
        target = np.zeros(nlabel)
        predict = np.zeros(nlabel)
        for t in target_batch[b]:
            if t>1:
                target[t]=1
        for t in predict_batch[b]:
            if t>1:
                predict[t]=1
        all_auc_macro.append(f1_score(target, predict, average='macro'))
        all_auc_micro.append(f1_score(target, predict, average='micro'))
    return np.mean(all_auc_macro),np.mean(all_auc_micro)




def set_score_pre(target_batch, predict_batch):
    s = []
    s2 = []
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        for t in target_batch[b]:
            if t > 1:
                trim_target.append(t)
        for t in predict_batch[b]:
            if t > 1:
                trim_predict.append(t)
        if np.random.rand()>1:
            print('{} vs {}'.format(trim_target, trim_predict))
        acc = len(set(trim_target).intersection(set(trim_predict)))/len(set(trim_target))
        acc2=0
        if len(set(trim_predict))>0:
            acc2 = len(set(trim_target).intersection(set(trim_predict))) / len(trim_predict)
        s.append(acc)
        s2.append(acc2)
    return np.mean(s2), np.mean(s)#prec, recall

def set_score_pre_jac(target_batch, predict_batch):
    s = []
    s2 = []
    s3 = []
    for b in range(target_batch.shape[0]):
        trim_target = []
        trim_predict = []
        for t in target_batch[b]:
            if t > 1:
                trim_target.append(t)
        for t in predict_batch[b]:
            if t > 1:
                trim_predict.append(t)
        if np.random.rand()>0.95:
            print('{} vs {}'.format(trim_target, trim_predict))
        acc = len(set(trim_target).intersection(set(trim_predict)))/len(set(trim_target))
        acc2=0
        if len(set(trim_predict))>0:
            acc2 = len(set(trim_target).intersection(set(trim_predict))) / len(trim_predict)
        acc3=len(set(trim_target).intersection(set(trim_predict)))/len(set(trim_target).union(set(trim_predict)))
        s.append(acc)
        s2.append(acc2)
        s3.append(acc3)
    return np.mean(s2), np.mean(s), np.mean(s3)#prec, recall, jaccard


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def load(path):
    return pickle.load(open(path, 'rb'))

def onehot(index, size):
    # print('-----')
    # print(index)
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def prepare_mimic_sample_dual(dig_list, proc_list, word_space_size_input1,
                              word_space_size_input2, word_space_size_output, index=-1, is_raw=False, multi=False):
    if index<0:
        index = int(np.random.choice(len(dig_list),1))
    input_seq = dig_list[index]
    # print(input_seq)
    i1=[]
    i2=[]
    isi1=True
    for c in input_seq:
        if c==0:
            isi1=False
        else:
            if isi1:
                i1.append(c)
            else:
                i2.append(c)

    o = proc_list[index]

    # print(i1)
    # print(i2)
    # print(o)

    if i2 is []:
        i2=[0]

    # raise  False

    maxl=max(len(i1),len(i2))
    seq_len = maxl+1+len(o)
    decoder_point = maxl + 1
    input_vec1 = np.zeros(seq_len,dtype=np.int32)
    input_vec2 = np.zeros(seq_len,dtype=np.int32)
    output_vec = np.zeros(seq_len,dtype=np.int32)

    for iii, token in enumerate(i1):
        input_vec1[maxl - len(i1) + iii] = token
    input_vec1[maxl] = 1

    for iii, token in enumerate(i2):
        input_vec2[maxl - len(i2) + iii] = token
    input_vec2[maxl] = 1

    for iii, token in enumerate(o):
        output_vec[decoder_point + iii] = token

    if is_raw:
        return input_vec1, input_vec2, output_vec,seq_len, decoder_point, maxl-len(i1), maxl-len(i2), o

    input_vec1 = np.array([onehot(code, word_space_size_input1) for code in input_vec1])
    input_vec2 = np.array([onehot(code, word_space_size_input2) for code in input_vec2])
    output_vec = np.array([onehot(code, word_space_size_output) for code in output_vec])

    if multi:
        # print(output_vec.shape)
        output_vec2 = np.zeros((decoder_point+1,word_space_size_output))
        for c in range(decoder_point):
            output_vec2[c,:]=output_vec[c,:]
        for c in range(decoder_point, seq_len):
            output_vec2[decoder_point,:]+=output_vec[c,:]
        output_vec=output_vec2
        # print(output_vec.shape)
        seq_len=decoder_point+1
        input_vec1 = input_vec1[:seq_len,:]
        input_vec2 = input_vec2[:seq_len, :]

    return np.reshape(input_vec1, (1, -1, word_space_size_input1)), \
           np.reshape(input_vec2, (1, -1, word_space_size_input2)), \
           np.reshape(output_vec, (1, -1, word_space_size_output)),\
           seq_len, decoder_point, maxl-len(i1), maxl-len(i2), o




def prepare_mimic_sample_dual_persist(patient_list, word_space_size_input1, word_space_size_input2,
                                      word_space_size_output, index=-1, multi=False):
    if index<0:
        index = int(np.random.choice(len(patient_list),1))

    # print('\n{}'.format(index))
    patient=patient_list[index]
    adms=[]
    for adm in patient:
        if len(adm)>2:
            input_seq = adm[0]
            input_seq2 = adm[1]
            output_seq = adm[2]
        else:
            input_seq = adm[0]
            input_seq2 = adm[0][::-1]
            output_seq = adm[1]
        adms.append(prepare_mimic_sample_dual([input_seq+[0]+input_seq2], [output_seq], word_space_size_input1, word_space_size_input2,
                                  word_space_size_output, 0,multi=multi))

    return adms

def mimic_task_persit_real_dual(args):

    dirname = os.path.dirname(os.path.abspath(__file__))+'/data/save/'
    print(dirname)
    ckpts_dir = os.path.join(dirname , 'checkpoints_{}_dual_in_single_out_persit'.format(args.task))


    llprint("Loading Data ... ")

    llprint("Done!\n")
    patient_records = pickle.load(open('./data/real_data/mimic/{}/patient_records.pkl'.format(args.task), 'rb'))
    str2tok_diag, str2tok_drug, str2tok_proc\
        = pickle.load(open('./data/real_data/mimic/{}/list_dict_str2token_no_adm.pkl'.format(args.task), 'rb'))
    tok2str_diag, tok2str_proc, tok2str_drug\
        = pickle.load(open('./data/real_data/mimic/{}/list_dict_token2str_no_adm.pkl'.format(args.task), 'rb'))


    all_index = list(range(len(patient_records)))
    train_index = all_index[:int(len(patient_records) * 2 / 3)]
    valid_index = all_index[int(len(patient_records) * 2 / 3):int(len(patient_records) * 5 / 6)]
    test_index = all_index[int(len(patient_records) * 5/6):int(len(patient_records) * 1)]

    patient_list_train = [patient_records[i] for i in train_index]

    patient_list_valid = [patient_records[i] for i in valid_index]

    patient_list_test = [patient_records[i] for i in test_index]

    print('num_patient {}'.format(len(patient_records)))
    print('num train {}'.format(len(patient_list_train)))
    print('num valid {}'.format(len(patient_list_valid)))
    print('num test {}'.format(len(patient_list_test)))
    print('dim in  {} {}'.format(len(str2tok_diag), len(str2tok_drug)))
    print('dim out {}'.format(len(str2tok_proc)))

    batch_size = 1
    input_size1 = len(str2tok_diag)
    input_size2 = len(str2tok_drug)
    output_size = len(str2tok_proc)
    sequence_max_length = 100

    words_count = args.mem_size
    word_size = args.word_size
    read_heads = args.read_heads

    learning_rate = 1e-4
    momentum = 0.9

    from_checkpoint = None
    iterations = args.iters
    start_step = 0



    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")
            if args.type=='no_cache':
                ncomputer = Dual_DNC(
                    StatelessRecurrentController,
                    input_size1,
                    input_size2,
                    output_size,
                    words_count,
                    word_size,
                    read_heads,
                    batch_size,
                    use_mem=args.use_mem,
                    hidden_controller_dim=args.hidden_dim,
                    decoder_mode=False,
                    write_protect=True,
                    dual_emb=args.dual_emb,
                    emb_size=args.emb_size,
                    share_mem=args.share_mem,
                    use_teacher=args.use_teacher,
                    persist_mode=args.persist,
                    attend_dim=args.attend
                )
            else:
                ncomputer = CachedDual_DNC(
                    CachedLSTMController,
                    input_size1,
                    input_size2,
                    output_size,
                    words_count,
                    word_size,
                    read_heads,
                    batch_size,
                    hidden_controller_dim=args.hidden_dim,
                    use_mem=args.use_mem,
                    decoder_mode=False,
                    write_protect=True,
                    dual_emb=args.dual_emb,
                    emb_size=args.emb_size,
                    share_mem=args.share_mem,
                    use_teacher=args.use_teacher,
                    persist_mode=args.persist,
                )
            multi=True

            if multi:
                output, prob, loss, apply_gradients = ncomputer.build_loss_function_multi_label()
            else:
                output, prob, loss, apply_gradients = ncomputer.build_loss_function()

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if args.from_checkpoint is not '':
                if args.from_checkpoint=='default':
                    from_checkpoint = ncomputer.print_config()
                else:
                    from_checkpoint = args.from_checkpoint
                llprint("Restoring Checkpoint %s ... " % from_checkpoint)
                ncomputer.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            start = 0 if start_step == 0 else start_step + 1
            end = start_step + iterations + 1
            if args.mode == 'test':
                end = start
                patient_list_valid = patient_list_test

            start_time_100 = time.time()
            avg_100_time = 0.
            avg_counter = 0
            if args.mode=='train':
                log_dir = './data/summary/log_mimic_{}_dual_in_single_out_persit/'.format(args.task)
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                log_dir='./data/summary/log_mimic_{}_dual_in_single_out_persit/{}/'.format(args.task, ncomputer.print_config())
                if not os.path.isdir(log_dir):
                    os.mkdir(log_dir)
                train_writer = tf.summary.FileWriter(log_dir, session.graph)
            min_tloss=0
            for i in range(start, end + 1):
                try:
                    llprint("\rIteration %d/%d" % (i, end))
                    # ncomputer.clear_current_mem(session)
                    adms = \
                        prepare_mimic_sample_dual_persist(patient_list_train, input_size1, input_size2, output_size, multi=multi)

                    summerize = (i % args.eval_step == 0)
                    if args.mode == 'train':
                        ncomputer.clear_current_mem(session)
                        for adm in adms:
                            input_vec1, input_vec2, output_vec, seq_len, decoder_point, e1, e2, rout \
                                =adm
                            if len(rout) == 1 and rout[0] == 0:
                                continue
                            loss_value, _= session.run([
                                loss,
                                apply_gradients
                            ], feed_dict={
                                ncomputer.input_data1: input_vec1,
                                ncomputer.input_data2: input_vec2,
                                ncomputer.target_output: output_vec,
                                ncomputer.sequence_length: seq_len,
                                ncomputer.encode1_point: e1,
                                ncomputer.encode2_point: e2,
                                ncomputer.decoder_point: decoder_point,
                                ncomputer.clear_mem: False,
                                ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(seq_len)
                            })
                            last_100_losses.append(loss_value)


                    tpre=0
                    if summerize:
                        llprint("\n\t episode %d -->Avg. Cross-Entropy: %.7f\n" % (i, np.mean(last_100_losses)))
                        summary = tf.Summary()
                        summary.value.add(tag='batch_train_loss', simple_value=np.mean(last_100_losses))
                        trscores=[]

                        for ii in range(10):
                            ncomputer.clear_current_mem(session)
                            adms = \
                                prepare_mimic_sample_dual_persist(patient_list_train, input_size1, input_size2,
                                                                  output_size,multi=multi)
                            # ncomputer.clear_current_mem(session)
                            for adm in adms:
                                input_vec1, input_vec2, output_vec, seq_len, decoder_point, e1, e2, rout \
                                    = adm
                                if len(rout)==1 and rout[0]==0:
                                    continue
                                out = session.run([prob],  feed_dict={
                                    ncomputer.input_data1: input_vec1,
                                    ncomputer.input_data2: input_vec2,
                                    ncomputer.target_output: output_vec,
                                    ncomputer.sequence_length: seq_len,
                                    ncomputer.encode1_point: e1,
                                    ncomputer.encode2_point: e2,
                                    ncomputer.decoder_point: decoder_point,
                                    ncomputer.clear_mem: False,
                                    ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(seq_len, prob_true_max=0)
                                                })

                                out = np.reshape(np.asarray(out),[-1, seq_len, output_size])

                                out_list=[]
                                # print('{} vs {}'.format(seq_len,out.shape[1]))
                                if multi:
                                    out=np.argsort(out, axis=-1)
                                    for io in range(len(rout)):
                                        out_list.append(out[0][decoder_point][-io-1])
                                else:
                                    out = np.argmax(out, axis=-1)
                                    for io in range(decoder_point, out.shape[1]):
                                        if out[0][io]<=1:
                                            break
                                        out_list.append(out[0][io])
                                pre,rec, jac=set_score_pre_jac(np.asarray([rout]),np.asarray([out_list]))
                                trscores.append(jac)

                        tescores = []
                        tescores2_1 = []
                        tescores2_2 = []
                        tescores2_3 = []
                        tescores2_5 = []
                        tescores2 = []
                        tescores3 = []
                        tescores4 = []
                        tescores5 = []
                        tescores6 = []
                        print('-----')
                        big_out_list=[]
                        losses=[]
                        single_best_loss=1000
                        best_mem_view=None
                        best_data=None
                        for ii in range(len(patient_list_valid)):
                            ncomputer.clear_current_mem(session)
                            adms = \
                                prepare_mimic_sample_dual_persist(patient_list_valid, input_size1, input_size2,
                                                                  output_size, ii,multi=multi)
                            # ncomputer.clear_current_mem(session)
                            for adm in adms:
                                input_vec1, input_vec2, output_vec, seq_len, decoder_point, e1, e2, rout \
                                    = adm
                                if len(rout)==1 and rout[0]==0:
                                    continue
                                out, loss_v, mem_view = session.run([prob, loss, ncomputer.packed_memory_view], feed_dict={
                                    ncomputer.input_data1: input_vec1,
                                    ncomputer.input_data2: input_vec2,
                                    ncomputer.target_output: output_vec,
                                    ncomputer.sequence_length: seq_len,
                                    ncomputer.encode1_point: e1,
                                    ncomputer.encode2_point: e2,
                                    ncomputer.decoder_point: decoder_point,
                                    ncomputer.clear_mem: False,
                                    ncomputer.teacher_force: ncomputer.get_bool_rand_incremental(seq_len, prob_true_max=0)
                                                })




                                # print(np.argmax(target_output, axis=-1))
                                # print(out)
                                # print(np.max(out, axis=-1))
                                # print(weights)
                                losses.append(loss_v)
                                out = np.reshape(np.asarray(out), [-1, seq_len, output_size])
                                pout=out
                                pind = np.argsort(out, axis=-1)
                                out_list = []
                                unorder_predict1=[]
                                if multi:
                                    io=1

                                    while len(out_list)<len(rout):
                                        if pind[0][decoder_point][-io]>1:
                                            out_list.append(pind[0][decoder_point][-io])
                                            unorder_predict1.append(pind[0][decoder_point][-io])
                                        io+=1
                                    unorder_predict1=unorder_predict1[::-1]

                                else:
                                    if args.use_beam_search==0:
                                        for io in range(decoder_point, seq_len):
                                            c = 1
                                            while 1:
                                                label = pind[0][io][-c]
                                                if label not in out_list:
                                                    out_list.append(label)
                                                    break
                                                c += 1
                                    else:
                                        out_list = beam_search.leap_beam_search(pout[0][decoder_point:seq_len],
                                                                     beam_size=args.use_beam_search,
                                                                                 is_set=True, is_fix_length=True)

                                    prob_pre=[]
                                    for ci,c in enumerate(out_list):
                                        prob_pre.append(pout[0][decoder_point+ci][c])
                                    unorder_predict1 = [x for _, x in sorted(zip(prob_pre, out_list))]
                                if args.mode=='test':
                                    avg_loss_v=loss_v/len(rout)
                                    if avg_loss_v<single_best_loss:
                                        single_best_loss=avg_loss_v
                                        best_mem_view=mem_view
                                        best_data=[input_vec1, input_vec2, unorder_predict1[::-1], rout, decoder_point]

                                big_out_list.append(out_list)
                                # tescores.append(bleu_score(np.asarray([rout]), np.asarray([out_list])))
                                pre, rec, jac = set_score_pre_jac(np.asarray([rout]), np.asarray([out_list]))
                                # print(pout.shape)
                                auc,auc2=roc_auc(np.asarray([rout]),pout[:,decoder_point])
                                f1, f2 = fscore(np.asarray([rout]), np.asarray([out_list]), output_size)
                                tescores.append(jac)
                                tescores2.append(pre)
                                tescores3.append(auc)
                                tescores4.append(auc2)
                                tescores5.append(f1)
                                tescores6.append(f2)
                                # at 1
                                # if args.mode=='test':
                                #     pre1=0
                                #     for pr in unorder_predict1[-2:]:
                                #         pre, rec = set_score_pre(np.asarray([rout]), np.asarray([[pr]]))
                                #         pre1=max(pre1,pre)
                                #         if pre1==1:
                                #             break
                                #     pre=pre1
                                # else:
                                pre, rec = set_score_pre(np.asarray([rout]), np.asarray([unorder_predict1[-1:]]))
                                tescores2_1.append(pre)

                                # at 2
                                pre, rec = set_score_pre(np.asarray([rout]), np.asarray([unorder_predict1[-2:]]))
                                tescores2_2.append(pre)

                                # at 3
                                pre, rec = set_score_pre(np.asarray([rout]), np.asarray([unorder_predict1[-3:]]))
                                tescores2_3.append(pre)

                                # at 5
                                pre, rec = set_score_pre(np.asarray([rout]), np.asarray([unorder_predict1[-5:]]))
                                tescores2_5.append(pre)
                        tloss=np.mean(losses)
                        tpre=np.mean(tescores2)
                        print('tr score {} vs te store {}'.format(np.mean(trscores),np.mean(tescores)))
                        print('test prec {} auc {} auc2 {} f1 {} f2 {}'.
                              format(np.mean(tescores2),
                                     np.mean(tescores3), np.mean(tescores4),
                                     np.mean(tescores5), np.mean(tescores6)))
                        print('test at 1 {}'.format(np.mean(tescores2_1)))
                        print('test at 2 {}'.format(np.mean(tescores2_2)))
                        print('test at 3 {}'.format(np.mean(tescores2_3)))
                        print('test at 5 {}'.format(np.mean(tescores2_5)))
                        print('test loss {}'.format(tloss))
                        if args.mode=='test':
                            print(best_mem_view['write_gates1'])
                            print('---')
                            print(best_mem_view['write_gates2'])
                            print('xxxx')
                            in1=convert_oh2raw(best_data[0][0], best_data[-1]-1)
                            in2=convert_oh2raw(best_data[1][0], best_data[-1]-1)
                            print(in1)
                            print(in2)
                            print(best_data[2])
                            print(best_data[3])
                            print('--translate---')
                            in12 = []
                            in22 = []
                            out12 = []
                            out22 = []
                            for c in in1:
                                if c>1:
                                    in12.append(tok2str_diag2[int(tok2str_diag[c])])
                            for c in in2:
                                if c>1:
                                    in22.append(tok2str_proc2[int(tok2str_proc[c])])
                            for c in best_data[2]:
                                if c > 1:
                                    out12.append(tok2str_drug2[int(tok2str_drug[c])])
                            for c in best_data[3]:
                                if c > 1:
                                    out22.append(tok2str_drug2[int(tok2str_drug[c])])
                            print(in12)
                            print(in22)
                            print(out12)
                            print(sorted(out22))

                        if args.mode=='train':
                            summary.value.add(tag='train_jac', simple_value=np.mean(trscores))
                            summary.value.add(tag='test_jac', simple_value=np.mean(tescores))
                            summary.value.add(tag='test_loss', simple_value=tloss)
                            summary.value.add(tag='test_recall', simple_value=np.mean(tescores3))
                            summary.value.add(tag='test_precision', simple_value=np.mean(tescores2))
                            train_writer.add_summary(summary, i)
                            train_writer.flush()

                        end_time_100 = time.time()
                        elapsed_time = (end_time_100 - start_time_100) / 60
                        avg_counter += 1
                        avg_100_time += (1. / avg_counter) * (elapsed_time - avg_100_time)
                        estimated_time = (avg_100_time * ((end - i) / 100.)) / 60.

                        print ("\tAvg. 100 iterations time: %.2f minutes" % (avg_100_time))
                        print ("\tApprox. time to completion: %.2f hours" % (estimated_time))

                        start_time_100 = time.time()
                        last_100_losses = []

                    if min_tloss<tpre:
                        min_tloss=tpre
                        if args.mode == 'train':
                            llprint("\nSaving Checkpoint ... ")
                            ncomputer.save(session, ckpts_dir, ncomputer.print_config())
                        llprint("Done!\n")

                except KeyboardInterrupt:
                    sys.exit(0)

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
    parser.add_argument('--mode', default="train")
    parser.add_argument('--use_mem', default=False, type=str2bool)
    parser.add_argument('--share_mem', default=True, type=str2bool)
    parser.add_argument('--use_teacher', default=False, type=str2bool)
    parser.add_argument('--persist', default=True, type=str2bool)
    parser.add_argument('--word_size', default=64, type=int)
    parser.add_argument('--mem_size', default=16, type=int)
    parser.add_argument('--read_heads', default=1, type=int)
    parser.add_argument('--emb_size', default=64, type=int)
    parser.add_argument('--dual_emb', default=True, type=str2bool)
    parser.add_argument('--attend', default=0, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--iters', default=42000, type=int)
    parser.add_argument('--eval_step', default=3000, type=int)
    parser.add_argument('--type', default="no_cache")
    parser.add_argument('--task', default="trim_diag_proc_drug_no_adm")
    parser.add_argument('--from_checkpoint', default="")
    parser.add_argument('--use_beam_search', default=0, type=int)
    args = parser.parse_args()
    print(args)
    mimic_task_persit_real_dual(args)

