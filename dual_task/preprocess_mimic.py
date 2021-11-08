import pickle
import numpy as np
import os

def preprocess(dict_patient, store_path="./data/real_data/mimic/trim_diag_proc_drug_no_adm"):
    """

    :param dict_patient: assume this is a dict with keys: pa, diag, proc, med, admit_time.
    dict_patient[pa]=list of raw patient ids, each for a patient
    dict_patient[diag]=list of raw diagnoses, each for a patient
    dict_patient[proc]=list of raw procedures, each for a patient
    dict_patient[med]=list of raw medicines, each for a patient
    dict_patient[admit_time]=list of admitted time, each for a patient
    The lists share patient order
    :return:
    stores preprocessed files
    """
    all_pats = {}

    for pa, d, p, m, t in zip(dict_patient['pa'], dict_patient['diag'], dict_patient['proc'], dict_patient['med'], dict_patient['admit_time']):
        if pa not in all_pats:
            all_pats[pa] = []
        if not m:
            m = ["PAD"]
        if not d:
            d = ["PAD"]
        if not p:
            p = ["PAD"]
        d = list(sorted(set(d), key=d.index))
        m = list(sorted(set(m), key=m.index))
        p = list(sorted(set(p), key=p.index))
        all_pats[pa].append([d, m, p, t])

    c = 0
    for k, v in all_pats.items():
        v2 = sorted(v, key=lambda x: x[3])
        # v2=v
        all_pats[k] = v2
        if c < 5:
            print(k)
            print(v2)
        c += 1
    print('---------------')
    print(len(all_pats))
    all_len = []
    chosen_patients = []
    diag_lens = []
    drug_lens = []
    proc_lens = []

    diag_str2token = {"PAD": 0, "EOS": 1}
    diag_token2str = {0: "PAD", 1: "EOS"}
    drug_str2token = {"PAD": 0, "EOS": 1}
    drug_token2str = {0: "PAD", 1: "EOS"}
    proc_str2token = diag_str2token
    proc_token2str = diag_token2str

    str2token = [diag_str2token, drug_str2token, proc_str2token]
    token2str = [diag_token2str, drug_token2str, proc_token2str]

    for k, v in all_pats.items():
        fail = False
        for adm in v:
            if min(len(adm[0]), len(adm[1]), len(adm[2])) == 0:
                fail = True
                break
        if len(v) < 2:
            fail = True
        if not fail:
            newv = []
            for ai, adm in enumerate(v[:-1]):
                newv.append([adm[0], adm[1], v[ai + 1][2]])
            v2 = newv
            newv = []
            for adm in v2:
                # print(adm)
                # print('....')
                diag_lens.append(len(adm[0]))
                drug_lens.append(len(adm[1]))
                proc_lens.append(len(adm[2]))
                new_adms = []
                for si, sadm in enumerate([adm[0], adm[1], adm[2]]):
                    new_adm = []
                    for d in sadm:
                        if d not in str2token[si]:
                            str2token[si][d] = len(str2token[si])
                            token2str[si][str2token[si][d]] = d
                        new_adm.append(str2token[si][d])
                    new_adms.append(new_adm)
                # print(new_adms)
                newv.append(new_adms)
            all_len.append(len(newv))
            chosen_patients.append(newv)
            # raise False

    print(len(chosen_patients))
    for ex in range(3):
        print(chosen_patients[ex])
    print('total adm {}'.format(np.sum(all_len)))
    print('min adm  {}'.format(np.min(all_len)))
    print('max adm  {}'.format(np.max(all_len)))
    print('avg adm  {}'.format(np.mean(all_len)))

    print('min max avg diag {} {} {}'.format(np.min(diag_lens), np.max(diag_lens), np.mean(diag_lens)))
    print('min max avg drug {} {} {}'.format(np.min(drug_lens), np.max(drug_lens), np.mean(drug_lens)))
    print('min max avg proc {} {} {}'.format(np.min(proc_lens), np.max(proc_lens), np.mean(proc_lens)))

    print('dict info')
    print(len(diag_token2str))
    print(len(drug_token2str))
    print(len(proc_token2str))
    print('done')

    if not os.path.isdir(store_path):
        os.mkdir(store_path)

    pickle.dump(str2token, open(store_path + '/list_dict_str2token_no_adm.pkl', 'wb'))
    pickle.dump(token2str, open(store_path + '/list_dict_token2str_no_adm.pkl', 'wb'))
    # np.random.shuffle(chosen_patients)
    print('dump sequence')
    pickle.dump(chosen_patients, open(store_path + '/patient_records.pkl', 'wb'))