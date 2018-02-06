# DMNC
source code for dual memory neural computer

# sum2 task
cd to dual_task folder  
run command examples:  
train concatenate LSTM>>python3 sum2_run.py --name=single --mode=train --seq_len=10 --use_mem=False  
train concatenate DNC>>python3 sum2_run.py --name=single --mode=train --seq_len=10 --use_mem=True  
train Dual LSTM>>python3 sum2_run.py --mode=train --seq_len=10 --use_mem=False --attend=0  
train WLAS>>python3 sum2_run.py --mode=train --seq_len=10 --use_mem=False --attend=128  
train DMNC_l>>python3 sum2_run.py --mode=train --seq_len=10 --use_mem=True --share_mem=False  
train DMNC_e>>python3 sum2_run.py --mode=train --seq_len=10 --use_mem=True --share_mem=True  
test: use --mode=test

# emr task (drug prescription and disease procession)
Please prepare the EMR data as described in the paper, which includes:  
+ list of 3 dictionaries mapping from token to view codes 
+ list of 3 dictionaries mapping from view code to token
+ list of patients, each consists of list of admissions, each consists of 3 sequences corresponding to 3 views  
Please modify the code in emr_run.py to point to your data location
run command examples:
train Dual LSTM>>python3 emr_run.py --mode=train --seq_len=10 --use_mem=False --attend=0  
train WLAS>>python3 emr_run.py --mode=train --seq_len=10 --use_mem=False --attend=128  
train DMNC_l>>python3 emr_run.py --mode=train --use_mem=True --share_mem=False  
train DMNC_e>>python3 emr_run.py --mode=train --use_mem=True --share_mem=True  
test: use --mode=test --from_checkpoint=default
Feel free to modify the hyper-parameters