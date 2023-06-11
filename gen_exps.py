#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 00:26:49 2019

@author: ilia10000
"""

import itertools as it
import numpy as np

epochs=20
exps={}
exps["dataset"]={"umsab"}
exps["softmax"]={1,0}
exps["network_init"]={"Fixed"}#,"Random"}
exps["label_init"]={"hard"}
exps["repl"]={1}#,2,3}
exps["add"]={0, -0.5}
exps["mult"]={10}
allNames=exps.keys()
combos = it.product(*(exps[Name] for Name in allNames))

def write_to_file():
    with open("label_exps.txt","w") as f:
        for combo in combos:
            command = "python3 main.py --mode distill_basic --dataset {0} --arch {1}Net {2} --distill_steps 1 --static_labels 0 --random_init_labels {3} --distill_lr 0.01  --decay_epochs 35 --epochs 350 --lr 0.01  --results_dir {4} --device_id 0 --add_label_scaling {5} --mult_label_scaling {6} --dist_metric {7} --invert_dist {8} --label_softmax {9}"
            random_init = "--train_nets_type known_init --n_nets 1 --test_nets_type same_as_train" if combo[2] == "Random" else ""
            network = "Le" if combo[0]=="MNIST" else "AlexCifar"
            dist_metric = "SSIM" if "SSIM" in combo[3] else "MSE"
            invert_dist = 1 if ("CNIDB" in combo[3] or "AIIDB" in combo[3]) else "''"
            results_dir = "~/label-init-exps/softmax_{0}-labinit_{1}-add_{2}-mult_{3}-repl_{4}".format(combo[1], combo[3], combo[5], combo[6], combo[4])
            labinit=combo[3]
            if "DB" in combo[3]:
                if "CNIDB"==combo[3]:
                    labinit = "CNDB"
                elif "IDB" in combo[3]:
                    labinit = "AIDB"
            command=command.format(combo[0],network,random_init, labinit,results_dir, combo[5], combo[6], dist_metric, invert_dist, combo[1])
            f.write(command+"\n\n")

def write_to_batch_files(batch_size=40, devices=8):
    i=0
    for combo in combos:
        with open("exp_scripts/label_exps_{0}.sh".format(int(np.floor(i/batch_size))),"a+") as f:
            command = "python3 main.py --mode distill_basic --dataset {0} --arch {1}Net {2} --distill_steps 1 --static_labels 0 --random_init_labels {3} --distill_lr {13}  --decay_epochs 35 --epochs {12} --lr 0.01  --results_dir {4} --device_id {11} --add_label_scaling {5} --mult_label_scaling {6} --dist_metric {7} --invert_dist {8} --label_softmax {9} > exp_scripts/batch_output_{10} 2>&1 &"
            random_init = "--train_nets_type known_init --n_nets 1 --test_nets_type same_as_train" if combo[2] == "Fixed" else "--test_n_nets 200"
            network = "Le" if combo[0]=="MNIST" else "AlexCifar"
            distill_lr = 0.01 if combo[0]=="MNIST" else 0.001
            dist_metric = "SSIM" if "SSIM" in combo[3] else "MSE"
            invert_dist = 1 if ("CNIDB" in combo[3] or "AIIDB" in combo[3]) else "''"
            results_dir = "~/soft-label-inits/full-softmax_{0}-labinit_{1}-add_{2}-mult_{3}-repl_{4}".format(combo[1], combo[3], combo[5], combo[6], combo[4])
            labinit=combo[3]
            if "DB" in combo[3]:
                if "CNIDB"==combo[3]:
                    labinit = "CNDB"
                elif "IDB" in combo[3]:
                    labinit = "AIDB"
            command=command.format(combo[0],network,random_init, labinit,results_dir, combo[5], combo[6], dist_metric, invert_dist, combo[1], int(np.remainder(i,batch_size)), i%devices, epochs, distill_lr)
            f.write(command+"\nsleep 2\n")
            if i==0:
                f.write("\nsleep 10\n") #to give time to get datasets
            i+=1
            
write_to_batch_files(batch_size=35)            
                        
            
            
            
            
    