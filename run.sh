#### ORIGINAL DATA TRAINING
CUDA_VISIBLE_DEVICES=4 python main.py --mode train --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --train_nets_type known_init --test_nets_type same_as_train --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 10\
 --epochs 120 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1 \
 --device_id 0 --phase train

#### UNK INIT VANILLADISTILL ----------------------------------------------------------------------------------
#1
CUDA_VISIBLE_DEVICES=0 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5\
 --epochs 25 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1 \
 --device_id 0 --phase train

# 10
CUDA_VISIBLE_DEVICES=4 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5\
 --epochs 25 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1 \
 --device_id 0 --phase train

# 100
CUDA_VISIBLE_DEVICES=1 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 100 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5\
 --epochs 30 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1 \
 --device_id 0 --phase train

# 1000
CUDA_VISIBLE_DEVICES=2 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 1000 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5\
 --epochs 30 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1 \
 --device_id 0 --phase train

#### FIXED INIT VANILLADISTILL ----------------------------------------------------------------------------------
#1 tdd6
CUDA_VISIBLE_DEVICES=3 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --train_nets_type known_init --test_nets_type same_as_train --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 10\
 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1 \
 --device_id 0 --phase train

# 10 tdd5
CUDA_VISIBLE_DEVICES=1 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT \
 --batch_size 1024 --distill_steps 1 --train_nets_type known_init --test_nets_type same_as_train --static_labels 0 --random_init_labels hard --textdata True --visualize ''\
 --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 10\
 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1 \
 --device_id 0 --phase train



#### UNK INIT VOCABDISTILL (SOFTMAX) ----------------------------------------------------------------------------------

# NO GUMBEL
CUDA_VISIBLE_DEVICES=0 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_NO_GUMBEL  --batch_size 2048 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 200 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=5 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_NO_GUMBEL  --batch_size 2048 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=0 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_NO_GUMBEL  --batch_size 512 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 50 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

#### UNK INIT SKIPLOOKUPDISTILL ----------------------------------------------------------------------------------

# INPUT EMBEDDINGS
python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_INPUT_EMBEDDINGS  --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 100 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=2 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_INPUT_EMBEDDINGS  --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=0 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_INPUT_EMBEDDINGS  --batch_size 512 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 50 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

#### UNK INIT VOCABDISTILL (GUMBEL) ----------------------------------------------------------------------------------

# GUMBEL 
CUDA_VISIBLE_DEVICES=0 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_MOD  --batch_size 1024 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=6 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_MOD  --batch_size 512 --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_unkinit_repl1  --device_id 0 --phase train


#### FIXED INIT VOCABDISTILL (SOFTMAX) ----------------------------------------------------------------------------------

# NO GUMBEL
CUDA_VISIBLE_DEVICES=2 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_NO_GUMBEL  --batch_size 2048 --train_nets_type known_init --test_nets_type same_as_train --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=5 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_NO_GUMBEL  --batch_size 512 --train_nets_type known_init --test_nets_type same_as_train --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1  --device_id 0 --phase train

#### FIXED INIT SKIPLOOKUPDISTILL ----------------------------------------------------------------------------------

# INPUT EMBEDDINGS
CUDA_VISIBLE_DEVICES=3 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_INPUT_EMBEDDINGS  --batch_size 2048 --train_nets_type known_init --test_nets_type same_as_train --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=0 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_INPUT_EMBEDDINGS  --batch_size 512 --train_nets_type known_init --test_nets_type same_as_train --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 768 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1  --device_id 0 --phase train

#### FIXED INIT VOCABDISTILL (SOFTMAX) ----------------------------------------------------------------------------------

# GUMBEL
CUDA_VISIBLE_DEVICES=2 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_MOD  --batch_size 1024 --train_nets_type known_init --test_nets_type same_as_train --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 1 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1  --device_id 0 --phase train

CUDA_VISIBLE_DEVICES=5 python main.py --mode distill_basic --dataset umsab --arch TextConvNet_BERT_MOD  --batch_size 512 --train_nets_type known_init --test_nets_type same_as_train --distill_steps 1 --static_labels 0 --random_init_labels hard --textdata True --visualize '' --distilled_images_per_class_per_step 10 --distill_epochs 5 --distill_lr 0.01 --decay_epochs 10 --log_interval 5 --epochs 250 --lr 0.01 --ntoken 10000 --ninp 119547 --maxlen 75 --results_dir text_results/umsab_20by1_knowninit_repl1  --device_id 0 --phase train