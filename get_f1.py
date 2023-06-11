import numpy as np
import re
unknownfiles = [
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_unkinit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_INPUT_EMBEDDINGS,xavier,1.0)_distillLR0.01_E(100,10,0.5)_lr0.01_B1x1x5_train(unknown_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_unkinit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_INPUT_EMBEDDINGS,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B10x1x5_train(unknown_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_unkinit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_MOD,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B1x1x5_train(unknown_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_unkinit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_NO_GUMBEL,xavier,1.0)_distillLR0.01_E(200,10,0.5)_lr0.01_B1x1x5_train(unknown_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_unkinit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_NO_GUMBEL,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B10x1x5_train(unknown_init)/output.log',
]

knownfiles = [
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_knowninit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_INPUT_EMBEDDINGS,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B1x1x5_train(known_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_knowninit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_INPUT_EMBEDDINGS,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B10x1x5_train(known_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_knowninit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_MOD,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B1x1x5_train(known_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_knowninit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_MOD,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B10x1x5_train(known_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_knowninit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_NO_GUMBEL,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B1x1x5_train(known_init)/output.log',
    '/data/harsh/dataset-distillation-2/dataset-distillation/text_results/umsab_20by1_knowninit_repl1/distill_basic/umsab/arch(TextConvNet_BERT_NO_GUMBEL,xavier,1.0)_distillLR0.01_E(250,10,0.5)_lr0.01_B10x1x5_train(known_init)/output.log',
]

for file in knownfiles:
    # open file
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # get f1
    f1 = []
    print(file)
    for line in lines:
        x = re.findall(r'\).         ([\d.]+)', line)
        if len(x) > 0:
            f1.append(float(x[0]))
            print(x[0])