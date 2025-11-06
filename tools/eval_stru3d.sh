#!/usr/bin/env bash

python eval.py --dataset_name=stru3d \
               --backbone=swinv2_L_192_22k \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=checkpoint/CAGE_stru3d_swinv2.pth \
               --output_dir=eval_stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
