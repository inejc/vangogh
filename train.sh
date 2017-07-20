#!/usr/bin/env bash

python src/analysis/generate_model.py --algorithm svm_linear --dir vgdb_2016/train/feats_bottleneck/ --model vgdb_2016/clf/svm_linear_bottleneck.pkl --verbose
python src/analysis/generate_model.py --algorithm svm_linear --dir vgdb_2016/train/feats_map/ --model vgdb_2016/clf/svm_linear_map.pkl --verbose
