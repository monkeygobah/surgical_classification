#!/bin/bash
# python main.py --root_dir none --model "LSTM" --batch_size 8 --device_num 0 --label 'train_easy_test_hard' --phases --train_easy
python main.py --root_dir none --model "LSTM" --batch_size 8 --device_num 0 --label 'train_hard_test_easy' --phases
