#!/bin/bash
cd ..
conda activate master_thesis
python main.py --train -n 2 -top_k 200
python main.py --train -n 2 -top_k 300
python main.py --train -n 2 -top_k 400
python main.py --train -n 3 -top_k 200
python main.py --train -n 3 -top_k 300
python main.py --train -n 3 -top_k 400
python main.py --train -n 4 -top_k 200
python main.py --train -n 4 -top_k 300
python main.py --train -n 4 -top_k 400