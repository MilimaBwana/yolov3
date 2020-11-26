#!bin/bash

source /home/jakob/Dokumente/INFORMATIK/_SOMMERSEMESTER_20/Masterarbeit/pollenanalysis2.0/detectionvenv/bin/activate

python train.py --img 640 --batch 4 --epochs 1 --data ./data/pollen15.yaml --weights ./weights/yolov3.pt
