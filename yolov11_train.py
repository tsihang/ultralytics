#!/usr/bin/env python3

import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg

    model = YOLO(yaml)
    #model = YOLO(weights)
    #model = YOLO(yaml).load(weights)

    model.info()

    results = model.train(data='coco128.yaml',
                        epochs=300, 
                        imgsz=640, 
                        workers=8, 
                        batch=4,
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/11/yolo11.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
