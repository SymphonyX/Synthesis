#!/bin/bash

python test_domain.py -t 0.0 -s 00.pkl
python test_domain.py -t 0.5 -s 05.pkl -p 00.pkl
python test_domain.py -t 1.0 -s 10.pkl -p 05.pkl
python test_domain.py -t 1.5 -s 15.pkl -p 10.pkl
python test_domain.py -t 2.0 -s 20.pkl -p 15.pkl
python test_domain.py -t 2.5 -s 25.pkl -p 20.pkl
python test_domain.py -t 3.0 -s 30.pkl -p 25.pkl
python test_domain.py -t 3.5 -s 35.pkl -p 30.pkl
python test_domain.py -t 4.0 -s 40.pkl -p 35.pkl
python test_domain.py -t 4.5 -s 45.pkl -p 40.pkl
python test_domain.py -t 5.0 -s 50.pkl -p 45.pkl
python test_domain.py -t 5.5 -s 55.pkl -p 50.pkl
python test_domain.py -t 6.0 -s 60.pkl -p 55.pkl