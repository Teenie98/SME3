# SME

## get dataset pkl
```commandline
python data_split.py
```
the data is in /data 

## get parameter
set '--pre_train' = 1 get target dataset train parameter in backbone model
set '--pre_train' = 2 get source dataset train parameter in backbone model
run /rec_models/__init__.py

## run model SME3
run  /gen_models/__init__.py
