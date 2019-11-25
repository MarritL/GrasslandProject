DATASET:
  segm_downsampling_rate: 4

MODEL:
  weights_encoder: ""
  weights_decoder: ""

TRAIN:
  lr_pow: 1
  beta1: 0.9
  deep_sup_scale: 0.4
  fix_bn: False
  loss: 100
  counter: 0
  
TEST:
  batch_size: 1
  result: "./"
