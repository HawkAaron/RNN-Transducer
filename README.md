# RNN Transducer
[MXNET GPU version of RNN Transducer loss is now available !](https://github.com/HawkAaron/mxnet-transducer/tree/add_network)

## File description
* eval.py: transducer decode
* model.py: rnn transducer refer to Graves2012
* DataLoader.py: data process
* train.py: rnnt training script, can be initialized from CTC and PM model

## Directory description
* conf: kaldi feature extraction config

## Reference Paper
* RNN Transducer (Graves 2012): [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* RNNT joint (Graves 2013): [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778 )

## Run
* Compile RNNT Loss
Follow the instructions in [here](https://github.com/HawkAaron/mxnet-transducer/tree/add_network) to compile MXNET with RNNT loss.

* Extract feature
link kaldi timit example dirs (`local` `steps` `utils` )
excute `run.sh` to extract 13 dim mfcc feature
run `feature_transform.sh` to get 26 dim feature as described in Graves2012

* Train RNNT model:
```bash
python train.py --lr 1e-3 --bi --dropout .5 --out exp/rnnt_bi_lr1e-3 --schedule
```

## Evaluation
Default only for RNNT
* Greedy decoding:
```
python eval.py <path to best model parameters> --bi
```
* Beam search:
```
python eval.py <path to best model parameters> --bi --beam <beam size>
```

## Results
* CTC

    | Decode | PER |
    |-------|---|
    | greedy | 22.27 |
    | beam 20 | **21.83** |

* Transducer

    | Decode | PER |
    |------|------|
    | greedy | 23.02 |
    | beam 20 | 22.45 |
    | beam 40 | 22.34 |
    | beam 60 | **21.92** |
    | beam 80 | 22.15 |

## TODO
* beam serach accelaration
* several baseline
* Seq2Seq with attention