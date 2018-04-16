## File description
* eval.py: rnnt joint model decode
* model.py: rnnt model, which contains acoustic / phoneme model
* model2012.py: rnnt model refer to Graves2012
* rnnt_np.py: rnnt loss function implementation on mxnet, support for both symbol and gluon [refer to PyTorch implementation](https://github.com/awni/transducer)
* DataLoader.py: data process
* train.py: rnnt training script, can be initialized from CTC and PM model

## Directory description
* conf: kaldi feature extraction config

## Reference Paper
* RNN Transducer (Graves 2012): [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* RNNT joint (Graves 2013): [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778 )
* E2E criterion comparison (Baidu 2017): [Exploring Neural Transducers for End-to-End Speech Recognition](https://arxiv.org/abs/1707.07413)

## Run
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

## Note
* Current implementation support batch training, but for TIMIT, only do online training.
* The implementation of Transduction loss is really slow, about 5 times running time of CTC.
* If you train RNNT ~~using `train_log.py`~~, the PER (calculated seperately in CTC training way.) doesn't change, it has nothing to do with RNNT joint PER.

## TODO
* RNNT loss accelaration using CPP
* beam serach accelaration
* several baseline
* Seq2Seq with attention