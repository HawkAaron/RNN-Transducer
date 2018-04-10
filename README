## File description
* eval.py: rnnt joint model decode
* joint_model.py: rnnt model, which contains acoustic / phoneme model
* rnnt_np.py: rnnt loss function implementation on mxnet, support for both symbol and gluon [refer to PyTorch implementation](https://github.com/awni/transducer)
* utils.py: data process
* train.py: rnnt training script, can be initialized from CTC and PM model
* train_log.py: rnnt training script with AM and PM PER log

## Directory description
* data-fbank: training data set 
* fbank: acoustic feature
* lang: phoneme units and mapping

## Reference Paper
* RNN Transducer (Graves 2012): [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
* RNNT joint (Graves 2013): [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778 )
* E2E criterion comparison (Baidu 2017): [Exploring Neural Transducers for End-to-End Speech Recognition](https://arxiv.org/abs/1707.07413)

# Run
* run local:
```bash
python train.py --init <path to initial rnnt model> --initam <path to initial CTC model> --initpm <path to initital PM model> \
--lr 1e-3 --out exp/rnnt_lr1e-3 --schedule
```

# Evaluation
Default only for RNNT
* Greedy decoding:
```
python eval.py <path to best model parameters>
```
* Beam search:
```
python eval.py <path to best model parameters> --beam <beam size>
```

# Note
* Current implementation support batch training, but for TIMIT, only do online training.
* The implementation of Transduction loss is really slow, about 5 times running time of CTC.
* If you train RNNT use `train_log.py`, the PER (calculated seperately in CTC training way.) doesn't change, it has nothing to do with RNNT joint PER

# TODO
* RNNT loss accelaration using CPP
* beam serach accelaration
* several baseline
* Seq2Seq with attention