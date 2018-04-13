#!/bin/bash

feature_transform_proto=/tmp/proto
splice=0
train_feats='ark:copy-feats scp:data/train/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/train/utt2spk scp:data/train/cmvn.scp ark:- ark:- | add-deltas --delta-order=1 ark:- ark:- |'

feat_dim=$(feat-to-dim "$train_feats" -)
echo "<Splice> <InputDim> $feat_dim <OutputDim> $(((2*splice+1)*feat_dim)) <BuildVector> -$splice:$splice </BuildVector>" >$feature_transform_proto

feature_transform=/tmp/nnet
nnet-initialize --binary=false $feature_transform_proto $feature_transform

nnet-forward --print-args=true $feature_transform "$train_feats" ark:- |\
compute-cmvn-stats ark:- - | cmvn-to-nnet --std-dev=1.0 - -| nnet-concat --binary=false $feature_transform - 'data/final.feature_transform'

# copy-feats scp:data/train/feats.scp ark:- | apply-cmvn --utt2spk=ark:data/train/utt2spk scp:data/train/cmvn.scp ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- | nnet-forward final.feature_transform ark:- ark:- |
