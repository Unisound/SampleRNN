# SampleRNN  

A Tensorflow implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837).

## Requirements
- tensroflow 1.0  
- python 2.7  
- librosa   
## Dataset  
We used the pinao music of 74 minutes as the training corpus, and you can use any corpus containing ".wav" files to instead as well.
## Example
## FEATURES
- [ ] 2-tier SampleRNN
- [x] 3-tier SampleRNN
- [ ] Quantization in linear. 
- [x] Quantization in mu-law. 

## Training 
```shell
python train.py \
	--data_dir=./pinao-corpus \
	--silence_threshold=0.1 \
	--sample_size=102408 \
	--big_frame_size=8 \
	--frame_size=2 \
	--q_levels=256 \
	--rnn_type=GRU \
	--dim=1024 \
	--n_rnn=1 \
	--seq_len=520 \
	--emb_size=256 \
	--batch_size=64 \
	--optimizer=adam \
	--num_gpus=4
```
or  
```shell
sh run.sh
```
## Related projects
This work is based on the flowing implementations with some modifications:  
- [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet), a TensorFlow implementation of WaveNet
- [sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017), a Theano implementation of sampleRNN
