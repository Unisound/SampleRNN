# SampleRNN  

A Tensorflow implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837).

## Requirements
- Tensroflow 1.0  
- Python 2.7  
- Librosa   
## Dataset  
We used the pinao music of 74 minutes as the training corpus, and you can use any corpus containing ".wav" files to instead as well.
For Mandarin samples, we used human voice of 7.7 hours as the training corpus .
## Samples 
- [Mandarin samples](https://pan.baidu.com/s/1o8M8bGI)
- [Pinao samples](https://soundcloud.com/xue-ruiqing/sets/tensorflow-samplernn)
- [Pinao samples_2](https://soundcloud.com/xue-ruiqing/sets/tensorflow-samplernn_2)
## Pretrained model
- [logdir(pinao)](https://drive.google.com/file/d/0B2MbqozKaoOQMW9PeHA1ZWNlTGc/view?usp=sharing)
- [logdir(mandarin)](https://pan.baidu.com/s/1i4WBq4X)
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
