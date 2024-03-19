# Raw waveform adaptation with SincNet
[![arXiv](https://img.shields.io/badge/arXiv-1909.13759-b31b1b.svg)](https://arxiv.org/abs/1909.13759)

This repository contains code for our ASRU 2019 paper titled ["Acoustic model adaptation from raw waveforms with SincNet"](http://arxiv.org/abs/1909.13759). The aim is to explore the adaptation of the SincNet layer (filter parameters and amplitudes) to speakers and domains.

The code is a little messy. I hope to clean it up soon, time permitting. Any questions or problems - please get in touch.

Much of the code is built on the work by Ondrej for [Learning to Adapt](https://github.com/ondrejklejch/learning_to_adapt).

This work is the result of a collaboration with my co-authors [Ondrej Klejch](http://www.ondrejklejch.cz), [Erfan Loweimi](https://www.research.ed.ac.uk/portal/eloweimi), [Peter Bell](http://homepages.inf.ed.ac.uk/pbell1), and [Steve Renals](https://homepages.inf.ed.ac.uk/srenals).

## Dependencies
The code has been run with:

 - [Keras](https://keras.io/) 2.2.2
 - [Tensorflow](https://www.tensorflow.org/) 1.10.0
 - [PyKaldi](https://github.com/pykaldi/pykaldi)
 - [Kaldi](https://github.com/kaldi-asr/kaldi)
 
## Usage
For training from scratch see `experiments/ami/train_sinc_40_flat_6epochs.sh`. For speaker adaptation see `experiments/ami/adapt_pfstar_40_flat_speaker_lhuc0+sinc.sh`. The layers to be adapted (LHUC0, LHUC1, LHUC0+Sinc, etc.) can are determined by an argument to `adapt_pfstar_40_flat_speaker.py`. The above scripts assume an existing tri3 model of AMI (or a different dataset). It will also look for `pdf_counts` in the main directory, which is equivalent to e.g. `tri3/final.occs`.

## Citation
For research using this work, please cite:
```
@inproceedings{Fainberg2019,
  author={Joachim Fainberg and Ondřej Klejch and Erfan Loweimi and Peter Bell and Steve Renals},
  title={{Acoustic Model Adaptation from Raw Waveforms with SincNet}},
  booktitle={ASRU},
  year=2019
}
```

## References
Our work builds on a paper by [Ravanelli and Bengio](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8639585). They have a [SincNet implementation for PyTorch](https://github.com/mravanelli/SincNet).
