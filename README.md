# Eyettention: An Attention-based Dual-Sequence Model for Predicting Human Scanpaths during Reading
[![paper](https://img.shields.io/static/v1?label=paper&message=download%20link&color=brightgreen)](https://arxiv.org/abs/2304.10784)

In this paper, we develop Eyettention, the first dual-sequence model that simultaneously processes the sequence of words and the chronological sequence of fixations. The alignment of the two sequences is achieved by a cross-sequence attention mechanism. We show that Eyettention outperforms state-of-the-art models in predicting scanpaths. We provide an extensive within- and across-data set evaluation on different languages. An ablation study and qualitative analysis support an in-depth understanding of the model's behavior.

## Setup

Clone repository:

```
git clone git@github.com:aeye-lab/Eyettention
```

or

```
git clone https://github.com/aeye-lab/Eyettention
```
and change to the cloned repo via `cd Eyettention`.

Install dependencies:

```
pip install -r requirements.txt
```

## Run Experiments
#For Chinese BSC dataset:
```
python main_BSC.py --test_mode='text'
python main_BSC.py --test_mode='subject'
python main_BSC_NRS_setting.py
python main_BSC_reader_identifier.py
```

#For English CELER dataset:
```
python main_celer.py --test_mode='text'
python main_celer.py --test_mode='subject'
python main_celer_NRS_setting.py
python main_celer_reader_identifier.py
```

## Cite our work
If you use our code for your research, please consider citing our paper:

```bibtex
@article{deng2023eyettention,
  title={Eyettention: An Attention-based Dual-Sequence Model for Predicting Human Scanpaths during Reading},
  author={Deng, Shuwen and Reich, David R and Prasse, Paul and Haller, Patrick and Scheffer, Tobias and J{\"a}ger, Lena A},
  journal={Proceedings of the ACM on Human-Computer Interaction},
  volume={7},
  number={ETRA},
  pages={1--24},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```
