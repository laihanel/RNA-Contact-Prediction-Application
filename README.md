# Using transformer to extract structure information for multi-sequences to boost function learning
Aihan Liu, Hsueh-Yi Lu

---
## Introduction
This project focuses on using transformers to extract structural information for
multi-sequences, which enhances function learning in RNA. RNA utilizes four bases,
and nucleotide-nucleotide interactions are crucial in determining RNA structure and
function. Transfer learning from protein-based models can improve RNA contact
prediction and deep learning algorithms can be used to identify and prioritize important
positions in genotype-phenotype associations. The study also explores graph
convolutional networks to learn complex structure-function relationships. The results
indicate that machine learning-based structure and function prediction can provide
explanations for RNA structure and open up avenues for new research in the field of
molecular biology.

## Environment Requirements
### CoT Transfer Learning
```bash
conda create -n pytorch-1.8 python=3.7
conda activate pytorch-1.8
pip install tqdm
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pydca
pip install tensorboard
```

### deepBreaks
```bash
conda create --name deepBreaks_env python=3.9
conda activate deepBreaks_env 
pip install deepBreaks
python -m pip install git+https://github.com/omicsEye/deepbreaks
```

### GCN
#### Installation
```
cd pygcn
python setup.py install
```
#### Requirements
  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Example
```bash
# contact prediction
cd CoT_Transfer_Learning
unzip data/HIV/HIV.zip -d data/HIV/
python tidyup_data.py --input data/HIV/hiv_V3_B_C_nu_clean.fasta -- output data/HIV/hiv.fasta
python run_inference.py --input_MSA data/HIV/hiv.fasta

# deepbreaks
cd deepbreaks/Examples


# GCN prediction
cd ../pygcn
python train.py
```

