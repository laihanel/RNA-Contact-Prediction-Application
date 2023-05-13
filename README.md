# Using transformer to extract structure information for multi-sequences to boost function learning
Aihan Liu, Hsueh-Yi Lu

---
<img src="presentation/Workflow.png">

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
## change to the cot_transfer_learning environment first
cd CoT_Transfer_Learning
unzip data/HIV/HIV.zip -d data/HIV/
python tidyup_data.py --input data/HIV/hiv_V3_B_C_nu_clean.fasta -- output data/HIV/hiv.fasta
python run_inference.py --input_MSA data/HIV/hiv.fasta

# deepbreaks
Follow these steps to use the deepBreaks environment and run the examples:
1. Make sure to set up the deepBreaks environment before running the examples.
2. To run the HIV.py file, execute the code and print out the plots to view the results.
3. For different results, you can modify the sequence length and the number of sequences used for training in the code. Feel free to experiment with different values.
4. Follow the steps of CoT-RNA-Transfer to get the contact matrix.

# GCN prediction
## change to the GCN environment first
cd ../pygcn
python train.py
```

## Results
The project achieved high accuracy in contact prediction and provided insights into important positions in the RNA sequence. The GCN model captured complex interactions within RNA structures, enhancing the accuracy of predictions. Results from different datasets and models are shown below:

| Data Description | Model Name                 | Training Size | Accuracy |
|------------------|----------------------------|---------------|----------|
| HIV-1 based on V3 | Logistic Regression        | 35,424, 105  | 0.9925   |
|                  | GCN (real contact)         | 35,424, 105  | 0.9924   |
|                  | GCN (random contact)       | 35,424, 105  | 0.9883   |
|                  | GCN (real contact)         | 100, 105     | 1.0000   |
|                  | GCN (random contact)       | 100, 105     | 0.8000   |
| SARS-CoV-2       | Extra Trees Classifier      | 900, 3822    | 0.9745   |
|                  | GCN (real contact)         | 900, 500     | 0.9495   |
|                  | GCN (random contact)       | 900, 500     | 0.9376   |
|                  | GCN (real contact)         | 900, 500     | 0.9376   |
|                  | GCN (random contact)      | 300, 500      | 0.3469   |

**Note:** The best ML method provided by deepBreaks was used for the logistic regression and extra trees classifier models.

## Structure and Function Relationship
The research indicates that with a large number of samples for training, the contact information has a minimal impact on prediction accuracy. However, the type and order of nucleotides become more critical. On the other hand, with smaller sample sizes, the contact information significantly influences the classification accuracy. The number and distribution of RNA cuts do not exhibit a significant pattern, and further investigation is required.

## Feature Importance
Certain positions in the sequence appear to be more relevant than others. Some of these positions align with the contact matrix, as shown by the red points in the figure. This provides an explainable result for RNA structure prediction. The research suggests that studying the feature importance can offer valuable insights into RNA structure and function.
<img src="presentation/Results.png">

## Future Work
The project opens up new avenues for further research and exploration. Future directions include:
- Optimizing CoT-Transfer learning by improving the mapping method and transfer learning network to overcome sequence length limitations.
- Conducting additional studies on RNA cuts to define appropriate criteria and examine the relationship between cuts and RNA structure complexity.
- Exploring alternative network architectures to extract a "feature map" for the sequence, potentially improving prediction results.
- Enhancing GCN explainability by employing gradient-based contrast or class activation mapping methods to identify important nodes in the network.

## Acknowledgements
We would like to express our sincere gratitude to our supervisor, Professor Chen Zeng, for his invaluable guidance and support throughout this project. We would also like to thank Professor Edwin Lo for providing valuable insights into our research. Special thanks to our classmates for their contributions and suggestions.

## References
1. Jian et al (Forthcoming), "Knowledge from Large-Scale Protein Contact Prediction Models can be Transferred to the Data-Scarce RNA Contact Prediction Task," Nature Machine Intelligence (submitted).
2. Rahnavard, A., Baghbanzadeh, M., Dawson, T., Sayoldin, B., Oakley, T., & Crandall, K. (2023). "deepBreaks: A Machine Learning Tool for Identifying and Prioritizing Genotype-Phenotype Associations."
3. Kipf, T. N., & Welling, M. (2016). "Semi-Supervised Classification with Graph Convolutional Networks." arXiv preprint arXiv:1609.02907.

