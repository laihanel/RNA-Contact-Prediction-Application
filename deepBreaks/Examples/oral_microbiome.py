from deepBreaks import preprocessing as prp
from deepBreaks import visualization as viz
from deepBreaks import models as ml
import os
import datetime
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

#Novel insights of niche associations in the oral microbiome

#3
# defining user params, file pathes, analysis type

# path to sequences
seqFileName = '/home/ubuntu/deepbreaks/data/s__Haemophilus_parainfluenzae.tsv'

# path to metadata
metaDataFileName = '/home/ubuntu/deepbreaks/data/my_HMP_metadata.tsv'

# name of the phenotype
mt = 'Body_site'

# type of the sequences
seq_type = 'nu'
# type of the analysis if it is a classification model, then we put cl instead of reg
anaType = 'cl'
sampleFrac=1

#4
# making a unique directory for saving the reports of the analysis
print('direcory preparation')
dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
seqFile = seqFileName.split('.')[0]
report_dir = str(seqFile +'_' + mt + '_' + dt_label)
os.makedirs(report_dir)

#5
print('reading meta-data')
# importing metadata
metaData = prp.read_data(metaDataFileName, seq_type = None, is_main=False)
print('metaData:', metaData.shape)

# importing sequences data
print('reading fasta file')
df = prp.read_data(seqFileName, seq_type = seq_type, is_main=True, gap_threshold=0.5)

positions = df.shape[1]
print('Done')
print('Shape of data is: ', df.shape)

#6
# selecting only the classes with enough number of samples
df = prp.balanced_classes(dat=df, meta_dat=metaData, feature=mt)

#7
print('metadata looks like this:')
print(metaData.head())

# Preprocessing
# In this step, we do all these steps:
#
# dropping columns with a number of missing values above a certain threshold
# dropping zero entropy columns
# imputing missing values with the mode of that column
# replacing cases with a frequency below a threshold (default 1.5%) with the mode of that column
# dropping zero entropy columns
# use statistical tests (each position against the phenotype) and drop columns with p-values below a threshold (default 0.25)
# one-hot encode the remaining columns
# calculate the pair-wise distance matrix for all of the columns
# use the distance matrix for DBSCAN and cluster the correlated positions together
# keep only one column (closes to center of each cluster) for each group and drop the rest from the training data set

#10
# taking care of missing data
print('Shape of data before missing/constant care: ', df.shape)
df = prp.missing_constant_care(df, missing_threshold=0.05)
print('Shape of data after missing/constant care: ', df.shape)

#11
# taking care of ultra-rare cases
print('Shape of data before imbalanced care: ', df.shape)
df = prp.imb_care(dat=df, imbalance_threshold=0.025)
print('Shape of data after imbalanced care: ', df.shape)

#12
# you may want to perform your analysis only on a random sample of the positions.
# Here you can have a random sample of your main data set.
print('number of columns of main data befor: ', df.shape[1])
df = prp.col_sampler(dat=df, sample_frac=sampleFrac)
print('number of columns of main data after: ', df.shape[1])

#13
# Use statistical tests to drop redundant features.
print('number of columns of main data befor: ', df.shape[1])
df_cleaned = prp.redundant_drop(dat=df, meta_dat=metaData,
                                feature=mt, model_type=anaType,
                                threshold=0.25,
                                report_dir=report_dir)
print('number of columns of main data after: ', df_cleaned.shape[1])

#14
print('one-hot encoding the dataset')
df_cleaned = prp.get_dummies(dat=df_cleaned, drop_first=True)

#15
print('calculating the distance matrix')
cr = prp.distance_calc(dat=df_cleaned,
                       dist_method='correlation',
                       report_dir=report_dir)
print(cr.shape)

#16
print('The distance matrix looks like this.\n The values are between 0 (exact the same) and 1 (non-related).')
cr.head()

#17
print('finding colinear groups')
dc_df = prp.db_grouped(dat = cr,
                       report_dir=report_dir,
                       threshold=.3)

#18
print('The result of the last step is a dataframe with two columns,\
1)feature and 2)group.\nif there are no groups, it will be an empty dataframe')
print(dc_df.head())

#19
print('grouping features')
dc = prp.group_features(dat=df_cleaned,
                        group_dat=dc_df,
                        report_dir=report_dir)

#20
print('dropping correlated features')
print('Shape of data before collinearity care: ', df_cleaned.shape)
df_cleaned = prp.cor_remove(df_cleaned, dc)
print('Shape of data after collinearity care: ', df_cleaned.shape)

#21
df = df.merge(metaData[mt], left_index=True, right_index=True)
df_cleaned = df_cleaned.merge(metaData[mt], left_index=True, right_index=True)

# Modelling
# In this step, we try to fit multiple models to the training dataset and rank them based on their performance. By default, we select the top 3 three models for further analysis.
# During this step, deepBreaks creates a CSV file containing all the fitted models with their performance metrics. These metrics are based on an average of 10-fold cross-validation.

#22
models_to_select = 5 # number of top models to select
trained_models = ml.model_compare(X_train=df_cleaned.loc[:, df_cleaned.columns != mt],
                                  y_train=df_cleaned.loc[:, mt],
                                  sort_by='F1',n_positions=positions,
                                  grouped_features=dc, report_dir=report_dir,
                                  ana_type=anaType, select_top=models_to_select)

#24
model_names = list(trained_models.keys())
print("Top model: ", model_names[0])
first_model_imp = viz._importance_to_df(trained_models[model_names[0]]['importance'])
first_model_imp.head()

#25
print('Available information for each model:')
print(trained_models[model_names[0]].keys())

#Interpretation
# In this step, we use the training data set, positions, and the top models to report the most discriminative positions in the sequences associated with the phenotype.
# we report the feature importances for all top models separately and make a box plot (regression) or stacked bar plot (classification) for the top 4 positions.

for key in trained_models.keys():
    if key == 'mean':
        # plot the mean importance
        viz.dp_plot(importance=trained_models[key], imp_col='mean', model_name=key, annotate=2, report_dir=report_dir)
    else:
        # importance plot (barplot)
        viz.dp_plot(importance= trained_models[key]['importance'], imp_col='standard_value',
                model_name=key, annotate=10,report_dir=report_dir)
        # top 4 position from each model
        viz.plot_imp_model(importance=trained_models[key]['importance'],
                           X_train=df.loc[:, df.columns != mt],
                           y_train=df.loc[:, mt],
                           model_name=key, meta_var=mt, model_type=anaType,
                           report_dir=report_dir)

plt.show()

for key in trained_models.keys():
    if key == 'mean':
        continue
    else:
        imp = trained_models[key]['importance']['standard_value']
        top_indices = []
        for i, x in enumerate(sorted(imp, reverse=True)[:10]):
            top_indices.append(imp.index(x))
        output = [i + 1 for i in top_indices]
        print(f"Selected features for {key}: {output}")

# visualizing top positions
plots = viz.plot_imp_all(trained_models=trained_models,
                         X_train=df.loc[:, df.columns != mt],
                         y_train=df.loc[:, mt],
                         meta_var=mt,
                         model_type=anaType,
                         report_dir=report_dir, max_plots=100,
                         figsize=(2, 3))

import pandas as pd

models = pd.read_csv(report_dir + '/model_performance.csv', index_col=0)
models

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=350)
fig = models.iloc[:6, ].plot(x="Model", y=["Accuracy", "AUC", "F1"],
                             kind="barh",
                             color=['#648FFF', '#DC267F', '#FFB000'],
                             ax=ax,
                             ylim=(0, 1))
ax.legend(bbox_to_anchor=(0.95, -.1), fontsize=6, ncol=3)
ax.set_title('Model performances', fontsize=10)
ax.set_xlabel('')
plt.xticks(fontsize=6)
plt.ylabel('')
plt.xlim(0, 1)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(str(report_dir + '/model_performances.pdf'),
            bbox_inches='tight')
