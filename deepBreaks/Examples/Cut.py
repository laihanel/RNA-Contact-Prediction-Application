from deepBreaks import preprocessing as prp
from deepBreaks import visualization as viz
from deepBreaks import models as ml
import os
import datetime
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

#3........................................................................
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

#4.........................................................................................
# making a unique directory for saving the reports of the analysis
print('direcory preparation')
dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
seqFile = seqFileName.split('.')[0]
report_dir = str(seqFile +'_' + mt + '_' + dt_label)
os.makedirs(report_dir)

#5...........................................................................................
print('reading meta-data')
# importing metadata
metaData = prp.read_data(metaDataFileName, seq_type = None, is_main=False)
print('metaData:', metaData.shape)

# importing sequences data
print('reading fasta file')
df = prp.read_data(seqFileName, seq_type = seq_type, is_main=True)

positions = df.shape[1]
print('Done')
print('Shape of data is: ', df.shape)

#7.................................................................................
# selecting only the classes with enough number of samples
df = prp.balanced_classes(dat=df, meta_dat=metaData, feature=mt)

#8...................................................................................
print(df.shape)

#9...................................................................................
print('metadata looks like this:')
print(metaData.head())

#10.................................................................................
print('sequence data looks like this:')
df.head()

#Preprocessing
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


#11......................................................................................
# taking care of missing data
print('Shape of data before missing/constant care: ', df.shape)
df = prp.missing_constant_care(df)
print('Shape of data after missing/constant care: ', df.shape)

#12.........................................................................................
# taking care of ultra-rare cases
print('Shape of data before imbalanced care: ', df.shape)
df = prp.imb_care(dat=df, imbalance_threshold=0.025)
print('Shape of data after imbalanced care: ', df.shape)

#13
# you may want to perform your analysis only on a random sample of the positions.
# Here you can have a random sample of your main data set.
print('number of columns of main data befor: ', df.shape[1])
df = prp.col_sampler(dat=df, sample_frac=sampleFrac)
print('number of columns of main data after: ', df.shape[1])

#14
# Use statistical tests to drop redundant features.
print('number of columns of main data befor: ', df.shape[1])
df_cleaned = prp.redundant_drop(dat=df, meta_dat=metaData,
                        feature=mt, model_type=anaType,
                        threshold=0.25,
                        report_dir=report_dir)
print('number of columns of main data after: ', df_cleaned.shape[1])

#15
print('one-hot encoding the dataset')
df_cleaned = prp.get_dummies(dat=df_cleaned, drop_first=True)

#16
print('calculating the distance matrix')
cr = prp.distance_calc(dat=df_cleaned,
                       dist_method='correlation',
                       report_dir=report_dir)
print(cr.shape)

#17
print('The distance matrix looks like this.\n The values are between 0 (exact the same) and 1 (non-related).')
print(cr.head())

#18
print('finding colinear groups')
dc_df = prp.db_grouped(dat = cr,
                       report_dir=report_dir,
                       threshold=.25,
                       # needs_pivot=False
                       )

#19
print('The result of the last step is a dataframe with two columns,\
1)feature and 2)group.\nif there are no groups, it will be an empty dataframe')
print(dc_df.head())

#20
print('grouping features')
dc = prp.group_features(dat=df_cleaned,
                        group_dat=dc_df,
                        report_dir=report_dir)

#21
print('dropping correlated features')
print('Shape of data before collinearity care: ', df_cleaned.shape)
df_cleaned = prp.cor_remove(df_cleaned, dc)
print('Shape of data after collinearity care: ', df_cleaned.shape)

#22
#merge with meta data
df = df.merge(metaData[mt], left_index=True, right_index=True)
df_cleaned = df_cleaned.merge(metaData[mt], left_index=True, right_index=True)


# Modelling
# In this step, we try to fit multiple models to the training dataset and rank them based on their performance. By default, we select the top 3 three models for further analysis.
# During this step, deepBreaks creates a CSV file containing all the fitted models with their performance metrics. These metrics are based on an average of 10-fold cross-validation.

#23
models_to_select = 3 # number of top models to select
top_models, train_cols, model_names = ml.fit_models(dat = df_cleaned,
                                                    meta_var=mt,
                                                    model_type=anaType,
                                                    models_to_select=models_to_select,
                                                    report_dir=report_dir)


# Interpretation
# In this step, we use the training data set, positions, and the top models to report the most discriminative positions in the sequences associated with the phenotype.
# we report the feature importances for all top models separately and make a box plot (regression) or stacked bar plot (classification) for the top 4 positions.
#

#24
for i in range(models_to_select):
    # calculate the featre importances
    imp = ml.fimp_single(trained_model=top_models[i], model_name=model_names[i],
                         train_cols=train_cols, grouped_features=dc,
                         n_positions=positions, report_dir=report_dir)
    # visualize the lollipop plot for features based on each model
    viz.dp_plot(dat=imp, model_name=model_names[i], imp_col='standard_value', report_dir=report_dir)

    # visualize the boxplots for features based on each model
    viz.plot_imp_model(dat=df, trained_model=top_models[i],
                       model_name=model_names[i],
                       train_cols=train_cols, grouped_features=dc,
                       meta_var=mt, n_positions=positions,
                       model_type=anaType, report_dir=report_dir)

#25
# merging the results for all the top models
mean_imp = ml.fimp_top_models(trained_models=top_models, model_names=model_names,
                              train_cols=train_cols,grouped_features=dc,
                              n_positions=positions,report_dir=report_dir)

# lollipop plot for the merged results
viz.dp_plot(dat=mean_imp,
            model_name= 'mean',
            imp_col='mean_imp',
            report_dir=report_dir)

#
# visualizing top positions

# viz.plot_imp_all(trained_models=top_models, dat=df, train_cols=train_cols,
#                  grouped_features=dc, meta_var=mt, model_type=anaType,
#                  n_positions=positions, report_dir=report_dir)











#...........................................................................................................
# deepBreaks reveals important SARS-CoV-2 regions associated with Alpha and Delta variants

seqFileName = '/home/ubuntu/deepbreaks/data/sarscov2_2.fasta'
meta_data = '/home/ubuntu/deepbreaks/data/meta_data_clean.csv'
# name of the phenotype
mt = 'variant_short'

# type of the sequences
seq_type = 'nu'
# type of the analysis if it is a classification model, then we put cl instead of reg
anaType = 'cl'
sampleFrac=1
# making a unique directory for saving the reports of the analysis
print('direcory preparation')
dt_label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
seqFile = seqFileName.split('.')[0]
report_dir = str(seqFile +'_' + mt + '_' + dt_label)
os.makedirs(report_dir)

#30
# importing sequences data
print('reading fasta file')
df = prp.read_data(seqFileName, seq_type = seq_type, is_main=True)
metaData = prp.read_data(meta_data, is_main=False)
positions = df.shape[1]
print('Done')
print('Shape of data is: ', df.shape)

#31
df.index = [ind[:-7] for ind in df.index]

#32
print('metadata looks like this:')
metaData.head()

#33
# selecting only the classes with enough number of samples
df = prp.balanced_classes(dat=df, meta_dat=metaData, feature=mt)

#
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


#34
# taking care of missing data
print('Shape of data before missing/constant care: ', df.shape)
df = prp.missing_constant_care(df, missing_threshold=0.25)
print('Shape of data after missing/constant care: ', df.shape)

#35
# taking care of ultra-rare cases
print('Shape of data before imbalanced care: ', df.shape)
df = prp.imb_care(dat=df)
print('Shape of data after imbalanced care: ', df.shape)

#36
print(df.head())

#37
# Use statistical tests to drop redundant features.
print('number of columns of main data befor: ', df.shape[1])
df_cleaned = prp.redundant_drop(dat=df, meta_dat=metaData,
                        feature=mt, model_type=anaType,
                        threshold=0.25,
                        report_dir=report_dir)
print('number of columns of main data after: ', df_cleaned.shape[1])

#38
print('one-hot encoding the dataset')
df_cleaned = prp.get_dummies(dat=df_cleaned, drop_first=True)

#39
print('calculating the distance matrix')
cr = prp.distance_calc(dat=df_cleaned,
                       dist_method='correlation',
                       report_dir=report_dir)
print(cr.shape)

#40
print('The distance matrix looks like this.\n The values are between 0 (exact the same) and 1 (non-related).')
print(cr.head())

#41
print('finding colinear groups')
dc_df = prp.db_grouped(dat = cr,
                       report_dir=report_dir,
                       threshold=.3)

#42
print('The result of the last step is a dataframe with two columns,\
1)feature and 2)group.\nif there are no groups, it will be an empty dataframe')
print(dc_df.head())

#43
# Then, we pass the above calculated groupes into the group_feature function. This function finds the distance of all the group members to the center of the group (median). The result will be a dictionary of columns like this:
#
# {
# group1_representativ:[member1, member2,...],
# group2_representativ:[member1, member2,...],...
# }
print('grouping features')
dc = prp.group_features(dat=df_cleaned,
                        group_dat=dc_df,
                        report_dir=report_dir)

#44
print('dropping correlated features')
print('Shape of data before collinearity care: ', df_cleaned.shape)
df_cleaned = prp.cor_remove(df_cleaned, dc)
print('Shape of data after collinearity care: ', df_cleaned.shape)

#45
#merge with meta data
df = df.merge(metaData[mt], left_index=True, right_index=True)
df_cleaned = df_cleaned.merge(metaData[mt], left_index=True, right_index=True)

# Modelling
# In this step, we try to fit multiple models to the training dataset and rank them based on their performance. By default, we select the top 3 three models for further analysis.
# During this step, deepBreaks creates a CSV file containing all the fitted models with their performance metrics. These metrics are based on an average of 10-fold cross-validation.
#46

models_to_select = 5 # number of top models to select
trained_models = ml.model_compare(X_train=df_cleaned.loc[:, df_cleaned.columns != mt],
                                  y_train=df_cleaned.loc[:, mt],
                                  sort_by='F1',n_positions=positions,
                                  grouped_features=dc, report_dir=report_dir,
                                  ana_type=anaType, select_top=models_to_select)

#47
# to access the importances
model_names = list(trained_models.keys())
print("Top model: ", model_names[0])
first_model_imp = viz._importance_to_df(trained_models[model_names[0]]['importance'])
first_model_imp.head()

#48
print('Available information for each model:')
print(trained_models[model_names[0]].keys())


# Interpretation
# In this step, we use the training data set, positions, and the top models to report the most discriminative positions in the sequences associated with the phenotype.
# we report the feature importances for all top models separately and make a box plot (regression) or stacked bar plot (classification) for the top 4 positions.
#49
for key in trained_models.keys():
    if key == 'mean':
        # plot the mean importance
        viz.dp_plot(importance=trained_models[key], imp_col='mean', model_name=key, annotate=2, report_dir=report_dir)
    else:
        # importance plot (barplot)
        viz.dp_plot(importance= trained_models[key]['importance'], imp_col='standard_value',
                model_name=key, annotate=2,report_dir=report_dir)
        # top 4 position from each model
        viz.plot_imp_model(importance=trained_models[key]['importance'],
                           X_train=df.loc[:, df.columns != mt],
                           y_train=df.loc[:, mt],
                           model_name=key, meta_var=mt, model_type=anaType,
                           report_dir=report_dir)

#50
# visualizing top positions
import matplotlib.pyplot as plt

plots = viz.plot_imp_all(trained_models=trained_models,
                         X_train=df.loc[:, df.columns != mt],
                         y_train=df.loc[:, mt],
                         meta_var=mt,
                         model_type=anaType,
                         report_dir=report_dir,max_plots=100,
                        figsize=(2,4))

plt.show()

import pandas as pd
models = pd.read_csv(report_dir+ '/model_performance.csv', index_col=0)

print(models)
