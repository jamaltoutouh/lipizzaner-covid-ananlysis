import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import figaspect
import seaborn as sns

from scipy.stats import shapiro, wilcoxon, f_oneway, iqr

input_csv = '../data/parametrization-results.csv'

dataset = pd.read_csv(input_csv)
parameters = ['network_name', 'batch_size', 'smote_augmentation', 'mutation_probability', 'adam_learning_rate']
parameters_name = ['Network architecture', 'Batch size', 'SMOTE augmentation', 'Mutation probability', 'Learning rate']
metrics = ['fid', 'execution_time']
metrics_name = ['Inception score', 'Execution time']

def normality_test(data, alpha): #Is it normal? If p-value > alpha it is normal
    stat, p = shapiro(data)
    return stat, p, p>alpha

def nonparametric_pairwise_test(data1, data2, alpha):
    val1, val2 = np.array(data1), np.array(data2)
    if np.median(val1) < np.median(val2):
        aux = data2
        data2 = data1
        data1 = aux
    stat, p = wilcoxon(data1, data2, alternative='greater')
    return stat, p, p<alpha

def parametric_pairwise_test(data1, data2, alpha):
    stat, p = f_oneway(data1, data2)
    return stat, p, p<alpha

def get_stats_normal(data_list):
    val = np.array(data_list)
    return val.min(), val.mean(), val.std(), val.max()

def get_stats_nonormal(data_list):
    val = np.array(data_list)
    return val.min(), np.median(val), iqr(val), val.max()

def get_dict_parameter_metric(parameter, metric):
    metric_values = list(dataset[parameter].unique())
    results = dict()
    for metric_value in metric_values:
        results[metric_value] = list(dataset[(dataset[parameter] == metric_value)][metric])
    return results

def get_df_for_boxplot(parameter, metric):
    return dataset[[parameter, metric]]

def create_boxplot(parameter, metric, parameter_name, metric_name):
    try:
        data = get_df_for_boxplot(parameter, metric)
    except:
        print('Error: There is no {} data.'.format(parameter))
        return -1
    w, h = figaspect(3 / 4)
    f, ax = plt.subplots(figsize=(w, h))
    sns.set(style="whitegrid")
    sns.boxplot(y=metric, x=parameter, data=data, showfliers=True).set(
        xlabel=parameter_name,
        ylabel=metric_name
    )
    plt.margins(x=0)
    plt.savefig('../figures/' + parameter + '-' + metric + '.png')
    plt.savefig('../figures/' + parameter + '-' + metric + '.pdf')
    plt.show()

def plot_inception_score_vs_time():
    values = dataset[metrics]
    score = list(values[metrics[0]].tolist())
    time =  list(values[metrics[1]].tolist())

    w, h = figaspect(3 / 4)
    plt.style.use('seaborn-whitegrid')
    f, ax = plt.subplots(figsize=(w, h))
    plt.plot(score, time, 'o', color='black')
    plt.xlabel('Inception score')
    plt.ylabel('Execution time (minutes)')
    plt.savefig('../figures/inceptrion-score_vs_execution-time.png')
    plt.savefig('../figures/inceptrion-score_vs_execution-time.pdf')
    plt.show()


def statistical_analysis(parameter, metric, parameter_name, metric_name):
    results = get_dict_parameter_metric(parameter, metric)

    print('Evaluating {} according to {}'.format(metric_name, parameter_name))

    # Test normality
    normality = True
    alpha = 0.01
    for parameter_value in results.keys():
        _, _, normality = normality_test(results[parameter_value], alpha)
        if not normality:
            print('{} distrinution is not normal.\nWe use non-parametric test.'.format(parameter_value))
            break
    if normality: print('All distributions are normal.\nWe use parametric test.')

    for parameter_value in results.keys():
        if len(results[parameter_value]) > 0:
            minn, med, dispersion, maxx = get_stats_nonormal(results[parameter_value]) if normality else get_stats_nonormal(results[parameter_value])
            print('{} & {:0.2f} & {:0.2f} & {:0.2f} & {:0.2f} \\\ '.format(parameter_value, minn, med, dispersion, maxx))
            print('FALTA POR HACER LA PARTE DE QUE HAGA LOS TEST PARAMETRICOS O NO PARAMETRICOS POR PARES!!!!')
            print('LA FUNCIONES YA TE LAS HE CREADO ARRIBA: parametric_pairwise_test Y nonparametric_pairwise_test')

    #--- FALTAN POR HACER LOS TESTS LOS TEST

for metric, metric_name in zip(metrics, metrics_name):
    for parameter, parameter_name in zip(parameters, parameters_name):
        create_boxplot(parameter, metric, parameter_name, metric_name)

for metric, metric_name in zip(metrics, metrics_name):
    for parameter, parameter_name in zip(parameters, parameters_name):
        statistical_analysis(parameter, metric, parameter_name, metric_name)


plot_inception_score_vs_time()



