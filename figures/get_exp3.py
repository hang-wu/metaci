#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("poster")
sns.set(style="whitegrid")
#%matplotlib inline

#%%
def draw_one(df, x, metric='shd', legend=False):
    new_df = df[df.name != 'ICP']
    sns.lineplot(x=x, y=metric, hue='name', data=new_df, legend=legend)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def draw():
    f = plt.figure(figsize=(22, 6))
    gs = f.add_gridspec(1, 3)

    for i, metric in enumerate(['fdr', 'shd', 'nnz']):
        ax = f.add_subplot(gs[0, i])
        draw_one(df, metric, legend=False if i!=2 else 'full')

#%%
df = pd.read_pickle('res/exp_ntasks_d30_ER_63872.pkl')
print(df.columns)
draw()
