#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("poster", font_scale=2.5, rc={"lines.linewidth": 3.5})
sns.set(style="whitegrid",)
#%matplotlib inline

#%%
def draw(figname, df, x, use_legend=False, hue='name'):

    NAMES = ['MTL', 'NoTears-L1', 'Unconstrained-L1', 'MetaNotears-L1', 'MetaUnconstrained-L1']

    def draw_one(df, metric='shd', legend=False):
        new_df = df[df.name != 'ICP']
        sns.lineplot(x=x, y=metric, hue=hue, hue_order= NAMES if hue else None ,data=new_df, legend=legend,ci=75, err_style='band', n_boot=1000,)
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #color=sns.color_palette()[3],

    if hue == 'name':

        df = df[df['name'].isin(NAMES)]

    f = plt.figure(figsize=(22, 6))
    gs = f.add_gridspec(1, 3)

    for i, metric in enumerate(['fdr', 'shd', 'nnz']):
        ax = f.add_subplot(gs[0, i])
        draw_one(df, metric, legend = 'full' if (use_legend and i==2) else False)
    plt.tight_layout()
    f.savefig(figname+'.pdf')

#%%
df = pd.read_pickle('res/exp_nouter_n500_d30_ER_96357.pkl')
draw('nouter', df, 'nouter', False, None)

df = pd.read_pickle('res/exp_lr_n50_d30_ER_15549.pkl')
draw('lr', df, 'lr', False, None)

df = pd.read_pickle('res/exp_lambda1_n50_d30_ER_52714.pkl')
draw('lambda1', df, 'lambda1', False, None)

df = pd.read_pickle('res/exp_ntasks_d30_ER_63872.pkl')
draw('ntasks', df, 'ntasks', True, 'name')

df = pd.read_pickle('res/exp_n_d30_ER_49858.pkl')
draw('n', df, 'n', True, 'name')

#%%
df = pd.read_pickle('res/exp_dim_n50_ntasks20_ER_52718.pkl')
df = pd.concat([df, pd.read_pickle('res/exp_dim_n50_ntasks20_ER_62269.pkl')], ignore_index=True)
draw('dim_n50_ER', df, 'd', True, 'name')

df = pd.read_pickle('res/exp_dim_n50_ntasks20_SF_32069.pkl')
df = pd.concat([df, pd.read_pickle('res/exp_dim_n50_ntasks20_SF_50923.pkl')], ignore_index=True)
draw('dim_n50_SF', df, 'd', True, 'name')

df = pd.read_pickle('res/exp_dim_n500_ntasks20_ER_86448.pkl')
df = pd.concat([df, pd.read_pickle('res/exp_dim_n500_ntasks20_ER_82814.pkl')], ignore_index=True)
draw('dim_n500_ER', df, 'd', True, 'name')

df = pd.read_pickle('res/exp_dim_n500_ntasks20_SF_54973.pkl')
df = pd.concat([df, pd.read_pickle('res/exp_dim_n500_ntasks20_SF_70572.pkl')], ignore_index=True)
draw('dim_n500_SF', df, 'd', True, 'name')


