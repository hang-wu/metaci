from data import LinearDataset, NonlinearDataset
import solvers
import torch
import pandas as pd
torch.set_default_dtype(torch.double)

train, test = NonlinearDataset(n_tasks=10).split()
print(len(train), len(test))

dfs = []

for solver in [solvers.MetaNotearsNonlinear(meta=True), solvers.MetaNotearsNonlinear(meta=False)]:
    print(solver.name_, solver.meta_)
    ave_shd, all_res = solver.solve(train, test)
    solver_df = pd.DataFrame(all_res)
    solver_df['solver'] = solver.name_ + '-' + str(solver.meta_) # Add hyper-parameters of interest
    print(solver_df)
    # Save the dataframe for plotting purposes
    dfs.append(solver_df)

df = pd.concat(dfs)
print(df)
df.to_pickle('outputs/exp.pkl')
