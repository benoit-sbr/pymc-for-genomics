from io import BytesIO
import numpy as np

def read(n_conditions, n_genes, n_genes_under_TF_control, n_replicats, n_timesteps, n_ignored_timesteps) :
    Binary = open('data.csv').read().replace(',', '.').encode()
    expr = np.genfromtxt(BytesIO(Binary), missing_values = 'NA', filling_values = np.nan, skip_header = 3)
#    print('expr.shape', expr.shape)

# We read the genes numbers in the data.
    genes_numbers = np.asarray(expr[ ::n_timesteps, 0], dtype = int)

    expr_genes_under_TF_control = expr[ :(n_timesteps*n_genes_under_TF_control) , : ]

    selector = [x for x in range(expr_genes_under_TF_control.shape[0]) if x % n_timesteps != n_timesteps-n_ignored_timesteps]
#    print('selector', selector)
    new_expr = expr_genes_under_TF_control[selector, :]

# We read the times in the data.
    times = expr[0:n_timesteps-n_ignored_timesteps, 1]
    
    replicat1 = new_expr[:, 2::n_replicats].flatten('F')
#    print('replicat1.shape', replicat1.shape)
    replicat1TG = replicat1.reshape(n_conditions*n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
#    print('replicat1TG.shape', replicat1TG.shape)
    replicat2 = new_expr[:, 3::n_replicats].flatten('F')
    replicat2TG = replicat2.reshape(n_conditions*n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    replicat3 = new_expr[:, 4::n_replicats].flatten('F')
    replicat3TG = replicat3.reshape(n_conditions*n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    replicat4 = new_expr[:, 5::n_replicats].flatten('F')
    replicat4TG = replicat4.reshape(n_conditions*n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    replicatsRTG = np.stack((replicat1TG, replicat2TG, replicat3TG, replicat4TG), )
    
    alginate1 = new_expr[:, 2]
    alginate1TG = alginate1.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    alginate2 = new_expr[:, 3]
    alginate2TG = alginate2.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    alginate3 = new_expr[:, 4]
    alginate3TG = alginate3.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    alginate4 = new_expr[:, 5]
    alginate4TG = alginate4.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    alginatesRTG = np.stack((alginate1TG, alginate2TG, alginate3TG, alginate4TG), )
    maltose1 = new_expr[:, 2+n_replicats]
    maltose1TG = maltose1.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    maltose2 = new_expr[:, 3+n_replicats]
    maltose2TG = maltose2.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    maltose3 = new_expr[:, 4+n_replicats]
    maltose3TG = maltose3.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    maltose4 = new_expr[:, 5+n_replicats]
    maltose4TG = maltose4.reshape(n_genes_under_TF_control, n_timesteps-n_ignored_timesteps).transpose()
    maltosesRTG = np.stack((maltose1TG, maltose2TG, maltose3TG, maltose4TG), )
    replicatsCRTG = np.stack((alginatesRTG, maltosesRTG), )

    replicatsRCTG = np.transpose(replicatsCRTG, axes=(1, 0, 2, 3))

    return genes_numbers, times, replicatsRTG, replicatsCRTG, replicatsRCTG
