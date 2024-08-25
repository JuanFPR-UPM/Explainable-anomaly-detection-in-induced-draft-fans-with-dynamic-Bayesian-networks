library(data.table)
library(dbnR)
library(reticulate)
use_condaenv("tfm")
source_python("./dataframe_folder.py")
#> Loading required package: bnlearn
#>
#> Attaching package: 'dbnR'
size <- 6
load(file = paste0("./nets/CV1/size", size, "_dmmhc_mmpc_hc_mi-g_bic-g_5e-06.RDS"))
# load(file=paste0("./nets/CV2/size", size, "_dmmhc_mmpc_hc_mi-g_bic-g_5e-04.RDS"))
n_cycles <- 685 # 1400 for IDF2
dt_train <- get_train_df(n_cycles)
f_dt_train <- fold_dt(dt_train, size)
dbn <- fit_dbn_params(net, f_dt_train, method = "mle-g")
# save(dbn, file=paste0("./nets/CV1/size", size, "_dmmhc_mmpc_hc_mi-g_bic-g_5e-06_fit.RDS"))
# save(dbn, file=paste0("./nets/CV2/size", size, "_dmmhc_mmpc_hc_mi-g_bic-g_5e-04_fit.RDS"))
