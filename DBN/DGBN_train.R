library(data.table)
library(dbnR)
#> Loading required package: bnlearn
#>
#> Attaching package: 'dbnR'
library(reticulate)
use_condaenv("tfm") # In order to run the following script
source_python("./dataframe_folder.py") # For IDF2, change input cycle datasets

n_cycles <- 300

size_list <- c(6) # c(2,3,4,5,6,7,8,9,10)
alpha_list <- c(5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7, 5e-8, 5e-9, 5e-10)
dt_train <- get_train_df(n_cycles)

for (size in size_list) {
        f_dt_train <- fold_dataframe(size, n_cycles)

        for (alpha in alpha_list) {
                net <- learn_dbn_struc(
                        dt_train,
                        size,
                        method = "dmmhc",
                        f_dt = f_dt_train,
                        restrict = "mmpc",
                        maximize = "hc",
                        restrict.args = list(test = "mi-g", alpha = alpha),
                        maximize.args = list(score = "bic-g", maxp = 15)
                )
                save(net, file = paste0("./nets/CV1/size6_dmmhc_mmpc_hc_mi-g_bic-g_", alpha, ".RDS")) # The serialized object will be named 'net'
        }
}
