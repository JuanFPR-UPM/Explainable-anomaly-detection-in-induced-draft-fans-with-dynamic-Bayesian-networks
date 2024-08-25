library(data.table)
library(dbnR)
library(bnlearn)
#> Loading required package: bnlearn
#>
#> Attaching package: 'dbnR'
load(file = paste0("./nets/CV1/size6_dmmhc_mmpc_hc_mi-g_bic-g.RDS"))
plot_dynamic_network(net, reverse = TRUE) # 'net' is the serialized object after training
# print(dbn)
# calc_mu(dbn)
# calc_sigma(dbn)
# sigma(dbn)
# nodes(dbn)
# plot(dbn)
# plot_dynamic_network(dbn)
# coef(dbn)
net
