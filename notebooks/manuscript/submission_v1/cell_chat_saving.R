library(CellChat)
library(patchwork)
options(stringsAsFactors = FALSE)

obj = readRDS("/data/estorrs/mushroom/data/projects/submission_v1/HT413C1-Th1k4A1/cellchat/results.rds")

pdf(file = "/data/estorrs/sandbox/fn1.pdf")
par(mfrow=c(1,1), xpd = TRUE) # `xpd = TRUE` should be added to show the title
netVisual_aggregate(obj, signaling = c("FN1"), layout = "circle")
dev.off()
