library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(EnsDb.Hsapiens.v75)
library(ggplot2)
library(patchwork)
library(Matrix)
setwd("./SE129785_scATAC-Hematopoiesis-All")
#load peaks, cells and matrix files.
peaks <- read.table("GSE129785_scATAC-Hematopoiesis-All.peaks.txt.gz", header = TRUE)
cells <- read.table("GSE129785_scATAC-Hematopoiesis-All.cell_barcodes.txt.gz", header = TRUE, stringsAsFactors = FALSE)
rownames(cells) <- make.unique(cells$Barcodes)
mtx <- readMM(file = "GSE129785_scATAC-Hematopoiesis-All.mtx.gz")
mtx <- as(object = mtx, Class = "dgCMatrix")
colnames(mtx) <- rownames(cells)
rownames(mtx) <- peaks$Feature

#load fragment files
files1 <- dir(pattern="*fragments.tsv.gz$")
files2 <- dir(pattern="*fragments.sorted.tsv.gz$")
files <- c(files1,files2)
tme <- grep("_SU0", files) # filter
pbmc <- grep("pbmc",files) # filter
retain_file <- files[-c(tme,pbmc,c(1:4,6:15,26:33,42:43))] # only retain new samples
retain_file_group <- c("PBMC_Rep1","Monocytes","B_Cells","CD34_Progenitors_Rep1",
                       "Regulatory_T_Cells","Naive_CD4_T_Cells_Rep1","Memory_CD4_T_Cells_Rep1","CD4_HelperT",
                       "NK_Cells","Naive_CD8_T_Cells","Memory_CD8_T_Cells",
                       "Bone_Marrow_Rep1","CD34_Progenitors_Rep2","Memory_CD4_T_Cells_Rep2",
                       "Naive_CD4_T_Cells_Rep2","PBMC_Rep3","PBMC_Rep2","PBMC_Rep4","Dendritic_cells")
retain_group <- unique(cells$Group)[1:19] # only retain new samples
retain_file <- retain_file[order(retain_file_group)] # sort file list and group id list to make them in the same order
retain_group <- retain_group[order(retain_group)]

#read all fragment files in a list
l <- list()
for (i in 1:length(retain_file)) {
  l[[i]] <- CreateFragmentObject(
    path = retain_file[i],
    validate.fragments = T,
    cells=cells[cells$Group_Barcode==retain_group[i],"Barcodes"] #define cells for a given fragment file from the cell annotation file
  )
}

#build Chromatin Assay
bone_assay <- CreateChromatinAssay(
  counts = mtx,
  min.cells = 5,
  fragments = l,
  sep = c("_", "_"),
  genome = "hg19"
)

#build Seurat object
bone <- CreateSeuratObject(
  counts = bone_assay,
  meta.data = cells,
  assay = "ATAC"
)

bone <- bone[, bone$Group_Barcode %in%retain_group] #filter

#Define cell types
cluster_names <- c("HSC",   "MEP",  "CMP-BMP",  "LMPP", "CLP",  "Pro-B",    "Pre-B",    "GMP",
                   "MDP",    "pDC",  "cDC",  "Monocyte-1",   "Monocyte-2",   "Naive-B",  "Memory-B",
                   "Plasma-cell",    "Basophil", "Immature-NK",  "Mature-NK1",   "Mature-NK2",   "Naive-CD4-T1",
                   "Naive-CD4-T2",   "Naive-Treg",   "Memory-CD4-T", "Treg", "Naive-CD8-T1", "Naive-CD8-T2",
                   "Naive-CD8-T3",   "Central-memory-CD8-T", "Effector-memory-CD8-T",    "Gamma delta T")

num.labels <- length(cluster_names)
names(cluster_names) <- paste0( rep("Cluster", num.labels), seq(num.labels) )
bone$celltype <- cluster_names[as.character(bone$Clusters)]

#load genome annotation
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v75)
seqlevelsStyle(annotations) <- 'UCSC'
Annotation(bone) <- annotations

#Quality control
bone <- TSSEnrichment(bone)
bone <- NucleosomeSignal(bone)
bone$blacklist_fraction <- FractionCountsInRegion(bone, regions = blacklist_hg19)
#save(bone, file = "GSE129785_scATAC-Hematopoiesis-All_raw_v2.Rdata")

bone <- bone[, (bone$nCount_ATAC < 50000) &
               (bone$TSS.enrichment > 2) & 
               (bone$nucleosome_signal < 5)]

#map all fragment files to the gene regions
gene.activities <- GeneActivity(bone)
gene.activities <- gene.activities[,bone$Barcodes]
library(rhdf5)
h5file = "GSE129785_scATAC-Hematopoiesis-all_PeaksToGenes_filtered.h5"
h5createFile(h5file)
h5write(as.matrix(gene.activities), h5file,"X")
h5write(rownames(gene.activities), h5file,"Genes")
h5write(colnames(gene.activities), h5file,"Barcodes")
h5write(as.character(bone$Group), h5file,"Group")
h5write(as.character(bone$Internal_Name), h5file,"Internal_Name")
h5write(as.character(bone$celltype), h5file,"Celltypes")
