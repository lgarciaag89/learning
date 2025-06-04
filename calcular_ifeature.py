import iFeatureOmegaCLI as iFeature


def get_names():
    return ["AAC",
            "EAAC",
            "CKSAAP type 1",
            "CKSAAP type 2",
            "DPC type 1",
            "DPC type 2",
            "TPC type 1",
            "TPC type 2",
            "CTDC",
            "CTDT",
            "CTDD",
            "CTriad",
            "KSCTriad",
            "ASDC",
            "DistancePair",
            "GAAC",
            "EGAAC",
            "CKSAAGP type 1",
            "CKSAAGP type 2",
            "GDPC type 1",
            "GDPC type 2",
            "GTPC type 1",
            "GTPC type 2",
            "Moran",
            "Geary",
            "NMBroto",
            "AC",
            "CC",
            "ACC",
            "SOCNumber",
            "QSOrder",
            "PAAC",
            "APAAC",
            "PseKRAAC type 1",
            "PseKRAAC type 2",
            "PseKRAAC type 3A",
            "PseKRAAC type 3B",
            "PseKRAAC type 4",
            "PseKRAAC type 5",
            "PseKRAAC type 6A",
            "PseKRAAC type 6B",
            "PseKRAAC type 6C",
            "PseKRAAC type 7",
            "PseKRAAC type 8",
            "PseKRAAC type 9",
            "PseKRAAC type 10",
            "PseKRAAC type 11",
            "PseKRAAC type 12",
            "PseKRAAC type 13",
            "PseKRAAC type 14",
            "PseKRAAC type 15",
            "PseKRAAC type 16",
            "binary",
            "binary_6bit",
            "binary_5bit type 1",
            "binary_5bit type 2",
            "binary_3bit type 1",
            "binary_3bit type 2",
            "binary_3bit type 3",
            "binary_3bit type 4",
            "binary_3bit type 5",
            "binary_3bit type 6",
            "binary_3bit type 7",
            "AESNN3",
            "OPF_10bit",
            "OPF_7bit type 1",
            "OPF_7bit type 2",
            "OPF_7bit type 3",
            "AAIndex",
            "BLOSUM62",
            "ZScale",
            "KNN"]


path_peptides = "D:\\OneDrive - CICESE\\Documentos\\00-WORK\\Docencia\\workspace\\learning\\data\\TR_starPep_AB_training.fasta"

features = iFeature.iProtein(path_peptides)

keys = get_names()

for key in keys:
    print(f"Calculating {key} features...")
    features.get_descriptor(key)
    features.to_csv(f"{path_peptides}_{key}.csv", "index=False", header=True)
