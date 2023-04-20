import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv("/home/ubuntu/CoT-RNA-Transfer/outputs/prob_hiv.txt", sep=",")
# # df = pd.read_csv("/home/ubuntu/ProteinFolding/coevolution_transformer/example/T0990-D1.contact", sep=" ")
# # important_aa = [50, 127, 124, 158, 165, 292, 198]  # eyes
# important_aa = [18, 37, 38, 39, 51, 52, 53, 60, 87, 102]  # HIV
# marked_pt = []
# rows = []
# cols = []
# for row in important_aa:
#     for col in important_aa:
#         contact = df.iloc[row, col]
#         if contact == 1 and row != col:
#             marked_pt.append((row, col))
#             print(f"({row}, {col})")
#
#
# plt.imshow(df)
# plt.colorbar()
# # plt.title("HIV [38, 39, 51]")
# for row, col in marked_pt:
#     plt.plot(row, col, marker="o", markersize=5, color="red")
# plt.show()
# #
# path = "/home/ubuntu/deepbreaks/data/HIV/hiv_V3_B_C_nu_clean.fasta"
# path2 = "/home/ubuntu/deepbreaks/data/HIV/hiv.fasta"
path = "/home/ubuntu/deepbreaks/data/Sars_Cov_2/sarscov2_2.fasta"
path2 = "/home/ubuntu/deepbreaks/data/Sars_Cov_2/sars_top500.fasta"
with open(path2, 'w') as f:
    with open(path) as fp:
        lins = fp.readlines()
        k = 0
        for i in range(len(lins)):
            line = lins[i]
            if i == len(lins)-1:
                f.write(lines[0:500])
                # print(lines)
            if line.startswith(">"):
                k += 1
                if i != 0:
                    f.write(lines[0:500])
                    f.write('\n')
                    # print(lines)
                f.write(line)
                # print(line.strip())
                lines = ""
            else:
                lines += line.strip()
        print(k)



#
# df = pd.read_csv("/home/ubuntu/pygcn/data/hmp/s__Haemophilus_parainfluenzae.tsv", sep="\t", header=None)
# print(df.head())
#

