import argparse
parser = argparse.ArgumentParser("RNA Contact Prediction by Efficient Protein Transformer Transfer")
parser.add_argument('--input', default="data/HIV/hiv_V3_B_C_nu_clean.fasta", type=str)
parser.add_argument('--output', default='data/HIV/hiv.fasta', type=str)
parser.add_argument('--maxlen', type=int, required=False) # number of nucleotide in each sequence

args = parser.parse_args()

with open(args.output, 'w') as f:
    with open(args.input) as fp:
        lins = fp.readlines()
        k = 0
        if args.maxlen:
            maxlen = args.maxlen
        else:
            maxlen = len(lins)
        for i in range(len(lins)):
            line = lins[i]
            if i == len(lins)-1:
                f.write(lines[0:maxlen])
                # print(lines)
            if line.startswith(">"):
                k += 1
                if i != 0:
                    f.write(lines[0:maxlen])
                    f.write('\n')
                    # print(lines)
                f.write(line)
                # print(line.strip())
                lines = ""
            else:
                lines += line.strip()
        print(k)