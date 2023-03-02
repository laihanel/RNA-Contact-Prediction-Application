from model.model import CoT_RNA_Transfer
import argparse
from create_dataset import *
from misc import *
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)
def msa_to_embed(msa_path, max_seqs=200, AminoAcids='HETL'):
    tmp_path = msa_path.replace('.faclean', '.fa')
    lines = []
    for line in open(msa_path):
        line = line.strip()
        if not line.startswith(">"):
            new_line = ''
            for l in line:
                if l == 'A':
                    new_line += AminoAcids[0]
                elif l == 'U':
                    new_line += AminoAcids[1]
                elif l == 'C':
                    new_line += AminoAcids[2]
                elif l == 'G':
                    new_line += AminoAcids[3]
                else:
                    new_line += '-'
            lines.append(new_line)
        else:
            lines.append(line)

    if max_seqs is not None:
        lines = lines[:2*max_seqs]     ### 2x for name and sequence

    with open(tmp_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

    if lines[0].startswith(">"):
        L = len(lines[1].strip())

    program = [
        os.path.join(os.path.dirname(__file__), "bin/a3m_to_feat"),
        "--input",
        tmp_path,
        "--max_gap",
        "7",
        "--max_keep",
        "5000",
        "--sample_ratio",
        "1.0",
    ]
    process = subprocess.run(program, capture_output=True)
    assert process.returncode == 0, "Invalid A3M file"
    x = np.copy(np.frombuffer(process.stdout, dtype=np.int8))
    x = x.reshape((-1, L, 7 * 2 + 3)).transpose((0, 2, 1))
    assert (x < 23).all(), "Internal error"
    seq = x[0][0]

    os.remove(tmp_path)

    # return {
    #     "seq": torch.tensor(seq).long()[None].cuda(),
    #     "msa": torch.tensor(x).long()[None].cuda(),
    #     "index": torch.arange(seq.shape[0]).long()[None].cuda(),
    # }

    return {
        "seq": torch.tensor(seq).long()[None],
        "msa": torch.tensor(x).long()[None],
        "index": torch.arange(seq.shape[0]).long()[None],
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_MSA', default='RNA_TESTSET/MSA_pydca/RF00001.faclean', type=str)
    parser.add_argument('--model', default='pretrained_models/model.chk', type=str)
    args = parser.parse_args()

    ### print args
    hparams_dict = dict()
    for arg in vars(args):
        hparams_dict[arg] = getattr(args, arg)
        print(arg, getattr(args, arg))

    ### model definition
    model = CoT_RNA_Transfer()

    ### load params
    weight_path = os.path.join(os.path.dirname(__file__), args.model)
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)

    ### move model to GPU
    # model = model.cuda()



    ### traslate MSA from nucletide to amino acids
    adapted_msa = msa_to_embed(args.input_MSA)

    ### evaluate model
    model.eval()
    with torch.no_grad():
        pred, feat = model(adapted_msa)
    pred = pred.cpu()

    L = pred.shape[0]
    mask = torch.full((L, L), -10000)
    for i in range(L):
        for j in range(L):
            if abs(i - j) > 4:
                mask[i, j] = 0
            else:
                pass

    pred = pred.cpu() + mask
    delta = torch.randn(L, L) * 1e-7
    pred = pred + delta + delta.T

    ### save raw output
    dist = pred
    np.savetxt('outputs/dist.txt', dist.numpy())

    ### save top-L prediction
    topk_values, _ = pred.reshape(-1).topk(k=int(2 * 1 * L))
    topk_value = topk_values[-1]
    pred[pred < topk_value] = -10000
    pred[pred >= topk_value] = 1
    pred[pred <= 0] = 0

    np.savetxt('outputs/pred.txt', pred.numpy().astype(int), fmt='%d', delimiter=",")

    ### get ground truth for input msa

    test_pdb_data_pickle_file = 'RNA_TESTSET_PDB_DATA.pickle'
    if os.path.exists(test_pdb_data_pickle_file):
        with open(test_pdb_data_pickle_file, 'rb') as handle:
            test_pdb_data = pickle.load(handle)
    rna_fam_name = args.input_MSA.split('/')[-1].split('.')[0]
    if rna_fam_name in test_pdb_data:
        test_label = np.ones((L, L)) * -100    ##### ignore index is -100
        for k, v in test_pdb_data[rna_fam_name].items():
            i, j = k[0], k[1]
            if abs(i-j) > 4:
                lbl = distance_to_37(v[-1])
                if lbl <= -1:
                    lbl2 = 100
                elif lbl <16:
                    lbl2 = 1  ##### lbl is 1 (contact) if distance is smaller than 10A, which corresponds to label 0,1,2,...,15
                elif lbl >= 16:
                    lbl2 = 0 ##### lbl is 0 (non-contact) if distance is larger than 10A, which corresponds to label 16,17,18,....
                test_label[i, j] = lbl2
                test_label[j, i] = lbl2

        test_label[test_label == -100] = 0

        # test_label = torch.from_numpy(test_label).long().unsqueeze(0)
        print(f"labels pre-processed: {L} x {L} matrix with 37/2 classes!")
        pred = pred.numpy().astype(int)

        element_compare = pred== test_label

        plt.imshow(element_compare * 1)
        plt.colorbar()
        plt.show()






if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
