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
    parser.add_argument('--input_MSA', default='RNA_TESTSET/MSA_pydca/RF01998.faclean', type=str)
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

    test_pdb_data_pickle_file = 'RNA_TESTSET_PDB_DATA_.pickle'
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
                test_label[i, j] = lbl
                test_label[j, i] = lbl
        test_label = torch.from_numpy(test_label).long().unsqueeze(0)

        test_label = test_label.cpu().squeeze(0) - mask
        test_label[test_label <= -1] = 100
        test_label[test_label < 16] = 1  ##### lbl is 1 (contact) if distance is smaller than 10A, which corresponds to label 0,1,2,...,15
        test_label[test_label >= 16] = 0
        ppv = (pred * test_label).sum() / int(2 * 1 * L)  ##### position-wise multiplication to find "positive prediction", divided by 2L (total number of predictions)
        print(rna_fam_name, ppv.item())
        test_label[test_label == -100] = 0

        # test_label = torch.from_numpy(test_label).long().unsqueeze(0)
        print(f"labels pre-processed: {L} x {L} matrix with 37/2 classes!")
        pred = pred.numpy().astype(int)
        test_label = test_label.numpy().astype(int)


        # #
        # plt.subplot(1, 2, 1)
        # plt.imshow(pred)
        # plt.colorbar()
        # plt.title("Prediction = 120")
        # #
        # plt.subplot(1, 2, 2)
        # plt.imshow(test_label)
        # plt.colorbar()
        # plt.title("GT")
        # plt.show()
        #
        # plt.imshow(element_compare * 1)
        # plt.colorbar()
        # plt.title("Prediction va GT")
        # plt.show()
        #
        # plt.imshow(element_compare * 1)
        # plt.colorbar()
        # plt.title("Prediction va GT")
        # plt.show()

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(test_label.flatten(), pred.flatten())/2
        cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['0', '1'])
        cmd_obj.plot()
        cmd_obj.ax_.set(
            title=f'Confusion Matrix for {rna_fam_name}, Total Length: {L}',
            xlabel='Predicted Contact',
            ylabel='Actual Contact')
        plt.show()

        result = np.zeros((L, L))

        i, j = np.where((pred == 1) & (pred != test_label))
        pred_wrong = list(zip(i.tolist(),j.tolist()))
        pred_wrong_list = []
        for (x, y) in pred_wrong:
            if (y, x) in pred_wrong:
                pred_wrong_list.append((x, y))
                pred_wrong.remove((y, x))

        k, l = np.where((pred == 1) & (pred == test_label))
        pred_correct = list(zip(k.tolist(), l.tolist()))
        pred_correct_list = []
        for (x, y) in pred_correct:
            if (y, x) in pred_correct:
                pred_correct_list.append((x, y))
                pred_correct.remove((y, x))

        n, m = np.where(pred == 1)
        prediction = list(zip(n.tolist(), m.tolist()))
        pred_list = []
        for (x, y) in prediction:
            if (y, x) in prediction:
                pred_list.append((x, y))
                prediction.remove((y, x))

        for index in pred_list:
            result[index] = 1

        # plt.imshow(result * 1)
        # plt.colorbar()
        # plt.title("Prediction Result")
        # plt.show()

        number_of_cuts1 = []
        cut_result = {}
        for contact in pred_list:
            c_i, c_j = contact
            number_of_cut = 0
            for predicted in pred_list:
                p_i, p_j = predicted
                if (c_i <= p_i and c_j < p_j and c_j > p_i) or (p_i < c_i and p_j <= c_j and p_j > c_i) :
                    result[c_i, c_j] = 1
                    number_of_cut += 1
                else:
                    continue
            cut_result[contact] = number_of_cut
            number_of_cuts1.append(number_of_cut)

        # plt.plot(np.linspace(0, L, num=L), number_of_cuts1, color='maroon')
        # plt.ylim((0, 30))
        # plt.title(f"Number of Cuts for {rna_fam_name}:{sum(number_of_cuts1)}")
        # plt.show()

        number_of_cuts2 = []
        cut_result = {}
        for contact in pred_wrong_list:
            c_i, c_j = contact
            number_of_cut = 0
            for predicted in pred_list:
                p_i, p_j = predicted
                if (c_i <= p_i and c_j < p_j and c_j > p_i) or (p_i < c_i and p_j <= c_j and p_j > c_i) :
                    result[c_i, c_j] = 1
                    number_of_cut += 1
                else:
                    continue
            cut_result[contact] = number_of_cut
            number_of_cuts2.append(number_of_cut)

        # plt.plot(np.linspace(0, len(pred_wrong_list), num=len(pred_wrong_list)), number_of_cuts2, color='maroon')
        # plt.ylim((0, 30))
        # plt.title(f"Number of Cuts for {rna_fam_name} for Wrong Prediction: {sum(number_of_cuts2)}")
        # plt.show()


        number_of_cuts3 = []
        cut_result = {}
        for contact in pred_correct_list:
            c_i, c_j = contact
            number_of_cut = 0
            for predicted in pred_list:
                p_i, p_j = predicted
                if (c_i <= p_i and c_j < p_j and c_j > p_i) or (p_i < c_i and p_j <= c_j and p_j > c_i) :
                    result[c_i, c_j] = 1
                    number_of_cut += 1
                else:
                    continue
            cut_result[contact] = number_of_cut
            number_of_cuts3.append(number_of_cut)


        # plt.plot(np.linspace(0, len(pred_correct_list), num=len(pred_correct_list)), number_of_cuts3, color='maroon')
        # plt.ylim((0, 30))
        # plt.title(f"Number of Cuts for {rna_fam_name} for Correct Prediction: {sum(number_of_cuts3)}")
        # plt.show()

        plt.figure(figsize=(10, 6))
        plt.boxplot([number_of_cuts1, number_of_cuts2, number_of_cuts3], vert=False, showmeans=True)
        plt.title(f"Total number of Cuts for {rna_fam_name}: {sum(number_of_cuts1)}, Wrong Prediction: {sum(number_of_cuts2)}, Correct Prediction: {sum(number_of_cuts3)}")
        plt.yticks([1, 2, 3], ['All', 'Wrong', 'Correct'])
        plt.show()

        plt.style.use('seaborn-deep')
        plt.figure(figsize=(10, 6))
        number_of_bins = int(max(number_of_cuts1 + number_of_cuts2 + number_of_cuts3)/2)
        kwargs = dict(alpha=0.5, bins=number_of_bins)
        hist_muti = [number_of_cuts1, number_of_cuts2, number_of_cuts3]
        plt.hist(hist_muti, bins=number_of_bins, histtype='bar', label=['all', 'wrong', 'collect'])
        # plt.hist(number_of_cuts1, **kwargs, color='r', label='all', histtype='bar')
        # plt.hist(number_of_cuts3, **kwargs, color='g', label='correct', histtype='bar')
        # plt.hist(number_of_cuts2, **kwargs, color='b', label='wrong', histtype='bar')
        plt.gca().set(title=f'Frequency Histogram of Number of Cuts \n Total number of Cuts for {rna_fam_name}: {sum(number_of_cuts1)}, Wrong Prediction: {sum(number_of_cuts2)}, Correct Prediction: {sum(number_of_cuts3)}', ylabel='Frequency', xlabel='Number of cuts')
        plt.xlim(0, number_of_bins*2+1)
        plt.legend()
        plt.tight_layout()
        plt.show()



        #
        # number_of_cuts = []
        # cut_result = {}
        # for contact in pred_list:
        #     c_i, c_j = contact
        #     number_of_cut = 0
        #     for predicted in pred_wrong_list:
        #         p_i, p_j = predicted
        #         if (c_i < p_i and c_j < p_j and p_j > c_i) or (p_i < c_i and p_j < c_j and p_j > c_i) :
        #             result[c_i, c_j] = 1
        #             number_of_cut += 1
        #         else:
        #             continue
        #     cut_result[contact] = number_of_cut
        #     number_of_cuts.append(number_of_cut)
        #
        # plt.plot(np.linspace(0, L, num=L), number_of_cuts, color='maroon')
        # plt.ylim((0, 30))
        # plt.title(f"Number of Cuts for {rna_fam_name} by Wrong Prediction: {sum(number_of_cuts)}")
        # plt.show()
        #
        #
        #
        # number_of_cuts = []
        # cut_result = {}
        # for contact in pred_list:
        #     c_i, c_j = contact
        #     number_of_cut = 0
        #     for predicted in pred_correct_list:
        #         p_i, p_j = predicted
        #         if (c_i < p_i and c_j < p_j and p_j > c_i) or (p_i < c_i and p_j < c_j and p_j > c_i) :
        #             result[c_i, c_j] = 1
        #             number_of_cut += 1
        #         else:
        #             continue
        #     cut_result[contact] = number_of_cut
        #     number_of_cuts.append(number_of_cut)
        #
        # plt.plot(np.linspace(0, L, num=L), number_of_cuts, color='maroon')
        # plt.ylim((0, 30))
        # plt.title(f"Number of Cuts for {rna_fam_name} by Correct Prediction: {sum(number_of_cuts)}")
        # plt.show()

        # plt.imshow(result * 1)
        # plt.colorbar()
        # plt.title(f"blue: wrong prediction - {result[result==1].sum()}, yellow: right GT crossed GT - {result[result==2].sum()}")
        # plt.show()





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
