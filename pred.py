# run this after prep.py

from util import *
from graph import *
from ae import *
from sage_ve import *
from lp import *

cur_fid = "GF"
losses = []

if __name__ == "__main__":
    E_list = []
    W_list = []
    end_cond = False

    for i in range(5):
        while (not end_cond) and (not bool(len(E_list))):
            cur_name = "input/GF/fingerprint{}.txt".format(i)
            observation_sign = "u"
            observation_ids = []
            pruned_observation_ids = []
            mac_all = []
            observation_ids, mac_all = load_file_comparison(cur_name, observation_ids, observation_sign, mac_all, file_type="db", prune=False)
            pruned_observation_ids, mac_all = load_file_comparison(cur_name, pruned_observation_ids, observation_sign, mac_all, file_type="db", prune=True)
            
            x_all = np.zeros((len(pruned_observation_ids), len(mac_all)))
            x_all = np.zeros((len(pruned_observation_ids), len(mac_all)))
            for i, item in enumerate(observation_ids):
                for j, mac in enumerate(mac_all):
                    if mac in item[3].keys():
                        x_all[i][j] = item[3][mac]/120
                    else:
                        x_all[i][j] = -120/120
            
            model = AE(len(mac_all))
            loss_function = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=0.01)
            
            for epoch in range(epochs):
                for record in x_all:
                    cur_record = record.reshape((1, 1, 1, len(record)))
                    reconstructed = model(torch.Tensor(cur_record))
                    loss = loss_function(reconstructed[1], torch.Tensor(cur_record))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss)
                    # print("Epoch: {}, Loss: {}".format(epoch, loss))
                outputs.append((epochs, torch.Tensor(cur_record), reconstructed[0], reconstructed[1]))
            with open("output/{}/output/cur_out.txt".format(cur_fid), 'w') as f:
                f.write(outputs)

            end_cond = True # can be modified for more training, shall be an int here, just make it bool for convenience

        cur_g = SAGE_VE(get_data("output/{}/raw_data/{}.edgelist".format(cur_fid)))[1]
        cur_g.run(runs=10, epochs=50, lr=0.01, dropout=0.5, out_fname="output/{}/output/out{}.txt".format(cur_fid, i))

        E_list, W_list = get_new_link("output/{}/raw_data/{}.edgelist".format(cur_fid), "output/{}/raw_data/{}.edgelist".format(cur_fid))