import pickle

with open('/scratch/gautschi/le317/PartialSpoof/03multireso/multi-reso/01/inferforfustion_new_0_to_90k/dev_score_ali_64_20251208.pkl', 'rb') as f:
    data1 = pickle.load(f)

with open('/scratch/gautschi/le317/PartialSpoof/03multireso/multi-reso/01/inferforfustion_new_90k_to_end/dev_score_ali_64_20251208.pkl', 'rb') as f:
    data2 = pickle.load(f)

data = data1 + data2

# create new pkl file with the name dev_score_ali_16_20251208_merged.pkl
with open('/scratch/gautschi/le317/PartialSpoof/03multireso/multi-reso/01/inferforfustion_new_0_to_90k_and_90k_to_end/dev_score_ali_64_20251208_merged.pkl', 'wb') as f:
    pickle.dump(data, f)