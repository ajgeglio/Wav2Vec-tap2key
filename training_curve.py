import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re 
import os

# Define the path to the latest model
# latest_model = '/work/ajgeglio/pretrained_models/checkpoint-3922_Feb-23-2023-17:54_tap2key' 
checkpoint_nums = []
latest_model = '/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-13-2023-13:38_tap2key'
for root, _, _ in os.walk(latest_model):
    cnum = re.split('\D+', root)[-1]
    checkpoint_nums.append(cnum)
latest_checkpoint = np.array(checkpoint_nums[1:], dtype=int).max()

model_checkpoints = [   
                        "/work/ajgeglio/pretrained_models/wav2vec2-base_Feb-16-2023-13:30_tap2key/checkpoint-3922",
                        "/work/ajgeglio/pretrained_models/checkpoint-3922_Feb-16-2023-17:03_tap2key/checkpoint-18500",
                        
                    ]
# This one with manual re-sampled data
model_checkpoints2 = [
                        "/work/ajgeglio/pretrained_models/wav2vec2-base_Feb-16-2023-13:30_tap2key/checkpoint-3922",
                        f"/work/ajgeglio/pretrained_models/checkpoint-3922_Feb-23-2023-17:54_tap2key"
]

# This one without resampled but with new data from group contributers and overall 3-way split of entire database
model_checkpoints = [
                        f"/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-06-2023-12:34_tap2key/checkpoint-17800"
]

# This one with resampled and with new data from group contributers and overall 3-way split of entire database
model_checkpoints = ['/work/ajgeglio/pretrained_models/wav2vec2-base_Mar-07-2023-11:42_tap2key/checkpoint-25350']

model_checkpoints2 = [
                        f"{latest_model}/checkpoint-{latest_checkpoint}"
]

print(model_checkpoints2)

def create_perf_df(checkpoint_list):
    with open(f"{checkpoint_list[0]}/trainer_state.json") as f:
        df = pd.DataFrame(json.load(f)['log_history'])
    for i in range(1,len(checkpoint_list)):
        with open(f"{checkpoint_list[i]}/trainer_state.json") as f:
            df = pd.concat([df, pd.DataFrame(json.load(f)['log_history'])])
    # df['iter'] = np.arange(1,df.shape[0]+1,1)
    val_hist = df[['epoch', 'step', 'eval_f1', 'eval_loss']].dropna() 
    train_hist = df[['epoch', 'step', 'loss']].dropna() 
    return val_hist, train_hist, df

val_hist1, train_hist1, _ = create_perf_df(model_checkpoints)
val_hist2, train_hist2, df = create_perf_df(model_checkpoints2)
max_f1 = val_hist2['eval_f1'].max()
max_epoch = val_hist2[val_hist2['eval_f1']==max_f1]['epoch'].values
print("Epoch", max_epoch, "f1", max_f1)

# Create a plot of the training loss over time
lw = 1
ax = train_hist1.plot('epoch', 'loss',  label="Training Loss (new)", linewidth=lw, legend=False)
ax2 = val_hist1.plot('epoch', 'eval_loss',  label="Validation Loss (new)", linewidth=lw, ax=ax, legend=False)
ax3 = val_hist1.plot("epoch", 'eval_f1', label="Validation F1 (new)", secondary_y=True,color='k', linewidth=lw, ax=ax, legend=False)

ax4 = train_hist2.plot('epoch', 'loss',  label="Training Loss (ovrsmple)", linewidth=lw, legend=False, ax=ax)
ax5 = val_hist2.plot('epoch', 'eval_loss',  label="Validation Loss (ovrsmple)", linewidth=lw, ax=ax, legend=False)
ax6 = val_hist2.plot("epoch", 'eval_f1', label="Validation F1 (ovrsmple)", secondary_y=True,color='dimgray', linewidth=lw, ax=ax, legend=False)

ax.set_xlabel("epoch")
# ax.set_xlim(0,1600)
ax.set_ylabel("Loss")
ax3.set_ylabel("Score")
ax.set_title(f"Learning curve \nmax validation F1 score: {max_f1:0.2f} at epoch {max_epoch}")
handles2, labels2 = ax2.get_legend_handles_labels()
handles3, labels3 = ax3.get_legend_handles_labels()
handles = [item for sublist in [handles2,handles3] for item in sublist]
labels = [item for sublist in [labels2,labels3] for item in sublist]

plt.legend(handles, labels, bbox_to_anchor=(0.4, 0.8))
plt.savefig('training_curve7.png')
