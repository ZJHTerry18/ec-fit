import os
import os.path as osp
import json
import sys
from tqdm import tqdm
import shutil

result_dir = '/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_fullvideos/2026-02-19_pico_stage3_rand100videos_wilorspace_maskv3_newsf_multimixed_con-p2-8.0_con-p3-80.0_silo0.03-occ_peno0.01_sc1.0_silh0.03_penh1.0_reg0.5_upd-h-trans'
separate_save = '/home/u5gi/jiahezhao25.u5gi/jiahe/data/epic-grasps/pico_v3/pico_stage3_fullvideos/2026-02-19_pico_stage3_rand100videos_wilorspace_maskv3_newsf_multimixed_con-p2-8.0_con-p3-80.0_silo0.03-occ_peno0.01_sc1.0_silh0.03_penh1.0_reg0.5_upd-h-trans_selbest'
rewrite = True

os.makedirs(separate_save, exist_ok=True)

def incremental_copy(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)

        if os.path.isdir(s):
            # Recursively call for subdirectories
            incremental_copy(s, d)
        else:
            # Check if destination file exists AND if source is newer
            # OR if the destination file simply doesn't exist
            if not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d):
                shutil.copy2(s, d)

sample_folders = [x for x in sorted(os.listdir(result_dir)) if osp.isdir(osp.join(result_dir, x))]
for sf in tqdm(sample_folders):
    try:
        sample_dir = osp.join(result_dir, sf)
        init_fols = [x for x in sorted(os.listdir(sample_dir)) if osp.isdir(osp.join(sample_dir, x))]

        losses_list = []
        for inif in init_fols:
            loss_file = osp.join(sample_dir, inif, "pred_phase2_noeval.json")
            with open(loss_file, 'r') as f:
                loss_dict = json.load(f)["loss"]
            total_loss = sum(list(loss_dict.values()))
            losses_list.append(total_loss)
        # choose the one with the smallest total loss
        min_loss_ind = min(enumerate(losses_list), key=lambda x: x[1])[0]
        best_init_fol = init_fols[min_loss_ind]

        with open(f"{sample_dir}/{best_init_fol}.txt", "w") as f:
            f.write("")

        if separate_save:
            src_fol = osp.join(sample_dir, best_init_fol)
            tgt_fol = osp.join(separate_save, sf)
            if not rewrite:
                incremental_copy(src_fol, tgt_fol)
            else:
                shutil.copytree(src_fol, tgt_fol)
    except Exception as e:
        print(f"{sf} has error: {e}. Skip.")
        continue
    