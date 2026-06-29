import os.path as osp
import argparse
import torch

from src.config_packs import epic_config
from src.dataset.hand_dataset import EpicDataset
from src.utils.save_results import save_phase_results, postprocess_results, exist_results
from src.utils.parse_config import update_config_from_args
from src.model.phase1_contact import optimize_phase1_contact
from src.model.phase2_image import optimize_phase2_image
from src.model.phase3_hand import optimize_phase3_hand

def run_sample(folder_name, output_path, sample, args, **kwargs):
    # save the initial pose to check the object pose initialization
    save_phase_results(folder_name, output_path, sample, phase=0)

    if not cfg.skip_phase_1:
        p1_object_params = optimize_phase1_contact(**kwargs)
        sample["object_params"].vertices = p1_object_params["vertices"]
        torch.cuda.empty_cache()
        # save results
        save_phase_results(folder_name, output_path, sample, object_phase_params=p1_object_params, phase=1)

    if not cfg.skip_phase_2:
        p2_object_params = optimize_phase2_image(**kwargs)
        sample["object_params"].vertices = p2_object_params['vertices']
        torch.cuda.empty_cache()
        # save results
        save_phase_results(folder_name, output_path, sample, object_phase_params=p2_object_params, phase=2)           

    if not cfg.skip_phase_3:
        p3_hand_params = optimize_phase3_hand(**kwargs)
        sample["hand_params"].vertices = p3_hand_params['vertices']
        torch.cuda.empty_cache()
        save_phase_results(folder_name, output_path, sample, hand_phase_params=p3_hand_params, phase=3)            

    postprocess_results(output_path)

def main(dataset, args, cfg = None):
    if cfg is None:
        cfg = epic_config

    cfg, args = update_config_from_args(cfg, args)

    for i, datas in enumerate(dataset):
        single_data = False
        if not isinstance(datas, list):
            single_data = True
            datas = [datas]
        for init_i, data in enumerate(datas):
            # if init_i != 0:
            #     continue
            sample, folder_name = data
            kwargs = {**sample, **vars(cfg)}
            if single_data:
                output_path = osp.join(args.output_dir, folder_name)
            else:
                output_path = osp.join(args.output_dir, folder_name, f"init{init_i}")

            # check whether to skip this task
            if not args.rewrite and exist_results(output_path, cfg):
                print(f"--> Skipping sample '{folder_name}' as it has already been processed.")
                torch.cuda.empty_cache()
                continue
            
            print(f"Run object pose initialization {init_i}")
            if args.debug:
                run_sample(folder_name, output_path, sample, args, **kwargs)    
            else:
                try:
                    run_sample(folder_name, output_path, sample, args, **kwargs)
                except Exception as e:
                    print(f"Sample {folder_name} induces error: {e}. Skip for now.")
                    continue
        print("Sample running finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PICO-fit-for-hand, input parameters")
    parser.add_argument("--data_dir", "-i", type=str, help="dataset directory")
    parser.add_argument("--output_dir", "-o", type=str, help="output directory")
    parser.add_argument("--rewrite", "-r", action="store_true", help="rewrite the outputs even if they exist")
    parser.add_argument("--start", type=int, default=0, help="start index of the dataset")
    parser.add_argument("--end", type=int, default=10**9, help="end index of the dataset")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    unknown_opts = [opt for opt in (args.opts or []) if opt.startswith("-")]
    if unknown_opts:
        parser.error(f"unrecognized arguments: {' '.join(unknown_opts)}")

    # Setup configurations and dataset
    cfg = epic_config
    hand_dataset = EpicDataset(
        data_dir=args.data_dir,
        start_idx=args.start,
        end_idx=args.end,
        cfg=cfg,
    )
    print(f"From {args.start} - {args.end}: {len(hand_dataset)} valid samples loaded.")
    
    main(hand_dataset, args, cfg=cfg)
