from src.constants import OUT_TEMPLATE

def parse_param_string(cfg, template):
    lws = cfg.loss_weights
    add_str = ""
    if cfg.phase_3_upd_trans:
        add_str += "_upd-h-trans"
    if cfg.phase_3_upd_rot:
        add_str += "_upd-h-rot"
    if cfg.phase_3_dyn_w:
        add_str += "_dyn-w"
    parsed_str = template.format(
        lws['lw_contact_p2'], lws['lw_contact_p3'], 
        lws['lw_silhouette'], lws['lw_collision_p2'], lws['lw_scale'],
        lws['lw_silhouette_hand'], lws['lw_collision_p3'], lws['lw_pose_reg'], add_str, 
    )
    return parsed_str

def update_config_from_args(config, args):
    # Rewrite default config file with command-line arg values
    if args.opts:
        for opt in args.opts:
            try:
                # Expect format: "path.key=value"
                if '=' in opt:
                    key_path, value_str = opt.split('=', 1)
                    if hasattr(config, key_path):
                        # Attempt to determine the correct type for the value
                        try:
                            new_value = float(value_str)
                        except ValueError:
                            try:
                                new_value = int(value_str)
                            except ValueError:
                                new_value = value_str
                        setattr(config, key_path, new_value)
                        print(f"Updated {key_path} to: {new_value}")
                    elif key_path in config.loss_weights.keys():
                        key_path, value_str = opt.split('=', 1)
                        new_value = float(value_str)
                        config.loss_weights[key_path] = new_value
                        print(f"Updated {key_path} to: {new_value}")
                    else:
                        print(f"Warning: Config attribute '{key_path}' not found. Skipping.")
                else:
                    print(f"Warning: Remainder argument '{opt}' skipped (no '=' found).")
                    
            except Exception as e:
                print(f"Error processing remainder option '{opt}': {e}")

    # Update template-based output directory
    args.output_dir = args.output_dir.replace("[template]", parse_param_string(config, template=OUT_TEMPLATE))

    return config, args