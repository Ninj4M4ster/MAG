import os
import os.path as path
import argparse
from ..utils import \
    get_config, update_config, save_config, \
    get_last_checkpoint, add_post, Logger
from data.deep_lesion import get_dataset
from ..models.adn import ADNTrain
from torch.utils.data import DataLoader
# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.cuda.amp import GradScaler


if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train an artifact disentanglement network")
    parser.add_argument("run_name", help="name of the run")
    parser.add_argument("--default_config", default="config/adn.yaml", help="default configs")
    parser.add_argument("--run_config", default="runs/adn.yaml", help="run configs")
    args = parser.parse_args()

    # Get ADN options
    opts = get_config(args.default_config)
    run_opts = get_config(args.run_config)
    if args.run_name in run_opts and "train" in run_opts[args.run_name]:
        run_opts = run_opts[args.run_name]["train"]
        update_config(opts, run_opts)
    run_dir = path.join(opts["checkpoints_dir"], args.run_name)
    if not path.isdir(run_dir): os.makedirs(run_dir)
    save_config(opts, path.join(run_dir, "train_options.yaml"))

    # Get dataset
    def get_image(data):
        dataset_type = dataset_opts['dataset_type']
        if dataset_type == "deep_lesion":
            if dataset_opts[dataset_type]['load_mask']: return data['lq_image'], data['hq_image'], data['mask']
            else: return data['lq_image'], data['hq_image']
        elif dataset_type == "spineweb":
            return data['a'], data['b']
        elif dataset_type == "nature_image":
            return data["artifact"], data["no_artifact"]
        else:
            raise ValueError("Invalid dataset type!")

    dataset_opts = opts['dataset']
    train_dataset = get_dataset(**dataset_opts)
    train_loader = DataLoader(train_dataset,
        batch_size=opts["batch_size"], num_workers=opts['num_workers'], shuffle=True)
    train_loader = add_post(train_loader, get_image)

    # Get checkpoint
    if opts['last_epoch'] == 'last':
        checkpoint, start_epoch = get_last_checkpoint(run_dir)
    else:
        start_epoch = opts['last_epoch']
        checkpoint = path.join(run_dir, "net_{}".format(start_epoch))
        if type(start_epoch) is not int: start_epoch = 0

    # Get model
    model = ADNTrain(opts['learn'], opts['loss'], **opts['model'])
    if opts['use_gpu']: model.cuda()
    if path.isfile(checkpoint): model.resume(checkpoint)
    
    # Scaler for faster training
    scaler = GradScaler()

    # Get logger
    logger = Logger(run_dir, start_epoch, args.run_name)
    logger.add_loss_log(model.get_loss, opts["print_step"], opts['window_size'])
    logger.add_iter_visual_log(model.get_visuals, opts['visualize_step'], "train_visuals")
    logger.add_save_log(model.save, opts['save_step'])
    
    log_file_path = path.join(run_dir, "training_log.txt")
    with open(log_file_path, "a") as f:
        f.write(f"--- Rozpoczęcie treningu: Run {args.run_name} ---\n")
        
    global_step = 0

    # Train the model
    # for epoch in range(start_epoch, opts['num_epochs']):
    #     for data in logger(train_loader):
    #         # model.optimize(*data)
    #         model.optimize(*data, scaler=scaler)
    #     model.update_lr()
    for epoch in range(start_epoch, opts['num_epochs']):
        for i, data in enumerate(logger(train_loader)):
            
            model.optimize(*data, scaler=scaler)
            
            if i % opts["print_step"] == 0:
                current_losses = model.get_loss()
                
                loss_str = " ".join([f"{k}: {v:.6f}" for k, v in current_losses.items()])
                log_entry = f"Epoch: {epoch} | Step: {i} | {loss_str}\n"
                
                with open(log_file_path, "a") as f:
                    f.write(log_entry)

            global_step += 1

        model.update_lr()
