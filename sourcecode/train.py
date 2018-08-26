""" script for training a Self Attention GAN on celeba images """

import argparse

import numpy as np
import torch as th
from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enable fast training
cudnn.benchmark = True

# set seed = 3
th.manual_seed(seed=3)


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--discriminator_file", action="store", type=str,
                        default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--images_dir", action="store", type=str,
                        default="../data/celeba",
                        help="path for the images directory")

    parser.add_argument("--sample_dir", action="store", type=str,
                        default="samples/1/",
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                        default="models/1/",
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="relativistic-hinge",
                        help="loss function to be used: 'hinge', 'relativistic-hinge'")

    parser.add_argument("--depth", action="store", type=int,
                        default=5,
                        help="Depth of the GAN")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=256,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=32,
                        help="batch_size for training")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=12,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=1041,
                        help="number of logs to generate per epoch")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=64,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=1,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.0001,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.0004,
                        help="learning rate for discriminator")

    parser.add_argument("--use_spectral_norm", action="store", type=bool,
                        default=True,
                        help="Whether to use spectral normalization or not")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    args = parser.parse_args()

    return args


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from Teacher.TeacherGAN import TeacherGAN
    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader
    from Teacher.Losses import HingeGAN, RelativisticAverageHingeGAN

    # create a data source:
    celeba_dataset = FlatDirectoryImageDataset(
        args.images_dir,
        transform=get_transform((int(np.power(2, args.depth + 1)),
                                 int(np.power(2, args.depth + 1)))))

    data = get_data_loader(celeba_dataset, args.batch_size, args.num_workers)

    # create a gan from these
    tgan = TeacherGAN(depth=args.depth, latent_size=args.latent_size, device=device)

    if args.generator_file is not None:
        # load the weights into generator
        tgan.gen.load_state_dict(th.load(args.generator_file))

    print("Generator Configuration: ")
    print(tgan.gen)

    if args.discriminator_file is not None:
        # load the weights into discriminator
        tgan.dis.load_state_dict(th.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(tgan.dis)

    # create optimizer for generator:
    gen_optim = th.optim.Adam(tgan.gen.parameters(), args.g_lr, [0, 0.99])

    dis_optim = th.optim.Adam(tgan.dis.parameters(), args.d_lr, [0, 0.99])

    loss_name = args.loss_function.lower()

    if loss_name == "hinge":
        loss = HingeGAN
    elif loss_name == "relativistic-hinge":
        loss = RelativisticAverageHingeGAN
    else:
        raise Exception("Unknown loss function requested")

    # train the GAN
    tgan.train(
        data,
        gen_optim,
        dis_optim,
        loss_fn=loss(device, tgan.dis),
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=args.num_samples,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.model_dir,
        start=args.start
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
