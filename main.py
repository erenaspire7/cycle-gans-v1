from config import (
    DEVICE,
    LEARNING_RATE,
    LOAD_MODEL,
    CHECKPOINT_DISCRIMINATOR_H,
    CHECKPOINT_DISCRIMINATOR_Z,
    BATCH_SIZE,
    NUM_EPOCHS,
    NUM_WORKERS,
    SAVE_MODEL,
    CHECKPOINT_GENERATOR_H,
    CHECKPOINT_GENERATOR_Z,
    TRAIN_DIR,
    transforms,
)

from classes.discriminator.Discriminator import Discriminator
from classes.generator.Generator import Generator
from classes.dataset.GANDataset import GANDataset

from helper.checkpoint_helper import load_checkpoint, save_checkpoint
from helper.train_helper import train_fn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main():
    disc_H = Discriminator(in_channels=3).to(DEVICE)
    disc_Z = Discriminator(in_channels=3).to(DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(DEVICE)

    # use Adam Optimizer for both generator and discriminator
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GENERATOR_H,
            gen_H,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GENERATOR_Z,
            gen_Z,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISCRIMINATOR_H,
            disc_H,
            opt_disc,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_DISCRIMINATOR_Z,
            disc_Z,
            opt_disc,
            LEARNING_RATE,
        )

    dataset = GANDataset(
        root_human=TRAIN_DIR + "/humans",
        root_statue=TRAIN_DIR + "/statues",
        transform=transforms,
    )
    # val_dataset = HorseZebraDataset(
    #     root_horse=VAL_DIR + "/horses",
    #     root_zebra=VAL_DIR + "/zebras",
    #     transform=transforms,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     pin_memory=True,
    # )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=CHECKPOINT_GENERATOR_H)
            save_checkpoint(gen_Z, opt_gen, filename=CHECKPOINT_GENERATOR_Z)
            save_checkpoint(disc_H, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_H)
            save_checkpoint(disc_Z, opt_disc, filename=CHECKPOINT_DISCRIMINATOR_Z)


main()
