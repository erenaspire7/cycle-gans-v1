import torch
from tqdm import tqdm
from torchvision.utils import save_image
from config import DEVICE, LAMBDA_CYCLE, LAMBDA_IDENTITY
from helper.image_helper import save_image_v2
import numpy as np


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (human, statue) in enumerate(loop):
        human = human.to(DEVICE)
        statue = statue.to(DEVICE)

        # Train discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_statue = gen_H(human)
            D_H_real = disc_H(statue)
            D_H_fake = disc_H(fake_statue.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_human = gen_Z(statue)
            D_Z_real = disc_Z(human)
            D_Z_fake = disc_Z(fake_human.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial losses
            D_H_fake = disc_H(fake_statue)
            D_Z_fake = disc_Z(fake_human)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle losses
            cycle_human = gen_Z(fake_statue)
            cycle_statue = gen_H(fake_human)
            cycle_human_loss = l1(human, cycle_human)
            cycle_statue_loss = l1(statue, cycle_statue)

            # identity losses
            identity_human = gen_Z(human)
            identity_statue = gen_H(statue)
            identity_human_loss = l1(human, identity_human)
            identity_statue_loss = l1(statue, identity_statue)

            # total loss
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_human_loss * LAMBDA_CYCLE
                + cycle_statue_loss * LAMBDA_CYCLE
                + identity_human_loss * LAMBDA_IDENTITY
                + identity_statue_loss * LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image_v2(
                [
                    statue.cpu().detach().numpy().astype(np.float32),
                    fake_statue.cpu().detach().numpy().astype(np.float32),
                ],
                "Statue to Human",
                f"outputs/statue_{idx}.png",
            )
            save_image_v2(
                [
                    human.cpu().detach().numpy().astype(np.float32),
                    fake_human.cpu().detach().numpy().astype(np.float32),
                ],
                "Human to Statue",
                f"outputs/human_{idx}.png",
            )

            print(f"Total Loss: {G_loss}")

            # save_image(fake_statue * 0.5 + 0.5, f"outputs/statue_{idx}.png")
            # save_image(fake_human * 0.5 + 0.5, f"outputs/human_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))
