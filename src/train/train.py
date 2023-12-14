import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from itertools import repeat
from src.wandb_logger.wandb import WanDBWriter
from torch import Tensor
from src.model.dcgan import DCGAN
import torch.nn as nn
import PIL

# legal reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def save_checkpoint(model, g_opt, d_opt, epoch, scheduler=None):
    state = {
        "arch": type(model).__name__,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "g_optimizer": g_opt.state_dict(),
        "d_optimizer": d_opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else ""
    }

    filename = "checkpoint.pth"
    torch.save(state, filename)


def get_grad_norm(model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()
        

def train(num_epochs, dataloader, model: DCGAN, g_opt, d_opt, device, log_step=50, start_step=0):
    iters = 0
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, model.generator.nz, 1, 1, device=device)
    writer = WanDBWriter()

    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader), 0):
            model.discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = model.discriminate(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, model.generator.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = model.generate(noise)
            label.fill_(0)
            # Classify all fake batch with D
            output = model.discriminate(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            d_opt.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            model.generator.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = model.discriminate(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            g_opt.step()

            # Output training stats
            if i % log_step == 0:
                save_checkpoint(model, g_opt, d_opt, epoch)
                writer.set_step(iters + start_step)
                writer.add_scalar("gen_loss", errG.item())
                writer.add_scalar("disc_loss", errD.item())
                writer.add_scalar("disc_loss", errD.item())
                writer.add_scalar("epoch", epoch)
                writer.add_scalar("grad_norm_g", get_grad_norm(model.generator))
                writer.add_scalar("grad_norm_d", get_grad_norm(model.discriminator))

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                writer.set_step(iters + start_step)
                with torch.no_grad():
                    fake = model.generate(fixed_noise[:5, :, :, :]).detach().cpu().numpy()

                images = [PIL.Image.fromarray(image.reshape(image.shape[1], image.shape[2], image.shape[0])) for image in fake]
                writer.add_images("example_images", images)
            iters += 1