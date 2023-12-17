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


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()    
    for i in arr:
        temp = (((i - arr.min()) * diff) / diff_arr) + t_min
        norm_arr.append(temp.unsqueeze(0))
    return norm_arr


def eval(model, dataloader, device, fixed_noise, fid_metric, ssim_metric):
    model.generator.eval()
    model.discriminator.eval()
    last_idx = 0
    real_imgs = []
    constructed_imgs = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            real = data[0].to(device)
            b_size = real.size(0)
            samples = model.generator(fixed_noise[last_idx:last_idx + b_size, ...])

            real_imgs.append(real.detach())
            constructed_imgs.append(samples.detach())
            last_idx += b_size

    print(len(real_imgs), real_imgs[0].shape)
    print(len(constructed_imgs), constructed_imgs[0].shape)

    real_imgs = torch.cat(real_imgs)
    real_imgs = normalize(real_imgs, 0, 1)
    real_imgs = torch.cat(real_imgs)

    constructed_imgs = torch.cat(constructed_imgs)
    print(constructed_imgs.shape)
    constructed_imgs = normalize(constructed_imgs, 0, 1)
    constructed_imgs = torch.cat(constructed_imgs)

    print(real_imgs.shape, constructed_imgs.shape)

    fid = fid_metric.compute_metric(real_imgs.flatten(1), constructed_imgs.flatten(1)).cpu().numpy(),
    ssim = ssim_metric(real_imgs, constructed_imgs).item()
    return fid, ssim


def train(num_epochs, dataloader, model: DCGAN, g_opt, d_opt, device, log_step=50, start_step=0):
    iters = 0
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, model.generator.nz, 1, 1, device=device)
    writer = WanDBWriter()

    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader), 0):
            model.discriminator.zero_grad()
            # Format batch
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = model.discriminate(real).view(-1)
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
                
                # images = []
                # for image in fake:
                #     image = (normalize(image.reshape(image.shape[1], image.shape[2], image.shape[0]), 0, 1) * 255).astype('uint8')
                #     images.append(PIL.Image.fromarray(image, 'RGB'))

                # writer.add_images("example_images", images)
            iters += 1