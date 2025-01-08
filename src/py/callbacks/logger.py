from lightning.pytorch.callbacks import Callback
import torchvision
import torch
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class SegImageLogger(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            img1 = batch["img"]
            img2 = batch["seg"]
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1)
            trainer.logger.experiment.add_image('img', grid_img1[0:max_num_image], pl_module.global_step)
            
            grid_img2 = torchvision.utils.make_grid(img2/torch.max(img2))
            trainer.logger.experiment.add_image('seg', grid_img2[0:max_num_image], pl_module.global_step) 

            with torch.no_grad():
                x_hat = pl_module(img1)
                grid_img3 = torchvision.utils.make_grid(x_hat/torch.max(x_hat))
                trainer.logger.experiment.add_image('x_hat', grid_img3[0:max_num_image], pl_module.global_step)

class SegImageLoggerNeptune(Callback):
    def __init__(self, num_images=12, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            img1 = batch["img"]
            img2 = batch["seg"]
            
            max_num_image = min(img1.shape[0], self.num_images)
            grid_img1 = torchvision.utils.make_grid(img1, nrow=max_num_image)
            
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(grid_img1.permute(1, 2, 0).cpu().numpy())
            trainer.logger.experiment["images/img"].upload(fig)
            plt.close()
            
            grid_img2 = torchvision.utils.make_grid(img2/torch.max(img2))
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(grid_img2.permute(1, 2, 0).cpu().numpy())
            trainer.logger.experiment["images/seg"].upload(fig)
            plt.close()

            with torch.no_grad():
                x_hat = pl_module(img1)
                grid_img3 = torchvision.utils.make_grid(x_hat/torch.max(x_hat))
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_img3.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_hat"].upload(fig)
                plt.close()


class MaskRCNNImageLoggerNeptune(Callback):
    def __init__(self, log_steps=100):
        self.log_steps = 200
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            img1 = batch["img"][0]
            img2 = batch["seg"][0]
                        
            fig = plt.figure(figsize=(7, 9))
            ax = plt.imshow(img2.permute(1, 2, 0).cpu().detach().numpy())
            trainer.logger.experiment["images/seg"].upload(fig)
            plt.close()

            with torch.no_grad():
                x_hat = pl_module(batch, mode='test')
                boxes = x_hat[0]['boxes']
                fig = plt.figure(figsize=(7, 9))
                plt.imshow(img1.permute(1,2,0).cpu().detach().numpy())
                ax = plt.gca()
                for box in boxes:
                    y1, x1, y2, x2 = box.cpu().detach().numpy()
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                    ax.add_patch(rect)

                trainer.logger.experiment["images/boxes"].upload(fig)
                plt.close()

class FasterRCNNImageLoggerNeptune(Callback):
    def __init__(self, log_steps=0,max_num_image=8):
        self.log_steps = log_steps
        self.max_num_image = max_num_image

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            imgs, targets = batch

            imgs = imgs[:self.max_num_image]
            targets = targets[:self.max_num_image]

            n_cols= int(self.max_num_image /2)
            fig, axs = plt.subplots(2, n_cols)
            axs = axs.flatten()
            images = [img for img in imgs]
            boxes_list = [target['boxes'] for target in targets]


            for ax, img, boxes in zip(axs, images, boxes_list):
                ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image

                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().detach().numpy()
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                    ax.add_patch(rect)  # Add the box
                ax.axis('off')  # Optional: Turn off axes for better visualization

            plt.tight_layout()
            trainer.logger.experiment["fig/train/input_boxes"].upload(fig)
            plt.close()

            with torch.no_grad():
                x_hat = pl_module(imgs, targets=None, mode='test')
                n_cols= int(self.max_num_image /2)

                fig, axs = plt.subplots(2, n_cols)
                axs = axs.flatten()
                boxes_list = [target['boxes'] for target in x_hat]

                for ax, img, boxes in zip(axs, images, boxes_list):
                    ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image
                    for box in boxes:
                        x1, y1, x2, y2 = box.cpu().detach().numpy()
                        width, height = x2 - x1, y2 - y1
                        rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                        ax.add_patch(rect)  # Add the box
                    ax.axis('off')  # Optional: Turn off axes for better visualization

                plt.tight_layout()
                trainer.logger.experiment["fig/train/predictions_boxes"].upload(fig)
                plt.close()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            
            imgs, targets = batch

            imgs = imgs[:self.max_num_image]
            targets = targets[:self.max_num_image]

            n_cols= int(self.max_num_image /2)
            fig, axs = plt.subplots(2, n_cols)
            axs = axs.flatten()
            images = [img for img in imgs]
            boxes_list = [target['boxes'] for target in targets]


            for ax, img, boxes in zip(axs, images, boxes_list):
                ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image

                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().detach().numpy()
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                    ax.add_patch(rect)  # Add the box
                ax.axis('off')  # Optional: Turn off axes for better visualization

            plt.tight_layout()
            trainer.logger.experiment["fig/val/input_boxes"].upload(fig)
            plt.close()

            with torch.no_grad():
                x_hat = pl_module(imgs, targets=None, mode='test')
                n_cols= int(self.max_num_image /2)

                fig, axs = plt.subplots(2, n_cols)
                axs = axs.flatten()
                boxes_list = [target['boxes'] for target in x_hat]

                for ax, img, boxes in zip(axs, images, boxes_list):
                    ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Display the image
                    for box in boxes:
                        x1, y1, x2, y2 = box.cpu().detach().numpy()
                        width, height = x2 - x1, y2 - y1
                        rect = Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3)
                        ax.add_patch(rect)  # Add the box
                    ax.axis('off')  # Optional: Turn off axes for better visualization

                plt.tight_layout()
                trainer.logger.experiment["fig/val/predictions_boxes"].upload(fig)
                plt.close()

class SegYOLOImageLogger(Callback):
    def __init__(self, num_images=2, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                x_bb, X_patches = pl_module(batch)

                X_patches = X_patches[0:self.num_images].cpu()

                for i in range(self.num_images):
                    x_p = X_patches[i]

                    grid_p = torchvision.utils.make_grid(x_p, nrow=pl_module.hparams.num_patches)
                    trainer.logger.experiment.add_image('grid_p{i}'.format(i=i), grid_p, pl_module.global_step)


class StackImageLogger(Callback):
    def __init__(self, num_images=2, log_steps=100):
        self.log_steps = log_steps
        self.num_images = num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        
        if batch_idx % self.log_steps == 0:
            img2 = batch["seg"]
            max_num_image = min(img2.shape[0], self.num_images)
            grid_p = torchvision.utils.make_grid(img2, nrow=max_num_image)
            fig = plt.figure(figsize=(7, 9))

            ax = plt.imshow(grid_p.permute(1, 2, 0).cpu().numpy())
            trainer.logger.experiment["images/seg"].upload(fig)
            plt.close()

            with torch.no_grad():
                x, X_patches, x_a, x_v, = pl_module(batch)

                X_patches = X_patches[0:self.num_images].cpu()

                for i in range(self.num_images):
                    x_p = X_patches[i]

                    grid_p = torchvision.utils.make_grid(x_p, nrow=pl_module.hparams.num_patches)
                    fig = plt.figure(figsize=(7, 9))

                    ax = plt.imshow(grid_p.permute(1, 2, 0).cpu().numpy())
                    trainer.logger.experiment["images/x"].upload(fig)
                    plt.close()
                    # trainer.logger.experiment.add_image('grid_p{i}'.format(i=i), grid_p, pl_module.global_step)