import wandb
import numpy as np
import datetime

class Logger:
    
    def __init__(self, args, extra_args =  None) -> None:
        time_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args = args
            
        wandb.init(project="diffpol", 
                    group = '{}_{}{}'.format(self.args.env_name, self.args.scheduler, self.args.tag),
                    name = 'seed{}_date{}'.format(self.args.seed, time_path),
                    )
        # Add extra arguments to existing args under separate header
        wandb.config.update(dict(args))
        
        if extra_args is not None:
            wandb.config.update(vars(extra_args)['_content'], allow_val_change=True)
        
        
        self.step = 0
        self.episode = 0
    
    
    def log(self, log, step):

        wandb.log(log, step)
    
    def log_video(self, video, step):

        video = [np.transpose(np.array(f), (2, 0, 1)) for f in video]
        video = np.stack(video, axis = 0)
        print
        wandb.log({"video": wandb.Video(video)}, step=step)
    

    def log_images(self, images, step):
        """
        Log images to WandB.

        Args:
            images (dict): A dictionary where keys are image names and values are numpy arrays representing images.
            step (int): The step at which to log the images.
        """
        for image_name, image in images.items():
            wandb.log({image_name: wandb.Image(image)}, step=step)
            
            
    def cleanup(self):
        if self.args.tb_plot:
            self.writer.close()
        if self.args.wandb:
            wandb.finish()