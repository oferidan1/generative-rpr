# generative_rpr
# final project in generative models course: training VAE to generate relative image 
train command:
python train_vae.py --bPoseCondition=0

test command: 
python train_vae.py --mode=test --checkpoint_path=out_vae/vae_model.pth --bPoseCondition=0
