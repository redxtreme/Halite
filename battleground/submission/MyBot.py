import tsmlstarterbot

# Load the model from the models directory. Models directory is created during training.
# Run "make" to download data and train.
tsmlstarterbot.Bot(location="model_cust_training.ckpt", name="MyBot_Leader_Smarter").play()
