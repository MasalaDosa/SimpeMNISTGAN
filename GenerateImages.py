import GAN
from keras.models import load_model
model_file = "test_model"
generator = load_model("test_model")

GAN.checkpoint_generate_images(generator,"from_saved")
