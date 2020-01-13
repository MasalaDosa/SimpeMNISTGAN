**Simple MNIST GAN**

An example generative adversarial network working with the MNIST dataset

To train this python3 program ensure you have the following libraries available in your environment:

numpy, tensorflow, keras and matplotlib

They can be installed by running the following commands in your environment:

pip install --upgrade pip

pip install numpy

pip install tensorflow

pip install keras

pip install matplotlib
`
Running GAN.py will run through a number of training epochs.
Every 40 batches and every 40 epochs example images will be written to the output_images folder.
As you run this you will see what initially is just noise gradually become more and more recognisable as digits.

This process can take some time to complete - if you are feeling impatient then you can instead run the GenerateImages.py script.
This uses a model saved after 100 training epochs using the GAN.py script and will write an example file into the output_images folder.
 
 

