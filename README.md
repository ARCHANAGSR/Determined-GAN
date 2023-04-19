# Determined-GAN
Explore how Determined AI can accelerate GAN.


STEPS TO EXECUTE:

1. Run the cgan_minist_collab.ipynb file in collab and get the generated images. The required images will be saved as mnistlikedataset.npz file.
2. This mnistlikedataset.npz file is passed to CNN classifer code which is run in Determined AI. Under DetAI_classifier/ folder we have the classifier code for Determined AI. 

Note :
We have done three types of data validation using CNN(step 2) in Determined AI as below, 
1. Only Mnist data, run: use data.py. 
2. 59k MNIST + 1k generated images. Rename 1kdata.py to data.py and using mnistlikedataset.npz, run the code.
3. 30k mnist + 30k generated images. Rename 30kdata.py to data.py and 30kmnistlikedataset.npz to mnistlikedataset.npz and run the code.

*For reference, we have placed partially ported CGAN code in determined at DetAI_cgan/ folder.
*Hackathon ppt and video presentation is uploaded in github.


# CGAN with Determined AI

This example demonstrates how to build a CGAN on the MNIST dataset using Determined's TensorFlow Keras API. This example is adapted from this [Keras Tutorial](https://keras.io/examples/generative/conditional_gan/).
The CGAN Keras model featured in this example subclasses `tf.keras.Model` and defines a custom `train_step()` and `test_step()`.

## Resources
[Keras c_gan](https://keras.io/examples/generative/conditional_gan/)

[Google Colab](https://colab.research.google.com/drive/1m2lUDfmX69iv2e0BXok0K63QQlr2iZ-L#scrollTo=NEY5JcHMo1qA) *Note can be ported to Determined AI JupyterLab.


## Files
* **c_gan.py**: The code code defining the model.
* **data.py**: The data loading and preparation code for the model.
* **model_def.py**: Organizes the model into Determined's TensorFlow Keras API.
* **export.py**: Exports a trained checkpoint and uses it to generate images.

## Configuration Files
* **const.yaml**: Train the model with constant hyperparameter values.
* **distributed.yaml**: Same as const.yaml, but instead uses multiple GPUs (distributed training).

## To Run
Installation instructions can be found under `docs/install-admin.html` or at [Determined installation page](https://docs.determined.ai/latest/index.html).
After configuring the settings in `const.yaml`, run the following command: `det -m <master host:port> experiment create -f const.yaml . `

## To Export
Once the model has been trained, its top checkpoint can be exported and used to generate images by running:
```bash
python export.py --experiment-id <experimend_id> --master-url <master:port>
```

## Demo

### Installtion instructions
*	Start
```bash
 det deploy local cluster-up --no-gpu
```

*	Port Forwarding
```bash
ssh -L 8080:localhost:8080 swarm@10.30.91.253
```

*	Login
http://localhost:8080/ (admin or determined user and empty password)

*	Run experiment from DAI folder

```bash
det experiment create const.yaml .
det experiment create adaptive.yaml .⁠
```

*	Experimentation around
    1.	Create workspace/Project and run experiment
    2.	Launch JupyterLab
    3.	Launch Tensorboard
    4.	Checkpointing model in model registry

![Training image](./DAI_integrated_cgan/images/c_gan.png)





