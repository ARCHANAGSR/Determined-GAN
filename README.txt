STEPS TO EXECUTE :

1. Run the cgan_minist_collab.ipynb file in collab and get the generated images. The required images will be saved as mnistlikedataset.npz file.
2. This mnistlikedataset.npz file is passed to CNN classifer code which is run in Determined AI. Under DetAI_classifier/ folder we have the classifier code for Determined AI. 


Note :
We have done three types of data validation using CNN(step 2) in Determined AI as below, 
1. Only Mnist data, run: use data.py. 
2. 59k MNIST + 1k generated images. Rename 1kdata.py to data.py and using mnistlikedataset.npz, run the code.
3. 30k mnist + 30k generated images. Rename 30kdata.py to data.py and 30kmnistlikedataset.npz to mnistlikedataset.npz and run the code.

*For reference, we have placed partially ported CGAN code in determined at DetAI_cgan/ folder.

* Hackathon ppt and video presentation is uploaded in github .

