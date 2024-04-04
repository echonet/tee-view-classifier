
# TEE View Classification

This repository contains inference code for the paper ["Deep learning for transesophageal echocardiography view classification" (Steffner et al., 2023)](https://www.nature.com/articles/s41598-023-50735-8)

To get started, clone this repository and navigate into it:
```
git clone https://github.com/echonet/tee-view-classifier.git
cd tee-view-classifier
```

Then, create a `conda` environment and install the required packages into it:
```
conda create --name tee-view-class python=3.8
conda activate tee-view-class
python -m pip install -r requirements.txt
```


Now that your environement is set up, download the model weights from the Releases tab on this repository. For this example, make a folder inside the repo called `weights` and place them inside it. Then, from the main repo folder, you should be able to run the inference script like this:

```
python -m src.inference \
--data_path "/path/to/folder/of/AVI/video/files" \
--weights_path "weights/weights.ckpt" 
```
