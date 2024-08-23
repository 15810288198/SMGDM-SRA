# Requirement
- Python 3.7
- Pytorch 1.7
- CUDA 11.1


# Datasets
- ISTD [link](https://github.com/DeepInsight-PCALab/ST-CGAN))
- ISTD+ [link](https://github.com/cvlab-stonybrook/SID))
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) [Testing](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view) [Mask](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_um_edu_mo/EZ8CiIhNADlAkA4Fhim_QzgBfDeI7qdUrt6wv2EVxZSc2w?e=wSjVQT) (detected by [DHAN](https://github.com/vinthony/ghost-free-shadow-removal))

# Pretrained models
[Link](https://pan.baidu.com/s/1X0hQMWJrIot9h3YjKs5USA?pwd=wb6r)<br>
Please download the corresponding pretrained model and modify the `resume_state` and `degradation_model_path` (optional) in `shadow.json`.

# Test

You can directly test the performance of the pre-trained model as follows

1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `shadow.json`

    ```text
    resume_state  # pretrain model or training state -- Line 12
    dataroot      # validation dataset path -- Line 30
    ```

2. Test the model

    ```bash
    python sr.py -p val -c config/shadow_SRD.json
    ```
# Train

1. Download datasets and set the following structure

    ```
    -- SRD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask
           |-- test_C  # shadow-free GT
    ```

2. You need to modify the following terms in `option.py`

    ```python
    "resume_state": null  # if train from scratch
    "dataroot"           # training and testing set path
    "gpu_ids": [0]       # Our model can be trained using a single RTX A5000 GPU. You can also train the model using multiple GPUs by changing this to [0, 1].
    ```

3. Train the network

    ```bash
    python sample.py -p train -c config/shadow.json
    ```

# Evaluation

The results reported in the paper are calculated by the `matlab` script used in [previous method](https://github.com/zhuyr97/AAAI2022_Unfolding_Network_Shadow_Removal/tree/master/codes).

### Testing results

The testing results on dataset ISTD, ISTD+, SRD are: [results](https://pan.baidu.com/s/12n9MvdLNvJSrY6-xhp3OOQ?pwd=34t2).

# References

Our implementation is based on [ShadowDiffusion]([link-to-ShadowDiffusion](https://github.com/GuoLanqing/ShadowDiffusion)) and [RePaint]([link-to-RePaint](https://github.com/andreas128/RePaint)). We would like to thank them.
