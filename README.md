
# E4040 2024 Fall Project
## Improving CNN Robustness via CS Shapley Value-guided Augmentation

# About my project
This is the README file of Group MYZH project: Improving CNN Robustness via CS Shapley Value-guided Augmentation.The project repo is a reproduction of the paper [Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain](https://papers.nips.cc/paper_files/paper/2022/hash/022abe84083d235f7572ca5cba24c51c-Abstract-Conference.html).

As shown in organization of this directory below，all the codes of the project is here in this repo.Notebooks marked as 1 show the traing of ResNet18, one using 140 epochs achiving higher accuracy of 92%, one using only 60 epochs achiving an accuracy rate of 91% ([checkpoints](https://drive.google.com/drive/folders/1jclMlmgVUmgL1ZkbfJ11zuk8kFul81qw?usp=drive_link) and [saved_model](https://drive.google.com/drive/folders/1ULBrfH6vbhICuajw08dnzZEB_HwW4tTz?usp=drive_link)). Notebooks marked as 2 show the traing of ResNet18/CSANet with/without adversarial attack ([checkpoints](https://drive.google.com/drive/folders/1jclMlmgVUmgL1ZkbfJ11zuk8kFul81qw?usp=drive_link) and [saved_model](https://drive.google.com/drive/folders/1ULBrfH6vbhICuajw08dnzZEB_HwW4tTz?usp=drive_link)).Both of the models are very large, so I have provided the links to the checkpoints and saved models.In the purpose of formatting the repo, the folders contain only a txt file showing the same link.
In the training of CSANet, it used reconstracted Shapley values of the images in a frequancy bias. The compuations can be made by using the Shapley_Value_Calculator.py file under the folder utils. However, it is very big and I utilized the [reconstructed images of the NFCs and the PFCs](https://drive.google.com/file/d/1do8KbtySg7vCZr0cXCQHZ4m0HViIPfdR/view?usp=sharing) offered by the arthurs of the paper [here](https://github.com/Ytchen981/CSA). The data I used in my trianing is [saved_values](https://drive.google.com/file/d/1_B5fJgK_Z6kerA6aHpD4E-BdqRI8yk65/view?usp=drive_link).
My report can be found [here](https://drive.google.com/file/d/1Dh23FTmvQfoXGufLji0uY21e6eF2ftFj/view?usp=sharing) or also in this repo.



# Organization of this directory
```
./
├── requirements.txt
├── E4040.2024Fall.MYZH.report.mz3088.pdf
├── Notebook1_140epoch.ipynb
├── Notebook1_60epoch.ipynb
├── Notebook2_pgd_attack_False_use_csa_False.ipynb
├── Notebook2_pgd_attack_False_use_csa_True.ipynb
├── Notebook2_pgd_attack_True_20_use_csa_False.ipynb
├── Notebook2_pgd_attack_True_20_use_csa_True.ipynb
├── Notebook2_pgd_attack_True_use_csa_False.ipynb
├── Notebook2_pgd_attack_True_use_csa_True.ipynb
├── Shapley_Value_Calculator.ipynb
├── README.md
├── checkpoints
│   └── checkpoints.txt
├── saved_model
│   └── saved_model.txt
├── saved_values
│   └── saved_values.txt
└── utils
    ├── CSANet_trainer.py
    ├── ResNet18_trainer.py
    ├── model_CSANet.py
    └── model_ResNet18.py

5 directories, 18 files

```






