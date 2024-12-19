[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cZnpr7Ns)
# E4040 2024 Fall Project
## TODO: Improving CNN Robustness via CS Shapley Value-guided Augmentation

Repository for E4040 2024 Fall Project
  - Distributed as Github repository and shared via Github Classroom
  - Contains only `README.md` file

## Project Instructions
Please read the project instructions carefully. In particular, pay extra attention to the following sections in the project instructions:
 - [Obligatory Github project updates](https://docs.google.com/document/d/1lQtsejwee6tpRIIi0f9AFF01tuUVpLd0Ui8VeN4zhhU/edit?tab=t.0#heading=h.m8ytn1ouhejl)
 - [Student Contributions to the Project](https://docs.google.com/document/d/1lQtsejwee6tpRIIi0f9AFF01tuUVpLd0Ui8VeN4zhhU/edit?tab=t.0#heading=h.m8ytn1ouhejl)

The project instructions can be found here:
https://docs.google.com/document/d/1lQtsejwee6tpRIIi0f9AFF01tuUVpLd0Ui8VeN4zhhU/edit?usp=sharing 

## TODO: This repository is to be used for final project development and documentation, by a group of students
  - Students must have at least one main Jupyter Notebook, and a number of python files in a number of directories and subdirectories such as `utils` or similar, as demonstrated in the assignments
  - The content of this `README.md` should be changed to describe the actual project
  - The organization of the directories has to be meaningful

## Detailed instructions how to submit this project:
1. The assignment will be distributed as a Github classroom assignment - as a special repository accessed through a link
2. A student's copy of the assignment gets created automatically with a special name
3. **Students must rename the repository per the instructions below**
5. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud
6. If some model is too large to be uploaded to Github - 1) create google (liondrive) directory; 2) upload the model and grant access to e4040TAs@columbia.edu; 3) attach the link in the report and this `README.md`
7. Submit the report as a PDF in the root of this Github repository
8. Also submit the report as a PDF in Courseworks
9. All contents must be submitted to Gradescope for final grading

## TODO: (Re)naming of a project repository shared by multiple students
Students must use a 4-letter groupID, the same one that was chosen in the class spreadsheet in Google Drive: 
* Template: e4040-2024Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2024Fall-Project-MEME-zz9999-aa9999-aa0000.

# Organization of this directory
To be populated by students, as shown in previous assignments.

TODO: Create a directory/file tree
```
./
├── Notebook1_140epoch.ipynb
├── Notebook1_60epoch.ipynb
├── Notebook2_pgd_attack_False_use_csa_False.ipynb
├── Notebook2_pgd_attack_False_use_csa_True.ipynb
├── Notebook2_pgd_attack_True_20_use_csa_False.ipynb
├── Notebook2_pgd_attack_True_20_use_csa_True.ipynb
├── Notebook2_pgd_attack_True_use_csa_False.ipynb
├── Notebook2_pgd_attack_True_use_csa_True.ipynb
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
    ├── Shapley_Value_Calculator.py
    ├── model_CSANet.py
    └── model_ResNet18.py

4 directories, 17 files

```

# About my project
This is the README file of Group MYZH project: Improving CNN Robustness via CS Shapley Value-guided Augmentation.The project repo is a reproduction of the paper [Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain](https://papers.nips.cc/paper_files/paper/2022/hash/022abe84083d235f7572ca5cba24c51c-Abstract-Conference.html).

As shown in organization of this directory below，all the codes of the project is here in this repo.Notebooks marked as 1 show the traing of ResNet18, one using 140 epochs achiving higher accuracy of 92%, one using only 60 epochs achiving an accuracy rate of 90% ([checkpoints](https://drive.google.com/drive/folders/1jclMlmgVUmgL1ZkbfJ11zuk8kFul81qw?usp=drive_link) and [saved_model](https://drive.google.com/drive/folders/1ULBrfH6vbhICuajw08dnzZEB_HwW4tTz?usp=drive_link)). Notebooks marked as 2 show the traing of ResNet18/CSANet with/without adversarial attack.([checkpoints](https://drive.google.com/drive/folders/1jclMlmgVUmgL1ZkbfJ11zuk8kFul81qw?usp=drive_link) and [saved_model](https://drive.google.com/drive/folders/1ULBrfH6vbhICuajw08dnzZEB_HwW4tTz?usp=drive_link)).Both of the models are very large, so I have provided the links to the checkpoints and saved models.In the purpose of formatting the repo, the folders contain only a txt file showing the same link.
In the training of CSANet, it used reconstracted Shapley values of the images in a frequancy bias. The compuations can be made by using the Shapley_Value_Calculator.py file under the folder utils. However, it is very big and I utilized the data [here](https://drive.google.com/file/d/1do8KbtySg7vCZr0cXCQHZ4m0HViIPfdR/view?usp=sharing). The data I used in my trianing is [saved_values](https://drive.google.com/file/d/1_B5fJgK_Z6kerA6aHpD4E-BdqRI8yk65/view?usp=drive_link).




