# Mental Workload Classification
Applying style transfer mapping on fNIRS data to predict mental workload level

![](https://images.unsplash.com/photo-1512187849-463fdb898f21?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1332&q=80)

## Abstract

Brain signals, such as functional near-infrared spectroscopy (fNIRS) and electroencephalogram (EEG) vary significantly from one person to another, thus collect sufficient data and building customized brain-computer interfaces for every user is a necessary, yet laborious thing to do. These variations have given rise to many studies on transfer learning models that can be trained on a sample of individuals to analyze the signals of a new subject. As part of the ongoing expansion of this line of work, this study seeks to evaluate a style transfer mapping (STM) method on a dataset of fNIRS signals in a mental workload intensity level classification task. This technique has previously been showed to produced improved precision on a similar task of recognizing emotions using EEG signals, and we want to examine its effectiveness can be extended to other types of classification tasks on brain signal data. This study inherits the fNIRS dataset and preprocessing pipeline from Huang et al. and Wang et al. (2021). We compare the fine-tuned STM model performance to our baseline metrics on the test data when being trained with different numbers of source subjects. We show that the accuracy of STM varies greatly on the target test data, which is likely due to the shift in distributions between the train and test set of the target data. In addition, we also assess the performance of a modified version of the original model and show that the algorithm with reduced complexity can perform just as well.

## $\textcolor{teal}{\textsf{You can read the full paper}}$ [here](Research_writeup.pdf)
