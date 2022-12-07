# Improving accuracy of myocardial T1 estimation in MyoMapNet

Please run /DemoCode/MyoMapNet_ImproveT1Estimation

For simulting MOLLI T1 mapping, please run MOLLI simulation


Purpose: To improve the accuracy and robustness of T1 estimation in MyoMapNet, a deep learning-based approach using four inversion-recovery T1-weighted images for cardiac T1 mapping. 


Method: MyoMapNet is a fully connected neural network for T1 estimation of LL4, a single inversion-recovery T1 mapping sequence collects four T1-weighted images. MyoMapNet was trained using in-vivo data from Modified Look-Locker (MOLLI) sequence, which resulted in significant bias and sensitivity to various confounders. This study sought to train MyoMapNet using signals generated from numerical simulations and phantom MR data under multiple simulated confounders. The trained model was then evaluated by collecting data in a new phantom study that differed from that used for training. The performance of the new model was compared with MOLLI and SAturation-recovery single-SHot Acquisition (SASHA) for measuring native and post-contrast T1 in 25 subjects. 


Results: In the phantom study, T1 values measured by LL4 with MyoMapNet were highly correlated with reference values from the spin-echo sequence. Furthermore, the estimated T1 had excellent robustness to changes in flip angle and off-resonance. The native and post-contrast myocardium T1 at 3T measured by SASHA, MOLLI, and MyoMapNet were 1483±46.6 ms and 791±45.8 ms, 1169±49.0 ms and 612±36.0 ms, and 1443±57.5 ms and 700±57.5 ms, respectively. The corresponding ECV was 22.90±3.20%, 28.88±3.48%, and 30.65±3.60%, respectively. 


Conclusion: 
Training MyoMapNet with numerical simulations and phantom data will improve the estimation of myocardial T1 values and increase its robustness to confounders while reducing overall T1 mapping estimation time to only four heartbeats. 



![Fig1_4HBs-02](https://user-images.githubusercontent.com/9512423/206268826-2eb38922-455a-43a5-8319-05791497d952.png)
