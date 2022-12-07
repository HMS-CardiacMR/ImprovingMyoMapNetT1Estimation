# ImprovingMyoMapNetT1Estimation

Please run /DemoCode/MyoMapNet_ImproveT1Estimation

For simulting MOLLI T1 mapping, please run MOLLI simulation



Purpose

To improve the accuracy and robustness of T1 estimation by MyoMapNet, a deep learning–based approach using 4 inversion-recovery T1-weighted images for cardiac T1 mapping.
Methods

MyoMapNet is a fully connected neural network for T1 estimation of an accelerated cardiac T1 mapping sequence, which collects 4 T1-weighted images by a single Look-Locker inversion-recovery experiment (LL4). MyoMapNet was originally trained using in vivo data from the modified Look-Locker inversion recovery sequence, which resulted in significant bias and sensitivity to various confounders. This study sought to train MyoMapNet using signals generated from numerical simulations and phantom MR data under multiple simulated confounders. The trained model was then evaluated by phantom data scanned using new phantom vials that differed from those used for training. The performance of the new model was compared with modified Look-Locker inversion recovery sequence and saturation-recovery single-shot acquisition for measuring native and postcontrast T1 in 25 subjects.
Results

In the phantom study, T1 values measured by LL4 with MyoMapNet were highly correlated with reference values from the spin-echo sequence. Furthermore, the estimated T1 had excellent robustness to changes in flip angle and off-resonance. Native and postcontrast myocardium T1 at 3 Tesla measured by saturation-recovery single-shot acquisition, modified Look-Locker inversion recovery sequence, and MyoMapNet were 1483 ± 46.6 ms and 791 ± 45.8 ms, 1169 ± 49.0 ms and 612 ± 36.0 ms, and 1443 ± 57.5 ms and 700 ± 57.5 ms, respectively. The corresponding extracellular volumes were 22.90% ± 3.20%, 28.88% ± 3.48%, and 30.65% ± 3.60%, respectively.
Conclusion

Training MyoMapNet with numerical simulations and phantom data will improve the estimation of myocardial T1 values and increase its robustness to confounders while also reducing the overall T1 mapping estimation time to only 4 heartbeats.
