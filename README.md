# phys-gmm-python
This package contains the inference implementation (Gibbs Sampler) for the "Physically Consistent Bayesian Non-Parametric Mixture Model" (PC-GMM) proposed in [1]. This approach is used to **automatically** (no model selection!) fit GMM on **trajectory data** while ensuring that the points clustered in each Gaussian represent/follow a linear dynamics model, in other words the points assigned to each Gaussian should be close in "position"-space and follow the same direction in "velocity"-space.

### Instructions and Content for Python User (Beta Edition)
This package offers a nice way to cluster the robot trajectory data in 2D and 3D space based on their similarity measured by their velocities and locations. 
If you want to see the demonstration set by us, you should open the demo_loadData.py. 

```Python
pkg_dir = r'E:\ds-opt-python\ds-opt-python\phys_gmm_python'
chosen_dataset = 12
sub_sample = 2  # % '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 7  # For real 3D data
```
### Image Gallary

![image](https://user-images.githubusercontent.com/97799818/190874177-67d995b9-b105-47f6-83b0-045c5b0d54f8.png)
<p align="center">
  **L-shape**
</>

![image](https://user-images.githubusercontent.com/97799818/190874280-3fdd430d-9e65-4756-96c7-9afc9697cbeb.png)
<p align="center">
**A-shape**
</>

![image](https://user-images.githubusercontent.com/97799818/190873962-0e82256d-5057-44bb-b33a-b9f18519bfb7.png)
<p align="center">
**S-shape**
</>

![90badeaf0e12caf784f65b6acd6cc2e](https://user-images.githubusercontent.com/97799818/190874080-d6599bec-161c-4075-955a-b799bb9d1062.jpg)
<p align="center">  
**multi-behavior**
</>


**References**    
> [1] Figueroa, N. and Billard, A. (2018) "A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning". In Proceedings of the 2nd Conference on Robot Learning (CoRL). 

Dependencies:
- Scipy: https://scipy.org/install/
- Gmr: https://github.com/AlexanderFabisch/gmr
