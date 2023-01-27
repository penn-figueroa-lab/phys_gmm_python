# phys-gmm-python
This package contains the inference implementation (Gibbs Sampler) for the "Physically Consistent Bayesian Non-Parametric Mixture Model" (PC-GMM) proposed in [1]. This approach is used to **automatically** (no model selection!) fit GMM on **trajectory data** while ensuring that the points clustered in each Gaussian represent/follow a linear dynamics model, in other words the points assigned to each Gaussian should be close in "position"-space and follow the same direction in "velocity"-space.

### Instructions and Content for Python User (Beta Edition)
This package offers a nice way to cluster the robot trajectory data in 2D and 3D space based on their similarity measured by their velocities and locations.  
A more user friendly interface will be developed soon.  

**To check the demonstrations, open the demo_loadData.py and check the following settings.**  
```Python
pkg_dir = r'E:\ds-opt-python\ds-opt-python\phys_gmm_python'
chosen_dataset = 12
sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 7  # For real 3D data
```
   - ``` pkg_dir ```  Change it to the package root.  
   - ```chosen_dataset``` We now offer datasets from 2D to 3D. Numbers 6-10 represent 5 different demo 2D datasets and numbers 11 and 12 represent two 3D datasets.  
   - ```sub_sample``` To reduce computation time, you may set the it to be 2 in the 3D dataset to reduce the number of samples.  
   - ```nb_trajectories``` In real 3D data, there will be many trajectories. To randomly pick a specific amount of them you could change the number of it.  
   - After finishing the above settings you could run the program and check the result.  

**To plug in your own data:**
- Arrange your data in this format
  <p align="center">
     <img src="https://user-images.githubusercontent.com/97799818/197041947-701c3b95-0772-44cf-8b93-3f1bc8188ace.jpg" width="660">
  </>  <br />
 
- Then do a little post process <br />
 ```Python
 M = len(Data)
N = len(Data[0]) # Number of samples
M = int(M / 2) # Dimension
Xi_ref = Data[0:M, :] # Position
Xi_dot_ref = Data[M:, :] # Velocity
est_options = Est_options() # create a empty option class
 ```  
 - Finally pass them into fit_gmm
 ```Python
 fig_gmm(Xi_ref, Xi_dot_ref, est_options)
 ```
 

### Image Gallary


<p align="center">
 <img src="https://user-images.githubusercontent.com/97799818/190874177-67d995b9-b105-47f6-83b0-045c5b0d54f8.png" width="220">
 <img src="https://user-images.githubusercontent.com/97799818/190874280-3fdd430d-9e65-4756-96c7-9afc9697cbeb.png" width="220">
 <img src="https://user-images.githubusercontent.com/97799818/190873962-0e82256d-5057-44bb-b33a-b9f18519bfb7.png" width="220">
 <img src="https://user-images.githubusercontent.com/97799818/190874080-d6599bec-161c-4075-955a-b799bb9d1062.jpg" width="220">
</>
<p align="center">  
**2D Datasets**
</>

<p align="center">
 <img src="https://user-images.githubusercontent.com/97799818/197040278-c8dfa696-b50a-4a07-b7b3-1d4b8f95ebfa.png" width="330">
 <img src="https://user-images.githubusercontent.com/97799818/197046561-e0cc334a-62c1-4c8e-a7cb-55615a54299a.png" width="330">
</>
<p align="center">  
**3D Datasets**
</>



**References**    
> [1] Figueroa, N. and Billard, A. (2018) "A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning". In Proceedings of the 2nd Conference on Robot Learning (CoRL). 

Dependencies:
- Scipy: https://scipy.org/install/
- Gmr: https://github.com/AlexanderFabisch/gmr
