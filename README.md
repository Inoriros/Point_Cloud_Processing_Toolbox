## Setting Up with CUDA 12.1

1. **Create a conda environment with Python 3.9:**

   ```bash
   conda create -n pc_tools python=3.9
   ```

2. **Activate the environment:**

   ```bash
   conda activate pc_tools
   ```

3. **Install PyTorch 2.1.0 (CUDA 12.1 wheels):**

   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install PyTorch3D (from GitHub):**

   ```bash
   pip install "git+https://github.com/facebookresearch/pytorch3d.git"
   ```

5. **Install Open3D and OpenCV:**

   ```bash
   pip install open3d opencv-python
   ```

---

**That's it!** Now your `pc_tools` environment is ready with PyTorch 2.1, CUDA 12.1, PyTorch3D, Open3D, and OpenCV.





### Aknowledgment 

Our code include the following PyTorch 3rd-party libraries: 

[ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), 

[Density_aware_Chamfer_Distance](https://github.com/wutong16/Density_aware_Chamfer_Distance), 

[emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion), and 

[Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch).

Thanks for their contribution to the community.

