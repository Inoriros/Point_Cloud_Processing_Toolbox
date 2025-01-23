# Density-aware Chamfer Distance

## Usage
### Density-aware Chamfer Distance
The function for **DCD** calculation is defined in `def calc_dcd()` in `utils/model_utils.py`.

Users of higher PyTorch versions may try `def calc_dcd()` in `utils_v2/model_utils.py`, which has been tested on PyTorch 1.6.0 .

## Citation
If you find the code or paper useful, please cite the paper:
```bibtex
@inproceedings{wu2021densityaware,
  title={Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion},
  author={Tong Wu, Liang Pan, Junzhe Zhang, Tai WANG, Ziwei Liu, Dahua Lin},
  booktitle={In Advances in Neural Information Processing Systems (NeurIPS), 2021},
  year={2021}
}
```
## Acknowledgement
The code is based on the [VRCNet](https://github.com/paul007pl/VRCNet) implementation. We include the following PyTorch 3rd-party libraries: 
[ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), 
[emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion), and 
[Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch).
Thanks for these great projects.

## Contact
Please contact [@wutong16](https://github.com/wutong16) for questions, comments and reporting bugs.


