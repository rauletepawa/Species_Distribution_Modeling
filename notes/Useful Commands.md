**Reloading shell**

```
source ~/.bashrc
```

**Create conda environment**

```
conda create -n pytorch_env python=3.10
```

**Install pytorch GPU cuda 12.6**

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Conda environment as a Jupyter Kernel**

1. Activate environment
```
conda activate pytorch
```
2.  Install ipykernel in the environment
```
pip install ipykernel
```
3. Add the environment to Jupyter as a kernel
```
python -m ipykernel install --user --name=pytorch --display-name "pytorch"
```

**How to save Conda environment into yml**

```
conda activate pytorch
conda env export > pytorch_env.yml
```
**Install pytorch CPU Conda**

```
conda install conda-forge::pytorch-cpu
```

**Install interesting libraries for data analysis**

```
pip install pandas numpy matplotlib scikit-learn seaborn captum geopandas
```
