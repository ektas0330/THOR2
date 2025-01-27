# THOR2: Topological Analysis for 3D Shape and Color-Based Human-Inspired Object Recognition in Unseen Environments

Code repository for THOR2 presented in a research article titled '[THOR2: Topological Analysis for 3D Shape and Color-Based Human-Inspired Object Recognition in Unseen Environments](https://advanced.onlinelibrary.wiley.com/journal/26404567)' published in the Advanced Intelligent Systems.

## Requirements
* [Panda3D](https://www.panda3d.org/)
* Open3D
* persim 0.3.1
* scikit-learn
* Keras
* Platform: This code has been tested on Ubuntu 18.04 (except synthetic data generation using Panda3D, which is done on a computer running Windows 10).

## Usage
* ### Synthetic data generation:

Follow instructions from [here](https://docs.panda3d.org/1.10/python/introduction/installation-windows) to install Panda3D for synthetic data generation. Create a new folder `syndata` in the directory where Panda3D is installed and place `generate_synthetic_data.py` from this repository into the `syndata` folder. Create subfolders `models` and `data` inside the `syndata` folder. Within `models`, create subfolders for all objects and place respective object meshes and texture maps inside them. Obtain synthetic depth images from the object meshes using Panda3D using the following command. 

```bash
python generate_synthetic_data.py --obj_name <obj_name> --h <h> --p <p> --r <r>
```
`<obj_name>` is the name of object for which data is to be generated, and the parameters `h`,`p`, and `r` are set to reorient the object mesh as required (details in the paper) before rendering. This command will create synthetic depth (and RGB) images for the object under a subfolder `<obj_name>` inside the `data` folder.

  * ### Training:

