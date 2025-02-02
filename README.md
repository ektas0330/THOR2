# THOR2: Topological Analysis for 3D Shape and Color-Based Human-Inspired Object Recognition in Unseen Environments

Code repository for THOR2 presented in a research article titled '[THOR2: Topological Analysis for 3D Shape and Color-Based Human-Inspired Object Recognition in Unseen Environments](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aisy.202400539)' published in the Advanced Intelligent Systems.

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
`<obj_name>` is the name of object for which data is to be generated, and the parameters `h`,`p`, and `r` are set to reorient the object mesh as required (details in the paper) before rendering. This command will create synthetic RGB and depth images for the object under a subfolder `<obj_name>` inside the `data` folder.

* ### Computing color regions and their similarity matrix:

Run the following to cluster colors in the sRGB color space using the Mapper algorithm. 
```bash
python3 get_color_regions_using_mapper.py
python3 get_similarity_matrix.py
```
As a result, a lookup table is generated containing membership information, mapping each color region (cluster) to its specific colors (members). A similarity matrix capturing the similarity and connectivity between the different color regions is also computed.
  
* ### Training:

  	1. From within the THOR2 directory run the following to generate point clouds corresponding to all the generated RGB-D images. 
		```bash
		python3 training/get_PCDs_from_synthetic_data.py --data_path <path_to_data_folder_from_above>
		```
	2. From within the THOR2 directory run the following to perform view normalization on the generated point clouds. 
		```bash
		python3 training/save_all_view_normalized_PCDs.py --data_path <path_to_data_folder_from_step_i>
		```
	3. From within the THOR2 directory run the following to generate Persistence Images (PIs) for the TOPS descriptor of all the point clouds. 
		```bash
		python3 training/compute_PIs_from_view_normalized_PCDs.py --data_path <path_to_data_folder_from_step_i>
		```
		A subfolder named `libpis` containing all the PIs will be generated inside the `training` folder.

	4. From within the THOR2 directory, run the following to generate color embeddings for the TOPS2 descriptor (i.e., descriptor with color embeddings interleaved with the TOPS descriptor) of all the point clouds. 
		```bash
		python3 training/compute_embeddings_from_view_normalized_PCDs.py --data_path <path_to_data_folder_from_step_i>
		```
		A subfolder named `libembeds` containing all the embeddings will be generated inside the `training` folder.

	5. Change the working directory to the `training` directory using `cd training` and run the following to train one multilayer perceptron (MLP) using the TOPS descriptor 
		```bash
  		python3 train_mlp_for_tops.py --data_path <path_to_data_folder_from_step_one> --random_state 2022
		```
  		and another MLP using the TOPS2 descriptor.
    
		```bash
  		python3 train_mlp_for_tops2.py --data_path <path_to_data_folder_from_step_one> --random_state 2022
		```

		 A folder `mlp_modela` will be created inside the `training` directory and trained models will be stored in it.

* ### Testing on the UW-IS Occluded Dataset:
	1. Download the UWISOccludedDataset.zip from [here](https://doi.org/10.6084/m9.figshare.20506506) and unzip it. Place `reogranizeUWISOccluded.sh` inside the `UWISOccludedDataset` folder and run the following from within that folder.

		```bash
		sh reorganizeUWISOccluded.sh
		```
  	2. 

