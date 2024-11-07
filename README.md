# AI_SEG

A machine learning model U2net and OpenCV-based color clustering method that performs object segmentation in a single shot

Author: Suxing Liu


##



Robust and parameter-free foreground object segmentation.



## Inputs 

   An image file 

## Results 

    Segmentation results and marker detection results


## Sample Test

![Sample Input](../main/sample_test/input.jpg)

Sample input image: a image capture by a portable scanner with marker


![Sample Output: mask](../main/sample_test/output.jpg)

Sample output: oreground object segmentation result

![Sample Output: mask](../main/sample_test/unit.xlsx)

Sample output: marker detection results



## Usage by cloning the  GitHub repo to local environment

Main pipeline: 

    python3 ai_seg.py -i /input/ -o /output/





1. Clone the repo into the local host PC:

```bash

    git clone https://github.com/Computational-Plant-Science/3d_sorghum_segmentation.git

```

   Now you should have a clone of the pipeline source code in your local PC, the folder path was:
```
   /$host_path/3d_sorghum_segmentation/
   
    Note: $host_path can be any path chosen by the user. 
```

2. Prepare your input image folder path,

   here we use the sample images inside the repo as input, the path was:
```
   /$host_path/3d_sorghum_segmentation/sample_test/
```

3. Main pipeline to compute the segmentation results:

```bash

   cd /$host_path/3d_sorghum_segmentation/

   
   python3 /$host_path/3d_sorghum_segmentation/ai_seg.py -i /$host_path/3d_sorghum_segmentation/sample_test/ -o /$host_path/3d_sorghum_segmentation/sample_test/

```
Results will be generated in the output folder by adding "/$host_path/3d_sorghum_segmentation/sample_test/"




## Usage for Docker container 


[Docker](https://www.docker.com/) is suggested to run this project in a Unix environment.

1. Download prebuilt docker container from DockerHub 

```bash

    docker pull computationalplantscience/3d_sorghum_segmentation

    docker run -v /$path_to_test_image:/images -it computationalplantscience/3d_sorghum_segmentation

Note: The "/" at the end of the path was NOT needed when mounting a host directory into a Docker container. Above command mount the local directory "/$path_to_test_image" inside the container path "/images"
Reference: https://docs.docker.com/storage/bind-mounts/
```

For example, to run the sample test inside this repo, under the folder "sample_test", first locate the local path 
```
    docker run -v /$path_to_test_image:/images -it computationalplantscience/3d_sorghum_segmentation
```

    then run the mounted input images inside the container:
``` 
    python3 /opt/code/ai_color_cluster_seg.py -i /images/ -o /images/results/
```
    or 
```
    docker run -v /$path_to_test_images:/images -it computationalplantscience/3d_sorghum_segmentation  python3 /opt/code/ai_seg.py -i /images/ -o /images/results/
```

2. Build your local container

```bash

    docker build -t test_container -f Dockerfile .

    docker run -v  /$path_to_test_images:/images -it test_container

```

3. Addition function to change the image format from 

```bash

    docker build -t test_container -f Dockerfile .

    docker run -v  /$path_to_test_images:/images -it test_container 
    
    docker run -v /$path_to_test_images:/images -it test_container  python3 /opt/code/ai_seg.py -i /images/ -o /images/results/

``` 



Results will be generated in the /images/results/ folder.

Note: They are processed copies of the original images, all the image content information was processed and extracted as traits information. 



<br/><br/> 




Reference:

    https://arxiv.org/pdf/2005.09007.pdf
    https://github.com/NathanUA/U-2-Net
    https://github.com/pymatting/pymatting
    https://github.com/danielgatis/rembg
    

## Citation
```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```
