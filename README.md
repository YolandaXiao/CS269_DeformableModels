# CS269_DeformableModels

## 1.Active Contours with CNN

## 2.Level-Set Active Contours

## 3.Graph-cut approach: graphcut_and_blur.py
1. What it does
   - takes an input image
   - runs slic algorithm -> output superpixel segments
   - runs graph-cut -> output segmentation mask
   - applies gaussian blur -> outputs final image with Bokeh effect

2. Requirements
   - opencv
   - numpy
   - scipy
   - scikit-image
   - PyMaxflow
    
3. How to run

   1) Single Image
       
        Run on image 1: 
        > 'python graphcut_and_blur.py -i 1'
        
        Defaults run on image 29
        > 'python graphcut_and_blur.py'
        
   2) All Images (1537) for mean IoU

        Run IoU on all images and save image outputs to folder 
        > 'python graphcut_and_blur.py -iou true'
        
        Run IoU on all images without saving image outputs 
        > 'python graphcut_and_blur.py -iou false'

