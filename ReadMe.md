# Attention U-Net For Satellite Imagery Segmentation
An implementation of an attention U-Net for semantic (binary) segmentation of satelite imagery over the CONUS.
Also includes some dataloaders designed for NAIP RBG + NIR imagery chips. 
This implementation was used to train a (mostly time aligned) building segmentation model tuned for the southeast U.S. on about 50GB of satelite imagery and building masks from OSM and MS Building footprints. It worked pretty well. Cheers!