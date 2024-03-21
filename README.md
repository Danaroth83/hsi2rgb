# Description

hsi2rgb is a Python script for the representation of widely known hyperspectral datasets for remote sensing into RGB images.
The library employs a CIE 1931 tristimulus color matching function with the D6500 standard illuminator and applies a gamma correction.

## How to use

This library requires Python 3.9+; the dependencies can be installed using:

```pip install -r requirements.txt```

Then the script can be launched by running the following line in the terminal:

```python src/hsi2rgb.py --i "pavia"```

where ```"pavia"``` is an example of a dataset name. The user can choose any of the datasets listed in the "Dataset list" section.
The script will download the raw dataset in the folder ```data/external```
and then generate numpy arrays in the ```data/outputs``` folder.

The generated numpy arrays are listed below:

- ```hsi.npy```: The original hyperspectral image in row, column, channel format;
- ```wavelengths.npy```: a list of the wavelengths of the hyperspectral image;
- ```rgb.npy```: a RGB representation of the original dataset;
- ```rgb.png```: The same RGB representation in PNG format.

It is also possible to generate a cropped version of the rgb image, which is useful when the original datasets are too large.
For example:

```python src/hsi2rgb.py --i "chikusei" --x 200 712 --y 200 712```

crops the ```"chikusei"``` dataset in the coordinates ranges given by the ```--x``` and ```--y``` options, that is ```(200, 712)``` in both cases.


## Dataset list

- ```"botswana"```
- ```"chikusei```
- ```"cuprite"```
- ```"indian_pines"```
- ```"jasper_ridge"```
- ```"kennedy_space_center"```
- ```"pavia```
- ```"salinas"```
- ```"samson"```
- ```"urban"```

The acknowledgments and licenses (where available) for each dataset are included in the readme file downloaded together with the data in ```data/external```.

## Acknowledgments

The implemented method is very well known specification given by the [CIE standards](https://standards.iteh.ai/catalog/tc/iso/23b8ea7c-5d92-446b-afd6-3dcd4a01fefc/cie).
Variations of this method are given on [this website](https://personalpages.manchester.ac.uk/staff/d.h.foster/Tutorial_HSI2RGB/Tutorial_HSI2RGB.html).
The specific implementation used in this library is also present on ```[Magnusson et al., IGARSS 2020]```.
If you find this library useful in your scientific publications, please cite:

```bibtex
@Online{web_HSI2RGB,
  author  = {Daniele Picone},
  title   = {Code repository for the hsi2rgb.},
  url     = {https://github.com/Danaroth83/hsi2rgb},
  urldate = {2024-03-13},
  year    = {2024},
}
```