# Fully-Automated Cartilage and Meniscus Segmentation for Knee MRI
Fully-automated segmentation of MSK tissues from MRI data.

This code is primarily designed to segment cartilage and meniscus from 3D DESS scans from the OAI. The model weights for this network are rather large (400MB) but they can be found by emailing akshaysc at stanford dot edu.

Here are some results from 14 subjects tested from the OAI iMorphics cohort:

| KLG         | Femoral Cartilage | Tibial Cartilage | Meniscus    |
|-------------|-------------------|------------------|-------------|
| 2           | 89.8 ± 2.0        | 87.9 ± 2.3       | 76.0 ± 3.5  |
| 3           | 90.0 ± 1.3        | 85.6 ± 3.8       | 75.1 ± 4.2  |
| 4           | 90.1 ± 1.8        | 76.1 ± 1.9       | 75.8 ± 3.7  |
| Average     | 89.9 ± 1.6        | 85.3 ± 4.7       | 75.4 ± 3.8  |
| Volume CV % | 1.7 ± 1.2         | 6.5 ± 5.8        | 9.0 ± 5.7   |


Things Planned For:

1. A detailed tutorial on how to use the provided code
2. Pre-processing pipeline (dicom read -> python preprocess -> segmentations in tiff)

If you find any issues with the code or want to contribute, please do let me know!
