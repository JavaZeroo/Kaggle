import dicom2nifti

dicom2nifti.dicom_series_to_nifti(
    r"E:\Code\Kaggle\RSNA_data\train_images\1.2.826.0.1.3680043.14", 
    r"E:\Code\Kaggle\ ", 
    reorient_nifti=True)
