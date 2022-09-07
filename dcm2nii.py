#coding=utf-8
import SimpleITK as sitk
import shutil
from alive_progress import alive_it
from multiprocessing import  Process
import time
import os
import yaml
import dicom2nifti


ret = dicom2nifti.convert_directory("E:\\Code\\Kaggle\\RSNA_data\\train_images\\1.2.826.0.1.3680043.14", 'E:\\Code\\Kaggle\\output.nii.gz', compression=True, reorient=True)
print(ret)
# def logerror(message):
#     with open("log.yaml", "a") as f:
#         data = yaml.load(f, Loader=yaml.Loader)
#     data['dir'].append(message)
#     yaml.dump(data, f, default_flow_style=False)


# def dcm2nii(dcms_path, nii_path):
#     try:
#         # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
#         reader = sitk.ImageSeriesReader()
#         dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
#         reader.SetFileNames(dicom_names)
#         image2 = reader.Execute()
#         # 2.将整合后的数据转为array，并获取dicom文件基本信息
#         image_array = sitk.GetArrayFromImage(image2)  # z, y, x
#         origin = image2.GetOrigin()  # x, y, z
#         spacing = image2.GetSpacing()  # x, y, z
#         direction = image2.GetDirection()  # x, y, z
#         # 3.将array转为img，并保存为.nii.gz
#         image3 = sitk.GetImageFromArray(image_array)
#         image3.SetSpacing(spacing)
#         image3.SetDirection(direction)
#         image3.SetOrigin(origin)
#         sitk.WriteImage(image3, nii_path)
#     except:
#         logerror(dcms_path)


# if __name__ == '__main__':
#     # dcms_path = r'E:\Code\Kaggle\RSNA_data\train_images\1.2.826.0.1.3680043.14'  # dicom序列文件所在路径
#     # nii_path = r'.\test.nii.gz'  # 所需.nii.gz文件保存路径
#     # dcm2nii(dcms_path, nii_path)
#     process_list = []

#     ROOT = os.getcwd()
#     DATA_DIR = os.path.join(ROOT, "RSNA_data/train_images")
    
#     OUTPUT = os.path.join(ROOT, 'RSNA_data_nii')
#     if os.path.exists(OUTPUT):
#         shutil.rmtree(OUTPUT)
#     os.mkdir(OUTPUT)

#     check = os.listdir(OUTPUT)

#     for root, dirs, _ in os.walk(DATA_DIR):
#         for dir in (dirs):
#             if f"{dir}.nii.gz" in check:
#                 continue
#             dcm_folder = os.path.join(root, dir)
#             print(dcm_folder)
#             nii_file = os.path.join(OUTPUT, f"{dir}.nii.gz")
#             p = Process(target=dcm2nii, args=(dcm_folder, nii_file, ))
#             process_list.append(p)

#     for index, p in enumerate(alive_it(process_list)):
#         if index %12 == 0 and index != 0:
#             time.sleep(30)
#         p.start()
#             # dcm2nii(dcm_folder, nii_file)
    