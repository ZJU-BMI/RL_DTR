import numpy as np
import pandas as pd
import datetime
import os
import csv



def main():
    data_root = os.path.abspath("..\\..\\Resource\\data_source")
    save_root = os.path.abspath('..\\..\\Resource\\cache')

    admission_path = os.path.join(data_root, 'DIAGNOSES_ICD.csv.gz')
    lab_test_path = os.path.join(data_root, 'LABEVENTS.csv.gz')
    vital_sign_path = os.path.join(data_root, 'CHARTEVENTS.csv')
    patients_path = os.path.join(data_root, 'PATIENTS.csv.gz')
    code_name_path = os.path.join(data_root, 'D_LABITEMS.csv.gz')
    extracted_id_path = os.path.join(data_root, '41591_2018_213_MOESM3_ESM.csv')

    subject_id = get_extracted_id(extracted_id_path)
    print('病人 Id 加载成功！')
    vital_sign_dict = get_vital_sign(subject_id, vital_sign_path, save_root, read_from_cache=True)
    print('vital sign load successfully!')

    age_sex_dict = get_age_and_sex(subject_id, patients_path, save_root, read_from_cache=False)
    print('age sex load successfully!')


# 获取符合标准的id
def get_extracted_id(extracted_id_path):
    id_dataframe = pd.read_csv(extracted_id_path, header=None)
    return id_dataframe.values


# 身高 体重
def get_vital_sign(subject_id, vital_sign_path, save_root, read_from_cache=True, file_name='vital_sign.csv'):
    if read_from_cache:
        vital_sign_dict = dict()
        vital_sign_all_data = pd.read_csv(vital_sign_path, iterator=True)
        while True:
            vital_sign_all_data_chunk = vital_sign_all_data.get_chunk(10000)
            for subject_id_ in subject_id:
                vital_sign_for_patient = vital_sign_all_data_chunk[vital_sign_all_data_chunk.HADM_ID== subject_id_[0]]
                print(vital_sign_for_patient.values)
                if vital_sign_for_patient.values != None:
                    patient_id, visit_id, item_id, value, unit, time = vital_sign_for_patient[1], \
                                                                       vital_sign_for_patient[2], \
                                                                       vital_sign_for_patient[4], \
                                                                       vital_sign_for_patient[9], \
                                                                       vital_sign_for_patient[10], \
                                                                       vital_sign_for_patient[5]

                    for visit_id_index in range(len(vital_sign_for_patient)):
                        patient_id_, visit_id_, item_id_, value_, unit_, time_ = patient_id[visit_id_index], visit_id[visit_id_index], \
                                                                                 item_id[visit_id_index], value[visit_id_index], \
                                                                                 unit[visit_id_index], \
                                                                                 time[visit_id_index]
                        # height
                        if item_id_ == '226707' or '226730' or '920' or '1394' or '4188' or '3486':
                            if unit_ == 'cm':
                                value_ = value_

                            elif unit_ == 'inch' or unit_ == 'inches':
                                value_ = value_ * 2.54

                            elif unit_ == 'feet' or unit_ == 'feets':
                                value_ = value_ * 30.48
                            else:
                                continue
                            if 250 > value_ > 50:
                                vital_sign_dict[patient_id_][visit_id_]['height'] = value_, time_

                        # weight
                        if item_id_ == '226512' or '762' or '763' or '226531':
                            if unit_ == 'kg':
                                value_ = value_
                            elif unit_ == 'lbs' or item_id_ == '226531':
                                value_ = value * 0.453592

                            elif unit_ == 'oz':
                                value_ = value_ * 0.0283495
                            else:
                                continue

                            if 300 > value_ > 20:
                                vital_sign_dict[patient_id_][visit_id_]['weight'] = value_, time_

        data_to_write = [['patient_id', 'visit_id', 'feature', 'value']]
        with open(os.path.join(save_root, file_name), 'w', encoding='utf-8-sig', newline='') as file:
            for patient_id_ in vital_sign_dict:
                for visit_id_ in vital_sign_dict[patient_id_]:
                    for feature in vital_sign_dict[patient_id_][visit_id_]:
                        value = vital_sign_dict[patient_id_][visit_id_][feature]
                        data_to_write.append([patient_id_, visit_id_, feature, value])
            csv.writer(file).writerrows(data_to_write)
        return vital_sign_dict
    else:
        return None


def get_age_and_sex(subject_id, patients_path, save_root, read_from_cache=False):
    age_and_sex_dict = dict()

    return age_and_sex_dict

if __name__ == '__main__':
    main()
