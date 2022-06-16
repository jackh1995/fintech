import numpy as np
import pandas as pd
import re
import nltk
import math
from constants import gov_occ_str_to_id, gov_occ_list, lead_job_str_to_id

def esun_occ_id_to_gov_occ_id(esun_occ_id):
    '''玉山產業id -> 政府產業id'''
    if esun_occ_id == 99 or math.isnan(esun_occ_id) or esun_occ_id is None:
        return 16 # '其他服務業'
    gov_str = df_gov_occ['gov_occ'][esun_occ_id - 1]
    if gov_str[0] != '*':
        gov_occ_id = gov_occ_str_to_id[gov_str]
    elif gov_str[1] == '*':
        return -int(gov_str[2:])
    else:
        return -int(gov_str[1:])
    return gov_occ_id

def get_job_list(raw_job_data_path='./data/職等分類依據.xlsx'):
    '''得到job_list與lead_list'''
    df_jobs = pd.read_excel(raw_job_data_path)
    raw_job_list = df_jobs['各業受僱員工(人數)(107年7月)(單位：人)'].values
    job_list = []
    lead_list = []
    for idx, x in enumerate(raw_job_list):
        if x[0]=='(':
            lead_list.append(idx)
        x = str(x)
        x = re.sub(r'[^\w]', '', x)
        x = re.compile(u'[\u4E00-\u9FA5|\s]').findall(x)
        x = "".join(x)
        if x:
            job_list.append(x)
        else:
            job_list.append('無')
    return job_list, lead_list

def get_lead_job_idx(sub_job_id: int, lead_list: list[int]) -> int: 
    '''返回某職業id對應的職業類別id'''
    for i in range(len(lead_list)-1):
        if lead_list[i] <= sub_job_id and sub_job_id < lead_list[i+1]:
            return lead_list[i]
    return lead_list[-1]

def find_most_similar_job(job_name: str, job_list: list[str]) -> int:
    '''給定職業名回傳最相似的職業id'''
    score_list = []
    for x in job_list:
        score_list.append(nltk.edit_distance(list(x), list(job_name))) # 字串編輯距離
    return np.argsort(score_list)[0]

def get_lead_job_name(job_name: str, job_list: list[str], lead_list: list[int]) -> str:
    '''給定職業名回傳最相似的職業類別名'''
    return job_list[get_lead_job_idx(find_most_similar_job(job_name, job_list), lead_list)]


def get_salary(job_name_str, job_list, lead_list, esun_occ_id, salary_backup):
    # Note: rel_val might lead to some negative salary values
    rel_val = [
        2 + 1.8282004788567674,
        2 + 0.6560752369233088,
        2 + 0.0062653633495642356,
        2 + -0.5315773430784837,
        2 + -0.2968502431305844,
        2 + -0.34361245855107936,
        2 + -1.3185010343694934
    ]
    lead_job_id = lead_job_str_to_id[get_lead_job_name(job_name_str, job_list, lead_list)]
    occ_id = esun_occ_id_to_gov_occ_id(esun_occ_id)
    if occ_id >= 0:
        return int(df_industry.iloc[lead_job_id, occ_id].replace(',', ""))
    else:
        if occ_id in {-12, -15}:
            return int(salary_backup.iloc[-occ_id, 1] * (1 + rel_val[lead_job_id])) // 12
        else: # -1
            return int(salary_backup.iloc[-occ_id, 1]) // 12

if __name__ == '__main__':

    # --------------------------------- READ DATA -------------------------------- #
    
    df_gov_occ = pd.read_csv('data/occ.csv')

    df_industry = pd.read_excel('./data/psdnquery1-9.xlsx')
    df_industry = df_industry.iloc[:, 1:].drop([0, 1, 2]).reset_index(drop=True)
    df_industry.columns = gov_occ_list
    
    salary_backup = pd.read_excel('data/Year19.xls', skiprows=11)

    # ----------------------------------- MAIN ----------------------------------- #
    
    job_name = '工程師'
    occ_id = 13
    
    job_list, lead_list = get_job_list()

    lead_job = get_lead_job_name(job_name, job_list, lead_list)
    
    salary = get_salary(job_name, job_list, lead_list, occ_id, salary_backup)

    print('Input:')
    print(f'    Job name: {job_name}')
    print(f'    Occupation ID: {occ_id}')
    print('Output:')
    print(f'    Job class (lead job ID): {lead_job}')
    print(f'    Estimated salary: {salary}')