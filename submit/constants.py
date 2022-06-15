esun_occ_id_to_str = {
    1: "營造／礦砂業",
    2: "製造業",
    3: "水電燃氣業",
    4: "批發／零售／貿易商",
    5: "旅宿／餐飲",
    6: "運輸倉儲",
    7: "農／林／漁／牧",
    8: "金融保險",
    9: "不動產／租賃",
    10: "軍／警／消",
    11: "公務人員",
    12: "律師／會計師／地政士",
    13: "醫藥服務",
    14: "休閒服務",
    15: "其他／家管",
    16: "自由業",
    17: "投資／自營商",
    18: "媒體文教",
    19: "學生",
    20: "學校教師（含行政人員）",
    21: "資訊科技",
    22: "公證人或記帳士",
    23: "國防工業",
    24: "投資或稅務顧問公司",
    25: "不動產仲介／代銷商",
    26: "大宗物資貿易商",
    27: "博弈業（網路／實體）",
    28: "八大特種行業",
    29: "宗教、慈善、基金會",
    30: "銀樓、珠寶商",
    31: "藝術品或古董買賣商",
    32: "當鋪",
    33: "實體或虛擬貨幣兌換所"
}

gov_occ_str_to_id = {
    '礦業及土石採取業' : 0,
    '製造業' : 1,
    '電力及燃氣供應業' : 2,
    '用水供應及污染整治業' : 3,
    '營建工程業' : 4,
    '批發及零售業' : 5,
    '運輸及倉儲業' : 6,
    '住宿及餐飲業' : 7,
    '出版、影音製作、傳播及資通訊服務業' : 8,
    '金融及保險業' : 9,
    '不動產業' : 10,
    '專業、科學及技術服務業' : 11,
    '支援服務業' : 12,
    '教育業' : 13,
    '醫療保健及社會工作服務業' : 14,
    '藝術、娛樂及休閒服務業' : 15,
    '其他服務業' : 16
}

gov_occ_list = ['礦業及土石採取業', '製造業', '電力及燃氣供應業', '用水供應及污染整治業', '營建工程業', '批發及零售業',
       '運輸及倉儲業', '住宿及餐飲業', '出版、影音製作、傳播及資通訊服務業', '金融及保險業', '不動產業',
       '專業、科學及技術服務業', '支援服務業', '教育業', '醫療保健及社會工作服務業', '藝術、娛樂及休閒服務業',
       '其他服務業']

lead_job_list = [
    '主管及監督人員',
    '專業人員',
    '技術員及助理專業人員',
    '事務支援人員',
    '服務及銷售工作人員',
    '技藝、機械設備操作及組裝人員',
    '基層技術工及勞力工',
]

lead_job_str_to_id = {
    '主管及監督人員' : 0,
    '專業人員' : 1,
    '技術員及助理專業人員' : 2,
    '事務支援人員' : 3,
    '服務及銷售工作人員' : 4,
    '技藝、機械設備操作及組裝人員' : 5,
    '基層技術工及勞力工' : 6 
}

selected_features =[
    'source',
    'age',
    'occupation',
    'hasOtherComAccount',
    'eduLevel',
    'isReject',
    'incomeYear',
    'totalWealth',
    'expInvestment',
    'yrsInvestment',
    'frqInvestment',
    'srcCapital',
    'quotaCredit',
    'quota_now',
    'quota_now_elec',
    'salary',
    'lead_job_id'
]

# FUGLE
selected_features_fugle =[
    'source',
    # 'age',
    'occupation',
    'hasOtherComAccount',
    # 'eduLevel',
    'isReject',
    'incomeYear',
    'totalWealth',
    'expInvestment',
    # 'yrsInvestment',
    # 'frqInvestment',
    # 'srcCapital',
    # 'quotaCredit',
    'quota_now',
    'quota_now_elec',
    'salary',
    'lead_job_id'
]

cat_features_fugle = ['occupation', 'hasOtherComAccount', 'lead_job_id']


# ESUN
selected_features_esun =[
    'source',
    'age',
    'occupation',
    'hasOtherComAccount',
    # 'eduLevel',
    'isReject',
    'incomeYear',
    'totalWealth',
    'expInvestment',
    # 'yrsInvestment',
    # 'frqInvestment',
    # 'srcCapital',
    'quotaCredit',
    'quota_now',
    'quota_now_elec',
    'salary',
    # 'lead_job_id'
]

cat_features_esun = ['occupation', 'hasOtherComAccount']