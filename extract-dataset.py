from pathlib import Path

import fire
import logging
import numpy as np
import pandas as pd
import re

DB_PATH = Path('databases')
MIMIC4_PATH = DB_PATH / 'mimic4'

logging.basicConfig(
    filename='extract-dataset.logs',
    filemode='a',
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG,
)
logger = logging.getLogger('default')

LABEVENTS_FEATURES = {
    'bicarbonate': [50882],  # mEq/L
    'chloride': [50902],
    'creatinine': [50912],
    'glucose': [50931],
    'magnesium': [50960],
    'potassium': [50822, 50971],  # mEq/L
    'sodium': [50824, 50983],  # mEq/L == mmol/L
    'bun': [51006],
    'hemoglobin': [51222],
    'platelets': [51265],
    'wbcs': [51300, 51301],
}
CHARTEVENTS_FEATURES = {
    'height': [
        226707,  # inches
        226730,  # cm
        1394,  # inches
    ],
    'weight': [
        763,  # kg
        224639,  # kg
    ]
}


def partition_rows(input_path, output_path):
    '''
    这个时候处理的文件应该是  icustay_id  item_id value
    一个icustay_id有多行，表示提取多个特征（每个特征对应一行）
    '''
    logger.info('`partition_rows` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # extract the day of the event
    #charttime保存的是2020-8-21 15:00：00 日期和时间是以空格为划分的
    #chartday保存日期，去掉时间
    df['chartday'] = df['charttime'].astype(
        'str').str.split(' ').apply(lambda x: x[0])

    # group day into a specific ICU stay
    #给病人一个新的stay_day，表示在chartday日期，病人在ICU并病房内 
    #所以可以通过查询某个病人的stay_day有多少个，确订病人在ICU病房内待多少天
    df['stay_day'] = df['stay_id'].astype('str') + '_' + df['chartday']

    # add feature label column
    #所有特征组成的字典
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    
    #通过item_id来查询对应的特征名称
    features_reversed = {v2: k for k, v1 in features.items() for v2 in v1}
    
    #新增feature一列，与item_id对应为其具体名称
    df['feature'] = df['itemid'].apply(lambda x: features_reversed[x])

    # save mapping of icu stay ID to patient ID
    #icu_subject_mapping={stay_day:subject_id}
    icu_subject_mapping = dict(zip(df['stay_day'], df['subject_id']))

    # convert height (inches to cm)
    mask = (df['itemid'] == 226707) | (df['itemid'] == 1394)
    df.loc[mask, 'valuenum'] *= 2.54

    # average all feature values each day
    '''
    透视表处理：
    以下处理得到的table是以stay_day为 行的index，列是feature，值是feature的value
    就是将所有stay_day相同的feature都列为行
                  feature1     feature2         ....          feature_3
    stay_id_1    feature1_value ... 
    stay_id_2
       ...
    stay_id_n

    fill_value是填充缺失值，这里使用NAN填充缺失值
    aggfunc是聚合函数，比如，当feature1在一天内测量了很多次时，将他们的平均值记为feature1_value
    '''
    
    df = pd.pivot_table(
        df,
        index='stay_day',
        columns='feature',
        values='valuenum',
        fill_value=np.nan,
        aggfunc=np.nanmean,
        dropna=False,
    )

    # insert back information related to the patient (for persistence)
    #将stay_day、stay_id、subject_id作为列添加到df数据中
    df['stay_day'] = df.index
    df['stay_id'] = df['stay_day'].str.split(
        '_').apply(lambda x: x[0]).astype('int')
    df['subject_id'] = df['stay_day'].apply(lambda x: icu_subject_mapping[x])

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`partition_rows` has ended')
    #以上，已经把原来的一行为一个feature的数据库原始的查询数据修改为一个表格数据。syay_day最为行的index，每行为features的集合


def impute_holes(input_path, output_path):
    logger.info('`impute_holes` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    # collect all feature keys
    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}.keys()

    # fill NaN values with the average feature value (only for the current ICU stay)
    # ICU stays with NaN average values are dropped
    #计算总共有多少条ICU记录
    stay_ids = pd.unique(df['stay_id'])
    logger.info(f'Total ICU stays: {len(stay_ids)}')

    #遍历每一条ICU记录
    for stay_id in stay_ids:
        # get mask for the current icu stay

        stay_id_mask = df['stay_id'] == stay_id
        '''
        stay_id_mask = df['stay_id'] == stay_id:
            stay_id_maks = (df['stay_id']==stay_id)
        对列"stay_id"进行操作：df['stay_id']==stay_id，列值等于stay_id的为true，否则为false

        '''
        # there are ICU stays that even though its los >= 3
        #los的登记可能会出错，所以我么去掉这些los>=3但实际上ICU住院<3的数据，
        # the actual measurements done in labevents or chartevents are fewer than that
        # so we drop them here
        if df[stay_id_mask].shape[0] < 3:
            logger.warning(f'ICU stay id={stay_id} has los<3 (dropped)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stays with no creatinine levels
        # after the first 48 hours
        #去掉48小时以后的所有creatinine都为空的数据记录
        #即，即使病人在ICU内住了三天以上，但是第三天以及以后的creatinine的值缺失了，我们无法判断真实情况下病人是否患有AKI
        if not np.isfinite(df[stay_id_mask]['creatinine'].values[2:]).any():
            logger.warning(f'ICU stay id={stay_id} creatinine levels'
                           + ' are all NaN after 48 hours (dropped)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stays with no creatinine levels
        # at the third day
        nan_index = get_nan_index(df[stay_id_mask]['creatinine'])
        if nan_index == 2:
            logger.warning(f'ICU stay id={stay_id} creatinine level'
                           + ' at 3rd day is not available (dropped)')
            df = df[~stay_id_mask]
            continue

        # drop ICU stay days (and onwards) with no creatinine levels defined
        if nan_index != -1:
            logger.warning(f'ICU stay id={stay_id} creatinine level'
                           + f' at {nan_index}th day is not available (dropped)')
            #加入病人第5天的creatinine的值确实了，去掉病人第5天以及以后的数据，保留病人前四天的数据
            nan_indices = df[stay_id_mask].index[nan_index:]
            df = df.drop(nan_indices)

        # fill feature missing values with the mean value
        # of the ICU stay, dropping ICU stays with missing values
        df = fill_nas_or_drop(df, stay_id, features)

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`impute_holes` has ended')


def fill_nas_or_drop(df, stay_id, features):
    '''
        对每一条ICU记录进行处理
         drop ICU stays with features that doesn't contain any 
            finite values (e.g., all values are NaN)
    '''

    # get mask for the current icu stay
    stay_id_mask = df['stay_id'] == stay_id

    for feature in features:
        #entity_features得到stay_id这条ICU记录的每个特征的所有天的某feature的value
        entity_features = df.loc[stay_id_mask, feature]
        #如果所有天的featrue的value都为空，去掉该条ICU记录
        if not np.isfinite(entity_features).any():
            logger.warning(f'ICU stay id={stay_id} feature={feature}'
                           + ' does not contain valid values (dropped)')
            return df[~stay_id_mask]

    # we impute feature values using forward/backward fills
    #优先使用前一天的数据进行填充缺失值
    df.loc[stay_id_mask] = df[stay_id_mask].ffill().bfill()

    return df


def add_patient_info(input_path, output_path):
    logger.info('`add_patient_info` has started')

    admissions_path = MIMIC4_PATH / 'filtered_admissions.csv'
    admissions = pd.read_csv(admissions_path)
    admissions.columns = map(str.lower, admissions.columns)

    icustays_path = MIMIC4_PATH / 'filtered_icustays.csv'
    icustays = pd.read_csv(icustays_path)
    icustays.columns = map(str.lower, icustays.columns)

    patients_path = MIMIC4_PATH / 'filtered_patients.csv'
    patients = pd.read_csv(patients_path)
    patients.columns = map(str.lower, patients.columns)

    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    stay_ids = pd.unique(df['stay_id'])
    logger.info(f'Total ICU stays: {len(stay_ids)}')

    # get auxiliary features
    hadm_id_mapping = dict(zip(icustays['stay_id'], icustays['hadm_id']))
    ethnicity_mapping = dict(
        zip(admissions['hadm_id'], admissions['ethnicity']))
    gender_mapping = dict(zip(patients['subject_id'], patients['gender']))
    age_mapping = dict(zip(patients['subject_id'], patients['anchor_age']))

    # retrieve admission ID from stay_day
    df['stay_id'] = df['stay_day'].str.split('_').apply(lambda x: x[0])
    df['stay_id'] = df['stay_id'].astype('int')
    df['hadm_id'] = df['stay_id'].apply(lambda x: hadm_id_mapping[x])

    # compute patient's age
    df['age'] = df['subject_id'].apply(lambda x: age_mapping[x])

    # add patient's gender
    df['gender'] = df['subject_id'].apply(lambda x: gender_mapping[x])
    df['gender'] = (df['gender'] == 'M').astype('int')

    # add patient's ethnicity (black or not)
    df['ethnicity'] = df['hadm_id'].apply(lambda x: ethnicity_mapping[x])
    df['black'] = df['ethnicity'].str.contains(
        r'.*black.*', flags=re.IGNORECASE).astype('int')

    # drop unneeded columns
    del df['ethnicity']

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`add_patient_info` has ended')


def add_aki_labels(input_path, output_path):
    logger.info('`add_aki_labels` has started')

    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    stay_ids = pd.unique(df['stay_id'])
    logger.info(f'Total ICU stays: {len(stay_ids)}')

    for stay_id in stay_ids:
        # get auxiliary variables
        stay_id_mask = df['stay_id'] == stay_id
        #得到患者黑人标志、年龄、性别
        black = df[stay_id_mask]['black'].values[0]
        age = df[stay_id_mask]['age'].values[0]
        gender = df[stay_id_mask]['gender'].values[0]

        # get difference of creatinine levels
        scr = df[stay_id_mask]['creatinine'].values
        #得到患者creatine的后一天与前一天的差值
        #scr[1,2,3,...n]-scr[0,1,2,3...n-1]
        diffs = scr[1:] - scr[:-1]

        # drop ICU stays with AKIs for the first 48 hours
        if (
            has_aki(diff=diffs[0])
            or has_aki(scr=scr[0], black=black, age=age, gender=gender)
            or has_aki(scr=scr[1], black=black, age=age, gender=gender)
        ):
            logger.warning(
                f'ICU stay id={stay_id} has AKI pre-48 (dropped)')
            df = df[~stay_id_mask]
            continue

        # we do next-day AKI prediction
        # use the 3rd day's creatinine level to get the AKI label of day 2 data
        '''
        得到第三天的aki label：使用diff[1]表示第三天的scr-第二天的scr来判断第三天的scr增量是否过大;
            或者使用每天的scr值是否大于baseline的1.5倍
        '''
        aki1 = pd.Series(diffs[1:]).apply(lambda x: has_aki(diff=x))
        aki2 = pd.Series(scr[2:]).apply(lambda x: has_aki(
            scr=x, black=black, age=age, gender=gender))
        aki = (aki1 | aki2).astype('int').values.tolist()

        # drop last day values
        last_day_index = df[stay_id_mask].index[-1]
        df = df.drop(last_day_index)

        # assign aki labels
        stay_id_mask = df['stay_id'] == stay_id
        aki_labels = [0] + aki
        df.loc[stay_id_mask, 'aki'] = aki_labels

        # truncate icu stays (retain first 8 days)
        to_truncate_indices = df[stay_id_mask].index[8:]
        if len(to_truncate_indices) > 0:
            logger.warning(
                f'ICU stay id={stay_id} will be truncated to 8 days.')
            df = df.drop(to_truncate_indices)

    # save results
    df.to_csv(output_path, index=False)
    logger.info('`add_aki_labels` has ended')


def has_aki(diff=None, scr=None, black=None, age=None, gender=None):
    # KDIGO criteria no. 1
    # Increase in SCr by >= 0.3 mg/dl (>= 26.5 lmol/l) within 48 hours
    if diff is not None:
        return diff >= 0.3

    # KDIGO criteria no. 2
    # increase in SCr to ≥1.5 times baseline, which is known or
    # presumed to have occurred within the prior 7 days
    if scr is not None:
        assert black is not None
        assert age is not None
        assert gender is not None

        baseline = get_baseline(black=black, age=age, gender=gender)
        return scr >= 1.5 * baseline

    # KDIGO criteria no. 3
    # Urine volume <0.5 ml/kg/h for 6 hours
    # not included since urine output data is scarce in MIMIC-III dataset

    raise AssertionError('ERROR - Should pass diff OR scr')


def get_baseline(*, black, age, gender):
    if 20 <= age <= 24:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.5 if gender == 1 else 1.2
        else:
            # other males: 1.3, other females: 1.0
            return 1.3 if gender == 1 else 1.0

    if 25 <= age <= 29:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.5 if gender == 1 else 1.1
        else:
            # other males: 1.3, other females: 1.0
            return 1.2 if gender == 1 else 1.0

    if 30 <= age <= 39:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.4 if gender == 1 else 1.1
        else:
            # other males: 1.3, other females: 1.0
            return 1.2 if gender == 1 else 0.9

    if 40 <= age <= 54:
        if black == 1:
            # black males: 1.5, black females: 1.2
            return 1.3 if gender == 1 else 1.0
        else:
            # other males: 1.3, other females: 1.0
            return 1.1 if gender == 1 else 0.9

    # for ages > 65
    if black == 1:
        # black males: 1.5, black females: 1.2
        return 1.2 if gender == 1 else 0.9
    else:
        # other males: 1.3, other females: 1.0
        return 1.0 if gender == 1 else 0.8


def get_nan_index(series):
    result = ~np.isfinite(series)
    for i, x in enumerate(result[2:]):
        if x:
            return i + 2

    return -1


def transform_outliers(input_path, output_path):
    logger.info('`transform_outliers` has started')
    df = pd.read_csv(input_path)
    df.columns = map(str.lower, df.columns)

    features = {**LABEVENTS_FEATURES, **CHARTEVENTS_FEATURES}
    for feature in features.keys():
        upper_bound = df[feature].mean() + 6 * df[feature].std()
        lower_bound = df[feature].mean() - 6 * df[feature].std()
        logger.info(f'Feature={feature} upper bound={upper_bound}')
        logger.info(f'Feature={feature} lower bound={lower_bound}')

        upper_mask = df[feature] > upper_bound
        lower_mask = df[feature] < lower_bound
        upper_ids = pd.unique(df.loc[upper_mask, 'stay_id'])
        lower_ids = pd.unique(df.loc[lower_mask, 'stay_id'])

        if len(upper_ids) > 0:
            # rescale values to the upper bound
            logger.info(f'Feature={feature}, {upper_ids} contains +outliers')
            df.loc[upper_mask, feature] = upper_bound

        if len(lower_ids) > 0:
            # rescale values to the lower bound
            logger.info(f'Feature={feature}, {lower_ids} contains -outliers')
            df.loc[lower_mask, feature] = lower_bound

    # save result
    df.to_csv(output_path, index=False)
    logger.info('`transform_outliers` has ended')


def extract_dataset(output_dir='dataset', redo=False):
    # create output dir if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=False, exist_ok=True)

    # partition features into days
    ipath = MIMIC4_PATH / 'filtered_events.csv'
    opath = output_dir / 'events_partitioned.csv'
    if redo or not opath.exists():
        partition_rows(ipath, opath)

    # fill empty holes with median values
    ipath = opath
    opath = output_dir / 'events_imputed.csv'
    if redo or not opath.exists():
        impute_holes(ipath, opath)

    # add patient info
    ipath = opath
    opath = output_dir / 'events_with_demographics.csv'
    if redo or not opath.exists():
        add_patient_info(ipath, opath)

    # add AKI labels
    ipath = opath
    opath = output_dir / 'events_with_labels.csv'
    if redo or not opath.exists():
        add_aki_labels(ipath, opath)

    # get rid of outliers
    ipath = opath
    opath = output_dir / 'events_complete.csv'
    if redo or not opath.exists():
        transform_outliers(ipath, opath)


if __name__ == '__main__':
    fire.Fire(extract_dataset)
