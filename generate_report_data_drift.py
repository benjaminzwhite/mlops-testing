import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# A sample selection of features that are important for our best prod model and that are explainable
FEATURES_FOR_DATA_DRIFT = [
    "CODE_GENDER",
    "POS_MONTHS_BALANCE_SIZE",
    "DAYS_BIRTH",
    "NAME_FAMILY_STATUS_Married",
    "CNT_CHILDREN",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT"
    ]

DATA_PATH = "./data"

df_train = pd.read_csv(DATA_PATH + '/df_train.csv')
df_holdout = pd.read_csv(DATA_PATH + '/df_holdout.csv')
# NOTE: the holdout CSV is data that was never used/seen by the model
# therefore we pretend that it represents "new" data that has arrived in our pipeline

pretend_old_data = df_train[FEATURES_FOR_DATA_DRIFT]
pretend_new_data = df_holdout[FEATURES_FOR_DATA_DRIFT]

# Create the report
# https://www.evidentlyai.com/ml-in-production/data-drift
# https://docs.evidentlyai.com/presets/data-drift
evidently_report = Report(metrics=[DataDriftPreset()])

evidently_report.run(reference_data=pretend_old_data, current_data=pretend_new_data)

# Save locally
evidently_report.save_html("evidently_report_data_drift_example.html")