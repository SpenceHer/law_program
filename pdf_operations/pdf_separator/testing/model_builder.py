import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from IPython import embed


import joblib
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
from collections import Counter


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc



import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import joblib
from imblearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler



import optuna
from optuna.integration import XGBoostPruningCallback


################################################################################################################################

df_391 = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/dataframe_391.csv")
df_1519 = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/dataframe_1519.csv")
df_1277 = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/dataframe_1277.csv")
df_4524 = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/dataframe_4524.csv")
df_475 = pd.read_csv(r"/Users/spencersmith/Desktop/CODING/Projects/law/page_dataframes/dataframe_475.csv")


################################################################################################################################


page_list_1519 = [1, range(19, 133+1), 145, 146, 159, 186, 230, 232, range(237, 241+1), 252, 263, 277, 289, 301, 303, 314, 326, 338, 339, 340, 352, 368, 369, 370, 372, 375, 378, 381, 384, 385, 386, 
                  389, 410, 454, 458, 462, 465, 469, 473, 479, 484, 491, 498, 507, 515, 526, 535, 547, 548, 559, 573, 585, 597, 613, 624, 638, 650, 662, 664, 675, 687, 699, 700, 701, 
                  713, 729, 730, 744, 758, 776, 779, 781, 782, 785, 788, 791, 796, 801, 805, 810, 816, 822, 828, 835, 836, 837, 839, 841, 844, 845, 848, 849, 854, 859, 863, 869, 874, 
                  881, 882, 883, 884, 928, 931, 934, 938, 963, 964, 965, 966, 967, 978, 987, 999, 1000, 1011, 1025, 1037, 1049, 1065, 1079, 1093, 1111, 1114, 1129, 1149, 1165, 1187, 
                  1189, 1191, 1193, 1194, 1198, 1202, 1207, 1212, 1217, 1224, 1230, 1238, 1243, 1253, 1264, 1269, 1270, 1272, 1277, 1280, 1284, 1285, 1289, 1293, 1301, 1307, 1312, 1314, 1316, 
                  1317, 1319, 1322, 1326, 1330, 1334, 1338, 1344, 1345, 1346, 1348, 1350, 1353, 1356, 1360, 1364, 1370, 1375, 1376, 1379, 1386, 1393, 1395, 1404, 1413, 1416, 1417, 1419, 1422, 1425, 
                  1428, 1431, 1432, 1433, 1436, 1457, 1501, 1505, 1509, 1510, 1511, 1512, 1513, 1514, 1515]

for i in page_list_1519:
    if isinstance(i, int):
        df_1519.loc[df_1519["page_num"] == i, "start_page"] = 1
    else:
        for x in i:
            df_1519.loc[df_1519["page_num"] == x, "start_page"] = 1 

df_1519.loc[~df_1519["page_num"].isin(page_list_1519), "start_page"] = 0


page_list_1277 = [1, 2, 3, 5, 8, 12, 16, 20, 24, 29, 34, 35, 41, 47, 54, 61, 69, 81, 86, 92, 98, 105, 113, 122, 131, 134, 135, 136, 137, 139, 141, 142, 143, 144, 146, 148, 153, 160, 166, 172, 
                  179, 180, 181, 182, 190, 191, 192, 193, 195, 196, 198, 199, 200, 207, 214, 215, 218, 220, 223, 227, 230, 232, 233, 239, 248, 249, 251, 252, 254, 255, 256, 259, 262, 263, 264, 279, 
                  281, 283, 284, 285, 286, 287, 288, 290, 291, 293, 295, 296, 298, 299, 300, 315, 316, 317, 318, 321, 322, 323, range(325, 332+1), 333, range(337, 367+1), 369, 370, 372, 
                  373, 375, 376, 377, 380, 381, 383, 384, 385, 386, 401, 402, 405, 406, 407, 408, 409, 410, 412, 413, 415, 421, 423, 424, 425, 426, 428, 429, 431, 432, 434, 440, 442, 443, 
                  range(446, 454+1), range(458, 478+1), 480, 482, 483, 484, 485, 486, 488, 497, 501, 503, 512, range(516,529+1), 534, 535, 537, 552, 554, range(557,564+1), range(568,586+1), 
                  range(588, 594+1), 597, 606, 608, 609, 610, 612, 613, 614, 617, range(620, 628+1), range(632, 652+1), range(654, 715+1), 720, 725, 727, 728, 731, 735, range(738, 745+1), 
                  range(749, 768+1), range(771, 861+1), 863, 865, 868, 873, 874, 876, 878, 880, range(883, 890+1), range(893, 913+1), range(916, 1007+1), range(1010, 1016), range(1019, 1038), 
                  range(1042, 1161), 1163, 1165, 1167, 1169, 1171, 1173, 1175, 1180, 1186, 1187, 1190, 1191, 1192, 1201, 1203, 1204, 1206, 1209, 1210, 1214, 1217, 1220, 1224, 1225, 1226, 1231, 
                  1236, 1239, 1242, 1243, 1244, 1250, 1261, 1263, 1272]

for i in page_list_1277:
    if isinstance(i, int):
        df_1277.loc[df_1277["page_num"] == i, "start_page"] = 1
    else:
        for x in i:
            df_1277.loc[df_1277["page_num"] == x, "start_page"] = 1

df_1277.loc[~df_1277["page_num"].isin(page_list_1277), "start_page"] = 0

page_list_391 = [1, 5, 16, 20, 30, 68, 71, 73, 84, 87, 95, 97, 100, 103, 107, 111, 116, 120, 124, 126, 129, 132, 136, 140, 142, 145, 150, 151, 152, 155, 157, 161, 166, 171, 177, 183, 190, 198, 
                 206, 215, 225, 235, 246, 257, 269, 270, 271, 273, 275, 287, 300, 313, 316, 318, 320, 322, 324, 327, 330, 335, 340, 345, 350, 355, 360, 366, 372, 374, 376, 384, 386, 389]

for i in page_list_391:
    if isinstance(i, int):
        df_391.loc[df_391["page_num"] == i, "start_page"] = 1
    else:
        for x in i:
            df_391.loc[df_391["page_num"] == x, "start_page"] = 1

df_391.loc[~df_391["page_num"].isin(page_list_391), "start_page"] = 0

page_list_4524 = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 20, 25, 26, 27, 29, 36, 37, 38, 40, 41, 42, 43, 52, 53, 55, 56, 57, 62, 63, 64, 69, 70, 71, 72, 73, 74, 78, 80, 83, 84, 89, 90, 92, 94, 95, 111, 112, 
                  114, 116, 117, 118, 124, 126, 129, 136, 137, 139, 141, 142, 145, 148, 149, 150, 151, 152, 153, 154, 156, 159, 161, 162, 163, 166, 167, 170, 173, 177, 184, 190, 191, 192, 193, 194, 195, 199, 
                  202, 203, 206, 207, 208, 210, 212, 213, 214, 215, 216, 218, 219, 221, 223, 227, 230, 231, 233, 234, 236, 237, 241, 242, 243, 244, 245, 246, 259, 260, range(265, 285+1), range(292, 302+1), 
                  331, 335, 336, 337, 338, 342, 343, 344, 345, 347, 348, 349, 350, 351, 353, 354, 355, 356, 357, 358, 361, 363, 364, 365, 368, 369, 372, 374, 375, 376, 377, 378, 380, 381, 382, 383, 385, 
                  386, 387, 388, 389, 391, 392, 394, 395, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 409, 412, 413, 414, 415, 416, 418, 421, 422, 423, 426, 427, 429, 430, 433, range(441, 445+1), 
                  449, 450, 475, 488, 489, range(497,506+1), 549, 550, 551, 554, 555, 584, 585, 620, 706, 748, 787, 790, 842, 845, 847, 912, 969, 1037, 1093, 1096, 1097, 1100, 1174, 1177, 1228, 1230, 1233, 
                  1235, 1256, 1263, 1264, 1267, 1268, 1269, 1344, 1350, 1383, 1458, 1517, 1519, 1542, 1545, 1566, 1568, 1571, 1657, 1660, 1681, 1710, 1733, 1819, 1888, 1891, 1926, 1929, 1963, 1966, 1969, 
                  1972, 1974, 2061, 2110, 2112, 2156, 2201, 2203, 2205, 2209, 2212, 2323, 2326, 2368, 2371, 2409, 2410, 2412, 2415, 2531, 2534, 2571, 2610, 2662, 2680, 2698, 2700, 2827, 2830, 2833, 5879, 
                  2882, 2944, 2946, 3082, 3142, 3147, 3152, 3154, 3155, 3156, 3230, 3232, 3382, 3386, 3388, 3449, 3453, 3457, 3461, 3464, 3475, 3486, 3488, 3617, 3619, 3669, 3673, 3677, 3679, 3733, 3737, 
                  3741, 3743, 3805, 3807, 3911, 3964, 4012, 4072, 4195, 4261, 4326, 4394, 4496, 4497, 4498, 4505, 4513, 4521, 4522]

for i in page_list_4524:
    if isinstance(i, int):
        df_4524.loc[df_4524["page_num"] == i, "start_page"] = 1
    else:
        for x in i:
            df_4524.loc[df_4524["page_num"] == x, "start_page"] = 1

df_4524.loc[~df_4524["page_num"].isin(page_list_4524), "start_page"] = 0



page_list_475 = [1, 11, 17, 19, 29, 35, 37, 38, 39, 40, 44, 50, 56, 60, 64, 68, 74, 78, 82, 86, 90, 94, 98, 102, 106, 112, 116, 124, 132, 142, 148, 156, 162, 170, 174, 180, 188, 196, 204, 210, 216, 
                 220, 224, 228, 232, 236, 240, 244, 250, 256, 260, 268, 276, 282, 284, 286, 288, 290, 300, 305, 306, 308, 310, 312, 313, 316, 317, 320, 321, 325, 326, 327, 329, 331, 332, 333, 337, 
                 338, 340, 341, 354, 355, 357, 358, 359, 360, 361, 365, 369, 373, 377, 381, 383, 386, 387, 389, 391, 392, 393, 394, 395, 396, 400, 404, 406, 408, 410, 411, 413, 415, 417, 418, 420, 
                 422, 423, 427, 431, 432, 435, 436, 437, 438, 440, 441, 443, 444, 445, 446, 447, 448, 449, 453, 455, 457, 459, 461, 463, 464, 466, 467, 468, 469, 471, 473]

for i in page_list_475:
    if isinstance(i, int):
        df_475.loc[df_475["page_num"] == i, "start_page"] = 1
    else:
        for x in i:
            df_475.loc[df_475["page_num"] == x, "start_page"] = 1

df_475.loc[~df_475["page_num"].isin(page_list_475), "start_page"] = 0





df_merged = pd.concat([df_1519, df_1277, df_391, df_4524, df_475])

df = df_merged.copy()
df = df.reset_index(drop=True)




target = "start_page"

features = [
    "word_count",
    "rows_of_text_count",
    "page_width",
    "page_height",
    "average_text_size",
    "text_density",
    "first_line_vertical_position",
    "vertical_margin_ratio",
    "horizontal_margin_ratio",
    "number_of_entities",
    "number_of_unique_entities",
    "whitespace_ratio",
    "has_images_or_tables",
    "average_x_position",
    "avg_x_first_10",
    "avg_x_last_10",
    "line_spacing",
    "has_page_1",
    "has_page_x",
    "has_page_x_of_x",
    "has_page_x_of_x_end",


    "next_page_width",
    "next_page_height",
    "next_page_width_diff",
    "next_page_height_diff",
    "next_page_vertical_margin_ratio",
    "next_page_horizontal_margin_ratio",
    "next_page_vertical_margin_ratio_diff",
    "next_page_horizontal_margin_ratio_diff",
    "next_page_whitespace_ratio",
    "next_page_has_images_or_tables",
    "next_page_average_x_position",
    "next_page_avg_x_first_10",
    "next_page_avg_x_last_10",
    "next_page_whitespace_ratio_diff",
    "next_page_has_images_or_tables_diff",
    "next_page_average_x_position_diff",
    "next_page_avg_x_first_10_diff",
    "next_page_avg_x_last_10_diff",
    "next_page_word_count",
    "next_page_first_line_vertical_position",
    "next_page_text_density",
    "next_page_average_text_size",
    "next_page_rows_of_text_count",
    "next_page_word_count_diff",
    "next_page_first_line_vertical_position_diff",
    "next_page_text_density_diff",
    "next_page_average_text_size_diff",
    "next_page_rows_of_text_count_diff",
    "next_page_bert_text_similarity",
    "next_page_idf_text_similarity",
    "next_page_header_similarity",
    "next_page_number_of_entities",
    "next_page_number_of_unique_entities",
    "next_page_shared_entities",
    "next_page_shared_entities_total",
    "next_page_has_page_1",
    "next_page_has_page_x",
    "next_page_has_page_x_of_x",
    "next_page_has_page_x_of_x_end",



    "prev_page_width",
    "prev_page_height",
    "prev_page_width_diff",
    "prev_page_height_diff",
    "prev_page_vertical_margin_ratio",
    "prev_page_horizontal_margin_ratio",
    "prev_page_vertical_margin_ratio_diff",
    "prev_page_horizontal_margin_ratio_diff",
    "prev_page_whitespace_ratio",
    "prev_page_has_images_or_tables",
    "prev_page_average_x_position",
    "prev_page_avg_x_first_10",
    "prev_page_avg_x_last_10",
    "prev_page_whitespace_ratio_diff",
    "prev_page_has_images_or_tables_diff",
    "prev_page_average_x_position_diff",
    "prev_page_avg_x_first_10_diff",
    "prev_page_avg_x_last_10_diff",
    "prev_page_word_count",
    "prev_page_first_line_vertical_position",
    "prev_page_text_density",
    "prev_page_average_text_size",
    "prev_page_rows_of_text_count",
    "prev_page_word_count_diff",
    "prev_page_first_line_vertical_position_diff",
    "prev_page_text_density_diff",
    "prev_page_average_text_size_diff",
    "prev_page_rows_of_text_count_diff",
    "prev_page_bert_text_similarity",
    "prev_page_idf_text_similarity",
    "prev_page_header_similarity",
    "prev_page_number_of_entities",
    "prev_page_number_of_unique_entities",
    "prev_page_shared_entities",
    "prev_page_shared_entities_total",
    "prev_page_has_page_1",
    "prev_page_has_page_x",
    "prev_page_has_page_x_of_x",
    "prev_page_has_page_x_of_x_end"


]











# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standard scaling
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Define the objective function for Optuna
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    model = xgb.XGBClassifier(**param, early_stopping_rounds=10)
    model.fit(X_train_res_scaled, y_train_res, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    preds = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, preds)
    return accuracy

# Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print best trial
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Train the final model with the best parameters
best_params = study.best_trial.params
best_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    **best_params
)

best_model.fit(X_train_res_scaled, y_train_res)

# Evaluate the model
y_pred = best_model.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(best_model, X_train_res_scaled, y_train_res, cv=5)
print(f"Cross-validation scores: {scores}")

# Feature importance
xgb.plot_importance(best_model)
plt.show()

# Save the model
joblib.dump(best_model, 'best_xgboost_model.pkl')