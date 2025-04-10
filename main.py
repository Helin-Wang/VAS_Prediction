from models.train import adaboost, ann, decision_tree, knn, logistic_regression, rf, svm, xgb
from utils.eval import eval_model
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "example_dataset.xlsx"  # 替换为实际的文件路径
df = pd.read_excel(file_path)
df.fillna(0, inplace=True)

input_columns = [
    'age', 'gender', 'height', 'weight', 'BMI', '高血压', '高血脂', '糖尿病', '肾病', '睡眠障碍',
    '吸烟史', '饮酒史', '病程', '影像学检查', 'Q角', '胫股关节间隙外侧（mm）', '胫股关节间隙内测（mm）',
    '胫股关节间隙外/内比值', '治疗前膝关节活动度（°）', '治疗前膝关节肿胀', '治疗前膝髌周压痛',
    '治疗前膝内侧副韧带压痛', '治疗前膝外侧副韧带压痛', '治疗前侧方挤压试验', '治疗前麦氏征',
    '治疗前抽屉试验', '治疗前浮髌试验', '治疗前研磨试验', '治疗前VAS评分'
]
target_column = 'VAS差值' 
# target_column = '治疗后VAS评分'

X = df[input_columns]
y = df[target_column]
y = y.apply(lambda x: 0 if x <= 3 else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_dict = {
    'adaboost': adaboost,
    'ann': ann,
    'decision_tree': decision_tree,
    'knn': knn,
    'logistic_regression': logistic_regression,
    'random_forest': rf,
    'svm': svm, 
    'xgboost': xgb
}
acc_list = []
f1_list = []
precision_list = []
recall_list = []
auc_list = []
model_names = []
for model_name, model in model_dict.items():
    best_model = model(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]  

    accuracy, f1, precision, recall, auc = eval_model(y_test, y_pred, y_pred_prob)
    model_names.append(model_name)
    acc_list.append(accuracy)
    f1_list.append(f1)
    precision_list.append(precision)
    recall_list.append(recall)
    auc_list.append(auc)
    
result_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': acc_list,
    'F1 Score': f1_list,
    'Precision': precision_list,
    'Recall': recall_list,
    'AUC': auc_list
})
result_df.to_csv("results.csv", index=False)