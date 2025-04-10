from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve
)

def eval_model(y_test, y_pred, y_pred_prob):
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    # 打印评估结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
