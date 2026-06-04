"""
modelling.py (MLProject version)
Mendukung parameter CLI untuk dijalankan via MLflow Project & GitHub Actions.
"""

import os
import argparse
import logging
import warnings
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_dir):
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test  = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').squeeze()
    y_test  = pd.read_csv(f'{data_dir}/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def main(args):
    logger.info("=" * 60)
    logger.info("MLFLOW PROJECT – HEART DISEASE TRAINING")
    logger.info("=" * 60)

    # Gunakan env var jika ada (DagsHub/CI), fallback ke lokal
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Heart_Disease_CI")

    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    mlflow.sklearn.autolog(log_model_signatures=True,
                           log_input_examples=True)

    with mlflow.start_run(run_name="RF_CI_Run"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'test_accuracy' : accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall'   : recall_score(y_test, y_pred),
            'test_f1'       : f1_score(y_test, y_pred),
            'test_roc_auc'  : roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tag("triggered_by", "github_actions")
        mlflow.set_tag("dataset", "heart_disease_uci")

        # ─── Artefak: Confusion Matrix ───
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Sehat', 'Sakit'], yticklabels=['Sehat', 'Sakit'])
        ax.set_title('Confusion Matrix'); ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        plt.tight_layout(); plt.savefig('confusion_matrix_ci.png', dpi=100); plt.close()
        mlflow.log_artifact('confusion_matrix_ci.png', 'plots')

        # ─── Artefak: ROC Curve ───
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'AUC = {roc_auc_score(y_test, y_prob):.3f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
        ax.legend()
        plt.tight_layout(); plt.savefig('roc_curve_ci.png', dpi=100); plt.close()
        mlflow.log_artifact('roc_curve_ci.png', 'plots')

        # ─── Artefak: Classification Report ───
        cr = classification_report(y_test, y_pred,
                                    target_names=['Sehat', 'Sakit Jantung'],
                                    output_dict=True)
        with open('classification_report_ci.json', 'w') as f:
            json.dump(cr, f, indent=4)
        mlflow.log_artifact('classification_report_ci.json', 'reports')

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID    : {run_id}")
        logger.info(f"Accuracy  : {metrics['test_accuracy']:.4f}")
        logger.info(f"F1-Score  : {metrics['test_f1']:.4f}")
        logger.info(f"ROC-AUC   : {metrics['test_roc_auc']:.4f}")

    logger.info("Training selesai!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     type=str,   default='heart_disease_preprocessing')
    parser.add_argument('--n_estimators', type=int,   default=100)
    parser.add_argument('--max_depth',    type=int,   default=10)
    parser.add_argument('--learning_rate',type=float, default=0.1)
    args = parser.parse_args()
    main(args)
