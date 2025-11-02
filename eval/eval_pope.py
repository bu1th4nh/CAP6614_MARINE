import os
import sys
import json
import glob
import mlflow
import logging
import pymongo
import argparse
from s3fs import S3FileSystem
from typing import List
from sklearn.metrics import confusion_matrix
sys.path.append(os.getcwd())



# -----------------------------------------------------------------------------------------------
# General Configurations
# -----------------------------------------------------------------------------------------------
from log_config import initialize_logging
import requests
initialize_logging()
S3_ENDPOINT_URL       = os.environ["S3_ENDPOINT_URL"]
S3_ACCESS_KEY_ID      = os.environ["S3_ACCESS_KEY_ID"]
S3_SECRET_ACCESS_KEY  = os.environ["S3_SECRET_ACCESS_KEY"]
MONGO_ENDPOINT        = os.environ["MONGO_ENDPOINT"]
MONGO_PORT            = os.environ["MONGO_PORT"]
MONGO_USERNAME        = os.environ["MONGO_USERNAME"]
MONGO_PASSWORD        = os.environ["MONGO_PASSWORD"]
N8N_ENDPOINT_URL      = os.environ["N8N_ENDPOINT_URL"]
N8N_WEBHOOK_ID        = os.environ["N8N_WEBHOOK_ID"]

mongo = pymongo.MongoClient(
    host=MONGO_ENDPOINT,
    port=MONGO_PORT,
    username=MONGO_USERNAME,
    password=MONGO_PASSWORD,
)
s3 = S3FileSystem(
    anon=False, 
    endpoint_url=S3_ENDPOINT_URL,
    key=S3_ACCESS_KEY_ID,
    secret=S3_SECRET_ACCESS_KEY,
    use_ssl=False
)
storage_options = {
    'key': S3_ACCESS_KEY_ID,
    'secret': S3_SECRET_ACCESS_KEY,
    'endpoint_url': S3_ENDPOINT_URL,
}


def report_message_to_n8n(message: str, msg_type: str = "info"):
    try:
        response = requests.post(
            f"{N8N_ENDPOINT_URL}/{N8N_WEBHOOK_ID}",
            json={"message": message, "type": msg_type}
        )
        if response.status_code == 200:
            logging.info("Successfully reported message to n8n.")
        else:
            logging.error(f"Failed to report message to n8n. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Exception occurred while reporting message to n8n: {e}")

def put_file_to_s3(local_path: str, s3_path: str):
    try:
        s3.put(local_path, s3_path)
        logging.info(f"Successfully uploaded {local_path} to {s3_path}.")
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to {s3_path}. Exception: {e}")
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------



def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate POPE binary QA performance.")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--label_dir", type=str, default="./data/marine_qa/label/")
    parser.add_argument("--label_file", type=str, default="pope_coco_detr_th0.95_ram_th0.68.json")

    return parser.parse_args()


def load_labels(label_path: str) -> List[int]:
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    return [0 if item["label"] == "no" else 1 for item in label_data]


def load_predictions(answer_path: str, model: str) -> List[int]:
    with open(answer_path, 'r') as f:
        answers = [json.loads(line)["text"] for line in f]

    preds = []
    for ans in answers:
        if not ans.strip():
            preds.append(-1)
        else:
            words = ans.split('.')[0].split()
            preds.append(0 if any(word.lower() in {"no", "not"} for word in words) else 1)
    return preds, answers


def compute_metrics(labels: List[int], preds: List[int], answers: List[str], answer_file: str):
    filtered = [(l, p) for l, p in zip(labels, preds) if p != -1]
    if not filtered:
        raise ValueError("No valid predictions to evaluate.")

    label_list, pred_list = zip(*filtered)
    tn, fp, fn, tp = confusion_matrix(label_list, pred_list).ravel()
    yes_ratio = pred_list.count(1) / len(pred_list)
    avg_len = round(sum(len(ans.split()) for ans in answers) / len(answers), 2)

    return {
        "answer_file": answer_file,
        "questions_num": len(answers),
        "length_response": avg_len,
        "overall_metrics": {
            "Accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
            "Yes_ratio": round(yes_ratio, 4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Precision": round(tp / (tp + fp) if tp + fp > 0 else 0, 4),
            "Recall": round(tp / (tp + fn) if tp + fn > 0 else 0, 4),
            "Specificity": round(tn / (tn + fp) if tn + fp > 0 else 0, 4),
            "F1": round(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0, 4),
        }
    }


def save_results(save_path: str, results: dict):
    with open(save_path, 'a+') as f:
        json.dump(results, f)
        f.write('\n')
        put_file_to_s3(save_path, f"s3://results/CAP6614_MARINE/evaluation/pope/{os.path.basename(save_path)}")



def pope(args):
    label_path = os.path.join(args.label_dir, args.label_file.replace('.json', '_label.json'))
    answer_path = os.path.join(args.answer_dir, args.answer_file)

    logging.info(f"== Evaluating POPE for {args.answer_file} ==")

    labels = load_labels(label_path)
    preds, answers = load_predictions(answer_path, args.model)

    if len(labels) != len(answers):
        logging.error(f"[ERROR] Answer count ({len(answers)}) does not match label count ({len(labels)}).")
        return

    results = compute_metrics(labels, preds, answers, answer_path)

    logging.info(f"{'Accuracy':<10}{'F1':<10}{'Yes_ratio':<10}")
    logging.info(f"{results['overall_metrics']['Accuracy']:<10}{results['overall_metrics']['F1']:<10}{results['overall_metrics']['Yes_ratio']:<10}")

    save_results(args.save_file, results)



if __name__ == "__main__":
    
    args = get_parser()

    logging.info(f"Evaluation Directory: {args.eval_dir}")
    logging.info(f"Label Directory: {args.label_dir}")
    logging.info(f"Save Directory: {args.save_dir}")
    

    report_message_to_n8n(f"Starting POPE evaluation for files in {args.eval_dir}.")


    os.makedirs(args.save_dir, exist_ok=True)
    args.save_file = os.path.join(args.save_dir, "pope_eval.jsonl")

    eval_files = glob.glob(os.path.join(args.eval_dir, "*.jsonl")) + glob.glob(os.path.join(args.eval_dir, "*.json"))

    for file in eval_files:

        logging.info(f"Processing file: {file}")
        args.answer_file = os.path.basename(file)
        args.answer_dir = args.eval_dir
        pope(args)
