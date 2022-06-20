from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

import argparse
from utils.training import Dataset, compute_metrics
from utils.prepare_data import get_data

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model-name', default="roberta-large", dest='model_name')
parser.add_argument('--dataset-path', default="./data/1preproc.tsv", dest='dataset_path')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_path = args.dataset_path
    model_name = args.model_name

    x_train, y_train, x_val, y_val, data = get_data(dataset_path)
    #uncomment for smaller dataset trainign
    # split = 0.3
    # n = int(x_train.shape[0] * split)
    # exp_train_data = x_train[:n]
    # exp_train_target = y_train[:n]
    num_labels = len(set(y_train.tolist()))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    X_train_tokenized = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=128)
    X_val_tokenized = tokenizer(x_val.tolist(), padding=True, truncation=True, max_length=128)

    train_dataset = Dataset(X_train_tokenized, y_train.tolist())
    val_dataset = Dataset(X_val_tokenized, y_val.tolist())

    args = TrainingArguments(
        output_dir="output_sm",
        evaluation_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
        fp16=True
        # report_to="wandb",
      #  no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    res = trainer.evaluate()
    print(res)
