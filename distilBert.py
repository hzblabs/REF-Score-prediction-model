from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


dataset = load_dataset("json", data_files="ref_training_data.json", split="train")


label_map = {"4*": 0, "3*": 1, "2*": 2, "1*": 3}
dataset = dataset.map(lambda x: {"label": label_map.get(x["label"], -1)})
dataset = dataset.filter(lambda x: x["label"] != -1)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
dataset = dataset.map(tokenize, batched=True)


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)


training_args = TrainingArguments(
    output_dir="./ref_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=10,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)


trainer.train()


trainer.save_model("ref_star_classifier")
tokenizer.save_pretrained("ref_star_classifier")

print("✅ Model training complete.")
