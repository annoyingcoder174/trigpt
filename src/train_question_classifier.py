# src/train_question_classifier.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from pathlib import Path


LABELS = ["summary", "explain", "compare", "definition", "other"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}


@dataclass
class Example:
    text: str
    label: str


def build_examples() -> List[Example]:
    """
    Synthetic dataset for question-type classification.
    """
    data: List[Example] = [
        # -------- summary (15) --------
        Example("Summarize this document for me.", "summary"),
        Example("Give me a brief summary of chapter 2.", "summary"),
        Example("Can you summarize the key ideas in this paper?", "summary"),
        Example("Short overview of this article?", "summary"),
        Example("What are the main points in this section?", "summary"),
        Example("Summarize the Bugs.pdf file in a few sentences.", "summary"),
        Example("I need a quick summary of this report.", "summary"),
        Example("Summarize the important parts only.", "summary"),
        Example("Can you give me a TL;DR of this document?", "summary"),
        Example("Write a short summary of the introduction.", "summary"),
        Example("Summarize this content in bullet points.", "summary"),
        Example("Give me a concise summary I can put into my notes.", "summary"),
        Example("What’s the overall message of this PDF?", "summary"),
        Example("Can you recap what this text is saying?", "summary"),
        Example("Summarize the conclusion section.", "summary"),

        # -------- explain (15) --------
        Example("Explain overfitting in simple terms.", "explain"),
        Example("Teach me how attention works in transformers.", "explain"),
        Example("Explain this like I'm 12: backpropagation.", "explain"),
        Example("Help me understand what regularization is.", "explain"),
        Example("Explain gradient descent step by step.", "explain"),
        Example("Explain what a learning rate is in training.", "explain"),
        Example("Can you explain this paragraph in easier words?", "explain"),
        Example("Explain what this formula means.", "explain"),
        Example("Explain the difference between precision and recall in a simple way.", "explain"),
        Example("Walk me through how this algorithm works.", "explain"),
        Example("Explain why this method is used here.", "explain"),
        Example("Explain the idea behind this theorem.", "explain"),
        Example("Break down this concept into simple pieces for me.", "explain"),
        Example("Explain what the author is trying to say in this sentence.", "explain"),
        Example("Explain this part of the document with an example.", "explain"),

        # -------- compare (15) --------
        Example("Compare supervised and unsupervised learning.", "compare"),
        Example("What is the difference between SVM and logistic regression?", "compare"),
        Example("Compare Python and Java for backend development.", "compare"),
        Example("How does RAG differ from fine-tuning?", "compare"),
        Example("Pros and cons of CNNs vs transformers?", "compare"),
        Example("Compare this method with the one in the previous section.", "compare"),
        Example("How is this algorithm different from the baseline?", "compare"),
        Example("Compare the approaches described in chapter 1 and chapter 3.", "compare"),
        Example("What’s the difference between accuracy and F1 score?", "compare"),
        Example("Compare this year’s results with last year’s.", "compare"),
        Example("How does this model compare to a simple linear model?", "compare"),
        Example("Compare the advantages and disadvantages of both options.", "compare"),
        Example("How does this technique differ from traditional methods?", "compare"),
        Example("Compare online learning and batch learning.", "compare"),
        Example("What are the main differences between the two definitions here?", "compare"),

        # -------- definition (15) --------
        Example("What is a convolutional neural network?", "definition"),
        Example("Define reinforcement learning.", "definition"),
        Example("What does 'epoch' mean in training?", "definition"),
        Example("What is dropout in neural networks?", "definition"),
        Example("What is a vector database?", "definition"),
        Example("What does 'overfitting' mean?", "definition"),
        Example("What is a loss function?", "definition"),
        Example("What is a gradient in machine learning?", "definition"),
        Example("What does 'embedding' mean in NLP?", "definition"),
        Example("What is a transformer model?", "definition"),
        Example("Define precision and recall.", "definition"),
        Example("What is a hyperparameter?", "definition"),
        Example("What does 'regularization' mean?", "definition"),
        Example("What is a dataset?", "definition"),
        Example("What is a confusion matrix?", "definition"),

        # -------- other (15) --------
        Example("Write a short email to my teacher.", "other"),
        Example("Generate a study schedule for my exams.", "other"),
        Example("Fix this Python bug for me.", "other"),
        Example("Create a list of practice problems about calculus.", "other"),
        Example("Help me brainstorm project ideas.", "other"),
        Example("Write a motivational message to myself.", "other"),
        Example("Draft a polite message to decline an invitation.", "other"),
        Example("Generate a to-do list for improving my CV.", "other"),
        Example("Rewrite this paragraph to sound more formal.", "other"),
        Example("Turn this text into a bullet-point checklist.", "other"),
        Example("Help me come up with interview questions.", "other"),
        Example("Write a short description for my GitHub project.", "other"),
        Example("Create a summary of my skills for my CV header.", "other"),
        Example("Suggest some titles for this report.", "other"),
        Example("Help me rephrase this sentence so it sounds better.", "other"),
    ]
    return data

    """
    Tiny synthetic dataset to start with.
    You can expand this list later with more variations.
    """
    data: List[Example] = [
        # summary
        Example("Summarize this document for me.", "summary"),
        Example("Give me a brief summary of chapter 2.", "summary"),
Example("Can you summarize the key ideas in this paper?", "summary"),
Example("Short overview of this article?", "summary"),
Example("What are the main points in this section?", "summary"),
Example("Summarize the Bugs.pdf file in a few sentences.", "summary"),
Example("I need a quick summary of this report.", "summary"),
Example("Summarize the important parts only.", "summary"),
Example("Can you give me a TL;DR of this document?", "summary"),
Example("Write a short summary of the introduction.", "summary"),
Example("Summarize this content in bullet points.", "summary"),
Example("Give me a concise summary I can put into my notes.", "summary"),
Example("What’s the overall message of this PDF?", "summary"),
Example("Can you recap what this text is saying?", "summary"),
Example("Summarize the conclusion section.", "summary"),

        # explain
Example("Explain overfitting in simple terms.", "explain"),
Example("Teach me how attention works in transformers.", "explain"),
Example("Explain this like I'm 12: backpropagation.", "explain"),
Example("Help me understand what regularization is.", "explain"),
Example("Explain gradient descent step by step.", "explain"),
Example("Explain what a learning rate is in training.", "explain"),
Example("Can you explain this paragraph in easier words?", "explain"),
Example("Explain what this formula means.", "explain"),
Example("Explain the difference between precision and recall in a simple way.", "explain"),
Example("Walk me through how this algorithm works.", "explain"),
Example("Explain why this method is used here.", "explain"),
Example("Explain the idea behind this theorem.", "explain"),
Example("Break down this concept into simple pieces for me.", "explain"),
Example("Explain what the author is trying to say in this sentence.", "explain"),
Example("Explain this part of the document with an example.", "explain"),


        # compare
        Example("Compare supervised and unsupervised learning.", "compare"),
        Example("What is the difference between SVM and logistic regression?", "compare"),
        Example("Compare Python and Java for backend development.", "compare"),
        Example("How does RAG differ from fine-tuning?", "compare"),
        Example("Pros and cons of CNNs vs transformers?", "compare"),

        # definition
        Example("What is a convolutional neural network?", "definition"),
        Example("Define reinforcement learning.", "definition"),
        Example("What does 'epoch' mean in training?", "definition"),
        Example("What is dropout in neural networks?", "definition"),
        Example("What is a vector database?", "definition"),

        # other
        Example("Write a short email to my teacher.", "other"),
        Example("Generate a study schedule for exams.", "other"),
        Example("Fix this Python bug for me.", "other"),
        Example("Create a list of practice problems about calculus.", "other"),
        Example("Help me brainstorm project ideas.", "other"),
    ]
    return data


def make_hf_dataset(examples: List[Example]) -> Dataset:
    texts = [ex.text for ex in examples]
    labels = [label2id[ex.label] for ex in examples]
    return Dataset.from_dict({"text": texts, "label": labels})


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def main():
    model_name = "distilbert-base-uncased"  # small and fast

    examples = build_examples()
    hf_dataset = make_hf_dataset(examples)

    # simple train / val split
    hf_dataset = hf_dataset.shuffle(seed=42)
    train_test = hf_dataset.train_test_split(test_size=0.2)
    train_ds = train_test["train"]
    val_ds = train_test["test"]  # not used for metrics now, just for possible future use

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = Path("models/question_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        # we could add eval_dataset later if your transformers version supports all options
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Also save labels mapping
    labels_path = output_dir / "labels.txt"
    with labels_path.open("w") as f:
        for label in LABELS:
            f.write(label + "\n")

    print("\n✅ Training complete. Saved to", output_dir)


if __name__ == "__main__":
    main()
