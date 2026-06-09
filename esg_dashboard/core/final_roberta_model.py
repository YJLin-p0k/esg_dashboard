from __future__ import annotations

import gc
from pathlib import Path
from typing import Any


ROBERTA_TASK_FIELDS = ["promise_status", "evidence_status"]

LABEL_LISTS = {
    "promise_status": ["Yes", "No"],
    "evidence_status": ["Yes", "No", "N/A"],
}

DEFAULT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL_DIR = Path("models") / "final_roberta_task13_A_stable_baseline"
DEFAULT_CHECKPOINT_NAMES = [
    "best_roberta_seed_42.pt",
    "best_roberta_seed_123.pt",
    "best_roberta_seed_2024.pt",
]


class FinalRoBERTaUnavailable(RuntimeError):
    pass


class FinalRoBERTaEnsemblePredictor:
    """Inference wrapper for the final notebook's RoBERTa task 1/3 ensemble."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        checkpoint_names: list[str] | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_len: int = DEFAULT_MAX_LEN,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.checkpoint_paths = [
            self.model_dir / name
            for name in (checkpoint_names or DEFAULT_CHECKPOINT_NAMES)
        ]
        self.tokenizer_dir = self.model_dir / "tokenizer"
        self.config_dir = self.model_dir / "base_model_config"
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self._loaded = False
        self._model_class: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._device: Any = None

    @property
    def is_available(self) -> bool:
        return all(path.exists() for path in self.checkpoint_paths)

    def predict(self, sentences: list[str]) -> list[dict[str, object]]:
        if not sentences:
            return []

        self._load_runtime()
        torch = self._torch
        assert torch is not None

        encoded = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        accumulated: dict[str, Any] = {
            task: torch.zeros((len(sentences), len(LABEL_LISTS[task])), dtype=torch.float32)
            for task in ROBERTA_TASK_FIELDS
        }
        with torch.inference_mode():
            for checkpoint_path in self.checkpoint_paths:
                model = self._load_checkpoint_model(checkpoint_path)
                for start in range(0, len(sentences), self.batch_size):
                    end = start + self.batch_size
                    input_ids = encoded["input_ids"][start:end].to(self._device)
                    attention_mask = encoded["attention_mask"][start:end].to(self._device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    for task in ROBERTA_TASK_FIELDS:
                        probs = torch.softmax(logits[task], dim=1).detach().cpu()
                        accumulated[task][start:end] += probs
                del model
                if str(self._device) == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        rows: list[dict[str, object]] = []
        for row_index in range(len(sentences)):
            row: dict[str, object] = {}
            confidences: list[float] = []
            for task in ROBERTA_TASK_FIELDS:
                probs = accumulated[task] / len(self.checkpoint_paths)
                label_id = int(torch.argmax(probs[row_index]).item())
                confidence = float(probs[row_index][label_id].item())
                row[task] = LABEL_LISTS[task][label_id]
                row[f"{task}_confidence"] = confidence
                confidences.append(confidence)
            row["confidence"] = min(confidences) if confidences else 0.0
            rows.append(row)

        return rows

    def _load_runtime(self) -> None:
        if self._loaded:
            return
        if not self.is_available:
            missing = [str(path) for path in self.checkpoint_paths if not path.exists()]
            raise FinalRoBERTaUnavailable(f"Missing final RoBERTa checkpoints: {missing}")

        try:
            import torch
            import torch.nn as nn
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise FinalRoBERTaUnavailable(
                "Install torch and transformers to use the final RoBERTa checkpoints."
            ) from exc

        class MultiTaskRoBERTa(nn.Module):
            def __init__(
                self,
                model_name: str,
                task_fields: list[str],
                label_lists: dict[str, list[str]],
                dropout_rate: float = 0.1,
                config_dir: Path | None = None,
            ) -> None:
                super().__init__()
                self.task_fields = task_fields
                if config_dir and config_dir.exists():
                    config = AutoConfig.from_pretrained(config_dir)
                    self.roberta = AutoModel.from_config(config)
                else:
                    self.roberta = AutoModel.from_pretrained(model_name)
                hidden_size = self.roberta.config.hidden_size
                self.dropout = nn.Dropout(dropout_rate)
                self.classifiers = nn.ModuleDict(
                    {
                        task: nn.Linear(hidden_size, len(label_lists[task]))
                        for task in task_fields
                    }
                )

            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
                return {
                    task: self.classifiers[task](cls_output)
                    for task in self.task_fields
                }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_source = self.tokenizer_dir if self.tokenizer_dir.exists() else self.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model_class = MultiTaskRoBERTa
        self._loaded = True

    def _load_checkpoint_model(self, checkpoint_path: Path) -> Any:
        torch = self._torch
        assert torch is not None
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        checkpoint_hp = checkpoint.get("hyperparameters", {})
        model = self._model_class(
            model_name=checkpoint.get("model_name", self.model_name),
            task_fields=checkpoint.get("roberta_task_fields", ROBERTA_TASK_FIELDS),
            label_lists=checkpoint.get("label_lists", LABEL_LISTS),
            dropout_rate=checkpoint_hp.get("DROPOUT_RATE", 0.1),
            config_dir=self.config_dir,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()
        return model
