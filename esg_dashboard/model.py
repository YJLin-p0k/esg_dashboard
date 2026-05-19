from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


LABELS = ["環境", "社會", "治理", "其他"]

ENV_KEYWORDS = ["碳", "排放", "能源", "氣候", "廢棄物", "水資源", "污染", "淨零", "再生能源", "減碳"]
SOC_KEYWORDS = ["員工", "人權", "職安", "安全", "供應鏈", "社區", "多元", "平等", "童工", "強迫勞動"]
GOV_KEYWORDS = ["董事", "治理", "稽核", "法遵", "風險", "反貪腐", "揭露", "股東", "薪酬", "內控"]
COMMITMENT_KEYWORDS = ["承諾", "保證", "確保", "將", "目標", "達成", "實現", "完成", "預計", "必須"]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    risk_score: float


class ESGSentenceModel(nn.Module):
    """Small PyTorch model over deterministic sentence features."""

    def __init__(self, input_dim: int = 10, num_labels: int = len(LABELS)) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_labels)
        self.risk_head = nn.Linear(input_dim, 1)
        self._init_default_weights()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(features)
        risk = torch.sigmoid(self.risk_head(features)).squeeze(-1)
        return logits, risk

    def _init_default_weights(self) -> None:
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias[:] = torch.tensor([-0.2, -0.2, -0.2, 0.1])
            self.classifier.weight[0, 0] = 2.4
            self.classifier.weight[0, 4] = 0.5
            self.classifier.weight[1, 1] = 2.4
            self.classifier.weight[1, 4] = 0.5
            self.classifier.weight[2, 2] = 2.4
            self.classifier.weight[2, 4] = 0.5
            self.classifier.weight[3, 3] = 1.8
            self.classifier.weight[3, 8] = -0.4

            self.risk_head.weight.zero_()
            self.risk_head.bias[:] = torch.tensor([-2.1])
            self.risk_head.weight[0, 4] = 1.4
            self.risk_head.weight[0, 5] = 1.0
            self.risk_head.weight[0, 6] = 0.8
            self.risk_head.weight[0, 7] = 0.8
            self.risk_head.weight[0, 8] = 0.5


class ESGTorchClassifier:
    """Run ESG sentence classification through a PyTorch model."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.model = ESGSentenceModel()
        path = Path(checkpoint_path) if checkpoint_path else Path("models/esg_classifier.pt")
        if path.exists():
            state_dict = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, sentences: list[str]) -> list[Prediction]:
        if not sentences:
            return []

        features = torch.stack([self._features(sentence) for sentence in sentences])
        logits, risks = self.model(features)
        probabilities = F.softmax(logits, dim=-1)
        confidences, indices = probabilities.max(dim=-1)

        return [
            Prediction(label=LABELS[index.item()], confidence=confidence.item(), risk_score=risk.item())
            for index, confidence, risk in zip(indices, confidences, risks)
        ]

    def _features(self, sentence: str) -> torch.Tensor:
        length = max(len(sentence), 1)
        env = _keyword_score(sentence, ENV_KEYWORDS)
        soc = _keyword_score(sentence, SOC_KEYWORDS)
        gov = _keyword_score(sentence, GOV_KEYWORDS)
        commitment = _keyword_score(sentence, COMMITMENT_KEYWORDS)
        has_year = 1.0 if any(str(year) in sentence for year in range(2020, 2051)) else 0.0
        has_percent = 1.0 if "%" in sentence or "％" in sentence else 0.0
        has_number = 1.0 if any(char.isdigit() for char in sentence) else 0.0
        esg_total = min(env + soc + gov, 3.0) / 3.0
        normalized_length = min(length / 120.0, 1.0)
        other_signal = 1.0 if esg_total == 0 else 0.0

        return torch.tensor(
            [
                env,
                soc,
                gov,
                other_signal,
                commitment,
                has_year,
                has_percent,
                has_number,
                esg_total,
                normalized_length,
            ],
            dtype=torch.float32,
        )


def _keyword_score(sentence: str, keywords: list[str]) -> float:
    hits = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
    return min(hits / 2.0, 1.0)

