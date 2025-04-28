import os
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from typing import List, Union

class SbertLargeNLU:
    def __init__(
        self,
        model_name: str = 'sberbank-ai/sbert_large_nlu_ru',
        model_dir: str = None,
        max_length: int = 512,
        enable_rocm_experimental: bool = True
    ):
        self.device = self._get_device(enable_rocm_experimental)
        self.model_name = model_name
        self.max_length = max_length
        
        # Определение корневой директории проекта
        try:
            current_file = Path(__file__).resolve()
            self.root_dir = current_file.parent.parent if "src" in current_file.parts else current_file.parent
        except NameError:
            self.root_dir = Path(os.getcwd()).resolve()
        
        # Установка пути для моделей
        self.model_dir = Path(model_dir) if model_dir else self.root_dir / "src/models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Загрузка модели и токенизатора
        self.tokenizer, self.model = self._load_model()

    def _get_device(self, enable_rocm: bool) -> str:
        """Автоматически определяет доступное устройство с поддержкой ROCm."""
        if torch.cuda.is_available():
            if enable_rocm and 'rocm' in torch.version.hip:
                os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
            return 'cuda'
        return 'cpu'

    def _load_model(self) -> tuple:
        """Загрузка модели с автоматическим выбором устройства."""
        local_path = self.model_dir / self.model_name.replace('/', '_')
        
        if (local_path / 'pytorch_model.bin').exists():
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            model = AutoModel.from_pretrained(local_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            
        return tokenizer, model.to(self.device)

    def create_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Создает эмбеддинги на доступном устройстве (GPU/CPU).
        """
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return self._mean_pooling(outputs, inputs['attention_mask'])

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Усреднение эмбеддингов с учетом маски."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
