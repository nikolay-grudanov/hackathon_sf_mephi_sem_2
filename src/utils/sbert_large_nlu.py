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
        enable_rocm: bool = True
    ):
        """
        Инициализация класса SbertLargeNLU.
        
        Args:
            model_name (str): Название модели для загрузки
            model_dir (str): Путь к директории для сохранения модели
            max_length (int): Максимальная длина текста
            enable_rocm (bool): Флаг для использования ROCm
        """
        self.device = self._get_device(enable_rocm)
        self.model_name = model_name
        self.max_length = max_length
        self.root_dir = self._get_root_dir()
        self.model_dir = self._init_model_dir(model_dir)
        self.tokenizer, self.model = self._load_model()

    def _get_device(self, enable_rocm: bool) -> str:
        """
        Получает устройство для работы модели.
        
        Args:
            enable_rocm (bool): Флаг для использования ROCm
            
        Returns:
            str: Устройство ('cuda' или 'cpu')
        """
        if enable_rocm and torch.cuda.is_available():
            # Отключаем проблемные бэкенды SDPA
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)  # Используем математический бэкенд
            os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TOKENIZERS_PARALLELISM'] = 'true'
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
            return 'cuda'
        return 'cpu'

    def _get_root_dir(self) -> Path:
        """
        Получает корневую директорию проекта.
        
        Returns:
            Path: Путь к корневой директории
        """
        try:
            current_file = Path(__file__).resolve()
            return current_file.parent.parent if "src" in current_file.parts else current_file.parent
        except NameError:
            return Path(os.getcwd()).resolve()

    def _init_model_dir(self, model_dir: str) -> Path:
        """
        Инициализирует директорию для сохранения модели.
        
        Args:
            model_dir (str): Путь к директории для сохранения модели
            
        Returns:
            Path: Путь к директории для сохранения модели
        """
        path = Path(model_dir) if model_dir else self.root_dir / "models/sbert"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_model(self):
        """
        Загружает модель и токенизатор.
        
        Returns:
            Tuple[PreTrainedTokenizer, PreTrainedModel]: Токенизатор и модель
        """
        local_path = self.model_dir / self.model_name.replace("/", "__")
        try:
            if (local_path / "config.json").exists():
                print(f"Loading model from local cache: {local_path}")
                tokenizer = AutoTokenizer.from_pretrained(local_path)
                model = AutoModel.from_pretrained(local_path)
            else:
                print(f"Downloading and saving model to: {local_path}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                model.save_pretrained(local_path, safe_serialization=True)
                tokenizer.save_pretrained(local_path)
                print(f"Model saved successfully at: {local_path}")
            
            model = model.to(self.device)
            print(f"Model loaded on device: {model.device}")
            return tokenizer, model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def create_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Создает эмбеддинги для текстов.
        
        Args:
            texts (Union[str, List[str]]): Тексты для создания эмбеддингов
            
        Returns:
            torch.Tensor: Тензор с эмбеддингами
        """
        if not texts:
            return torch.empty((0,))
            
        if isinstance(texts, str):
            texts = [texts]
            
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return torch.empty((0,))
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Пулинг на GPU
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # Конвертация в numpy с учетом устройства
        if embeddings.device.type == 'cuda':
            return embeddings.cpu().numpy()
        return embeddings.numpy()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """
        Вычисляет среднее значение эмбеддингов.
        
        Args:
            model_output: Выход модели
            attention_mask: Маска внимания
            
        Returns:
            torch.Tensor: Тензор со средним значением эмбеддингов
        """
        if attention_mask.sum() == 0:
            return torch.zeros((1, model_output.last_hidden_state.size(-1)))
            
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)