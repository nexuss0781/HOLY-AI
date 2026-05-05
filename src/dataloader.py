"""
High-performance DataLoader with enhanced Parquet support.
Automatically ingests all supported files from data/ directory.
"""

import os
import glob
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Iterator
import itertools

import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import PreTrainedTokenizer
import torch


class AutoIngestDataset(Dataset):
    """
    A unified dataset that automatically ingests multiple file formats from a directory.
    Supports: .parquet, .json, .jsonl, .csv, .txt
    """
    
    SUPPORTED_EXTENSIONS = {'.parquet', '.json', '.jsonl', '.csv', '.txt'}
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        text_column: Optional[str] = None,
        preprocessing_num_workers: int = 1,
    ):
        self.data_dir = Path(data_dir).resolve()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        if not self.data_dir.exists():
            # Try to create it if it doesn't exist
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created data directory: {self.data_dir}")
            except Exception as e:
                raise FileNotFoundError(f"Data directory {self.data_dir} does not exist and cannot be created: {e}")
        
        # Discover all files
        self.files = self._discover_files()
        print(f"Discovered {len(self.files)} files to process")
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"No supported data files found in {self.data_dir}. Supported formats: .parquet, .json, .jsonl, .csv, .txt")
        
        # Load all data
        self.data = self._load_all_data()
        print(f"Loaded {len(self.data)} total samples")
        
        # Tokenize all data
        self.tokenized_data = self._tokenize_all(preprocessing_num_workers)
        
    def _discover_files(self) -> List[Path]:
        """Discover all supported files in the data directory recursively."""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = f"**/*{ext}"
            matched = list(self.data_dir.glob(pattern))
            files.extend(matched)
            print(f"Found {len(matched)} files with extension {ext}")
        return sorted(files)
    
    def _load_file(self, file_path: Path) -> List[str]:
        """Load a single file and extract text content."""
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.parquet':
                return self._load_parquet(file_path)
            elif ext in ['.json', '.jsonl']:
                return self._load_json(file_path)
            elif ext == '.csv':
                return self._load_csv(file_path)
            elif ext == '.txt':
                return self._load_txt(file_path)
            else:
                print(f"Warning: Unsupported file type {ext}, skipping {file_path}")
                return []
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def _load_parquet(self, file_path: Path) -> List[str]:
        """Load parquet file with intelligent column detection."""
        df = pd.read_parquet(file_path)
        
        # Determine text column
        text_col = self._find_text_column(df)
        
        if text_col is None:
            print(f"Warning: No suitable text column found in {file_path}")
            return []
        
        texts = df[text_col].dropna().astype(str).tolist()
        print(f"Loaded {len(texts)} samples from {file_path.name} (column: {text_col})")
        return texts
    
    def _load_json(self, file_path: Path) -> List[str]:
        """Load JSON/JSONL file."""
        if file_path.suffix.lower() == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        else:
            df = pd.read_json(file_path)
        
        text_col = self._find_text_column(df)
        if text_col is None:
            return []
        
        texts = df[text_col].dropna().astype(str).tolist()
        print(f"Loaded {len(texts)} samples from {file_path.name}")
        return texts
    
    def _load_csv(self, file_path: Path) -> List[str]:
        """Load CSV file."""
        df = pd.read_csv(file_path)
        text_col = self._find_text_column(df)
        if text_col is None:
            return []
        
        texts = df[text_col].dropna().astype(str).tolist()
        print(f"Loaded {len(texts)} samples from {file_path.name}")
        return texts
    
    def _load_txt(self, file_path: Path) -> List[str]:
        """Load plain text file (each line is a sample)."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} samples from {file_path.name}")
        return texts
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """Intelligently find the best text column in a DataFrame."""
        if self.text_column and self.text_column in df.columns:
            return self.text_column
        
        # Common text column names
        priority_columns = ['text', 'content', 'body', 'passage', 'sentence', 
                           'paragraph', 'article', 'title', 'description']
        
        for col in priority_columns:
            if col in df.columns:
                return col
        
        # Find first string column
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it has reasonable text content
                sample = df[col].dropna().head(10)
                if len(sample) > 0 and all(isinstance(x, str) for x in sample):
                    return col
        
        return None
    
    def _load_all_data(self) -> List[str]:
        """Load data from all discovered files."""
        all_texts = []
        for file_path in self.files:
            texts = self._load_file(file_path)
            all_texts.extend(texts)
        return all_texts
    
    def _tokenize_all(self, num_workers: int = 1) -> List[torch.Tensor]:
        """Tokenize all text data with fixed length."""
        print(f"Tokenizing {len(self.data)} samples...")
        
        tokenized = []
        for i, text in enumerate(self.data):
            if i % 1000 == 0 and i > 0:
                print(f"Tokenized {i}/{len(self.data)} samples")
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',  # Pad to max_length for consistent batching
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            if len(input_ids) > 0:
                tokenized.append(input_ids)
        
        print(f"Tokenization complete: {len(tokenized)} samples")
        return tokenized
    
    def __len__(self) -> int:
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.tokenized_data[idx]
        return {
            'input_ids': item,
            'labels': item.clone(),
            'attention_mask': torch.ones_like(item)  # Add attention mask for padded sequences
        }


class StreamingParquetDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large parquet files.
    Processes files on-the-fly without loading everything into memory.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        text_column: Optional[str] = None,
        buffer_size: int = 10000,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.buffer_size = buffer_size
        
        self.files = self._discover_files()
        print(f"Streaming mode: Found {len(self.files)} files")
    
    def _discover_files(self) -> List[Path]:
        """Discover all parquet files."""
        files = []
        for ext in ['.parquet', '.json', '.jsonl', '.csv', '.txt']:
            files.extend(self.data_dir.glob(f"**/*{ext}"))
        return sorted(files)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            file_iter = iter(self.files)
        else:
            # Split files across workers
            files_per_worker = self.files[worker_info.id::worker_info.num_workers]
            file_iter = iter(files_per_worker)
        
        for file_path in file_iter:
            yield from self._stream_file(file_path)
    
    def _stream_file(self, file_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream samples from a single file."""
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.parquet':
                yield from self._stream_parquet(file_path)
            elif ext in ['.json', '.jsonl']:
                yield from self._stream_json(file_path)
            elif ext == '.csv':
                yield from self._stream_csv(file_path)
            elif ext == '.txt':
                yield from self._stream_txt(file_path)
        except Exception as e:
            print(f"Error streaming {file_path}: {e}")
    
    def _stream_parquet(self, file_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream from parquet file in chunks."""
        parquet_file = pq.ParquetFile(file_path)
        
        text_col = None
        schema = parquet_file.schema_arrow
        if self.text_column and self.text_column in schema.names:
            text_col = self.text_column
        else:
            priority_cols = ['text', 'content', 'body', 'passage']
            for col in priority_cols:
                if col in schema.names:
                    text_col = col
                    break
            if text_col is None:
                for col in schema.names:
                    text_col = col
                    break
        
        if text_col is None:
            return
        
        for batch in parquet_file.iter_batches(batch_size=self.buffer_size):
            df = batch.to_pandas()
            texts = df[text_col].dropna().astype(str).tolist()
            
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0)
                if len(input_ids) > 0:
                    yield {
                        'input_ids': input_ids,
                        'labels': input_ids.clone()
                    }
    
    def _stream_json(self, file_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream from JSON/JSONL file."""
        if file_path.suffix.lower() == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        import json
                        data = json.loads(line)
                        text = data.get('text', data.get('content', str(data)))
                        yield self._tokenize_text(text)
                    except:
                        continue
        else:
            df = pd.read_json(file_path, chunksize=self.buffer_size)
            for chunk in df:
                text_col = 'text' if 'text' in chunk.columns else chunk.columns[0]
                for text in chunk[text_col].dropna().astype(str):
                    yield self._tokenize_text(text)
    
    def _stream_csv(self, file_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream from CSV file."""
        df = pd.read_csv(file_path, chunksize=self.buffer_size)
        for chunk in df:
            text_col = 'text' if 'text' in chunk.columns else chunk.columns[0]
            for text in chunk[text_col].dropna().astype(str):
                yield self._tokenize_text(text)
    
    def _stream_txt(self, file_path: Path) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream from text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text:
                    yield self._tokenize_text(text)
    
    def _tokenize_text(self, text: str) -> Optional[Dict[str, torch.Tensor]]:
        """Tokenize a single text."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        if len(input_ids) > 0:
            return {
                'input_ids': input_ids,
                'labels': input_ids.clone()
            }
        return None


def create_dataloader(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    streaming: bool = False,
    num_workers: int = 4,
    text_column: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with automatic format detection.
    
    Args:
        data_dir: Directory containing data files
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        streaming: If True, use streaming mode for large datasets
        num_workers: Number of data loading workers
        text_column: Specific column name to use for text (optional)
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        Configured DataLoader
    """
    if streaming:
        dataset = StreamingParquetDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=text_column,
        )
    else:
        dataset = AutoIngestDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length,
            text_column=text_column,
        )
    
    # Collate function for variable length sequences - optimized for CPU
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad to max length in batch (more efficient than fixed length)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        attention_mask = input_ids != tokenizer.pad_token_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,  # 0 for single-process on 2-core CPU
        'collate_fn': collate_fn,
        'pin_memory': False,  # No benefit on CPU
        'persistent_workers': False if num_workers == 0 else (num_workers > 0),
        'prefetch_factor': 2 if num_workers > 0 else None,
        **kwargs
    }
    
    if streaming:
        loader_kwargs['persistent_workers'] = False
    
    return DataLoader(dataset, **loader_kwargs)


if __name__ == "__main__":
    # Test the dataloader
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test with data directory
    data_dir = "data"
    
    print("\n=== Testing AutoIngestDataset ===")
    dataset = AutoIngestDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=512
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512,
        num_workers=0
    )
    
    print("\n=== Testing DataLoader ===")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        if i >= 2:
            break
    
    print("\n✓ DataLoader test passed!")
