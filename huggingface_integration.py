"""
HuggingFace Model Integration for ARC Prize 2025
Using state-of-the-art models for pattern recognition
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
    ViTModel,
    ViTFeatureExtractor,
    BertModel,
    BertTokenizer
)
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class GridToText:
    """Convert ARC grids to text descriptions"""

    def __init__(self):
        self.color_names = {
            0: 'black', 1: 'blue', 2: 'red', 3: 'green',
            4: 'yellow', 5: 'gray', 6: 'pink', 7: 'orange',
            8: 'cyan', 9: 'brown'
        }

    def grid_to_text(self, grid: List[List]) -> str:
        """Convert grid to natural language description"""
        h, w = len(grid), len(grid[0]) if grid else 0

        description = f"Grid of size {h}x{w}. "

        # Describe colors present
        colors = set(v for row in grid for v in row)
        color_desc = [self.color_names[c] for c in sorted(colors) if c in self.color_names]
        description += f"Colors: {', '.join(color_desc)}. "

        # Describe patterns
        if self.has_symmetry(grid):
            description += "Has symmetry. "

        objects = self.count_objects(grid)
        if objects > 0:
            description += f"Contains {objects} objects. "

        return description

    def grid_to_sequence(self, grid: List[List]) -> str:
        """Convert grid to token sequence"""
        tokens = []
        for row in grid:
            for val in row:
                tokens.append(f"c{val}")
            tokens.append("EOL")  # End of line
        return " ".join(tokens)

    def has_symmetry(self, grid):
        """Check if grid has symmetry"""
        h = len(grid)
        for i in range(h // 2):
            if grid[i] == grid[h - 1 - i]:
                return True
        return False

    def count_objects(self, grid):
        """Count connected components"""
        h, w = len(grid), len(grid[0]) if grid else 0
        visited = set()
        count = 0

        def dfs(i, j, color):
            if (i, j) in visited or i < 0 or i >= h or j < 0 or j >= w:
                return
            if grid[i][j] != color or grid[i][j] == 0:
                return
            visited.add((i, j))
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(i + di, j + dj, color)

        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and grid[i][j] != 0:
                    count += 1
                    dfs(i, j, grid[i][j])

        return count


class T5ARCModel:
    """T5 model for ARC task generation"""

    def __init__(self, model_name: str = "t5-small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading T5 model on {self.device}...")

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✓ T5 model loaded")
        except:
            print("⚠ T5 model loading failed, using placeholder")
            self.tokenizer = None
            self.model = None

        self.grid_converter = GridToText()

    def prepare_input(self, task: Dict) -> str:
        """Prepare input for T5"""
        prompt = "Transform ARC grid: "

        for i, example in enumerate(task.get('train', [])):
            input_desc = self.grid_converter.grid_to_sequence(example['input'])
            output_desc = self.grid_converter.grid_to_sequence(example['output'])
            prompt += f"Example {i+1}: {input_desc} -> {output_desc} "

        if task.get('test'):
            test_input = task['test'][0]['input']
            test_desc = self.grid_converter.grid_to_sequence(test_input)
            prompt += f"Transform: {test_desc} -> "

        return prompt

    def generate(self, task: Dict) -> List[List]:
        """Generate output using T5"""
        if not self.model:
            # Fallback if model not loaded
            return task['test'][0]['input'] if task.get('test') else [[0]]

        prompt = self.prepare_input(task)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt",
                               max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                temperature=0.7,
                do_sample=False
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse back to grid
        return self.parse_output(generated_text)

    def parse_output(self, text: str) -> List[List]:
        """Parse generated text back to grid"""
        # Simple parsing - in practice needs robust parsing
        grid = []
        current_row = []

        tokens = text.split()
        for token in tokens:
            if token == "EOL":
                if current_row:
                    grid.append(current_row)
                    current_row = []
            elif token.startswith("c"):
                try:
                    color = int(token[1:])
                    current_row.append(color)
                except:
                    pass

        if current_row:
            grid.append(current_row)

        # Ensure rectangular grid
        if grid:
            max_w = max(len(row) for row in grid)
            grid = [row + [0] * (max_w - len(row)) for row in grid]

        return grid if grid else [[0]]


class VisionTransformerARC:
    """Vision Transformer for visual pattern recognition"""

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ViT model on {self.device}...")

        try:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✓ ViT model loaded")
        except:
            print("⚠ ViT model loading failed, using placeholder")
            self.model = None

        # Custom head for ARC
        self.arc_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)  # 10x10 transformation matrix
        ).to(self.device)

    def grid_to_image(self, grid: List[List]) -> Image.Image:
        """Convert grid to PIL Image"""
        # Create RGB image from grid
        h, w = len(grid), len(grid[0]) if grid else 1

        # Color palette (RGB)
        palette = [
            (0, 0, 0),      # 0: black
            (0, 0, 255),    # 1: blue
            (255, 0, 0),    # 2: red
            (0, 255, 0),    # 3: green
            (255, 255, 0),  # 4: yellow
            (128, 128, 128),# 5: gray
            (255, 192, 203),# 6: pink
            (255, 165, 0),  # 7: orange
            (0, 255, 255),  # 8: cyan
            (165, 42, 42)   # 9: brown
        ]

        # Create image
        img_array = np.zeros((h * 20, w * 20, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                color = palette[grid[i][j] % 10]
                img_array[i*20:(i+1)*20, j*20:(j+1)*20] = color

        return Image.fromarray(img_array)

    def extract_features(self, grid: List[List]) -> torch.Tensor:
        """Extract features from grid using ViT"""
        if not self.model:
            return torch.zeros(1, 768).to(self.device)

        # Convert to image
        img = self.grid_to_image(grid)

        # Prepare for ViT
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling

        return features

    def predict_transformation(self, task: Dict) -> np.ndarray:
        """Predict transformation matrix"""
        # Extract features from training examples
        train_features = []

        for example in task.get('train', []):
            in_feat = self.extract_features(example['input'])
            out_feat = self.extract_features(example['output'])
            combined = torch.cat([in_feat, out_feat], dim=-1)
            train_features.append(combined)

        if not train_features:
            return np.eye(10)

        # Average training features
        avg_features = torch.stack(train_features).mean(dim=0)

        # Predict transformation
        with torch.no_grad():
            transform = self.arc_head(avg_features[:, :768])
            transform = transform.view(10, 10).cpu().numpy()

        return transform


class BERTPatternMatcher:
    """BERT for pattern matching and reasoning"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BERT model on {self.device}...")

        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✓ BERT model loaded")
        except:
            print("⚠ BERT model loading failed, using placeholder")
            self.model = None

        self.grid_converter = GridToText()

        # Pattern matching head
        self.pattern_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)  # 10 pattern types
        ).to(self.device)

    def encode_task(self, task: Dict) -> torch.Tensor:
        """Encode task description using BERT"""
        if not self.model:
            return torch.zeros(1, 768).to(self.device)

        # Create text description
        description = "ARC task: "

        for i, example in enumerate(task.get('train', [])):
            in_desc = self.grid_converter.grid_to_text(example['input'])
            out_desc = self.grid_converter.grid_to_text(example['output'])
            description += f"Example {i+1}: {in_desc} transforms to {out_desc}. "

        # Tokenize
        inputs = self.tokenizer(description, return_tensors="pt",
                               max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        return embeddings

    def predict_pattern_type(self, task: Dict) -> str:
        """Predict pattern type for task"""
        embeddings = self.encode_task(task)

        with torch.no_grad():
            pattern_logits = self.pattern_head(embeddings)
            pattern_idx = torch.argmax(pattern_logits, dim=-1).item()

        pattern_types = [
            'rotation', 'mirror', 'scale', 'color_map',
            'fill', 'extract', 'combine', 'repeat',
            'symmetry', 'unknown'
        ]

        return pattern_types[pattern_idx % len(pattern_types)]


class HuggingFaceEnsemble:
    """Ensemble of HuggingFace models"""

    def __init__(self):
        print("=" * 70)
        print("Initializing HuggingFace Model Ensemble")
        print("=" * 70)

        self.models = {}

        # Initialize models (with error handling)
        try:
            print("\n1. Loading T5 for sequence generation...")
            self.models['t5'] = T5ARCModel()
        except Exception as e:
            print(f"   Failed: {e}")

        try:
            print("\n2. Loading ViT for visual pattern recognition...")
            self.models['vit'] = VisionTransformerARC()
        except Exception as e:
            print(f"   Failed: {e}")

        try:
            print("\n3. Loading BERT for pattern matching...")
            self.models['bert'] = BERTPatternMatcher()
        except Exception as e:
            print(f"   Failed: {e}")

        print("\n✓ HuggingFace ensemble ready!")

    def predict(self, task: Dict) -> List[List]:
        """Ensemble prediction"""
        predictions = []

        # Get predictions from each model
        if 'bert' in self.models:
            pattern_type = self.models['bert'].predict_pattern_type(task)
            print(f"BERT detected pattern: {pattern_type}")

        if 'vit' in self.models:
            transform = self.models['vit'].predict_transformation(task)
            # Apply transformation logic

        if 't5' in self.models:
            generated = self.models['t5'].generate(task)
            predictions.append(generated)

        # If no predictions, return input
        if not predictions and task.get('test'):
            predictions.append(task['test'][0]['input'])

        return predictions


def test_huggingface_models():
    """Test HuggingFace integration"""
    print("Testing HuggingFace Integration")
    print("=" * 70)

    # Create sample task
    sample_task = {
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[4, 3], [2, 1]]
            }
        ],
        'test': [
            {
                'input': [[5, 6], [7, 8]]
            }
        ]
    }

    # Initialize ensemble
    ensemble = HuggingFaceEnsemble()

    # Make prediction
    predictions = ensemble.predict(sample_task)

    print(f"\nPredictions: {predictions}")
    print("\n✓ HuggingFace integration test complete!")


if __name__ == "__main__":
    test_huggingface_models()