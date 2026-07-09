# Architecture Diagrams

This directory contains architecture diagrams for the components of the Footfall Analysis project.

## Gender Model Architecture

The gender model is a binary classifier based on a Vision Transformer (ViT) backbone. It is implemented in [model.py](file:///e:/projects/Footfall-Analysis/src/training/model.py) as the [GenderClassifier](file:///e:/projects/Footfall-Analysis/src/training/model.py#L9) class.

### 1. End-to-End Inference Pipeline
During video inference, we combine person detection with gender classification:
1. **Person Detection**: YOLOv8n detects persons in each video frame.
2. **Crop & Preprocess**: The bounding boxes are cropped and resized to $224 \times 224$ pixels, then normalized using ImageNet mean/standard deviation.
3. **Gender Classification**: The preprocessed crop is passed through the `GenderClassifier` model to predict the gender (`female` or `male`).
4. **Annotation**: The frame is annotated with bounding boxes (colored by gender) and labels.

### 2. Model Architecture
- **Backbone**: `google/vit-base-patch16-224` (Vision Transformer).
- **Classification Head**: A single Linear layer mapping the pooler output ($768 \to 2$).

### Diagram

Here is the architecture flowchart. You can view the raw mermaid source code in [gender_model_architecture.mmd](file:///e:/projects/Footfall-Analysis/docs/architecture/gender_model_architecture.mmd).

```mermaid
flowchart TB
    %% Styling
    classDef input fill:#D2E5FF,stroke:#2B78E4,stroke-width:2px;
    classDef process fill:#F9F9F9,stroke:#CCCCCC,stroke-width:1px;
    classDef model fill:#E2F0D9,stroke:#385723,stroke-width:2px;
    classDef layer fill:#FFF2CC,stroke:#D6B656,stroke-width:1px;
    classDef head fill:#FCE4D6,stroke:#C65911,stroke-width:2px;
    classDef output fill:#E1D5E7,stroke:#9673A6,stroke-width:2px;

    subgraph InferencePipeline ["1. End-to-End Inference Pipeline"]
        direction LR
        Frame["Input Video Frame"]:::input --> Detection["YOLOv8 Person Detection"]:::process
        Detection --> Crop["Crop Person Bounding Box"]:::process
        Crop --> Preprocess["Preprocess Crop<br>(Resize to 224x224, Normalize)"]:::process
        Preprocess --> Classifier["GenderClassifier Model"]:::model
        Classifier --> Annotate["Annotate Frame<br>(Colored Box & Label)"]:::output
    end

    subgraph ModelArchitecture ["2. GenderClassifier Model Architecture"]
        direction TB
        InputTensor["Input Tensor<br>(Batch Size, 3, 224, 224)"]:::input --> ViT["ViT Backbone<br>(google/vit-base-patch16-224)"]:::model
        
        subgraph ViTDetails ["ViT Backbone Internal Flow"]
            direction TB
            Patchify["Patch & Position Embedding<br>(Split to 16x16 patches, Project to 768, Add CLS token)"]:::layer --> Encoder["12x Transformer Encoder Layers<br>(Multi-Head Self-Attention + MLP)"]:::layer
            Encoder --> CLS["Extract CLS Token Output<br>(1 x 768)"]:::layer
            CLS --> Pooler["Pooler Layer<br>(Dense Layer + Tanh Activation)"]:::layer
        end

        ViT -.-> Patchify
        Pooler -.-> PoolerOutput["pooler_output<br>(Batch Size, 768)"]:::layer
        
        PoolerOutput --> ClassifierHead["Classifier Head<br>(nn.Linear: 768 -> 2)"]:::head
        ClassifierHead --> Logits["Raw Logits<br>(Batch Size, 2)"]:::output
        Logits --> Softmax["Softmax Activation"]:::output
        Softmax --> FinalOutput["Gender Prediction<br>(female: 0, male: 1)"]:::output
    end
```

## Age Model Architecture

The age model is a four-class classifier based on classical computer vision features (color and texture histograms) with one SVM per age attribute. It is implemented in the `age_classifier_v3.ipynb` notebook, replicating the original PETA benchmark methodology (Deng et al., 2014) rather than a dedicated `model.py` class.

### 1. End-to-End Inference Pipeline
During video inference, we combine person detection with age classification:
1. **Person Detection**: A Faster R-CNN (ResNet-50 FPN) detector detects persons in each video frame.
2. **Crop & Region Split**: Each bounding box is cropped, then split into horizontal body regions.
3. **Feature Extraction**: Per region, 16-bin RGB and HSV color histograms plus a uniform LBP texture histogram are computed and L1-normalized, then concatenated into one feature vector per crop.
4. **Age Classification**: The feature vector is scored by all four per-attribute SVMs (`Age16-30`, `Age31-45`, `Age46-60`, `AgeAbove61`); the attribute with the highest decision function score is predicted.
5. **Annotation**: The frame is annotated with bounding boxes and the predicted age label.

### 2. Model Architecture
- **Features**: Region-based 16-bin RGB and HSV color histograms plus uniform LBP texture histograms (radii 1, 2, 3), all L1-normalized and concatenated across regions.
- **Classification Heads**: Four independent one-vs-rest SVMs, one per age bucket, each using either a histogram-intersection kernel or an RBF kernel (selected per attribute via grid search over `C` and `gamma`).
- **Decision Rule**: The age bucket whose SVM produces the highest decision function score is chosen as the final prediction.

### Diagram

Here is the architecture flowchart. You can view the raw mermaid source code in [age_model_architecture.mmd](file:///e:/projects/Footfall-Analysis/docs/architecture/age_model_architecture.mmd).

```mermaid
flowchart TB
    %% Styling
    classDef input fill:#D2E5FF,stroke:#2B78E4,stroke-width:2px;
    classDef process fill:#F9F9F9,stroke:#CCCCCC,stroke-width:1px;
    classDef model fill:#E2F0D9,stroke:#385723,stroke-width:2px;
    classDef layer fill:#FFF2CC,stroke:#D6B656,stroke-width:1px;
    classDef head fill:#FCE4D6,stroke:#C65911,stroke-width:2px;
    classDef output fill:#E1D5E7,stroke:#9673A6,stroke-width:2px;

    subgraph InferencePipeline ["1. End-to-End Inference Pipeline"]
        direction LR
        Frame["Input Video Frame"]:::input --> Detection["Faster R-CNN (ResNet-50 FPN) Person Detection"]:::process
        Detection --> Crop["Crop Person Bounding Box"]:::process
        Crop --> Regions["Split Crop into Horizontal Body Regions"]:::process
        Regions --> Features["Extract Region Color + LBP Texture Histograms"]:::process
        Features --> Classifier["Per-Attribute SVM Ensemble"]:::model
        Classifier --> Annotate["Annotate Frame<br>(Box & Age Label)"]:::output
    end

    subgraph ModelArchitecture ["2. Age Attribute Model Architecture"]
        direction TB
        InputImage["Input Crop<br>(Pedestrian Image)"]:::input --> RegionSplit["Region Split<br>(N horizontal body regions)"]:::process

        subgraph FeatureDetails ["Per-Region Feature Extraction"]
            direction TB
            ColorHist["Color Histogram<br>(16-bin RGB + HSV, L1-normalized)"]:::layer
            TextureHist["Texture Histogram<br>(Uniform LBP, radii 1/2/3, L1-normalized)"]:::layer
        end

        RegionSplit --> ColorHist
        RegionSplit --> TextureHist
        ColorHist --> Concat["Concatenate All Region Histograms<br>(Single Feature Vector)"]:::layer
        TextureHist --> Concat

        subgraph SVMDetails ["Per-Attribute SVM Bank (One-vs-Rest)"]
            direction TB
            SVM1["Age16-30 SVM<br>(Histogram Intersection / RBF)"]:::head
            SVM2["Age31-45 SVM<br>(Histogram Intersection / RBF)"]:::head
            SVM3["Age46-60 SVM<br>(Histogram Intersection / RBF)"]:::head
            SVM4["AgeAbove61 SVM<br>(Histogram Intersection / RBF)"]:::head
        end

        Concat --> SVM1
        Concat --> SVM2
        Concat --> SVM3
        Concat --> SVM4

        SVM1 --> Scores["Decision Function Scores<br>(one per attribute)"]:::output
        SVM2 --> Scores
        SVM3 --> Scores
        SVM4 --> Scores
        Scores --> FinalOutput["Age Prediction<br>(argmax over attribute scores)"]:::output
    end
```
