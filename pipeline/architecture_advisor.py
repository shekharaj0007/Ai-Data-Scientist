"""
Architecture advisor — recommends ANN, CNN, RNN, Transformers, LLMs, encoders/decoders
based on detected data modality and ML problem type.
"""


def _torch_ready() -> bool:
    try:
        from . import deep_learning
        return deep_learning.is_available()
    except ImportError:
        return False


def recommend_architectures(
    problem_type: str,
    modality_info: dict,
    n_rows: int = 0,
    n_classes: int | None = None,
) -> dict:
    primary = modality_info.get("primary_modality", "tabular")
    modalities = modality_info.get("modalities", ["tabular"])
    is_multimodal = modality_info.get("is_multimodal", False)

    recommendations = []
    recommendations.extend(_tabular_recs(problem_type, n_rows))
    if "text" in modalities or modality_info.get("text_columns"):
        recommendations.extend(_text_recs(problem_type, n_rows, n_classes))
    if "image" in modalities or modality_info.get("image_columns"):
        recommendations.extend(_image_recs(problem_type, n_rows, n_classes))
    recommendations.extend(_timeseries_recs(problem_type))
    recommendations.extend(_multimodal_recs(is_multimodal, modalities))
    recommendations.extend(_generative_recs(problem_type, modalities))

    # Deduplicate by architecture name, keep highest priority
    seen = set()
    unique = []
    for rec in sorted(recommendations, key=lambda r: r["priority_score"], reverse=True):
        if rec["architecture"] in seen:
            continue
        seen.add(rec["architecture"])
        unique.append(rec)

    top_pick = unique[0] if unique else None
    by_family = _group_by_family(unique)

    return {
        "primary_modality": primary,
        "modalities_detected": modalities,
        "is_multimodal": is_multimodal,
        "top_recommendation": top_pick,
        "recommendations": unique[:12],
        "by_family": by_family,
        "decision_summary": _build_summary(problem_type, primary, top_pick, unique[:3]),
    }


def _rec(
    architecture: str,
    family: str,
    reason: str,
    when_to_use: str,
    implemented: bool,
    priority_score: int,
    alternatives: list[str] | None = None,
) -> dict:
    return {
        "architecture": architecture,
        "family": family,
        "reason": reason,
        "when_to_use": when_to_use,
        "implemented_in_pipeline": implemented,
        "priority_score": priority_score,
        "alternatives": alternatives or [],
    }


def _tabular_recs(problem_type: str, n_rows: int) -> list[dict]:
    if problem_type not in ("classification", "regression", "clustering"):
        return []

    recs = [
        _rec(
            "Gradient Boosting (XGBoost / LightGBM / CatBoost)",
            "tree_ensemble",
            "Best default for structured tabular data on most benchmarks.",
            "Medium/large tabular datasets with mixed numeric & categorical features.",
            True,
            95,
            ["Random Forest", "Extra Trees"],
        ),
        _rec(
            "Random Forest / Extra Trees",
            "tree_ensemble",
            "Robust, low tuning, handles non-linearities and interactions well.",
            "Noisy tabular data, feature interactions, when interpretability matters.",
            True,
            88,
        ),
        _rec(
            "Linear / Logistic Regression",
            "linear",
            "Fast, interpretable baseline; strong when relationships are mostly linear.",
            "Small data, high interpretability, sparse linear patterns.",
            True,
            75,
        ),
        _rec(
            "SVM (Linear / RBF Kernel)",
            "kernel",
            "Powerful for medium-sized data with clear margin structure.",
            "Small-to-medium datasets (<100k rows), text-like sparse inputs.",
            True,
            70,
        ),
        _rec(
            "K-Nearest Neighbors (KNN)",
            "instance_based",
            "Non-parametric; good local decision boundaries.",
            "Small datasets, low dimensionality after encoding.",
            True,
            60,
        ),
        _rec(
            "Naive Bayes",
            "probabilistic",
            "Very fast baseline; works well for high-dimensional sparse features.",
            "Text-like counts, simple classification baselines.",
            True,
            58,
        ),
        _rec(
            "MLP — Artificial Neural Network (ANN)",
            "ann",
            "Feed-forward network; learns non-linear tabular patterns.",
            "Large tabular data, complex non-linear boundaries; use when tree models plateau.",
            True,
            72,
            ["TabNet", "FT-Transformer"],
        ),
        _rec(
            "Deep Tabular Transformer (FT-Transformer / TabNet)",
            "transformer",
            "State-of-art for some tabular tasks using attention over features.",
            "Large tabular datasets, budget for GPU training & tuning.",
            False,
            65,
            ["MLP ANN", "XGBoost"],
        ),
    ]

    if problem_type == "clustering":
        recs = [
            _rec("K-Means", "clustering", "Fast, scalable centroid clustering.", "Spherical clusters, known K.", True, 90),
            _rec("Agglomerative / Hierarchical", "clustering", "Builds cluster hierarchy.", "Small data, unknown K.", True, 75),
            _rec("Gaussian Mixture Model (GMM)", "clustering", "Soft clustering with probabilistic assignments.", "Overlapping clusters.", True, 70),
            _rec("DBSCAN", "clustering", "Density-based; finds arbitrary shapes.", "Spatial data, outliers, unknown K.", True, 68),
            _rec("Autoencoder + K-Means (Deep Clustering)", "ann", "Neural embedding then cluster.", "High-dimensional tabular/image features.", False, 60),
        ]
    return recs


def _text_recs(problem_type: str, n_rows: int, n_classes: int | None) -> list[dict]:
    if problem_type not in ("classification", "regression"):
        return []

    return [
        _rec(
            "TF-IDF + Linear Model (LogReg / Linear SVM)",
            "linear",
            "Strong classical baseline for text classification.",
            "Short-to-medium text, limited data, need fast training.",
            True,
            90,
        ),
        _rec(
            "RNN / LSTM / GRU",
            "rnn",
            "Sequence models capture word order for text.",
            "Sequential text, moderate data, order matters (sentiment, NER).",
            _torch_ready(),
            78,
            ["BiLSTM + Attention"],
        ),
        _rec(
            "CNN (TextCNN)",
            "cnn",
            "Convolving n-grams; fast and effective for sentence classification.",
            "Fixed-length text, keyword/phrase patterns matter.",
            _torch_ready(),
            74,
        ),
        _rec(
            "Transformer Encoder (BERT / RoBERTa / DistilBERT)",
            "transformer_encoder",
            "Pre-trained bidirectional context; SOTA for most NLP tasks.",
            "Text classification, NER, QA — use when accuracy > speed.",
            False,
            92,
            ["Fine-tune encoder head"],
        ),
        _rec(
            "LLM Fine-tuning (LoRA / QLoRA on Llama, Mistral)",
            "llm",
            "Instruction-tuned LLMs for complex language tasks.",
            "Generation, summarization, complex reasoning on text.",
            False,
            85,
            ["Prompt engineering", "RAG"],
        ),
        _rec(
            "Encoder-Decoder (T5 / BART)",
            "encoder_decoder",
            "Seq2seq for translation, summarization, text-to-text.",
            "Input text → output text (not just classification).",
            False,
            88,
        ),
    ]


def _image_recs(problem_type: str, n_rows: int, n_classes: int | None) -> list[dict]:
    return [
        _rec(
            "CNN (ResNet / EfficientNet / MobileNet)",
            "cnn",
            "Industry standard for image classification & feature extraction.",
            "Image paths/arrays, object recognition, medical imaging.",
            _torch_ready(),
            95,
            ["Transfer learning", "pytorch_image_cnn in pipeline"],
        ),
        _rec(
            "Vision Transformer (ViT / Swin)",
            "transformer",
            "Attention-based image model; strong on large image datasets.",
            "Large labeled image sets, when CNNs plateau.",
            False,
            88,
        ),
        _rec(
            "Transfer Learning + Classifier Head",
            "cnn",
            "Freeze pretrained backbone, train small head on your labels.",
            "Small/medium image datasets (<50k images).",
            False,
            92,
        ),
        _rec(
            "Autoencoder / VAE (Unsupervised)",
            "encoder_decoder",
            "Learn compressed image representations.",
            "Anomaly detection, denoising, dimensionality reduction on images.",
            False,
            70,
        ),
        _rec(
            "U-Net / Diffusion Models",
            "generative",
            "Segmentation and image generation.",
            "Segmentation masks, inpainting, synthetic image generation.",
            False,
            75,
        ),
    ]


def _timeseries_recs(problem_type: str) -> list[dict]:
    if problem_type != "time_series":
        return []
    return [
        _rec("ARIMA / SARIMA", "statistical", "Classical forecasting with trend/seasonality.", "Univariate series, interpretability.", True, 85),
        _rec("Prophet / Exponential Smoothing", "statistical", "Handles seasonality & holidays.", "Business forecasting.", False, 80),
        _rec("RNN / LSTM / GRU", "rnn", "Learns temporal dependencies.", "Multivariate series, long sequences.", False, 78),
        _rec("Temporal CNN (TCN)", "cnn", "Convolutions over time; parallelizable.", "Local temporal patterns.", False, 72),
        _rec("Temporal Transformer (Informer, PatchTST)", "transformer", "Attention over time steps.", "Long-horizon forecasting.", False, 82),
    ]


def _multimodal_recs(is_multimodal: bool, modalities: list[str]) -> list[dict]:
    if not is_multimodal:
        return []
    return [
        _rec(
            "Early Fusion (concat tabular + text/image embeddings)",
            "multimodal",
            "Combine features from each modality into one vector.",
            "Tabular + text columns in same CSV.",
            False,
            80,
        ),
        _rec(
            "Late Fusion (ensemble per modality)",
            "multimodal",
            "Train separate models, combine predictions.",
            "Different modalities with different best algorithms.",
            False,
            78,
        ),
        _rec(
            "Cross-Modal Transformer (CLIP-style)",
            "transformer",
            "Joint embedding space for text + image.",
            "Image + caption, image + metadata text.",
            False,
            85,
        ),
        _rec(
            "Multimodal LLM (GPT-4V, LLaVA)",
            "llm",
            "Large models understanding text + images together.",
            "Complex QA over images and text.",
            False,
            82,
        ),
    ]


def _generative_recs(problem_type: str, modalities: list[str]) -> list[dict]:
    if problem_type != "regression" and "text" not in modalities:
        return []
    return [
        _rec(
            "Decoder-only Transformer (GPT-style LLM)",
            "transformer_decoder",
            "Autoregressive text generation.",
            "Chatbots, code gen, creative writing.",
            False,
            70,
        ),
        _rec(
            "Encoder-Decoder (T5, BART, MarianMT)",
            "encoder_decoder",
            "Text-to-text transformation.",
            "Summarization, translation, data-to-text.",
            False,
            72,
        ),
    ]


def _group_by_family(recs: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for r in recs:
        grouped.setdefault(r["family"], []).append(r["architecture"])
    return grouped


def _build_summary(problem_type: str, primary: str, top, top3: list[dict]) -> str:
    if not top:
        return "Upload data to receive architecture recommendations."

    implemented = [r for r in top3 if r.get("implemented_in_pipeline")]
    lines = [
        f"Detected primary modality: **{primary}** with task **{problem_type.replace('_', ' ')}**.",
        f"Top pick: **{top['architecture']}** ({top['family'].upper()}) — {top['reason']}",
    ]
    if implemented:
        lines.append(
            f"This pipeline will auto-train: {', '.join(r['architecture'] for r in implemented[:3])}."
        )
    not_impl = [r for r in top3 if not r.get("implemented_in_pipeline")]
    if not_impl:
        lines.append(
            f"For production-grade deep learning consider: {not_impl[0]['architecture']}."
        )
    return " ".join(lines)
