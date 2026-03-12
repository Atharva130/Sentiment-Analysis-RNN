# 🎬 Sentiment Analysis with RNN, LSTM & GRU

A deep learning project that compares three recurrent neural network architectures — **SimpleRNN**, **LSTM**, and **GRU** — for sentiment classification on the **SST-2 (Stanford Sentiment Treebank)** dataset.

---

## 📁 Project Structure

```
RNNSentiment.ipynb   # Main notebook with all models and comparisons
```

---

## 📊 Dataset

- **Name:** SST-2 (Stanford Sentiment Treebank 2)
- **Source:** [HuggingFace Datasets — GLUE Benchmark](https://huggingface.co/datasets/glue)
- **Task:** Binary sentiment classification (Positive / Negative)
- **Training samples:** ~67,349
- **Validation samples:** ~872

---

## 🧠 Models Compared

All three models share the same architecture, differing only in the recurrent layer:

| Layer       | Config                        |
|-------------|-------------------------------|
| Embedding   | `vocab=10000`, `output_dim=32` |
| Recurrent   | `units=32`, `dropout=0.5`     |
| Dense       | `1 unit`, `sigmoid` activation |

### Architectures
- ✅ **SimpleRNN** — Baseline recurrent model
- ✅ **LSTM** — Long Short-Term Memory (handles long-range dependencies)
- ✅ **GRU** — Gated Recurrent Unit (efficient alternative to LSTM)

---

## ⚙️ Training Config

| Parameter     | Value                  |
|---------------|------------------------|
| Optimizer     | Adam                   |
| Loss          | Binary Crossentropy    |
| Epochs        | 5                      |
| Batch Size    | 32                     |
| Max Seq Len   | 40                     |
| Vocab Size    | 10,000                 |
| Train/Test    | 80% / 20% split        |

---

## 📈 Results (SimpleRNN)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1     | 68.45%    | 86.47%  |
| 2     | 89.93%    | 90.48%  |
| 3     | 92.54%    | 91.05%  |
| 4     | 93.53%    | 91.14%  |
| 5     | 94.15%    | 91.39%  |

> LSTM and GRU results will appear after running those cells.

---

## 🚀 How to Run

1. Open `RNNSentiment.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter
2. Run all cells in order
3. The notebook will:
   - Download the SST-2 dataset automatically
   - Tokenize and pad sequences
   - Train SimpleRNN, LSTM, and GRU models
   - Plot accuracy & loss comparisons
   - Accept custom text input for live prediction

---

## 🔮 Custom Prediction

At the end of the notebook, you can test the model with your own sentence:

```
Enter a sentence: i loved this movie
Sentiment: Positive :)
```

---

## 🛠️ Dependencies

```
tensorflow
datasets (HuggingFace)
scikit-learn
pandas
numpy
matplotlib
```

Install via:
```bash
pip install tensorflow datasets scikit-learn pandas numpy matplotlib
```

---

## 📚 References

- [GLUE Benchmark — SST-2](https://gluebenchmark.com/tasks)
- [TensorFlow Keras Docs](https://www.tensorflow.org/api_docs/python/tf/keras)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
