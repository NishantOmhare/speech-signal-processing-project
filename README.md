# Noise-Robust Spoken Digit Recognition

**Spectral subtraction vs. multi-condition training for MFCC + CNN speech recognition in noise**

This project asks one question:

> **Does spectral-subtraction speech enhancement make an MFCC-based speech recogniser more robust to noise, and how does it compare with training the recogniser on noisy data?**

Spoken digits from the Free Spoken Digit Dataset are corrupted with white and babble noise at controlled SNRs. They are then cleaned with spectral subtraction, scored with objective quality metrics, and classified by a CNN. Everything runs in a single reproducible Kaggle notebook, with a step-by-step explanation in every section.

---

## Key Findings

- **Spectral subtraction raises SNR but barely improves recognition.** For white noise it gains 4–5 dB of SNR, but STOI (intelligibility) only rises by about 0.01–0.02. For babble noise it does almost nothing.
- **A model trained only on clean speech fails in mismatched noise.** Accuracy falls from 100% to 56% in white noise at 0 dB, and to 57% in babble noise even at 20 dB.
- **Enhancement helps the clean-trained model only in white noise**, by up to +7 points at 0 dB. It slightly hurts accuracy in babble noise.
- **Multi-condition training is far more effective.** It adds an average of +10.5 points in white noise and +27 points in babble noise, with no loss on clean speech.
- **Enhancing the input of the noise-trained model doesn't help.** Its artefacts are a new mismatch, and they cost up to 10 points in babble noise.

---

## Pipeline

```
FSDD .wav ─► preprocess ─► add noise (white / babble, 0–20 dB SNR)
                               │
                               ├─► spectral subtraction ─► SNR · SegSNR · PESQ · STOI
                               │
                               └─► MFCC + Δ + ΔΔ ─► CNN (Model A: clean / Model B: multi-condition)
                                                     └─► accuracy vs SNR, noisy vs enhanced
```

| Step | Description |
|---|---|
| **Data** | [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset): 3,000 recordings, 6 speakers, digits 0–9, 8 kHz |
| **Split** | Official FSDD split: index 0–4 for test, 5–9 for validation, the rest for training. An optional unseen-speaker split is included. |
| **Preprocessing** | DC removal, peak normalisation, fixed 1 s length, 0.25 s of silence padding on each side |
| **Analysis** | Waveform, STFT spectrogram (32 ms Hann window, 10 ms hop), 40-band mel spectrogram, MFCC |
| **Noise** | White noise and 6-talker babble (built from training speech only) at 20, 10, 5 and 0 dB SNR |
| **Enhancement** | Power-domain spectral subtraction (Boll / Berouti) with over-subtraction α and spectral floor β. Noise is estimated from the lowest-energy frames. α and β are tuned on the validation set. |
| **Features** | MFCC 1–20 + Δ + ΔΔ (20 × 151 × 3), standardised with training-set statistics |
| **Model** | 4-block CNN (BatchNorm, global average pooling, dropout), about 105k parameters, SpecAugment, early stopping |
| **Evaluation** | SNR, SegSNR, PESQ (narrow-band), STOI, accuracy, and confusion matrices across 18 test conditions |

---

## Results

### Recognition Accuracy (%) on the 300-clip Test Set

| Condition | Model A, noisy | Model A, enhanced | Model B, noisy | Model B, enhanced |
|---|---|---|---|---|
| Clean | 100.0 | 100.0 | 100.0 | – |
| White, 20 dB | 98.3 | 98.0 | 99.0 | 99.0 |
| White, 10 dB | 91.7 | 93.7 | 97.0 | 97.3 |
| White, 5 dB | 78.0 | 81.0 | 90.7 | 91.3 |
| White, 0 dB | 56.0 | 63.3 | 79.3 | 80.7 |
| Babble, 20 dB | 57.3 | 53.7 | 94.0 | 92.3 |
| Babble, 10 dB | 41.3 | 38.7 | 78.3 | 72.3 |
| Babble, 5 dB | 31.3 | 29.7 | 56.0 | 46.3 |
| Babble, 0 dB | 20.7 | 19.0 | 30.7 | 28.7 |

*Model A is trained on clean speech only. Model B is trained on clean speech plus noisy copies at random SNRs between 0 and 20 dB.*

### Enhancement Quality (noisy → enhanced)

| Condition | SNR (dB) | SegSNR (dB) | PESQ | STOI |
|---|---|---|---|---|
| White, 20 dB | 14.4 → 18.5 | 13.1 → 15.5 | 2.78 → 3.10 | 0.95 → 0.96 |
| White, 10 dB | 4.4 → 9.0 | 3.9 → 6.9 | 1.99 → 2.13 | 0.84 → 0.85 |
| White, 0 dB | −5.7 → −0.8 | −3.6 → −0.8 | 1.67 → 1.70 | 0.65 → 0.67 |
| Babble, 20 dB | 14.4 → 16.2 | 14.1 → 14.8 | 3.08 → 3.16 | 0.97 → 0.97 |
| Babble, 10 dB | 4.4 → 6.3 | 5.0 → 5.9 | 2.18 → 2.19 | 0.86 → 0.86 |
| Babble, 0 dB | −5.7 → −3.6 | −2.9 → −1.8 | 1.58 → 1.54 | 0.62 → 0.61 |

The best enhancement parameters on the validation set were **α = 4, β = 0.3**. Gentle subtraction (a high spectral floor) consistently beat aggressive subtraction.

---

## How to Run

### On Kaggle (recommended)
1. Upload `speech_noise_robustness.ipynb` as a new Kaggle notebook.
2. **Add data** → [Free Spoken Digit Dataset (FSDD)](https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd).
3. **Settings** → Internet **On** (needed for `pip install pesq pystoi`) and Accelerator **GPU T4** (optional).
4. **Run All.** Outputs are written to `/kaggle/working/outputs/`.

### Locally
```bash
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
pip install numpy pandas matplotlib seaborn librosa soundfile scikit-learn tensorflow joblib pesq pystoi
jupyter notebook speech_noise_robustness.ipynb
```
If the dataset isn't found under `/kaggle/input`, the notebook clones it automatically.

**Runtime:** about 30 minutes on a 2-core CPU, and much faster on a GPU. All parameters are in a single configuration cell. Set `SPLIT_MODE = "speaker"` to test on an unseen speaker.

---

## Repository Structure

```
├── speech_noise_robustness.ipynb               # main notebook (no outputs)
├── speech_noise_robustness_with_outputs.ipynb  # executed reference run
└── README.md
```

Running the notebook produces figures, CSV tables (`accuracy_results.csv`, `enhancement_metrics.csv`, `alpha_beta_tuning.csv`), the trained models (`.keras`), example audio clips and a `summary.txt`.

---

## Limitations

- The results come from a single run on 300 test clips, so each clip is worth 0.33 points and differences of 1–2 points may be chance.
- The default split is speaker-dependent: test speakers also appear in training.
- The noise is synthetic and added digitally. The babble is made of digits from the same speakers, which makes it an especially confusable interferer.

## Future Work

- Wiener filtering and MMSE-STSA enhancement
- Cepstral mean and variance normalisation (CMVN)
- Real recorded noise (MUSAN, DEMAND)
- Training on enhanced speech
- Speaker-independent evaluation with multiple seeds

---

## References

1. S. F. Boll, "Suppression of acoustic noise in speech using spectral subtraction," *IEEE Trans. ASSP*, 27(2), 1979.
2. M. Berouti, R. Schwartz, J. Makhoul, "Enhancement of speech corrupted by acoustic noise," *ICASSP*, 1979.
3. S. Davis, P. Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences," *IEEE Trans. ASSP*, 28(4), 1980.
4. ITU-T Rec. P.862, *Perceptual evaluation of speech quality (PESQ)*, 2001.
5. C. H. Taal et al., "An algorithm for intelligibility prediction of
