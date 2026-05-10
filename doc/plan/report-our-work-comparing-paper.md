# After RunPod: How to Report Our Results

This document is a step-by-step guide for what to do after running the pipeline on RunPod
and getting real EM numbers.

---

## Phase A — On RunPod (GPU machine)

### A1. Spin up a RunPod instance
- Recommended: A100 80GB or H100 (Qwen2.5-7B needs ~16GB VRAM in fp16)
- Template: PyTorch 2.3 / CUDA 12.1
- Disk: at least 100GB (models + Wikipedia corpus)

### A2. Clone the repo and install
```bash
git clone https://github.com/<your-repo>.git
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### A3. Download the Wikipedia corpus
```bash
mkdir -p dataset/wikipedia-dump
wget -P dataset/wikipedia-dump \
  https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```
TriviaQA and NaturalQA download automatically from HuggingFace when the pipeline runs.

### A4. Run the full pipeline
```bash
python src/run_pipeline.py \
  --data_dir dataset/ \
  --device cuda \
  --epochs 3 \
  --batch_size 16 \
  --resume
```

The `--resume` flag means if the run crashes partway through, re-running the same command
will skip steps whose output files already exist and pick up where it left off.

**Intermediate files saved automatically:**
| File | Created after |
|---|---|
| `outputs/qa_train.pkl` | Step 0 — TriviaQA train set loaded |
| `outputs/qa_eval.pkl` | Step 0 — TriviaQA eval set loaded |
| `outputs/triplets_train.pkl` | Step 0 — Contriever retrieval done |
| `outputs/dig_results.pkl` | Step 2 — DIG scoring done |
| `outputs/reranker.pt` | Step 3 — Reranker training done |
| `outputs/results.json` | Step 4 — Inference + EM evaluation done |

### A5. Check your results
```bash
cat outputs/results.json
```
You should see something like:
```json
[
  {
    "model": "Qwen2.5-7B",
    "dataset": "TriviaQA",
    "approach": "infogain_rag",
    "em_score": 0.7134,
    "n_samples": 11313
  }
]
```

---

## Phase B — Copy results back to your local machine

Only copy the small files — models and dataset do not need to come back.

### B1. Download from RunPod
In the RunPod file manager or via scp, download these files:
```
outputs/results.json          ← THE most important file
outputs/reranker.pt           ← trained model weights (upload to HuggingFace later)
```

### B2. Place `results.json` in your local `outputs/` folder
```
/Users/keita-katsumi/.../Final/outputs/results.json
```

### B3. Run main.py locally
```bash
python src/main.py
```

Now the terminal output will say:
```
[outputs/results.json found — showing our reproduced results alongside paper numbers]

  Dataset: TriviaQA
    [paper] naive_rag         EM=0.529  ███████████████
    [paper] bge_reranker      EM=0.670  ████████████████████
    [paper] infogain_rag      EM=0.720  █████████████████████
    [ours]  infogain_rag      EM=0.713  █████████████████████  (Δ -0.007 vs paper)
```

The `Δ` column tells you exactly how close your reproduction is to the paper number.

---

## Phase C — Update the README results table

Open `README.md` and find the **Results** section. Replace the placeholder table with:

```markdown
### Main comparison (Exact Match %)

| Model | Approach | TriviaQA (paper) | TriviaQA (ours) | Gap |
|---|---|:---:|:---:|:---:|
| Qwen2.5-7B | Naive RAG      | 52.9 | _TBD_ | — |
| Qwen2.5-7B | BGE-Reranker   | 67.0 | _TBD_ | — |
| Qwen2.5-7B | InfoGain-RAG   | **72.0** | **XX.X** | Δ X.X pp |
```

Fill in the `ours` column from `outputs/results.json`. The `Gap` column is paper minus ours.

**If your number is close (within ~2 pp):**
> "Our reproduction achieves 71.3 EM on TriviaQA, 0.7 pp below the paper's 72.0.
> The small gap is likely due to [fewer training epochs / smaller batch size / fp16 precision]."

**If your number is further off (>3 pp):**
> "Our reproduction achieves X.X EM on TriviaQA, Y.Y pp below the paper's 72.0.
> Possible reasons: training data sampling differs from the paper's 88K triplets,
> our Wikipedia corpus version, or compute constraints limiting training epochs."

The rubric rewards honest gap analysis — a well-explained gap is **not penalized**.

---

## Phase D — Commit everything and push

```bash
# Stage the new results and updated charts
git add outputs/results.json outputs/*.png outputs/sample-run.log
git add README.md

git commit -m "Add reproduced results from full pipeline run"
git push
```

Do NOT commit `outputs/reranker.pt` (it's in `.gitignore` as `*.pt`).
Instead, upload it to HuggingFace Hub and add the download link to the README's
Dataset & Model Setup table.

### Upload reranker to HuggingFace Hub
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload <your-hf-username>/infogain-rag-reranker outputs/reranker.pt
```
Then in README.md update:
```markdown
| **Trained reranker checkpoint** | Download from HuggingFace Hub: `<your-hf-username>/infogain-rag-reranker` |
```

---

## Phase E — What to write in the paper (Replication section)

The paper's Replication section should cover:

1. **Implementation decisions** — which components you implemented, which hyperparameters you used exactly vs. approximated
2. **Results table** — paper numbers vs. your numbers side by side
3. **Gap analysis** — explain each difference:
   - Did you use the same dataset split?
   - Did you train for the same number of epochs?
   - Did you use the same Wikipedia corpus?
   - Any compute constraints?
4. **What matched well** — call out where your implementation closely matched

Template:
> "We replicate the InfoGain-RAG pipeline as described in Wang et al. (2025), achieving
> an Exact Match of X.X on TriviaQA (paper: 72.0, gap: Y.Y pp). The gap is attributable to
> [reason]. Our ablation study results (CE-only: X.X, Margin-only: X.X, Multi-task: X.X)
> are consistent with the paper's reported trend, confirming that the combined loss objective
> outperforms either component alone."

---

## Summary checklist

- [ ] A. RunPod run complete → `outputs/results.json` exists on RunPod
- [ ] B. `results.json` copied to local `outputs/` folder
- [ ] B. `python src/main.py` shows `[paper] vs [ours]` comparison
- [ ] C. README results table updated with real numbers + gap analysis
- [ ] D. Committed and pushed to GitHub (repo is public)
- [ ] D. `reranker.pt` uploaded to HuggingFace Hub, link added to README
- [ ] E. Paper Replication section written with honest gap analysis
