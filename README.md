# POISE-Bench

**Do LLM Agents Know What to Look Up? Certified Information Seeking under Partial Observability and a Budget**

Shahriar Shakir Sumit, Murat Tahtali, Seyedali Mirjalili and Md Tauhidul Islam.
Submitted to the Journal of Data-centric Machine Learning Research (DMLR), 2026.

This repository has the code of POISE-Bench (Partially Observable Information-Seeking Evaluation). The questions, the
naturalistic tasks and every answer our models gave are on Hugging Face at
[huggingface.co/datasets/Shahriar10/POISE-Bench](https://huggingface.co/datasets/Shahriar10/POISE-Bench).

## Why we built this

Language-model agents now look things up before they answer. They search, open documents and ask follow-up questions,
and only then do they reply. Most benchmarks still check the final answer alone, so an agent that read the right
record and an agent that guessed can end up with the same score. We wanted to see what an agent looked at, whether it
needed it, and how much it spent to get there.

So every question in POISE-Bench comes from a small hidden world that we generate. Because we know that world, we can
work out exactly which records settle the answer and what the cheapest way to find them is. We keep a question only
when that cheapest set is unique. An agent sees the question and a list of hidden records, opens them one at a time
within a budget, and then answers or says that the records cannot decide. We score the answer, the records it opened
and the cost.

When we gave 14 open-weight models the exact records they needed, most of them answered well. When they had to find
those records themselves, their accuracy fell by 17 to 55 points, so for most models the search is the harder part.
The ranking of the models on POISE-Bench also agreed closely with their ranking on real Wikipedia paragraphs from
MuSiQue, where the agents write their own search queries.

## What the benchmark contains

There are 11,600 multiple-choice questions in four families. Retrieval questions follow chains of facts about invented
organisations, people and cities. Spatial questions ask about landmarks hidden on a town map. Procedural questions ask
which steps of a maintenance procedure depend on which. Abstention questions ask the agent to name an item from a
reference table, and sometimes the honest answer is that the notes cannot tell. The questions are split into 8,000 for
training, 1,600 for development and 2,000 for testing, and each split uses its own names, sentence templates and
question forms. We also release 78 human-curated tasks from areas such as medicine, law and engineering, and the list
of 600 MuSiQue questions used in the real-text study.

## How the code is organised

The generator lives in `code/poise2`. Each family has its own file (`fam_retrieval.py`, `fam_spatial.py`,
`fam_procedural.py` and `fam_abstention.py`), and they all share `core.py`, which finds the cheapest set of records
for a question and checks every item before it is kept. `names.py` makes up the names, `textfmt.py` holds the sentence
templates, `metrics.py` computes the scores, `baselines.py` has the simple non-language-model agents, and `build.py`
puts the three splits together.

The scripts in `code/scripts_v2` run the experiments and compute the results. `run_llm_agent.py` runs a language
model as an agent, `run_musique_agent.py` does the same on MuSiQue, and `run_algorithmic.py` runs the simple agents.
`musique_prep.py` builds the MuSiQue sample, `adapt_human_set.py` converts the naturalistic tasks, and
`model_registry.py` lists the models we tested. `make_authoritative_results.py`, with help from `analysis_extra.py`
and `musique_metrics.py`, turns the saved answers into the scores, tests and summaries we report, all collected in
`results/authoritative.json`.

The unit tests are in `code/tests_v2`. The two scripts in `audit/` check every released item again from its text alone
and look for overlap between the splits, and the scripts in `annotation/` built and scored our human check of the
evidence.

## Getting started

We used Python 3.11. The commands below set up the code, fetch the data from Hugging Face into the same folder, run
the tests and rebuild the three splits, which come out identical to the released files.

```bash
git clone https://github.com/ShahriarShakirSumit/POISE-Bench.git
cd POISE-Bench
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('Shahriar10/POISE-Bench', repo_type='dataset', local_dir='.', allow_patterns=['data/*', 'results/*', 'annotation/*'])"
cd code
python -m pytest -q tests_v2
python -m poise2.build --out ../data/v2
python ../audit/v2_certify_and_shortcuts.py
python scripts_v2/run_algorithmic.py --split test --out ../results/algorithmic_test
```

## Running a language model

The agents run on a GPU with PyTorch and Hugging Face Transformers. We used one to four NVIDIA V100 cards with 32 GB
each. This command runs a model on the test split in all five conditions. The agent sees record names only, short
record titles, no records at all, exactly the records it needs, or every record.

```bash
pip install -r requirements-agents.txt
python scripts_v2/run_llm_agent.py --model Qwen/Qwen3-8B --out ../results/llm/qwen3-8b
```

For the MuSiQue study, download the official MuSiQue v1.0 release, unzip it into `data/musique/`, build the sample,
and run the search agent on it.

```bash
python scripts_v2/musique_prep.py
python scripts_v2/run_musique_agent.py --model Qwen/Qwen3-8B --out ../results/musique/qwen3-8b
```

## Reproducing our results

All our results come from the saved answers in `results/`, so you can reproduce them on a normal CPU without running
any model. Build the MuSiQue sample first as shown above, then run these two commands from the `code` folder. The
first writes every score, test and summary to `results/authoritative.json`, and the second scores the human check of
the evidence.

```bash
python scripts_v2/make_authoritative_results.py
python ../annotation/score_annotations.py
```

## Citation

If POISE-Bench is useful in your work, please cite our paper.

```bibtex
@misc{sumit2026poisebench,
  title  = {Do {LLM} Agents Know What to Look Up? Certified Information Seeking under Partial Observability and a Budget},
  author = {Sumit, Shahriar Shakir and Tahtali, Murat and Mirjalili, Seyedali and Islam, Md Tauhidul},
  year   = {2026},
  note   = {Submitted to the Journal of Data-centric Machine Learning Research}
}
```

## Licence and contact

The code and data are released under the MIT licence. The MuSiQue questions and paragraphs belong to their authors and
keep their CC BY 4.0 licence. Please do not train on the test split, so that the results stay comparable. For questions, write to Shahriar Shakir Sumit at
shahriar9121@gmail.com or m.sumit@unsw.edu.au.
