# Project Tree

```text
submit/
├── README.md
├── requirements.txt
├── KG_construct/
│   ├── build_oag_triples.py
│   ├── bulid_oag_stream.py
│   └── subgraph_sample.py
├── data_filter/
│   ├── correctness_filter.py
│   └── hardness_filter.py
├── data_synthesis/
│   ├── adj_generator.py
│   ├── gen_qa_from_subgraphs.py
│   ├── gen_vis_qa.py
│   ├── graph2qa.sh
│   ├── graph2visqa.sh
│   └── paper_summary.py
├── database/
│   ├── db_start.py
│   └── load_and_run.sh
├── evaluation/
│   ├── EmbeddingSearchTool.py
│   ├── GetInDegreeTool.py
│   ├── GetKhopTool.py
│   ├── GetNeighborTool.py
│   ├── KeyWordSearchTool.py
│   ├── PaperInfoTool.py
│   ├── PlotTools.py
│   ├── ResearchToolBox.py
│   ├── ShortestPathTool.py
│   ├── SummarizeTool.py
│   ├── draw_picture.py
│   ├── faiss_server.py
│   ├── message_processor.py
│   ├── qwen2-5-eval.py
│   ├── qwen2-5-serve.sh
│   ├── qwen2-5_run_eval.sh
│   ├── qwen2-5_run_evalbenchmark.sh
│   ├── qwen3-30b_eval.py
│   ├── qwen3-30b_run_eval.sh
│   ├── qwen3-30b_run_evalbenchmark.sh
│   ├── qwen3-30ba3b-serve.sh
│   ├── qwen3-8b-serve.sh
│   ├── qwen3_eval.py
│   ├── qwen3_run_eval.sh
│   ├── qwen3_run_evalbenchmark.sh
│   ├── qwq-32b-serve.sh
│   ├── qwq_eval.py
│   ├── qwq_run_eval.sh
│   ├── qwq_run_evalbenchmark.sh
│   ├── serve-qwen3-8b-embedding.sh
│   ├── start_faiss_server.sh
│   ├── stop_vllm_servers.sh
│   ├── tool.py
│   └── utils/
│       └── db_tool.py
├── text_benchmark/
│   ├── judge.py
│   └── text_benchmark.jsonl
└── vis_benchmark/
    ├── judge.py
    └── vis_benchmark.jsonl
```

## Quick Usage

Install dependencies: `pip install -r requirements.txt`.

Run rollout in `evaluation/` to generate model trajectories/results.

Evaluate the rollout outputs with `text_benchmark/judge.py` and `vis_benchmark/judge.py`.

Data generation scripts are under `data_synthesis/`; filtering scripts are under `data_filter/`.
