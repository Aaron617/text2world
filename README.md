<h1 align="center">
	🌍 Text2World: Benchmarking Large Language Models for Symbolic World Model Generation<br>
</h1>
<p align="center">
  <!-- <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/Aaron617/text2world?color=blue">
  </a> -->
  <a href="https://github.com/Aaron617/text2world/pulls">
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">
  </a>
  <a href="https://github.com/Aaron617/text2world/commits">
    <img alt="Last Commit" src="https://img.shields.io/github/last-commit/Aaron617/text2world">
  </a>
  <!-- <a href="https://github.com/Aaron617/text2world">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Aaron617/text2world?style=social">
  </a> -->
  <a href="https://hits.seeyoufarm.com">
    <img alt="vies" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FAaron617%2Ftext2world&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false">
  </a>
  <a href="https://arxiv.org/abs/2502.13092">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2502.13092-b31b1b.svg">
  </a>
  <a href="https://huggingface.co/datasets/xdzouyd/text2world">
    <img alt="Dataset" src="https://img.shields.io/badge/🤗-Dataset-ffce44">
  </a>
  <!-- <a href="https://huggingface.co/spaces/xdzouyd/text2world">
    <img alt="Demo" src="https://img.shields.io/badge/🤗-Data_Viewer-ffce44">
  </a>
   -->
   Webpage: <a href="https://text-to-world.github.io/">text-to-world.github.io</a>
</p>


# 📚 Overview
![](./assets/main.png)

# 📝 Updates

- 2025-02-18: Add [data viewer](https://huggingface.co/spaces/xdzouyd/text2world) and [dataset](https://huggingface.co/datasets/xdzouyd/text2world) for better visualization of Text2World.

# 💻 Installation
```
conda create -n text2world python=3.8 -y
conda activate text2world

pip install -r requirements.txt
```

# 🏃 Generate PDDL
Running the following command will generate files with `PROMPT_TYPE="desc2domain_zeroshot_cot"` and `DESCRIPTION_TYPE="corrected_description"` in the `_generated_pddl/_all_gen` directory.  
```
bash generate.sh ${MODEL} ${CORRECTION_TIME}
```
To try different values for `$PROMPT_TYPE` and `$DESCRIPTION_TYPE`, you can manually modify them in the `generate.sh` script.

## Complete `utils/.env` file
```
OPENAI_API_TYPE="open_ai"
OPENAI_API_BASE=...
OPENAI_API_KEY=...
```

## Available `$MODEL`
```
# OpenAI O-series
o1-mini
o1-preview
o3-mini

# OpenAI GPT-4
gpt-4o
gpt-4o-mini
gpt-4-turbo
chatgpt-4o-latest

# OpenAI GPT-3.5
gpt-3.5-turbo-0125
gpt-3.5-turbo-1106

# Anthropic Claude
claude-3.5-sonnet

# Meta Llama-2
llama2-7b
llama2-13b
llama2-70b

# Meta LlaMA-3.1
llama3.1-8b
llama3.1-70b

# DeepSeek
deepseek-reasoner
deepseek-v3

# Meta CodeLlaMA
codellama-7b
codellama-13b
codellama-34b
codellama-70b
```

## Define your own `$MODEL`
If you need to configure your own LLM, you can modify `utils/text2world.yaml` to define your custom model, as an example:
```
${NAME}:
    name: gpt # API type
    engine: ${ENGINE}
    context_length: 128000
    use_azure: False
    temperature: 0.
    top_p: 1
    retry_delays: 20
    max_retry_iters: 100
    stop: 
    max_tokens: 4000
    use_parser: False
```

# 🧑‍🏫 Evaluate Generated PDDL
First, create a project using the following command. It will create a folder with the same name under `_generated_pddl`. Note that `$PROJECT_NAME` cannot be `_all_gen`.
```
bash create_project.sh $PROJECT_NAME
```
Next, please manually copy the generated content of the models you are interested in evaluating from `_generated_pddl/_all_gen` to `_generated_pddl/$PROJECT_NAME`.

Finally, run the following command to evaluate all models in the project. The evaluation results will be generated in `_generated_pddl/_eval_result/$PROJECT_NAME`, including detailed scores for all PDDL files generated by each model and an overall leaderboard in the `_result_board.txt` file.
```
bash evaluate.sh $PROJECT_NAME
```

# 📝 Citation

If you find this work useful, please consider citing the following papers:

Text2World: Benchmarking Large Language Models for Symbolic World Model Generation:
```
@misc{hu2025text2worldbenchmarkinglargelanguage,
      title={Text2World: Benchmarking Large Language Models for Symbolic World Model Generation}, 
      author={Mengkang Hu and Tianxing Chen and Yude Zou and Yuheng Lei and Qiguang Chen and Ming Li and Yao Mu and Hongyuan Zhang and Wenqi Shao and Ping Luo},
      year={2025},
      eprint={2502.13092},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.13092}, 
}
```
