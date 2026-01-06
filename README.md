# Is this Idea Novel? Improving LLMs as Judges of Research Idea Novelty Through Probing Latent Judgments During Reasoning


> This repository accompanies the paper *Is this Idea Novel? Improving LLMs as Judges of Research Idea Novelty Through Probing Latent Judgments During Reasoning*.  
> It presents a new evaluation benchmark including a dataset of 1,381 research ideas derived from and judged by human experts as well as automated evaluation metrics designed to assess both rubric-based novelty scores and textual justifications of novelty judgments.
>In addition, we introduce Think-Probe-Respond (TPR), a novel approach for research idea novelty judgment that leverages the hidden representations of LLMs during reasoning to generate more accurate judgments.
---

## 🦏 RINoBench 

![example_novelty_judgment.png](figures/example_novelty_judgment.png)

Judging the novelty of research ideas is crucial for advancing science, enabling the identification of unexplored directions, and ensuring contributions meaningfully extend existing knowledge rather than reiterate minor variations. However, given the exponential growth of scientific literature, manually judging the novelty of research ideas through literature reviews is labor-intensive, subjective, and infeasible at scale. Therefore, recent efforts have proposed automated approaches for research idea novelty judgment. Yet, evaluation of these approaches remains largely inconsistent and is typically based on non-standardized human evaluations, hindering large-scale, comparable evaluations. To address this, we introduce **RINoBench 🦏**, the first comprehensive benchmark for large-scale evaluation of research idea novelty judgments.
Our benchmark unifies approaches for judging the novelty of research ideas by formalizing the task, illustrated in the Figure above, as the process of comparing a proposed idea with existing work to identify meaningful differences. Further, the task requires predicting a rubric-based novelty score (1–5) alongside a textual justification that grounds the judgment in related literature. This task design enables fine-grained, interpretable judgments of novelty and provides actionable feedback, empowering researchers to iteratively refine their ideas towards greater innovation and impact.
---
## Think-Probe-Respond (TPR)

![TPR.png](figures/TPR.png)

Building on our finding that LLMs internalize beliefs about research idea novelty that closely mirror those of human experts—and therefore generate comparable novelty rationales—yet often fail to map these beliefs to accurate numerical judgments, we propose the \acl{tpr} approach for research idea novelty judgment. TPR explicitly exploits the model’s internal beliefs during reasoning about novelty to yield more accurate quantitative judgments, which are then reused as conditioning signals to generate textual justifications that are coherent and well aligned with the predicted numerical scores.

As illustrated in the Figure above, TPR consists of three stages.

1. **Think:** We instruct an LLM to evaluate the novelty of a research idea and to think step by step before producing a final response. Importantly, we provide only textual descriptions of the novelty categories—without numerical scores—and instruct the model to evaluate novelty solely based on these descriptions without producing numerical judgments. For reasoning models that generate think tokens by default, we omit any explicit “think step by step” instruction.
2. **Probe:** Given an LLM with $L$ hidden layers, let $H = h^{(1)}, \dots, h^{(L)}$ represent the stack of hidden states. For a generated sequence of reasoning (“think”) tokens $T = t_{1}, \dots, t_{n}$, we terminate generation upon the production of the final think token $t_{n}$ and extract the hidden state $h^{(L)}_{t_{n}}$. This representation $h^{(L)}_{t_{n}}$ is then used as the input feature vector for a logistic regression probing classifier.
3. **Respond:** We append the textual description of the predicted novelty class to the model’s generated output and resume generation. Conditioned on both its prior reasoning and the predicted novelty signal, the \ac{llm} generates the final response, which is used as the justification of the novelty judgment. This stage is *critical for ensuring alignment between the numerical novelty prediction and the generated textual justification*. Without explicitly conditioning the response on the predicted novelty class, the model may arrive at a different conclusion during response generation, resulting in justifications that are inconsistent with the underlying novelty judgment predicted by the probing classifier.

TPR is computationally efficient and lightweight. All LLM parameters remain frozen and are used only at inference. Training is limited to a simple logistic regression classifier, which can be learned efficiently on CPU, in contrast to resource-intensive fine-tuning that requires substantial GPU resources.

---

## 📂 Repository Structure

```
NoveltyBench/
├── data/
│   ├── final_benchmark_dataset                # includes the dataset of RINoBench
│   ├── evaluations                            # incudes the evaluation results of various state-of-the-art LLMs on RINoBench
│   └── novelty_predictions                    # incudes the reserach idea novelty judgments of various state-of-the-art LLMs on RINoBench
│
├── figures/                                   # includes figures from the paper
│
├── src/
│   ├── data_processing                        # Scripts and LLM prompts used to construct our dataset
│   └── experiments                            # Scripts and LLM prompts used to generate LLM predictions as well as for evaluating the predictions
│
├── .gitignore
├── README.md 
└── requirements.txt  
```

Clean implementations of TPR and baseline appraoches will be made available for the camera-ready version of the paper.

## 🧱 Data Description

| File                            | Description                     |
|---------------------------------|---------------------------------|
| `label_descriptions.json`       | The novelty judgment rubric.    |
| `train.json`                    | The train split of our dataset. |
| `test.json`                     | The test split of our dataset.  |

Each train and test split contains research ideas, gold novelty scores, gold textual judgment justifications, related works, and the respective sources from OpenReview.
