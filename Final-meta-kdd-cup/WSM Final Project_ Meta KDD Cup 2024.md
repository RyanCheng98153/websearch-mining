---
title: 'WSM Final Project: Meta KDD Cup 2024'

---

# WSM Final Project: Meta KDD Cup 2024

## 1. Introduction

Large language models (LLMs) have demonstrated impressive performance across various domains. However, they come with notable limitations. For instance, their knowledge is confined to the training data, which means they cannot accurately answer questions about events or developments that occurred after their training was completed. Another widely recognized limitation is the issue of hallucinations. As LLMs function as next-token predictors, they generate the token with the highest probability at each step. This often leads them to produce what they believe is a correct answer, even if the response is incorrect or entirely fabricated, especially when faced with challenging questions.

To address these issues, a new approach called Retrieval-Augmented Generation (RAG) has emerged. RAG works by retrieving relevant information based on the input prompt and feeding it to the LLM. This enables the model to leverage in-context learning to incorporate knowledge beyond its training data. Consequently, the LLM can generate answers grounded in the retrieved information, providing responses with references to ensure accuracy and reduce the likelihood of fabrication.

In this report, we implemented a Retrieval-Augmented Generation (RAG) pipeline for Task 1 of Meta's CRAG competition. Our performance achieved a score of approximately 0.3.

### 1.1 Problem Statement

In the task, we are provided with up to five web pages for each question. While these web pages are likely, but not guaranteed, to be relevant.  Scoring works as follows: a correct answer earns 1 point, while an incorrect answer deducts 1 point. Alternatively, the model can output "I don't know," which neither earns nor deducts points.

Since the model's output may not perfectly match the correct answer, ChatGPT-4 acts as the evaluator, determining whether the model's response conveys the same meaning as the expected answer.

## 2. Literature Review

- **2024 KDD Cup CRAG Workshop Papers: UM6P Team Technical Report**
  - Focused on cross encoding to enhance retrieval relevance.
  - Employed query classification to refine results.
- **Winning Solution For Meta KDD Cup’ 24**
  - Proposed a finetune dataset construction for RAG that significantly reduces hallucinations.
- **KDD Cup Meta CRAG Technical Report: Three-step Question-Answering Framework**
  - Introduced a three-step question-answering framework leveraging category classification, Chain of Thought (CoT) reasoning, and voting mechanisms.

- **Honest AI: Fine-Tuning "Small" Language Models to Say "I Don't Know", and Reducing Hallucination in RAG**
  - Proposed methods to minimize hallucination in generative responses.
  - Differentiated between easy and hard questions to guide response strategies. For questions involving comparison or false premises, they advocated for answers like “I don’t know” to maintain reliability.

**Notable Research Insights:**

1. Cross encoding ensures enhanced context understanding, leading to higher retrieval accuracy.
1. Query classification by domain and complexity enables targeted solutions.
1. Fine-tuning “small” language models can significantly reduce hallucination in generated responses.

## 3. Experiment Design

### 3.1 Preprocessing

Effective preprocessing is the foundation for reliable RAG systems.

1. **HTML Parsing**:
    - Utilized BeautifulSoup4 to extract meaningful content from web pages and ensured accurate conversion of HTML structures into readable text for downstream tasks.
3. **Chunking**
    - Implemented a cross-encoding strategy to divide content into manageable, coherent pieces.
    - Improved retrieval by maintaining contextual integrity within chunks.

### 3.2 Retrieval

- **Key Comparisons:**

  - **BGE-M3** demonstrated superior performance over:
    - OpenAI text-embedding.
    - Microsoft E5-mistral-7b.
    - Nomic-embed-text-v1.
    - Jina embedding
  - Evaluation metrics showed BGE-M3’s effectiveness in languages such as English, Czech, French, and Hungarian.
  - Source: 
      - BGE's paper [arxiv](https://arxiv.org/abs/2402.03216v3) 
      - BGE's huggingface [huggingface](https://huggingface.co/BAAI/bge-m3)

- **BGE-M3’s Strengths:**
  - Robust embeddings tailored for diverse linguistic datasets.
  - Enhanced precision in semantic similarity tasks.

![CRAG](https://hackmd.io/_uploads/r1AtOP-8yg.png)

### 3.3 Reranker

Reranking is crucial to prioritize the most relevant results.

1. **Llama3 LLM Reranker:**
   - Leveraged llama3 8b-instruct models to re-evaluate retrieval outputs, using the same language model as generate model, no need to install additional LM.
    - The result is worse than OpenAI LLM Reranker; therefore, we change the LLM reranker model to OPENAI.

2. **OpenAI LLM Reranker:**
   - Leveraged OpenAI’s advanced models to re-evaluate retrieval outputs.

3. **Sentence Transformer Reranker:**
   - Utilized **cross-encoder/ms-marco-MiniLM** for reranking, emphasizing contextual relevance.


4. **Comparison of Reranking Models:**
  - **Exact Accuracy:** OpenAI Reranker performed consistently but required substantial computational resources.
  - **Hallucination Rates:** BGE-M3 Reranker exhibited the lowest hallucination, enhancing answer reliability.
  - **Miss Rates:** Sentence Transformer models demonstrated higher miss rates but were faster.

### 3.4 Query Classification

- **Classifier training:**
    For the purpose of skipping difficult questions acrroding to the three fields:
    - domain: "sports", "movie", "finance", "open", "music"
    - question_type: "post-processing", "simple_w_condition", "multi-hop", "simple", "set", "false_premise", "comparison", "aggregation"
    - static_or_dynamic: "static", "changing", "real-time", "fast-changing"

    which are given in the dataset, but not given to the system when generating answer, so we trained the three classifier base on DistilBERT.
- **Filtering:**
    Acrroding to the Static scores of differente classes of question base on the three field mention above. During the process of generation, we first use the pretrained classifiers to predict the questions' classes, then we performed two kinds of filtering.
    - strict filter: Skipped the question and answer "IDK" if the questions are: 
        - finance
        - set
        - real-time
        - open & fast-changing
    - loose filter: Skipped the question and answer "IDK" if the questions are: 
        - finance & ( set | open | real-time | fast-changing )
        - set & fast-changing
### 3.5 Answer Generation
The prompt is as following.
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful and honest assistant. Please, respond concisely and truthfully in 70 words or less. Now is {time}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context information is below.
<DOC>
{doc1}
</DOC>
<DOC>
{doc2}
</DOC>
...
<DOC>
{doc10}
</DOC>

Given the context information and using your prior knowledge, please provide your answer in concise style. End your answer with a period. Answer the question in one line only.

Question: {question}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```



### 3.6 Model Finetune
Our fine-tuning dataset is from this [Paper](https://openreview.net/attachment?id=oWNPeoP1uC&name=pdf), the dataset is constructed as follows:

First passed the original model through a complete RAG pipeline, which generated an initial dataset for fine-tuning.
Each piece of data in the dataset can be categorized into four categories:

|                | Has Relevant | No Relevant |
| -------------- | ------------ | ----------- |
| Answer Correct | A            | B           |
| Answer Wrong   | C            | D           |

Next, modify the dataset based on the category it belongs to:

A: Leave it unchanged.
B: Leave it unchanged because we want the model to answer correctly using its own knowledge, even if the RAG pipeline returns irrelevant data.
C: Correct the answer.
D: Change the answer to "I don't know" because we want the model to respond honestly with "I don't know."

Then the model is finetuned using SFTTrainer from huggingface transformers library with peft.

## 4. Experiment Result

### Reranker Comparision Experiment

> [!Important]
> All the experiment are using **BGE as Retriever**, 
> The following results are testing different **Rerankers**

> [!Note]
> `top-k rerank-n`: means BGE retrieve $k$ most relevence documents, rerank them then pass $n$ most relevnce documents to the LLM.

- **Only Retriever (BGE retriever with no reranker)**

| Metric            | Retrieve 3 | Retrieve 5 | Retrieve 10 | Retrieve 15 | Retrieve 20 | Retrieve 30 |
| ----------------- | ---------- | ---------- | ----------- | ----------- | ----------- | ----------- |
| **Score**         | 0.28       | 0.32       | 0.28        | -0.36       | -0.48       | -0.52       |
| **Exact AC**      | 0.27       | 0.26       | 0.26        | 0.08        | 0.07        | 0.07        |
| **Overall AC**    | 0.37       | 0.4        | 0.37        | 0.17        | 0.14        | 0.12        |
| **Hallucination** | 0.09       | 0.08       | 0.09        | 0.53        | 0.62        | 0.64        |
| **Miss**          | 0.54       | 0.52       | 0.54        | 0.3         | 0.24        | 0.24        |

- **OpenAI Reranker for BGE Retrieve top-k = 5, 10, 15**

| Metric            | top-5 rerank-3 | top-10 rerank-3 | top-15 rerank-3 | top-15 rerank-5 |
| ----------------- | -------------- | --------------- | --------------- | --------------- |
| **Score**         | 0.28           | 0.3             | 0.3             | 0.28            |
| **Exact AC**      | 0.3            | 0.3             | 0.3             | 0.28            |
| **Overall AC**    | 0.4            | 0.4             | 0.41            | 0.39            |
| **Hallucination** | 0.12           | 0.1             | 0.11            | 0.11            |
| **Miss**          | 0.48           | 0.5             | 0.48            | 0.5             |

- **OpenAI Reranker for BGE Retrieve top-k = 20**

| Metric            | Rerank 3 | Rerank 5 | Rerank 10 |
| ----------------- | -------- | -------- | --------- |
| **Score**         | 0.35     | 0.32     | 0.31      |
| **Exact AC**      | 0.29     | 0.28     | 0.28      |
| **Overall AC**    | 0.43     | 0.41     | 0.4       |
| **Hallucination** | 0.08     | 0.09     | 0.09      |
| **Miss**          | 0.49     | 0.5      | 0.51      |

- **OpenAI Reranker for BGE Retrieve top-k = 30**

| Metric            | Rerank 3 | Rerank 5 | Rerank 10 | Rerank 15 |
| ----------------- | -------- | -------- | --------- | --------- |
| **Score**         | 0.29     | 0.31     | 0.3       | 0.28      |
| **Exact AC**      | 0.27     | 0.28     | 0.27      | 0.28      |
| **Overall AC**    | 0.38     | 0.39     | 0.38      | 0.39      |
| **Hallucination** | 0.09     | 0.08     | 0.08      | 0.11      |
| **Miss**          | 0.53     | 0.53     | 0.54      | 0.5       |

- **OpenAI Reranker for BGE Retrieve top-k = 100**

| Metric            | Rerank 3 | Rerank 10 | Rerank 15 | Rerank 30 |
| ----------------- | -------- | --------- | --------- | --------- |
| **Score**         | 0.34     | 0.29      | 0.26      | 0.23      |
| **Exact AC**      | 0.28     | 0.26      | 0.27      | 0.25      |
| **Overall AC**    | 0.4      | 0.38      | 0.36      | 0.34      |
| **Hallucination** | 0.06     | 0.09      | 0.1       | 0.11      |
| **Miss**          | 0.54     | 0.53      | 0.54      | 0.55      |

- **Sentence Transformer Reranker for BGE Retrieve top-k = 5, 10, 15**

| Metric            | top-5 rerank-3 | top-10 rerank-3 | top-15 rerank-3 | top-15 rerank-5 |
| ----------------- | -------------- | --------------- | --------------- | --------------- |
| **Score**         | 0.26           | 0.26            | 0.25            | 0.28            |
| **Exact AC**      | 0.27           | 0.26            | 0.26            | 0.28            |
| **Overall AC**    | 0.38           | 0.38            | 0.38            | 0.39            |
| **Hallucination** | 0.12           | 0.12            | 0.13            | 0.11            |
| **Miss**          | 0.5            | 0.5             | 0.49            | 0.5             |

- **Sentence Transformer Reranker for BGE Retrieve top-k = 20**

| Metric            | Rerank 3 | Rerank 5 | Rerank 10 |
| ----------------- | -------- | -------- | --------- |
| **Score**         | 0.26     | 0.3      | 0.29      |
| **Exact AC**      | 0.25     | 0.28     | 0.24      |
| **Overall AC**    | 0.38     | 0.4      | 0.36      |
| **Hallucination** | 0.12     | 0.1      | 0.07      |
| **Miss**          | 0.5      | 0.5      | 0.57      |

- **Sentence Transformer Reranker for BGE Retrieve top-k = 30**

| Metric            | Rerank 3 | Rerank 5 | Rerank 10 | Rerank 15 |
| ----------------- | -------- | -------- | --------- | --------- |
| **Score**         | 0.24     | 0.27     | 0.27      | -0.4      |
| **Exact AC**      | 0.25     | 0.26     | 0.24      | 0.08      |
| **Overall AC**    | 0.37     | 0.38     | 0.35      | 0.14      |
| **Hallucination** | 0.13     | 0.11     | 0.08      | 0.54      |
| **Miss**          | 0.5      | 0.51     | 0.57      | 0.32      |

- **Sentence Transformer Reranker for BGE Retrieve top-k =  100**

| Metric            | Rerank 3 | Rerank 10 | Rerank 15 | Rerank 30 |
| ----------------- | -------- | --------- | --------- | --------- |
| **Score**         | 0.26     | 0.27      | -0.34     | -0.47     |
| **Exact AC**      | 0.25     | 0.24      | 0.09      | 0.07      |
| **Overall AC**    | 0.38     | 0.35      | 0.17      | 0.14      |
| **Hallucination** | 0.12     | 0.08      | 0.51      | 0.61      |
| **Miss**          | 0.5      | 0.57      | 0.32      | 0.25      |

- **Best 8 Results for different Rerankers (No classifier)**

| Experiment Type          | Retrieve | Rerank | Score | Exact AC | Overall AC | Hallucination | Miss |
| ------------------------ | -------- | ------ | ----- | -------- | ---------- | ------------- | ---- |
| **OpenAI Reranker**      | 20       | 3      | 0.35  | 0.29     | 0.43       | 0.08          | 0.49 |
| **OpenAI Reranker**      | 100      | 3      | 0.34  | 0.28     | 0.4        | 0.06          | 0.54 |
| **No Reranker**          | 5        | none   | 0.32  | 0.26     | 0.4        | 0.08          | 0.52 |
| **OpenAI Reranker**      | 5        | 3      | 0.3   | 0.3      | 0.4        | 0.12          | 0.48 |
| **OpenAI Reranker**      | 15       | 3      | 0.3   | 0.3      | 0.41       | 0.11          | 0.48 |
| **Sentence Transformer** | 20       | 5      | 0.3   | 0.28     | 0.4        | 0.1           | 0.5  |
| **No Reranker**          | 3        | none   | 0.28  | 0.27     | 0.37       | 0.09          | 0.54 |
| **No Reranker**          | 10       | none   | 0.28  | 0.26     | 0.37       | 0.09          | 0.54 |

### Classifier Comparision Experiment
#### Training the classifiers

| Metric        | domain | static_or_dynamic |
| ------------- | ------ | ----------------- |
| **Accuracy**  | 0.98   | 0.91              |
| **Precision** | 0.98   | 0.90              |
| **Recall**    | 0.97   | 0.91              |
| **F1-score**  | 0.98   | 0.90              |

![domain_non_bal (1)_T](https://hackmd.io/_uploads/rymN7Hl8kl.png)
![static_non_bal (1)_T](https://hackmd.io/_uploads/SkOS7Bg8Jx.png)

| Metric        | question_type | question_type(balanced) |
| ------------- | ------------- | ----------------------- |
| **Accuracy**  | 0.95          | 0.94                    |
| **Precision** | 0.94          | 0.93                    |
| **Recall**    | 0.93          | 0.93                    |
| **F1-score**  | 0.94          | 0.93                    |

![type_non_bal (1)_T](https://hackmd.io/_uploads/BktJ8BlUkl.png)

**Macro accuracy of three classifier : 0.9467**

Additionally, we observed that the data balancing didn't work well, since that the origin data is relatively balance. Therefore, we use the model trained by original dataset without balancing for later works.

#### Filtering

We choose the parameters(OpenAI Reranker=3 for BGE Retrieve topk= 20) that performed well in the previous Reranker Comparision Experiment

| Metric            | no filter | loose filter | strict filter |
| ----------------- | --------- | ------------ | ------------- |
| **Score**         | 0.31      | 0.32         | 0.31          |
| **Exact AC**      | 0.3       | 0.3          | 0.27          |
| **Overall AC**    | 0.41      | 0.4          | 0.36          |
| **Hallucination** | 0.1       | 0.08         | 0.05          |
| **Miss**          | 0.49      | 0.52         | 0.59          |

The stricter filter system used, the lower Hallucination score can be obtained. The tradeoff is lower AC score, which is trievial. The result proof that it is possible to get higher comprehensive by implementing filter, while it is not promised. The adjustment of filter function is necessary. The filter we proposed at this paper is different from the previous one to adapt the updated RAG system. You can adjust the filter by modify *model/rag_llama_baseline.py* and change the calculation of difficulty in function *classify()*, the system decide to answer "IDK" if the function return False

## 5. Variance of Language model
> After organizing the code, we attempted to reproduce our data and results but found that the outcomes were different. Consequently, we conducted an experiment by running the same model test four times consecutively under identical parameter settings.

```
Fixed parameters: 
BGE Retrieve top-20 Openai LLM-Rerank top-3
classify difficulty < 2 (strict filter)
```

| Metric            | Iter 1 | Iter 2 | Iter 3 | Iter 4 |
| ----------------- | ------ | ------ | ------ | ------ |
| **Score**         | 0.33   | 0.32   | 0.32   | 0.31   |
| **Exact AC**      | 0.28   | 0.27   | 0.29   | 0.27   |
| **Overall AC**    | 0.37   | 0.37   | 0.37   | 0.36   |
| **Hallucination** | 0.04   | 0.05   | 0.05   | 0.05   |
| **Miss**          | 0.59   | 0.58   | 0.58   | 0.59   |

> [!Caution]
> Therefore, even if we fix other control variables, we still cannot determine whether the changes in scores are caused by different parameters or the model itself. We only record the experimental results obtained at that time. Due to hardware limitations, we were only able to test each experiment once, making it difficult to conduct multiple trials.

## 6. Conclusion

### Key Takeaways:
- In previous experiment, we observe that not using Reranker is a better way compare with Llama LLM Reranker and sentence-transformer Reranker, since retrieve top 5 with no reranker has the highest score of 0.32.
- However, after we optimize LLM Reranker by using OpenAI model, we found that when retrieve top 20 Openai LLM rerank 3 it has a new best result.
- The later experiment shows that rerankers is still needed in particularly situation.
- The query classify technique can provide minor performance improvement, but the classifier need to be trained for every new tasks. The filter function should be carefully defined and validated to avoid overfitting.

### Challenges and Future Work:

- Addressing high miss rates remains a priority.
- Further exploration of hybrid retrieval-generation pipelines could yield additional gains.
- Automatically fintuning classifier and defining the filter function.

## 6. Team Organization

(Equal contribution)

- **Brian J. Chan\***
- **Chao-Ting Chen\***
- **Jui-Hung Cheng\***
- **Kuei-Chung Chen\***

**References**

- 2024 KDD Cup CRAG Workshop Papers. [KDD cup](https://openreview.net/group?id=KDD.org/2024/Workshop/KDD_Cup_CRAG#tab-accept)
- What is Reranking in Retrieval-Augmented Generation (RAG)?[Medium Article](https://medium.com/@sahin.samia/what-is-reranking-in-retrieval-augmented-generation-rag-ee3dd93540ee)
- Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks [arXiv](https://arxiv.org/abs/2412.15605)

**Appendix**

- [Repository Link](https://gitlab.aicrowd.com/chaoting_chen/meta-comphrehensive-rag-benchmark-starter-kit)
