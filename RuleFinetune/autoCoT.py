import json
from random import random
from typing import List, Dict, Union

import numpy as np
# import torch

from utils.data import DatasetLoader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def fix_seed(seed): # 来源于autoCoT
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def demo_cluster(args, dataset: DatasetLoader):
    encoder = SentenceTransformer(args.encoder)

    corpus = []
    questions = []
    rationales = []
    predictions = []
    gold_labels = []

    for example in dataset.data:
        c_question = example.question
        c_rationale = example.rationale.rationale
        c_pred_ans = example.rationale.prediction
        c_gold_ans = example.gold_label

        c_rationale = c_rationale.replace("A: Let's think step by step.", "Let's think step by step.") #虽然应该不会被触发
        c_question = c_question + "\nA:"

        corpus.append(c_question)
        questions.append(c_question)
        rationales.append(c_rationale)
        predictions.append(c_pred_ans)
        if args.debug:
            gold_labels.append(c_gold_ans)

    corpus_embeddings = encoder.encode(corpus)  # 只用question来分类

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=args.num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for _ in range(args.num_clusters)]

    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for _ in range(args.num_clusters)]
    clustered_idx = [[] for _ in range(args.num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    demos = []

    for i in range(len(clustered_dists)): # 全会用的到吗？
        print("Cluster ", i+ 1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        # if not args.sampling == "center":
        random.shuffle(top_min_dist)
        for element in top_min_dist:
            min_idx = element[0]
            c_rationale = rationales[clustered_idx[i][min_idx]].strip()
            c_pred_ans = predictions[clustered_idx[i][min_idx]].strip()

            if len(questions[clustered_idx[i][min_idx]].strip().split()) <= 300 and c_rationale[
                -1] == "." and c_pred_ans != "": # 太长的样例也放不下

                c_question = questions[clustered_idx[i][min_idx]]
                c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                c_rationale = " ".join(c_rationale.split())

                if args.debug:
                    c_gold_ans = gold_labels[clustered_idx[i][min_idx]]
                else:
                    c_gold_ans = None
                demo_element = {
                    "question": c_question,
                    "rationale": c_rationale,
                    "pred_ans": c_pred_ans,
                    "gold_ans": c_gold_ans,
                }
                demos.append(demo_element)
                print(c_question)
                print(c_rationale)
                print(c_pred_ans)
                print(c_gold_ans)
                print("")
                break

    demos_save = {"demo": demos}

    with open(args.demo_save_dir, 'w', encoding="utf-8") as write_f:
        json.dump(demos_save, write_f, indent=4, ensure_ascii=False)

    y_km = clustering_model.fit_predict(corpus_embeddings)
    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(corpus_embeddings)
    centers = pca_model.transform(clustering_model.cluster_centers_)

    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
    plt.scatter(centers[:, 0], centers[:, 1],
                s=250, marker='*', label='centroids',
                edgecolor='black',
                c=np.arange(0, args.num_clusters), cmap=plt.cm.Paired, )
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.save_dir + "cluster.png", dpi=600)

    return demos


def create_demo_text(args, demos: List[Dict[str, str]]) -> str:
    x, z, y = [], [], []

    # with open(args.demo_path, encoding="utf-8") as f:
    #     json_data = json.load(f)
    #     json_data = json_data["demo"]
    for line in demos:
        x.append(line["question"])
        z.append(line["rationale"])
        y.append(line["pred_ans"])

    index_list = list(range(len(x)))

    demo_text = ""
    for i in index_list:
        demo_text += x[i] + " " + z[i] + " " + \
                     args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text


def llm_n_shot_CoT(args, llm, input_text: str, demos: Union[str, List[Dict[str, str]]]):
    """
    目前是直接用了autoCoT的写法
    demos既可以指用于n-shot个题目+rationale，也可以是个生成好的
    """
    if isinstance(demos, str):
        demo = demos
    elif isinstance(demos, list):
        demo = create_demo_text(args, demos) #  auto n-shot CoT
    else:
        raise TypeError("Incorrect type.")

    input_text = demo + input_text

    max_length = 2048
    z = llm.generate_single(input_text=input_text, temperature=0.3, max_length=max_length)

    pred = z

    return z, pred