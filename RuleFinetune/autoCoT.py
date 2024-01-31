import json
import itertools, operator, random
import re
from typing import List, Dict, Union, Tuple

import numpy as np
# import torch

from utils.data import DatasetLoader, Rule, Rationale
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def fix_seed(seed):  # 来源于autoCoT
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

class Demo:
    """
    一个question + 一个rationale + 一个prediction + 一个gold_label
    """

    def __init__(self, question: str, rationale: str, gold_label: str, cnt: float = 0):
        self.question = question
        self.rationale = rationale
        self.gold_label = gold_label


class DemoBase:
    """
    每个Demo看成多臂老虎机的一个臂
    """

    def __init__(self, demos: List[Demo] = None):
        if demos is None:
            demos = []
        self.demos = demos

        self.K = len(demos)
        self.probs = np.array([1 / self.K] * self.K)  # 每个臂的获奖概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率

class DemoBaseMAB:
    def __init__(self, bandit: DemoBase): # 可能需要做一个add bandit的操作，也可以就用zero的
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆的尝试次数

        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def add_demo(self, demo: Demo):
        self.bandit.demos.append(demo)
        self.bandit.K += 1
        # 将新引入的demo赋予一个初始概率
        self.bandit.probs = np.append(self.bandit.probs, 1 / self.bandit.K)
        self.bandit.probs /= np.sum(self.bandit.probs)
        self.bandit.best_idx = np.argmax(self.bandit.probs)
        self.bandit.best_prob = self.bandit.probs[self.bandit.best_idx]
        self.counts = np.append(self.counts, 0)

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def forward_step(self, topk: int = 5):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def backward_step(self, k, r):
        # 更新拉杆k的获奖概率,由每个具体的策略实现
        raise NotImplementedError

    def forward(self, topk: int = 5) -> List[int]:
        # 运行一定次数,num_steps为总运行次数
        # for _ in range(num_steps):
        k = self.forward_step(topk)

        return k

    def backward(self, k: list, r: Union[list, float]):
        if not isinstance(r, list):
            r = [r] * len(k)

        self.backward_step(k, r)
        for i in range(len(k)):
            self.counts[k[i]] += 1
            self.update_regret(k[i])
            self.actions.append(k[i])

    def sample_demos(self, topk: int = 5) -> Tuple[List[int], List[Demo]]:
        raise NotImplementedError

    def save(self, save_path):
        # 保存相关信息
        with open(save_path, 'w', encoding="utf-8") as write_f:
            # demo+对应的概率
            demo_prob = {self.bandit.demos[i].question: self.bandit.probs[i] for i in range(self.bandit.K)}
            json.dump(demo_prob, write_f, indent=4, ensure_ascii=False)


class ThompsonSampling(DemoBaseMAB):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def add_demo(self, demo: Demo):
        super(ThompsonSampling, self).add_demo(demo)
        self._a = np.append(self._a, 1)
        self._b = np.append(self._b, 1)

    def forward_step(self, topk: int = 5) -> List[int]:
        # 为并行的不同步做妥协。等待的话就会出现多个限速环节（因为希望第t个训练集可能用到t-x轮更新后的参数）
        min_length = min(len(self._a), len(self._b), len(self.bandit.probs))
        used_a = self._a[:min_length]
        used_b = self._b[:min_length]
        samples = np.random.beta(used_a, used_b)  # 按照Beta分布采样一组奖励样本
        # 选出采样奖励最大的前topk个拉杆
        k = np.argpartition(samples, -topk)[-topk:]

        return k

    def backward_step(self, k: list, r: list):
        for i in range(len(k)):
            self._a[k[i]] += r[i]
            self._b[k[i]] += 1 - r[i]

    def sample_demos(self, topk: int = 5) -> Tuple[List[int], List[Demo]]:
        topk = min(topk, len(self.bandit.demos))

        k = self.forward(topk)
        demos = [self.bandit.demos[i] for i in k]

        return k, demos

def demo_cluster(args, dataset: DatasetLoader):
    encoder = SentenceTransformer(args.encoder)
    # encoder = SentenceTransformer(generate_func_mapping[args.llm_model])
    # encoder = generate_func_mapping[args.llm_model]

    corpus = []
    questions = []
    rationales = []
    predictions = []
    gold_labels = []

    for example in dataset:
        c_question = example.question
        c_gold_ans = example.gold_label
        c_question = c_question.strip()

        corpus.append(c_question)
        questions.append(c_question)

        if example.rationale:
            for r in example.Top_k_rationale(k=1):
                c_rationale = r.rationale
                c_pred_ans = r.prediction
                c_rationale = c_rationale.replace("Answer: Let's think step by step.",
                                                  "Let's think step by step.")  # 虽然应该不会被触发
                rationales.append(c_rationale)
                predictions.append(c_pred_ans)
        else:
            rationales.append(None)
            predictions.append(None)

        if args.debug:
            gold_labels.append(c_gold_ans)

    corpus_embeddings = encoder.encode(corpus)  # 只用question来分类

    # Perform kmean clustering s
    clustering_model = KMeans(n_clusters=args.num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for _ in range(args.num_clusters)]

    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for _ in range(args.num_clusters)]
    clustered_idx = [[] for _ in range(args.num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

        if not rationales[sentence_id]:
            continue

        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    demos = {}
    question2cluster = {sentence: f"cluster_{cluster_id}" for cluster_id, sentences in enumerate(clustered_sentences)
                        for sentence in
                        sentences}

    for i in range(len(clustered_dists)):  # 全会用的到吗？
        cluster_demo = []

        print("Cluster ", i + 1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        print(top_min_dist)
        # if not args.sampling == "center":
        # groups = [list(g) for _, g in itertools.groupby(top_min_dist, operator.itemgetter(1))]
        # random.shuffle(groups)
        # top_min_dist = [item for group in groups for item in group]

        for element in top_min_dist:
            min_idx = element[0]
            c_rationale = rationales[clustered_idx[i][min_idx]].strip()
            c_gold_ans = Rationale.clean_prediction(gold_labels[clustered_idx[i][min_idx]])
            c_pred_ans = predictions[clustered_idx[i][min_idx]].strip()
            if c_gold_ans.strip().lower() != c_pred_ans.strip().lower():
                continue

            if len(questions[clustered_idx[i][min_idx]].strip().split()) <= 300 and \
                    c_rationale[-1] == "." and \
                    c_pred_ans != "":  # 太长的样例也放不下

                c_question = questions[clustered_idx[i][min_idx]]
                c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                c_rationale = " ".join(c_rationale.split())

                if args.debug:
                    c_gold_ans = gold_labels[clustered_idx[i][min_idx]]
                else:
                    c_gold_ans = None
            #     demo_element = {
            #         "question": c_question,
            #         "rationale": c_rationale,
            #         "pred_ans": c_pred_ans,
            #         "gold_ans": c_gold_ans,
            #     }
                demo_element = Demo(c_question, c_rationale, c_gold_ans)
                cluster_demo.append(demo_element)
            #     print(c_question)
            #     print(c_rationale)
            #     print(c_pred_ans)
            #     print(c_gold_ans)
            #     print("")
            #
            # if len(cluster_demo) >= 5:
            #     break

        if not cluster_demo:
            print("No demo found in cluster {}".format(i + 1))
        #
        demos[f"cluster_{i}"] = cluster_demo

    for cluster in demos:
        if len(demos[cluster]) == 0:
            demos[cluster] = [demos[d][0] for d in demos if len(demos[d]) > 0]

    # 统一修改一遍score
    with open(args.demo_save_dir, 'w', encoding="utf-8") as write_f:
        save_demos = {k: [d.__dict__ for d in v] for k, v in demos.items()}
        json.dump(save_demos, write_f, indent=4, ensure_ascii=False)

    y_km = clustering_model.fit_predict(corpus_embeddings)
    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(corpus_embeddings)
    centers = pca_model.transform(clustering_model.cluster_centers_)

    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
    plt.scatter(centers[:, 0], centers[:, 1],
                s=250, marker='*', label='centroids',
                edgecolor='black',
                c=np.arange(0, args.num_clusters), cmap=plt.cm.Paired)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.save_dir + "cluster.png", dpi=600)

    # 为每个cluster准备一个MAB，然后cluster的分布就不变了
    bandits = {}
    for cluster in demos:
        bandits[cluster] = ThompsonSampling(DemoBase(demos[cluster]))

    return bandits, question2cluster

def sample_demos_from_MAB(args, bandits: Dict[str, DemoBaseMAB], question2cluster: Dict[str, str]):
    """
    从每个cluster的MAB中采样一个demo
    """
    demos = []
    for cluster in bandits:
        # 从每个cluster的MAB中采样一个demo
        k = bandits[cluster].forward(topk=1)[0]
        demo = bandits[cluster].bandit.demos[k]
        demo_element = {
            "question": demo.question,
            "rationale": demo.rationale,
            "pred_ans": demo.gold_label,
            "gold_ans": demo.gold_label,
        }

        demos.append(demo_element)

    return demos

def create_demo_text(args, demos: List[Demo]) -> str:
    x, z, y = [], [], []

    # with open(args.demo_path, encoding="utf-8") as f:
    #     json_data = json.load(f)
    #     json_data = json_data["demo"]
    for line in demos:
        x.append(line.question)
        z.append(line.rationale)
        y.append(line.gold_label)

    index_list = list(range(len(x)))

    demo_text = ""
    for i in index_list:
        demo_text += x[i].strip() + "\nAnswer:" + " " + z[i].strip() + " " + \
                     args.pred_trigger + " " + y[i].strip() + ".\n\n"

    return demo_text


def n_shot_prompt(args, rules: List[Rule], demos: Union[str, List[Demo]]):
    """
    不包括具体问题的提示词
    demos既可以指用于n-shot个题目+rationale，也可以是个生成好的
    """
    if isinstance(demos, str):
        demo = demos
    elif isinstance(demos, list):
        demo = create_demo_text(args, demos)  # auto n-shot CoT
    else:
        raise TypeError("Incorrect type.")

    added_rules = prompt_rules(rules)
    demos, added_rules = prompt_demos(args, demo, added_rules)

    return demos, added_rules


def llm_n_shot_CoT(llm, added_rules: str, input_text: str, demos: str, **kwargs):
    """
    目前是直接用了autoCoT的写法
    """
    demo_prompt = '''Next, I will give you some examples to indicate how to use existed rules by <retrieved_rule> ''' \
                  '''<retrieved_rule>tag and add the new rules into knowledge base by <new_rule> <new_rule> tag.
                  Examples: \n'''

#     demos = """Context: Donald wanted to go visit his grandfather Jason. His dad Michael said okay and drove him to over to Jason house.
# Question: Jason is Michael's what?
# Answer: <retrieved_rule> Children's grandparents are their parents' parents <retrieved_rule> Given the context, Donald is visiting his grandfather Jason. Michael, Donald's father, is driving him there. According to the rule, if Jason is Donald's grandfather, then Jason must be Michael's father. Therefore, Jason is Michael's father. The answer is father.
#
# Context: Michael is taking his son Donald out for coffee. Donald invited his uncle Christopher to dinner
# Question: Christopher is Michael's what?
# Answer: <new_rule> Brothers' children are your nieces or nephews. <new_rule> <new_rule> Your child's uncle is either your brother or your spouse's brother. <new_rule> Given the context: Michael is taking his son Donald out for coffee, and Donald invited his uncle Christopher to dinner. Applying the rules: Since Christopher is Donald's uncle, we apply the second rule: Your child's uncle is either your brother or your spouse's brother. Since the relationship is through Michael's son, we can infer that Christopher is Michael's brother. Therefore, Christopher is Michael's brother. The answer is brother.
#
# Context: Christopher went out for pizza with his daughter Lucille and his brother Dwight.
# Question: Lucille is Dwight's what?
# Answer: <retrieved_rule> Children of siblings are cousins to each other. <retrieved_rule> <retrieved_rule> Siblings are individuals who share at least one parent. <retrieved_rule> <retrieved_rule> The child of one's sibling is one's niece or nephew. <retrieved_rule> Given the context: 1. Christopher is the father of Lucille. 2. Dwight is the brother of Christopher. Applying the rules: - Since Dwight is Christopher's brother, they are siblings (Rule 2). - Lucille, being the daughter of Christopher, is the child of Dwight's sibling (Christopher). - Therefore, Lucille is Dwight's niece (Rule 3). The answer is niece.
#
# Context: Michael and his son Donald went to the store to by bread. Donald took his brother Russell out to get drinks after a long work week.
# Question: Russell is Michael's what?
# Answer: <retrieved_rule> Children of the same parents are siblings to each other. <retrieved_rule> <new_rule> If Person A is the parent of Person B, and Person B is a sibling of Person C, then Person A is the parent of Person C. <new_rule> From the context: 1. Donald is Michael's son. 2. Donald took his brother Russell out. Applying the rules: - Since Donald is Michael's son, and Russell is Donald's brother, we apply the first rule and acknowledge that Donald and Russell are siblings. - Applying the second rule, since Michael is the parent of Donald, and Donald is a sibling of Russell, it follows that Michael is also the parent of Russell. Answer: Russell is Michael's son. The answer is son.
#
# Context: Jason went bike riding his grandson Joe. Joe's father Dwight was skateboarding.
# Question: Jason is Dwight's what?
# Answer: <retrieved_rule> Parents of one's parents are called grandparents. <retrieved_rule> <new_rule> Fathers are male parents. <new_rule> Since Joe is Jason's grandson, Jason must be the grandparent of Joe. Given that Dwight is Joe's father and therefore a male parent, we can deduce that Jason is Dwight's father or mother. Since the question asks specifically about Jason, who is involved in an activity with his grandson, it is most likely that Jason is male (although the information provided does not explicitly state Jason's gender). Assuming Jason is male: <new_rule> Male grandparents are called grandfathers. <new_rule> Therefore, Jason is Dwight's father. To summarize: Jason (grandfather of Joe) --> Dwight's father. The answer is father."""

    demos = """Context: Asia is a proud sibling of the amazing Donald. Donald has mixed feelings towards his mother, Alma.
Question: Alma is Asia's what?
Answer: <retrieved_rule> If Asia is a sibling of Donald, then Asia and Donald have at least one parent in common. <retrieved_rule> <new_rule> If Donald has mixed feelings towards his mother, Alma, then Alma is Donald's mother. <new_rule> <new_rule> If Alma is Donald's mother and Asia is Donald's sibling, then Alma is also Asia's mother, assuming no step relationships or adoptions are mentioned. <new_rule> Therefore, Alma is Asia's mother. The answer is mother.

Context: Lewis and his brother Scott are fighting again. Lisa took her son Scott to school this morning because he missed the bus.
Question: Lewis is Lisa's what?
Answer: <retrieved_rule> Lisa is Scott's mother. <retrieved_rule> <retrieved_rule> Scott is Lewis's brother. <retrieved_rule> <new_rule> If two people are siblings, they share the same parents. <new_rule> <new_rule> Therefore, if Lisa is Scott's mother and Scott is Lewis's brother, then Lisa is Lewis's mother. <new_rule> Answer: Lewis is Lisa's son. The answer is son.

Context: Jason's wife, Gabrielle, could not figure why Jason was coming home so late. It turned out Jason's grandson, Dan, had asked Jason to teach him how to drive.
Question: Dan is Gabrielle's what?
Answer: <new_rule> If Jason is Dan's grandfather, then Dan is the child of one of Jason's children. <new_rule> <retrieved_rule> If Gabrielle is Jason's wife, and assuming that the family structure follows typical naming conventions without any step-relatives or adoptions, Gabrielle would be Dan's grandmother. <retrieved_rule> Given these rules, Dan is Gabrielle's grandson. The answer is grandson.

Context: Dorothy and her daughter Aida went out to dinner. Aida took her uncle Michael to the grocery store.
Question: Michael is Dorothy's what?
Answer: <new_rule> If Aida is Dorothy's daughter and Aida took her uncle Michael to the grocery store, then Michael is the brother of one of Aida's parents. <new_rule> <new_rule> Since Aida is Dorothy's daughter, if Michael is Aida's uncle, then Michael must be Dorothy's brother. <new_rule> Answer: Michael is Dorothy's brother. The answer is brother.

Context: Dorothy was so proud of her daughter Aida for getting straight A's this semester. Dorothy and her father Jason went for a hike in the mountains.
Question: Jason is Aida's what?
Answer: <retrieved_rule> Dorothy's father is Aida's grandfather. <retrieved_rule> Given the context, Jason is Dorothy's father. Therefore, applying the rule, Jason is Aida's grandfather. The answer is grandfather."""

    input_text = added_rules + \
                 demo_prompt + demos + '\n' + \
                 'Then, please answer this question: \n' + input_text.strip() + "\nAnswer: "

    max_length = 2048
    z = llm.generate_single(input_text=input_text, **kwargs)

    print("输入：", input_text)

    pred = z

    return z, pred


def prompt_rules(rules: List[Rule]) -> str:
    """
    返回所有rules作为prompt

    下面有xx规则，优先从中找，找得到的话，格式是Existed Rule xxx；找不到就自己生成，生成的格式是Rulexxxx。
    write_rules(建议排序)

    加一个选择或者说删除机制，比如score=0的rule直接丢弃
    stages: 把所有的rule按分数排序分成3个stage
    proportion: 根据这个比例进行采样
    第n个阶段中采出rule_num*proportion的样放到rule_name里

    相关prompt部分已经被移除放到autoCoT里
    """
    # out = 'Instruction: '
    # out += 'For your guidance, here are several reference rules. '
    # out += 'Endeavor to adhere to these rules when responding to queries in <retrieved_rule>xxx<retrieved_rule> format. '
    # out += 'However, if adherence is unfeasible, you are permitted to establish your own rules in <new_rule>xxx<new_rule> format. '

    out = '''Instruction: Following are several existed knowledge in knowledge base. When you answer the questions, ''' \
          '''try to use the provided knowledge whenever possible in <retrieved_knowledge>xxx<retrieved_knowledge> format. ''' \
          '''Try not to invent knowledge by yourself unless necessary. But if so, you are permitted to ''' \
          '''establish your own rules in <new_knowledge>xxx<new_knowledge> format.\n'''

    out += 'Knowledge base:\n'
    out += '\n'.join([
        # str(idx+1)+':\t'+rn.content
        rn.content
        for idx, rn in enumerate(rules)])

    out += '\n\n'
    return out


def prompt_demos(args, demos: str, added_rules: str) -> Tuple[str, str]:
    assert args.train | args.test
    demos = demos.replace("<retrieved_rule>", "<rule>")
    demos = demos.replace("<new_rule>", "<rule>")
    demos = demos.replace("<B>", "<rule>").replace("<E>", "<rule>")

    rule_pattern = re.compile(r"<rule>(.+?)<rule>")
    rules = rule_pattern.findall(demos)

    cnt = 0
    for rule in rules:
        if rule.strip() in added_rules:
            cnt += 1

    if cnt < 5:  # 这个阈值可以考虑随epoch增大
        chosen_rules = random.sample(rules, min(len(rules), 5 - cnt))
        added_rules = added_rules.strip() + "\n" + "\n".join(chosen_rules) + "\n\n"

    for rule in rules:
        if rule.strip() in added_rules:
            demos = demos.replace("<rule>" + rule + "<rule>", "<retrieved_rule> " + rule.strip() + " <retrieved_rule>")
        else:
            demos = demos.replace("<rule>" + rule + "<rule>", "<new_rule> " + rule.strip() + " <new_rule>")

    out = ''
    out += demos
    return out, added_rules
