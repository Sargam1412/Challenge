# ✨ GNN Challenge: Liar Nodes✨

##  The Problem


For most neural networks, classification tasks are made individually based on the embedding of each data point. Samples that belong to the same class tend to have similar embeddings and therefore lie close to each other in the embedding space.

**Graph Neural Networks (GNNs)** fundamentally alter this paradigm by performing **collective decision making**. Rather than relying solely on the representation (embedding) of an individual node, the classification decision is influenced by the structure of the graph and by aggregated information from neighboring nodes.



### 💡 The Core Question

**This raises an important question: what should a model do when there is a contradiction between the information provided by a node's embedding and the information coming from its neighborhood? AKA adversary learning**

---
![Alt text](picture.png)

## 🎮 Challenge Overview

In this challenge, the goal is to implement mechanisms that balance these two sources of information that might contradict each other. Node features are manually corrupted, while graph connectivity encodes contextual relationships that may either reinforce or contradict a node's individual features.

We apply this concept to Cancer biology! A cell's environment can give misleading cues , a malignant cell might “look normal” in isolation, or neighboring immune cells might influence its signaling,the node’s own features (gene expression) might conflict with neighborhood information (similar cells in the microenvironment), mimicking real biological “contradictions.”

Participants must adapt GNN models using techniques such as **neighborhood sampling** and **aggregation** to learn when to trust the node embedding, when to trust the neighborhood, and how to effectively combine both in order to perform node classification and detect the cancerous cells.



###  Critical Constraint

**The provided node embeddings are fixed and cannot be modified or recomputed**; participants must rely solely on graph-based aggregation and sampling strategies to resolve conflicting signals.

---

## 💭 Hint

> **This challenge is not about inventing a new GNN, it is about choosing the right way to listen.**
> 
> You can’t tell which cells are sending corrupted signals , you must be clever about how you gather and trust information from the tumor microenvironment!

---

##  What Data You Will Use

The Cancer Single-cell Expression Map (CancerSCEM) dataset provides comprehensive single-cell RNA-seq data across multiple cancer types. Instead of raw sequencing reads, the dataset offers UMI count matrices in standard .tsv format for downstream analysis. Each dataset includes:
Gene Expression Matrices per cell  enabling gene descriptio/Cell-Type Annotations and Components /Functional Molecules /Cell-Cell Interaction Data .
You can read more about this data source on https://ngdc.cncb.ac.cn/cancerscem/index .
As for this task, we have altered samples from this data to present it for a  GNN training tasks, you will be given:

### 🔗 **A (Adjacency Matrix)**
- **Absolute truth** ✓
- Contains information about the connections and structure of your input graph
- This is your reliable source for understanding interractions between the cells , you are encouraged to read about cancer cells behavior for a better solution!

### 📈 **X (Node Embeddings)**
- Embeddings of nodes == Genes encodings
- ⚠️ **Some nodes have been corrupted to trick you!**
- You cannot modify these embeddings

### 🎭 **Mask (Corruption Indicator)**
- Gives you a hint about which nodes have been corrupted
- Available in `train.csv`
- **NOT available in `test.csv`** (this is what you need to test on)

### 📁 **Data Files**
- **`train.npz`** - Contains features AND corruption mask(part1)
- **`test.csv`** - Contains features only (no labels, no mask)
- **`labels.csv`** - Contains labels for the full training data
- **`edges.csv`** - Describes the communication between the cells
  For dimentions issues we split the training data features in two , you would find additional code to help you import the data as a dataframe.
---

##  Example Techniques

You can use the notebook in baseline as your starting point! It is a simple example solution  that I made!
We want the GNN to  learn when to rely on a node’s own features versus its neighbors’ right ?. An example of a solution is a trainable gate mechanism that dynamically balances these two signals, adapting to potential feature corruption without modifying the fixed embeddings. The model is trained on the provided graph and features, then predicts on test data using the same adaptive trust logic.
You can use the same GNN as your baseline model , same for the training loop , get creative with your aggregation mecanism!

---

## 🚫 Constraints

###  You CANNOT
- Remove the noisy nodes based on the mask from the training data
- Alter the embedding of the nodes
- Use external data

###  You CAN:
- Use the mask to guide your aggregation strategy
- Implement creative neighborhood sampling
- Design custom aggregation functions
- Use attention mechanisms
- Combine multiple GNN layers strategically
- Look up additional resources to understand the bahavior of the cancerous cells.
---

##  Your Mission ✨✨

Build a GNN that can navigate the noise and make accurate predictions by:

1. **Learning when to trust** individual node features
2. **Learning when to trust** neighborhood aggregation
3. **Build** robust GNNs can detect anomalies or corrupted signals — similar to spotting a misbehaving cell in a tumor.
## Evaluation Metrics
Your submissions will be evaluated on the test set using the overall classification correctness 

## 🏆 Winning Criteria
The winner will be determined by the submission that achieves the best classification accuracy .
###  Submission Process

1. Fork this repository
2. Add your predictions: `submissions/submission_<username>.csv`
3. Create Pull Request:
   *Base : Main
   *Name Submission: <your_username>
   *Compare: your fork/branch with the submission file
5. Leaderboard updated within 24 hours :https://noormajdoub.github.io/Challenge/ 

Good luck, and may the best aggregation strategy win! 
## Resources

Convex Adversarial Collective Classification ;MohamadAli Torkamani, Daniel Lowd Proceedings of the 30th International Conference on Machine Learning, PMLR 28(1):642-650, 2013.

Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

The Reason Why Cancer is so Hard to Beat -Kurzgesagt – In a Nutshell

A Multimodal Graph Neural Network Framework of Cancer Molecular Subtype Classification

Cancer Single-cell Expression Map

Graph Neural Networks in Cancer and Oncology Research: Emerging and Future Trends Grigoriy Gogoshin 1,*, Andrei S Rodin 1,

---
PS: the code used to create the challenge dataset is defined in challenge_data notebook , as a participants please ignore it.

