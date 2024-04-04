# aes-papers
Paper list for automated essay scoring (AES) from 2015 to now.

## 2024
+ **Unveiling the Tapestry of Automated Essay Scoring: A Comprehensive Investigation of Accuracy, Fairness, and Generalizability**
  + Authors: Kaixun Yang, Mladen Raković, Yuyang Li, Quanlong Guan, Dragan Gašević, Guangliang Chen
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/30254
  + <details>
      <summary>Abstract</summary>
      Automatic Essay Scoring (AES) is a well-established educational pursuit that employs machine learning to evaluate student-authored essays. While much effort has been made in this area, current research primarily focuses on either (i) boosting the predictive accuracy of an AES model for a specific prompt (i.e., developing prompt-specific models), which often heavily relies on the use of the labeled data from the same target prompt; or (ii) assessing the applicability of AES models developed on non-target prompts to the intended target prompt (i.e., developing the AES models in a cross-prompt setting). Given the inherent bias in machine learning and its potential impact on marginalized groups, it is imperative to investigate whether such bias exists in current AES methods and, if identified, how it intervenes with an AES model's accuracy and generalizability. Thus, our study aimed to uncover the intricate relationship between an AES model's accuracy, fairness, and generalizability, contributing practical insights for developing effective AES models in real-world education. To this end, we meticulously selected nine prominent AES methods and evaluated their performance using seven distinct metrics on an open-sourced dataset, which contains over 25,000 essays and various demographic information about students such as gender, English language learner status, and economic status. Through extensive evaluations, we demonstrated that: (1) prompt-specific models tend to outperform their cross-prompt counterparts in terms of predictive accuracy; (2) prompt-specific models frequently exhibit a greater bias towards students of different economic statuses compared to cross-prompt models; (3) in the pursuit of generalizability, traditional machine learning models (e.g., SVM) coupled with carefully engineered features hold greater potential for achieving both high accuracy and fairness than complex neural network models.
    </details>
+ **Did the Names I Used within My Essay Affect My Score? Diagnosing Name Biases in Automated Essay Scoring**
  + Authors: Ricardo Muñoz Sánchez, Simon Dobnik, Maria Irena Szawerna, Therese Lindström Tiedemann, Elena Volodina
  + Conference: EACL
  + Link: https://aclanthology.org/2024.caldpseudo-1.10/
  + <details>
      <summary>Abstract</summary>
      Automated essay scoring (AES) of second-language learner essays is a high-stakes task as it can affect the job and educational opportunities a student may have access to. Thus, it becomes imperative to make sure that the essays are graded based on the students’ language proficiency as opposed to other reasons, such as personal names used in the text of the essay. Moreover, most of the research data for AES tends to contain personal identifiable information. Because of that, pseudonymization becomes an important tool to make sure that this data can be freely shared. Thus, our systems should not grade students based on which given names were used in the text of the essay, both for fairness and for privacy reasons. In this paper we explore how given names affect the CEFR level classification of essays of second language learners of Swedish. We use essays containing just one personal name and substitute it for names from lists of given names from four different ethnic origins, namely Swedish, Finnish, Anglo-American, and Arabic. We find that changing the names within the essays has no apparent effect on the classification task, regardless of whether a feature-based or a transformer-based model is used.
    </details>
+ **Autoregressive Score Generation for Multi-trait Essay Scoring**
  + Authors: Heejin Do, Yunsu Kim, Gary Lee
  + Conference: EACL Findings
  + Link: https://aclanthology.org/2024.findings-eacl.115/
  + <details>
      <summary>Abstract</summary>
      Recently, encoder-only pre-trained models such as BERT have been successfully applied in automated essay scoring (AES) to predict a single overall score. However, studies have yet to explore these models in multi-trait AES, possibly due to the inefficiency of replicating BERT-based models for each trait. Breaking away from the existing sole use of *encoder*, we propose an autoregressive prediction of multi-trait scores (ArTS), incorporating a *decoding* process by leveraging the pre-trained T5. Unlike prior regression or classification methods, we redefine AES as a score-generation task, allowing a single model to predict multiple scores. During decoding, the subsequent trait prediction can benefit by conditioning on the preceding trait scores. Experimental results proved the efficacy of ArTS, showing over 5% average improvements in both prompts and traits.
    </details>
## 2023
+ **Aggregating Multiple Heuristic Signals as Supervision for Unsupervised Automated Essay Scoring**
  + Authors: Cong Wang, Zhiwei Jiang, Yafeng Yin, Zifeng Cheng, Shiping Ge, Qing Gu
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.782/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Scoring (AES) aims to evaluate the quality score for input essays. In this work, we propose a novel unsupervised AES approach ULRA, which does not require groundtruth scores of essays for training. The core idea of our ULRA is to use multiple heuristic quality signals as the pseudo-groundtruth, and then train a neural AES model by learning from the aggregation of these quality signals. To aggregate these inconsistent quality signals into a unified supervision, we view the AES task as a ranking problem, and design a special Deep Pairwise Rank Aggregation (DPRA) loss for training. In the DPRA loss, we set a learnable confidence weight for each signal to address the conflicts among signals, and train the neural AES model in a pairwise way to disentangle the cascade effect among partial-order pairs. Experiments on eight prompts of ASPA dataset show that ULRA achieves the state-of-the-art performance compared with previous unsupervised methods in terms of both transductive and inductive settings. Further, our approach achieves comparable performance with many existing domain-adapted supervised models, showing the effectiveness of ULRA. The code is available at https://github.com/tenvence/ulra.
    </details>
+ **Automated evaluation of written discourse coherence using GPT-4**
  + Authors: Ben Naismith, Phoebe Mulcaire, Jill Burstein
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.32/
  + <details>
      <summary>Abstract</summary>
      The popularization of large language models (LLMs) such as OpenAI’s GPT-3 and GPT-4 have led to numerous innovations in the field of AI in education. With respect to automated writing evaluation (AWE), LLMs have reduced challenges associated with assessing writing quality characteristics that are difficult to identify automatically, such as discourse coherence. In addition, LLMs can provide rationales for their evaluations (ratings) which increases score interpretability and transparency. This paper investigates one approach to producing ratings by training GPT-4 to assess discourse coherence in a manner consistent with expert human raters. The findings of the study suggest that GPT-4 has strong potential to produce discourse coherence ratings that are comparable to human ratings, accompanied by clear rationales. Furthermore, the GPT-4 ratings outperform traditional NLP coherence metrics with respect to agreement with human ratings. These results have implications for advancing AWE technology for learning and assessment.
    </details>
+ **Improving Domain Generalization for Prompt-Aware Essay Scoring via Disentangled Representation Learning**
  + Authors: Zhiwei Jiang, Tianyi Gao, Yafeng Yin, Meng Liu, Hua Yu, Zifeng Cheng, Qing Gu
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.696/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Scoring (AES) aims to score essays written in response to specific prompts. Many AES models have been proposed, but most of them are either prompt-specific or prompt-adaptive and cannot generalize well on “unseen” prompts. This work focuses on improving the generalization ability of AES models from the perspective of domain generalization, where the data of target prompts cannot be accessed during training. Specifically, we propose a prompt-aware neural AES model to extract comprehensive representation for essay scoring, including both prompt-invariant and prompt-specific features. To improve the generalization of representation, we further propose a novel disentangled representation learning framework. In this framework, a contrastive norm-angular alignment strategy and a counterfactual self-training strategy are designed to disentangle the prompt-invariant information and prompt-specific information in representation. Extensive experimental results on datasets of both ASAP and TOEFL11 demonstrate the effectiveness of our method under the domain generalization setting.
    </details>
+ **Modeling Structural Similarities between Documents for Coherence Assessment with Graph Convolutional Networks**
  + Authors: Wei Liu, Xiyan Fu, Michael Strube
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.431/
  + <details>
      <summary>Abstract</summary>
      Coherence is an important aspect of text quality, and various approaches have been applied to coherence modeling. However, existing methods solely focus on a single document’s coherence patterns, ignoring the underlying correlation between documents. We investigate a GCN-based coherence model that is capable of capturing structural similarities between documents. Our model first creates a graph structure for each document, from where we mine different subgraph patterns. We then construct a heterogeneous graph for the training corpus, connecting documents based on their shared subgraphs. Finally, a GCN is applied to the heterogeneous graph to model the connectivity relationships. We evaluate our method on two tasks, assessing discourse coherence and automated essay scoring. Results show that our GCN-based model outperforms all baselines, achieving a new state-of-the-art on both tasks.
    </details>
+ **PMAES: Prompt-mapping Contrastive Learning for Cross-prompt Automated Essay Scoring**
  + Authors: Yuan Chen, Xia Li
  + Conference: ACL
  + Link: https://aclanthology.org/2023.acl-long.83/
  + <details>
      <summary>Abstract</summary>
      Current cross-prompt automated essay scoring (AES) is a challenging task due to the large discrepancies between different prompts, such as different genres and expressions. The main goal of current cross-prompt AES systems is to learn enough shared features between the source and target prompts to grade well on the target prompt. However, because the features are captured based on the original prompt representation, they may be limited by being extracted directly between essays. In fact, when the representations of two prompts are more similar, we can gain more shared features between them. Based on this motivation, in this paper, we propose a learning strategy called “prompt-mapping” to learn about more consistent representations of source and target prompts. In this way, we can obtain more shared features between the two prompts and use them to better represent the essays for the target prompt. Experimental results on the ASAP++ dataset demonstrate the effectiveness of our method. We also design experiments in different settings to show that our method can be applied in different scenarios. Our code is available at https://github.com/gdufsnlp/PMAES.
    </details>
+ **Rating Short L2 Essays on the CEFR Scale with GPT-4**
  + Authors: Kevin P. Yancey, Geoffrey Laflair, Anthony Verardi, Jill Burstein
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.49/
  + <details>
      <summary>Abstract</summary>
      Essay scoring is a critical task used to evaluate second-language (L2) writing proficiency on high-stakes language assessments. While automated scoring approaches are mature and have been around for decades, human scoring is still considered the gold standard, despite its high costs and well-known issues such as human rater fatigue and bias. The recent introduction of large language models (LLMs) brings new opportunities for automated scoring. In this paper, we evaluate how well GPT-3.5 and GPT-4 can rate short essay responses written by L2 English learners on a high-stakes language assessment, computing inter-rater agreement with human ratings. Results show that when calibration examples are provided, GPT-4 can perform almost as well as modern Automatic Writing Evaluation (AWE) methods, but agreement with human ratings can vary depending on the test-taker’s first language (L1).
    </details>
+ **Span Identification of Epistemic Stance-Taking in Academic Written English**
  + Authors: Masaki Eguchi, Kristopher Kyle
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.35/
  + <details>
      <summary>Abstract</summary>
      Responding to the increasing need for automated writing evaluation (AWE) systems to assess language use beyond lexis and grammar (Burstein et al., 2016), we introduce a new approach to identify rhetorical features of stance in academic English writing. Drawing on the discourse-analytic framework of engagement in the Appraisal analysis (Martin & White, 2005), we manually annotated 4,688 sentences (126,411 tokens) for eight rhetorical stance categories (e.g., PROCLAIM, ATTRIBUTION) and additional discourse elements. We then report an experiment to train machine learning models to identify and categorize the spans of these stance expressions. The best-performing model (RoBERTa + LSTM) achieved macro-averaged F1 of .7208 in the span identification of stance-taking expressions, slightly outperforming the intercoder reliability estimates before adjudication (F1 = .6629).
    </details>
+ **Towards Extracting and Understanding the Implicit Rubrics of Transformer Based Automatic Essay Scoring Models**
  + Authors: James Fiacco, David Adamson, Carolyn Ros
  + Conference: ACL
  + Link: https://aclanthology.org/2023.bea-1.20/
  + <details>
      <summary>Abstract</summary>
      By aligning the functional components derived from the activations of transformer models trained for AES with external knowledge such as human-understandable feature groups, the proposed method improves the interpretability of a Longformer Automatic Essay Scoring (AES) system and provides tools for performing such analyses on further neural AES systems. The analysis focuses on models trained to score essays based on organization, main idea, support, and language. The findings provide insights into the models’ decision-making processes, biases, and limitations, contributing to the development of more transparent and reliable AES systems.
    </details>
+ **A Comparative Analysis of the Effectiveness of Rare Tokens on Creative Expression using ramBERT**
  + Authors: Youbin Lee, Deokgi Kim, Byung-Won On, Ingyu Lee
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.639/
  + <details>
      <summary>Abstract</summary>
      Until now, few studies have been explored on Automated Creative Essay Scoring (ACES), in which a pre-trained model automatically labels an essay as a creative or a non-creative. Since the creativity evaluation of essays is very subjective, each evaluator often has his or her own criteria for creativity. For this reason, quantifying creativity in essays is very challenging. In this work, as one of preliminary studies in developing a novel model for ACES, we deeply investigate the correlation between creative essays and expressiveness. Specifically, we explore how rare tokens affect the evaluation of creativity for essays. For such a journey, we present five distinct methods to extract rare tokens, and conduct a comparative study on the correlation between rare tokens and creative essay evaluation results using BERT. Our experimental results showed clear correlation between rare tokens and creative essays. In all test sets, accuracies of our rare token masking-based BERT (ramBERT) model were improved over the existing BERT model up to 14%.
    </details>
+ **Prompt- and Trait Relation-aware Cross-prompt Essay Trait Scoring**
  + Authors: Heejin Do, Yunsu Kim, Gary Geunbae Lee
  + Conference: ACL Findings
  + Link: https://aclanthology.org/2023.findings-acl.98/
  + <details>
      <summary>Abstract</summary>
      Automated essay scoring (AES) aims to score essays written for a given prompt, which defines the writing topic. Most existing AES systems assume to grade essays of the same prompt as used in training and assign only a holistic score. However, such settings conflict with real-education situations; pre-graded essays for a particular prompt are lacking, and detailed trait scores of sub-rubrics are required. Thus, predicting various trait scores of unseen-prompt essays (called cross-prompt essay trait scoring) is a remaining challenge of AES. In this paper, we propose a robust model: prompt- and trait relation-aware cross-prompt essay trait scorer. We encode prompt-aware essay representation by essay-prompt attention and utilizing the topic-coherence feature extracted by the topic-modeling mechanism without access to labeled data; therefore, our model considers the prompt adherence of an essay, even in a cross-prompt setting. To facilitate multi-trait scoring, we design trait-similarity loss that encapsulates the correlations of traits. Experiments prove the efficacy of our model, showing state-of-the-art results for all prompts and traits. Significant improvements in low-resource-prompt and inferior traits further indicate our model’s strength.
    </details>
+ **A Multi-Task Dataset for Assessing Discourse Coherence in Chinese Essays: Structure, Theme, and Logic Analysis**
  + Authors: Hongyi Wu, Xinshu Shen, Man Lan, Shaoguang Mao, Xiaopeng Bai, Yuanbin Wu
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.412/
  + <details>
      <summary>Abstract</summary>
      This paper introduces the Chinese Essay Discourse Coherence Corpus (CEDCC), a multi-task dataset for assessing discourse coherence. Existing research tends to focus on isolated dimensions of discourse coherence, a gap which the CEDCC addresses by integrating coherence grading, topical continuity, and discourse relations. This approach, alongside detailed annotations, captures the subtleties of real-world texts and stimulates progress in Chinese discourse coherence analysis. Our contributions include the development of the CEDCC, the establishment of baselines for further research, and the demonstration of the impact of coherence on discourse relation recognition and automated essay scoring. The dataset and related codes is available at https://github.com/cubenlp/CEDCC_corpus.
    </details>
+ **TCFLE-8: a Corpus of Learner Written Productions for French as a Foreign Language and its Application to Automated Essay Scoring**
  + Authors: Rodrigo Wilkens, Alice Pintard, David Alfter, Vincent Folny, Thomas François
  + Conference: EMNLP
  + Link: https://aclanthology.org/2023.emnlp-main.210/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Scoring (AES) aims to automatically assess the quality of essays. Automation enables large-scale assessment, improvements in consistency, reliability, and standardization. Those characteristics are of particular relevance in the context of language certification exams. However, a major bottleneck in the development of AES systems is the availability of corpora, which, unfortunately, are scarce, especially for languages other than English. In this paper, we aim to foster the development of AES for French by providing the TCFLE-8 corpus, a corpus of 6.5k essays collected in the context of the Test de Connaissance du Français (TCF - French Knowledge Test) certification exam. We report the strict quality procedure that led to the scoring of each essay by at least two raters according to the CEFR levels and to the creation of a balanced corpus. In addition, we describe how linguistic properties of the essays relate to the learners’ proficiency in TCFLE-8. We also advance the state-of-the-art performance for the AES task in French by experimenting with two strong baselines (i.e. RoBERTa and feature-based). Finally, we discuss the challenges of AES using TCFLE-8.
    </details>
+ **Learning to love diligent trolls: Accounting for rater effects in the dialogue safety task**
  + Authors: Michael Ilagan
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2023.findings-emnlp.928/
  + <details>
      <summary>Abstract</summary>
      Chatbots have the risk of generating offensive utterances, which must be avoided. Post-deployment, one way for a chatbot to continuously improve is to source utterance/label pairs from feedback by live users. However, among users are trolls, who provide training examples with incorrect labels. To de-troll training data, previous work removed training examples that have high user-aggregated cross-validation (CV) error. However, CV is expensive; and in a coordinated attack, CV may be overwhelmed by trolls in number and in consistency among themselves. In the present work, I address both limitations by proposing a solution inspired by methodology in automated essay scoring (AES): have multiple users rate each utterance, then perform latent class analysis (LCA) to infer correct labels. As it does not require GPU computations, LCA is inexpensive. In experiments, I found that the AES-like solution can infer training labels with high accuracy when trolls are consistent, even when trolls are the majority.
    </details>
+ **H-AES: Towards Automated Essay Scoring for Hindi**
  + Authors: Shubhankar Singh, Anirudh Pupneja, Shivaansh Mital, Cheril Shah, Manish Bawkar, Lakshman Prasad Gupta, Ajit Kumar, Yaman Kumar, Rushali Gupta, Rajiv Ratn Shah
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/26894
  + <details>
      <summary>Abstract</summary>
      The use of Natural Language Processing (NLP) for Automated Essay Scoring (AES) has been well explored in the English language, with benchmark models exhibiting performance comparable to human scorers. However, AES in Hindi and other low-resource languages remains unexplored. In this study, we reproduce and compare state-of-the-art methods for AES in the Hindi domain. We employ classical feature-based Machine Learning (ML) and advanced end-to-end models, including LSTM Networks and Fine-Tuned Transformer Architecture, in our approach and derive results comparable to those in the English language domain. Hindi being a low-resource language, lacks a dedicated essay-scoring corpus. We train and evaluate our models using translated English essays and empirically measure their performance on our own small-scale, real-world Hindi corpus. We follow this up with an in-depth analysis discussing prompt-specific behavior of different language models implemented.
    </details>
+ **On the Effectiveness of Curriculum Learning in Educational Text Scoring**
  + Authors: Zijie Zeng, Dragan Gasevic, Guangliang Chen
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/26707
  + <details>
      <summary>Abstract</summary>
      Automatic Text Scoring (ATS) is a widely-investigated task in education. Existing approaches often stressed the structure design of an ATS model and neglected the training process of the model. Considering the difficult nature of this task, we argued that the performance of an ATS model could be potentially boosted by carefully selecting data of varying complexities in the training process. Therefore, we aimed to investigate the effectiveness of curriculum learning (CL) in scoring educational text. Specifically, we designed two types of difficulty measurers: (i) pre-defined, calculated by measuring a sample's readability, length, the number of grammatical errors or unique words it contains; and (ii) automatic, calculated based on whether a model in a training epoch can accurately score the samples. These measurers were tested in both the easy-to-hard to hard-to-easy training paradigms. Through extensive evaluations on two widely-used datasets (one for short answer scoring and the other for long essay scoring), we demonstrated that (a) CL indeed could boost the performance of state-of-the-art ATS models, and the maximum improvement could be up to 4.5%, but most improvements were achieved when assessing short and easy answers; (b) the pre-defined measurer calculated based on the number of grammatical errors contained in a text sample tended to outperform the other difficulty measurers across different training paradigms.
    </details>
+ **Towards automatic essay scoring of Basque language texts from a rule-based approach based on curriculum-aware systems**
  + Authors: Jose Maria Arriola, Mikel Iruskieta, Ekain Arrieta, Jon Alkorta
  + Conference: WS
  + Link: https://aclanthology.org/2023.nodalida-cgmta.4/
  + <details>
      <summary>Abstract</summary>
      Although the Basque Education Law mentions that students must finish secondary compulsory education at B2 Basque level and their undergraduate studies at the C1 level, there are no objective tests or tools that can discriminate between these levels. This work presents the first rule-based method to grade written Basque learner texts. We adapt the adult Basque learner curriculum based on the CEFR to create a rule-based grammar for Basque. This paper summarises the results obtained in different classification tasks by combining information formalised through CG3 and different machine learning algorithms used in text classification. Besides, we perform a manual evaluation of the grammar. Finally, we discuss the informa- tiveness of these rules and some ways to further improve assisted text grading and combine rule-based approaches with other approaches based on readability and complexity measures.
    </details>
## 2022
+ **Many Hands Make Light Work: Using Essay Traits to Automatically Score Essays**
  + Authors: Rahul Kumar, Sandeep Mathias, Sriparna Saha, Pushpak Bhattacharyya
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.106/
  + <details>
      <summary>Abstract</summary>
      Most research in the area of automatic essay grading (AEG) is geared towards scoring the essay holistically while there has also been little work done on scoring individual essay traits. In this paper, we describe a way to score essays using a multi-task learning (MTL) approach, where scoring the essay holistically is the primary task, and scoring the essay traits is the auxiliary task. We compare our results with a single-task learning (STL) approach, using both LSTMs and BiLSTMs. To find out which traits work best for different types of essays, we conduct ablation tests for each of the essay traits. We also report the runtime and number of training parameters for each system. We find that MTL-based BiLSTM system gives the best results for scoring the essay holistically, as well as performing well on scoring the essay traits. The MTL systems also give a speed-up of between 2.30 to 3.70 times the speed of the STL system, when it comes to scoring the essay and all the traits.
    </details>
+ **On the Use of Bert for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation**
  + Authors: Yongjie Wang, Chuang Wang, Ruobing Li, Hui Lin
  + Conference: NAACL
  + Link: https://aclanthology.org/2022.naacl-main.249/
  + <details>
      <summary>Abstract</summary>
      In recent years, pre-trained models have become dominant in most natural language processing (NLP) tasks. However, in the area of Automated Essay Scoring (AES), pre-trained models such as BERT have not been properly used to outperform other deep learning models such as LSTM. In this paper, we introduce a novel multi-scale essay representation for BERT that can be jointly learned. We also employ multiple losses and transfer learning from out-of-domain essays to further improve the performance. Experiment results show that our approach derives much benefit from joint learning of multi-scale essay representation and obtains almost the state-of-the-art result among all deep learning models in the ASAP task. Our multi-scale essay representation also generalizes well to CommonLit Readability Prize data set, which suggests that the novel text representation proposed in this paper may be a new and effective choice for long-text tasks.
    </details>
+ **Analytic Automated Essay Scoring Based on Deep Neural Networks Integrating Multidimensional Item Response Theory**
  + Authors: Takumi Shibata, Masaki Uto
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.257/
  + <details>
      <summary>Abstract</summary>
      Essay exams have been attracting attention as a way of measuring the higher-order abilities of examinees, but they have two major drawbacks in that grading them is expensive and raises questions about fairness. As an approach to overcome these problems, automated essay scoring (AES) is in increasing need. Many AES models based on deep neural networks have been proposed in recent years and have achieved high accuracy, but most of these models are designed to predict only a single overall score. However, to provide detailed feedback in practical situations, we often require not only the overall score but also analytic scores corresponding to various aspects of the essay. Several neural AES models that can predict both the analytic scores and the overall score have also been proposed for this very purpose. However, conventional models are designed to have complex neural architectures for each analytic score, which makes interpreting the score prediction difficult. To improve the interpretability of the prediction while maintaining scoring accuracy, we propose a new neural model for automated analytic scoring that integrates a multidimensional item response theory model, which is a popular psychometric model.
    </details>
+ **Automated Chinese Essay Scoring from Multiple Traits**
  + Authors: Yaqiong He, Feng Jiang, Xiaomin Chu, Peifeng Li
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.266/
  + <details>
      <summary>Abstract</summary>
      Automatic Essay Scoring (AES) is the task of using the computer to evaluate the quality of essays automatically. Current research on AES focuses on scoring the overall quality or single trait of prompt-specific essays. However, the users not only expect to obtain the overall score but also the instant feedback from different traits to help their writing in the real world. Therefore, we first annotate a mutli-trait dataset ACEA including 1220 argumentative essays from four traits, i.e., essay organization, topic, logic, and language. And then we design a hierarchical multi-task trait scorer HMTS to evaluate the quality of writing by modeling these four traits. Moreover, we propose an inter-sequence attention mechanism to enhance information interaction between different tasks and design the trait-specific features for various tasks in AES. The experimental results on ACEA show that our HMTS can effectively score essays from multiple traits, outperforming several strong models.
    </details>
+ **Automated Essay Scoring via Pairwise Contrastive Regression**
  + Authors: Jiayi Xie, Kaiwei Cai, Li Kong, Junsheng Zhou, Weiguang Qu
  + Conference: COLING
  + Link: https://aclanthology.org/2022.coling-1.240/
  + <details>
      <summary>Abstract</summary>
      Automated essay scoring (AES) involves the prediction of a score relating to the writing quality of an essay. Most existing works in AES utilize regression objectives or ranking objectives respectively. However, the two types of methods are highly complementary. To this end, in this paper we take inspiration from contrastive learning and propose a novel unified Neural Pairwise Contrastive Regression (NPCR) model in which both objectives are optimized simultaneously as a single loss. Specifically, we first design a neural pairwise ranking model to guarantee the global ranking order in a large list of essays, and then we further extend this pairwise ranking model to predict the relative scores between an input essay and several reference essays. Additionally, a multi-sample voting strategy is employed for inference. We use Quadratic Weighted Kappa to evaluate our model on the public Automated Student Assessment Prize (ASAP) dataset, and the experimental results demonstrate that NPCR outperforms previous methods by a large margin, achieving the state-of-the-art average performance for the AES task.
    </details>
+ **ImageArg: A Multi-modal Tweet Dataset for Image Persuasiveness Mining**
  + Authors: Zhexiong Liu, Meiqi Guo, Yue Dai, Diane Litman
  + Conference: WS
  + Link: https://aclanthology.org/2022.argmining-1.1/
  + <details>
      <summary>Abstract</summary>
      The growing interest in developing corpora of persuasive texts has promoted applications in automated systems, e.g., debating and essay scoring systems; however, there is little prior work mining image persuasiveness from an argumentative perspective. To expand persuasiveness mining into a multi-modal realm, we present a multi-modal dataset, ImageArg, consisting of annotations of image persuasiveness in tweets. The annotations are based on a persuasion taxonomy we developed to explore image functionalities and the means of persuasion. We benchmark image persuasiveness tasks on ImageArg using widely-used multi-modal learning methods. The experimental results show that our dataset offers a useful resource for this rich and challenging topic, and there is ample room for modeling improvement.
    </details>
+ **MWE for Essay Scoring English as a Foreign Language**
  + Authors: Rodrigo Wilkens, Daiane Seibert, Xiaoou Wang, Thomas François
  + Conference: LREC
  + Link: https://aclanthology.org/2022.readi-1.9/
  + <details>
      <summary>Abstract</summary>
      Mastering a foreign language like English can bring better opportunities. In this context, although multiword expressions (MWE) are associated with proficiency, they are usually neglected in the works of automatic scoring language learners. Therefore, we study MWE-based features (i.e., occurrence and concreteness) in this work, aiming at assessing their relevance for automated essay scoring. To achieve this goal, we also compare MWE features with other classic features, such as length-based, graded resource, orthographic neighbors, part-of-speech, morphology, dependency relations, verb tense, language development, and coherence. Although the results indicate that classic features are more significant than MWE for automatic scoring, we observed encouraging results when looking at the MWE concreteness through the levels.
    </details>
## 2021
+ **IFlyEA: A Chinese Essay Assessment System with Automated Rating, Review Generation, and Recommendation**
  + Authors: Jiefu Gong, Xiao Hu, Wei Song, Ruiji Fu, Zhichao Sheng, Bo Zhu, Shijin Wang, Ting Liu
  + Conference: ACL
  + Link: https://aclanthology.org/2021.acl-demo.29/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Assessment (AEA) aims to judge students’ writing proficiency in an automatic way. This paper presents a Chinese AEA system IFlyEssayAssess (IFlyEA), targeting on evaluating essays written by native Chinese students from primary and junior schools. IFlyEA provides multi-level and multi-dimension analytical modules for essay assessment. It has state-of-the-art grammar level analysis techniques, and also integrates components for rhetoric and discourse level analysis, which are important for evaluating native speakers’ writing ability, but still challenging and less studied in previous work. Based on the comprehensive analysis, IFlyEA provides application services for essay scoring, review generation, recommendation, and explainable analytical visualization. These services can benefit both teachers and students during the process of writing teaching and learning.
    </details>
+ **Countering the Influence of Essay Length in Neural Essay Scoring**
  + Authors: Sungho Jeon, Michael Strube
  + Conference: EMNLP
  + Link: https://aclanthology.org/2021.sustainlp-1.4/
  + <details>
      <summary>Abstract</summary>
      Previous work has shown that automated essay scoring systems, in particular machine learning-based systems, are not capable of assessing the quality of essays, but are relying on essay length, a factor irrelevant to writing proficiency. In this work, we first show that state-of-the-art systems, recent neural essay scoring systems, might be also influenced by the correlation between essay length and scores in a standard dataset. In our evaluation, a very simple neural model shows the state-of-the-art performance on the standard dataset. To consider essay content without taking essay length into account, we introduce a simple neural model assessing the similarity of content between an input essay and essays assigned different scores. This neural model achieves performance comparable to the state of the art on a standard dataset as well as on a second dataset. Our findings suggest that neural essay scoring systems should consider the characteristics of datasets to focus on text quality.
    </details>
+ **Automated Cross-prompt Scoring of Essay Traits**
  + Authors: Robert Ridley, Liang He, Xin-yu Dai, Shujian Huang, Jiajun Chen
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/17620
  + <details>
      <summary>Abstract</summary>
      The majority of current research in Automated Essay Scoring (AES) focuses on prompt-specific scoring of either the overall quality of an essay or the quality with regards to certain traits. In real-world applications obtaining labelled data for a target essay prompt is often expensive or unfeasible, requiring the AES system to be able to perform well when predicting scores for essays from unseen prompts. As a result, some recent research has been dedicated to cross-prompt AES. However, this line of research has thus far only been concerned with holistic, overall scoring, with no exploration into the scoring of different traits. As users of AES systems often require feedback with regards to different aspects of their writing, trait scoring is a necessary component of an effective AES system. Therefore, to address this need, we introduce a new task named Automated Cross-prompt Scoring of Essay Traits, which requires the model to be trained solely on non-target-prompt essays and to predict the holistic, overall score as well as scores for a number of specific traits for target-prompt essays. This task challenges the model's ability to generalize in order to score essays from a novel domain as well as its ability to represent the quality of essays from multiple different aspects. In addition, we introduce a new, innovative approach which builds on top of a state-of-the-art method for cross-prompt AES. Our method utilizes a trait-attention mechanism and a multi-task architecture that leverages the relationships between each trait to simultaneously predict the overall score and the score of each individual trait. We conduct extensive experiments on the widely used ASAP and ASAP++ datasets and demonstrate that our approach is able to outperform leading prompt-specific trait scoring and cross-prompt AES methods.
    </details>
+ **Hierarchical Coherence Modeling for Document Quality Assessment**
  + Authors: Dongliang Liao, Jin Xu, Gongfu Li, Yiru Wang
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/17576
  + <details>
      <summary>Abstract</summary>
      Text coherence plays a key role in document quality assessment. Most existing text coherence methods only focus on similarity of adjacent sentences. However, local coherence exists in sentences with broader contexts and diverse rhetoric relations, rather than just adjacent sentences similarity. Besides, the highlevel text coherence is also an important aspect of document quality. To this end, we propose a hierarchical coherence model for document quality assessment. In our model, we implement a local attention mechanism to capture the location semantics, bilinear tensor layer for measure coherence and max-coherence pooling for acquiring high-level coherence. We evaluate the proposed method on two realistic tasks: news quality judgement and automated essay scoring.   Experimental results demonstrate the validity and superiority of our work.
    </details>
+ **Essay Quality Signals as Weak Supervision for Source-based Essay Scoring**
  + Authors: Haoran Zhang, Diane Litman
  + Conference: EACL
  + Link: https://aclanthology.org/2021.bea-1.9/
  + <details>
      <summary>Abstract</summary>
      Human essay grading is a laborious task that can consume much time and effort. Automated Essay Scoring (AES) has thus been proposed as a fast and effective solution to the problem of grading student writing at scale. However, because AES typically uses supervised machine learning, a human-graded essay corpus is still required to train the AES model. Unfortunately, such a graded corpus often does not exist, so creating a corpus for machine learning can also be a laborious task. This paper presents an investigation of replacing the use of human-labeled essay grades when training an AES system with two automatically available but weaker signals of essay quality: word count and topic distribution similarity. Experiments using two source-based essay scoring (evidence score) corpora show that while weak supervision does not yield a competitive result when training a neural source-based AES model, it can be used to successfully extract Topical Components (TCs) from a source text, which are required by a supervised feature-based AES model. In particular, results show that feature-based AES performance is comparable with either automatically or manually constructed TCs.
    </details>
+ **A Prompt-independent and Interpretable Automated Essay Scoring Method for Chinese Second Language Writing**
  + Authors: Wang Yupei, Hu Renfen
  + Conference: CCL
  + Link: https://aclanthology.org/2021.ccl-1.107/
  + <details>
      <summary>Abstract</summary>
      With the increasing popularity of learning Chinese as a second language (L2) the development of an automatic essay scoring (AES) method specially for Chinese L2 essays has become animportant task. To build a robust model that could easily adapt to prompt changes we propose 90linguistic features with consideration of both language complexity and correctness and introducethe Ordinal Logistic Regression model that explicitly combines these linguistic features and low-level textual representations. Our model obtains a high QWK of 0.714 a low RMSE of 1.516 anda considerable Pearson correlation of 0.734. With a simple linear model we further analyze the contribution of the linguistic features to score prediction revealing the model’s interpretability and its potential to give writing feedback to users. This work provides insights and establishes asolid baseline for Chinese L2 AES studies.
    </details>
## 2020
+ **Annotation and Classification of Evidence and Reasoning Revisions in Argumentative Writing**
  + Authors: Tazin Afrin, Elaine Lin Wang, Diane Litman, Lindsay Clare Matsumura, Richard Correnti
  + Conference: ACL
  + Link: https://aclanthology.org/2020.bea-1.7/
  + <details>
      <summary>Abstract</summary>
      Automated writing evaluation systems can improve students’ writing insofar as students attend to the feedback provided and revise their essay drafts in ways aligned with such feedback. Existing research on revision of argumentative writing in such systems, however, has focused on the types of revisions students make (e.g., surface vs. content) rather than the extent to which revisions actually respond to the feedback provided and improve the essay. We introduce an annotation scheme to capture the nature of sentence-level revisions of evidence use and reasoning (the ‘RER’ scheme) and apply it to 5th- and 6th-grade students’ argumentative essays. We show that reliable manual annotation can be achieved and that revision annotations correlate with a holistic assessment of essay improvement in line with the feedback provided. Furthermore, we explore the feasibility of automatically classifying revisions according to our scheme.
    </details>
+ **Automated Evaluation of Writing – 50 Years and Counting**
  + Authors: Beata Beigman Klebanov, Nitin Madnani
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-main.697/
  + <details>
      <summary>Abstract</summary>
      In this theme paper, we focus on Automated Writing Evaluation (AWE), using Ellis Page’s seminal 1966 paper to frame the presentation. We discuss some of the current frontiers in the field and offer some thoughts on the emergent uses of this technology.
    </details>
+ **Automated Topical Component Extraction Using Neural Network Attention Scores from Source-based Essay Scoring**
  + Authors: Haoran Zhang, Diane Litman
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-main.759/
  + <details>
      <summary>Abstract</summary>
      While automated essay scoring (AES) can reliably grade essays at scale, automated writing evaluation (AWE) additionally provides formative feedback to guide essay revision. However, a neural AES typically does not provide useful feature representations for supporting AWE. This paper presents a method for linking AWE and neural AES, by extracting Topical Components (TCs) representing evidence from a source text using the intermediate output of attention layers. We evaluate performance using a feature-based AES requiring TCs. Results show that performance is comparable whether using automatically or manually constructed TCs for 1) representing essays as rubric-based features, 2) grading essays.
    </details>
+ **Can Neural Networks Automatically Score Essay Traits?**
  + Authors: Sandeep Mathias, Pushpak Bhattacharyya
  + Conference: ACL
  + Link: https://aclanthology.org/2020.bea-1.8/
  + <details>
      <summary>Abstract</summary>
      Essay traits are attributes of an essay that can help explain how well written (or badly written) the essay is. Examples of traits include Content, Organization, Language, Sentence Fluency, Word Choice, etc. A lot of research in the last decade has dealt with automatic holistic essay scoring - where a machine rates an essay and gives a score for the essay. However, writers need feedback, especially if they want to improve their writing - which is why trait-scoring is important. In this paper, we show how a deep-learning based system can outperform feature-based machine learning systems, as well as a string kernel system in scoring essay traits.
    </details>
+ **Exploiting Personal Characteristics of Debaters for Predicting Persuasiveness**
  + Authors: Khalid Al Khatib, Michael Völske, Shahbaz Syed, Nikolay Kolyada, Benno Stein
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-main.632/
  + <details>
      <summary>Abstract</summary>
      Predicting the persuasiveness of arguments has applications as diverse as writing assistance, essay scoring, and advertising. While clearly relevant to the task, the personal characteristics of an argument’s source and audience have not yet been fully exploited toward automated persuasiveness prediction. In this paper, we model debaters’ prior beliefs, interests, and personality traits based on their previous activity, without dependence on explicit user profiles or questionnaires. Using a dataset of over 60,000 argumentative discussions, comprising more than three million individual posts collected from the subreddit r/ChangeMyView, we demonstrate that our modeling of debater’s characteristics enhances the prediction of argument persuasiveness as well as of debaters’ resistance to persuasion.
    </details>
+ **Exploring the Effect of Author and Reader Identity in Online Story Writing: the STORIESINTHEWILD Corpus.**
  + Authors: Tal August, Maarten Sap, Elizabeth Clark, Katharina Reinecke, Noah A. Smith
  + Conference: ACL
  + Link: https://aclanthology.org/2020.nuse-1.6/
  + <details>
      <summary>Abstract</summary>
      Current story writing or story editing systems rely on human judgments of story quality for evaluating performance, often ignoring the subjectivity in ratings. We analyze the effect of author and reader characteristics and story writing setup on the quality of stories in a short storytelling task. To study this effect, we create and release STORIESINTHEWILD, containing 1,630 stories collected on a volunteer-based crowdsourcing platform. Each story is rated by three different readers, and comes paired with the author’s and reader’s age, gender, and personality. Our findings show significant effects of authors’ and readers’ identities, as well as writing setup, on story writing and ratings. Notably, compared to younger readers, readers age 45 and older consider stories significantly less creative and less entertaining. Readers also prefer stories written all at once, rather than in chunks, finding them more coherent and creative. We also observe linguistic differences associated with authors’ demographics (e.g., older authors wrote more vivid and emotional stories). Our findings suggest that reader and writer demographics, as well as writing setup, should be accounted for in story writing evaluations.
    </details>
+ **LinggleWrite: a Coaching System for Essay Writing**
  + Authors: Chung-Ting Tsai, Jhih-Jie Chen, Ching-Yu Yang, Jason S. Chang
  + Conference: ACL
  + Link: https://aclanthology.org/2020.acl-demos.17/
  + <details>
      <summary>Abstract</summary>
      This paper presents LinggleWrite, a writing coach that provides writing suggestions, assesses writing proficiency levels, detects grammatical errors, and offers corrective feedback in response to user’s essay. The method involves extracting grammar patterns, training models for automated essay scoring (AES) and grammatical error detection (GED), and finally retrieving plausible corrections from a n-gram search engine. Experiments on public test sets indicate that both AES and GED models achieve state-of-the-art performance. These results show that LinggleWrite is potentially useful in helping learners improve their writing skills.
    </details>
+ **Multiple Instance Learning for Content Feedback Localization without Annotation**
  + Authors: Scott Hellman, William Murray, Adam Wiemerslage, Mark Rosenstein, Peter Foltz, Lee Becker, Marcia Derr
  + Conference: ACL
  + Link: https://aclanthology.org/2020.bea-1.3/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Scoring (AES) can be used to automatically generate holistic scores with reliability comparable to human scoring. In addition, AES systems can provide formative feedback to learners, typically at the essay level. In contrast, we are interested in providing feedback specialized to the content of the essay, and specifically for the content areas required by the rubric. A key objective is that the feedback should be localized alongside the relevant essay text. An important step in this process is determining where in the essay the rubric designated points and topics are discussed. A natural approach to this task is to train a classifier using manually annotated data; however, collecting such data is extremely resource intensive. Instead, we propose a method to predict these annotation spans without requiring any labeled annotation data. Our approach is to consider AES as a Multiple Instance Learning (MIL) task. We show that such models can both predict content scores and localize content by leveraging their sentence-level score predictions. This capability arises despite never having access to annotation training data. Implications are discussed for improving formative feedback and explainable AES models.
    </details>
+ **Should You Fine-Tune BERT for Automated Essay Scoring?**
  + Authors: Elijah Mayfield, Alan W Black
  + Conference: ACL
  + Link: https://aclanthology.org/2020.bea-1.15/
  + <details>
      <summary>Abstract</summary>
      Most natural language processing research now recommends large Transformer-based models with fine-tuning for supervised classification tasks; older strategies like bag-of-words features and linear models have fallen out of favor. Here we investigate whether, in automated essay scoring (AES) research, deep neural models are an appropriate technological choice. We find that fine-tuning BERT produces similar performance to classical models at significant additional cost. We argue that while state-of-the-art strategies do match existing best results, they come with opportunity costs in computational resources. We conclude with a review of promising areas for research on student essays where the unique characteristics of Transformers may provide benefits over classical methods to justify the costs.
    </details>
+ **Centering-based Neural Coherence Modeling with Hierarchical Discourse Segments**
  + Authors: Sungho Jeon, Michael Strube
  + Conference: EMNLP
  + Link: https://aclanthology.org/2020.emnlp-main.604/
  + <details>
      <summary>Abstract</summary>
      Previous neural coherence models have focused on identifying semantic relations between adjacent sentences. However, they do not have the means to exploit structural information. In this work, we propose a coherence model which takes discourse structural information into account without relying on human annotations. We approximate a linguistic theory of coherence, Centering theory, which we use to track the changes of focus between discourse segments. Our model first identifies the focus of each sentence, recognized with regards to the context, and constructs the structural relationship for discourse segments by tracking the changes of the focus. The model then incorporates this structural information into a structure-aware transformer. We evaluate our model on two tasks, automated essay scoring and assessing writing quality. Our results demonstrate that our model, built on top of a pretrained language model, achieves state-of-the-art performance on both tasks. We next statistically examine the identified trees of texts assigned to different quality scores. Finally, we investigate what our model learns in terms of theoretical claims.
    </details>
+ **Multi-Stage Pre-training for Automated Chinese Essay Scoring**
  + Authors: Wei Song, Kai Zhang, Ruiji Fu, Lizhen Liu, Ting Liu, Miaomiao Cheng
  + Conference: EMNLP
  + Link: https://aclanthology.org/2020.emnlp-main.546/
  + <details>
      <summary>Abstract</summary>
      This paper proposes a pre-training based automated Chinese essay scoring method. The method involves three components: weakly supervised pre-training, supervised cross- prompt fine-tuning and supervised target- prompt fine-tuning. An essay scorer is first pre- trained on a large essay dataset covering diverse topics and with coarse ratings, i.e., good and poor, which are used as a kind of weak supervision. The pre-trained essay scorer would be further fine-tuned on previously rated es- says from existing prompts, which have the same score range with the target prompt and provide extra supervision. At last, the scorer is fine-tuned on the target-prompt training data. The evaluation on four prompts shows that this method can improve a state-of-the-art neural essay scorer in terms of effectiveness and domain adaptation ability, while in-depth analysis also reveals its limitations..
    </details>
+ **Enhancing Automated Essay Scoring Performance via Fine-tuning Pre-trained Language Models with Combination of Regression and Ranking**
  + Authors: Ruosong Yang, Jiannong Cao, Zhiyuan Wen, Youzheng Wu, Xiaodong He
  + Conference: EMNLP Findings
  + Link: https://aclanthology.org/2020.findings-emnlp.141/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Scoring (AES) is a critical text regression task that automatically assigns scores to essays based on their writing quality. Recently, the performance of sentence prediction tasks has been largely improved by using Pre-trained Language Models via fusing representations from different layers, constructing an auxiliary sentence, using multi-task learning, etc. However, to solve the AES task, previous works utilize shallow neural networks to learn essay representations and constrain calculated scores with regression loss or ranking loss, respectively. Since shallow neural networks trained on limited samples show poor performance to capture deep semantic of texts. And without an accurate scoring function, ranking loss and regression loss measures two different aspects of the calculated scores. To improve AES’s performance, we find a new way to fine-tune pre-trained language models with multiple losses of the same task. In this paper, we propose to utilize a pre-trained language model to learn text representations first. With scores calculated from the representations, mean square error loss and the batch-wise ListNet loss with dynamic weights constrain the scores simultaneously. We utilize Quadratic Weighted Kappa to evaluate our model on the Automated Student Assessment Prize dataset. Our model outperforms not only state-of-the-art neural models near 3 percent but also the latest statistic model. Especially on the two narrative prompts, our model performs much better than all other state-of-the-art models.
    </details>
+ **Neural Automated Essay Scoring Incorporating Handcrafted Features**
  + Authors: Masaki Uto, Yikuan Xie, Maomi Ueno
  + Conference: COLING
  + Link: https://aclanthology.org/2020.coling-main.535/
  + <details>
      <summary>Abstract</summary>
      Automated essay scoring (AES) is the task of automatically assigning scores to essays as an alternative to grading by human raters. Conventional AES typically relies on handcrafted features, whereas recent studies have proposed AES models based on deep neural networks (DNNs) to obviate the need for feature engineering. Furthermore, hybrid methods that integrate handcrafted features in a DNN-AES model have been recently developed and have achieved state-of-the-art accuracy. One of the most popular hybrid methods is formulated as a DNN-AES model with an additional recurrent neural network (RNN) that processes a sequence of handcrafted sentence-level features. However, this method has the following problems: 1) It cannot incorporate effective essay-level features developed in previous AES research. 2) It greatly increases the numbers of model parameters and tuning parameters, increasing the difficulty of model training. 3) It has an additional RNN to process sentence-level features, enabling extension to various DNN-AES models complex. To resolve these problems, we propose a new hybrid method that integrates handcrafted essay-level features into a DNN-AES model. Specifically, our method concatenates handcrafted essay-level features to a distributed essay representation vector, which is obtained from an intermediate layer of a DNN-AES model. Our method is a simple DNN-AES extension, but significantly improves scoring accuracy.
    </details>
+ **An Exploratory Study into Automated Précis Grading**
  + Authors: Orphee De Clercq, Senne Van Hoecke
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.50/
  + <details>
      <summary>Abstract</summary>
      Automated writing evaluation is a popular research field, but the main focus has been on evaluating argumentative essays. In this paper, we consider a different genre, namely précis texts. A précis is a written text that provides a coherent summary of main points of a spoken or written text. We present a corpus of English précis texts which all received a grade assigned by a highly-experienced English language teacher and were subsequently annotated following an exhaustive error typology. With this corpus we trained a machine learning model which relies on a number of linguistic, automatic summarization and AWE features. Our results reveal that this model is able to predict the grade of précis texts with only a moderate error margin.
    </details>
+ **Automated Essay Scoring System for Nonnative Japanese Learners**
  + Authors: Reo Hirao, Mio Arai, Hiroki Shimanaka, Satoru Katsumata, Mamoru Komachi
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.157/
  + <details>
      <summary>Abstract</summary>
      In this study, we created an automated essay scoring (AES) system for nonnative Japanese learners using an essay dataset with annotations for a holistic score and multiple trait scores, including content, organization, and language scores. In particular, we developed AES systems using two different approaches: a feature-based approach and a neural-network-based approach. In the former approach, we used Japanese-specific linguistic features, including character-type features such as “kanji” and “hiragana.” In the latter approach, we used two models: a long short-term memory (LSTM) model (Hochreiter and Schmidhuber, 1997) and a bidirectional encoder representations from transformers (BERT) model (Devlin et al., 2019), which achieved the highest accuracy in various natural language processing tasks in 2018. Overall, the BERT model achieved the best root mean squared error and quadratic weighted kappa scores. In addition, we analyzed the robustness of the outputs of the BERT model. We have released and shared this system to facilitate further research on AES for Japanese as a second language learners.
    </details>
+ **Computing with Subjectivity Lexicons**
  + Authors: Caio L. M. Jeronimo, Claudio E. C. Campelo, Leandro Balby Marinho, Allan Sales, Adriano Veloso, Roberta Viola
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.400/
  + <details>
      <summary>Abstract</summary>
      In this paper, we introduce a new set of lexicons for expressing subjectivity in text documents written in Brazilian Portuguese. Besides the non-English idiom, in contrast to other subjectivity lexicons available, these lexicons represent different subjectivity dimensions (other than sentiment) and are more compact in number of terms. This last feature was designed intentionally to leverage the power of word embedding techniques, i.e., with the words mapped to an embedding space and the appropriate distance measures, we can easily capture semantically related words to the ones in the lexicons. Thus, we do not need to build comprehensive vocabularies and can focus on the most representative words for each lexicon dimension. We showcase the use of these lexicons in three highly non-trivial tasks: (1) Automated Essay Scoring in the Presence of Biased Ratings, (2) Subjectivity Bias in Brazilian Presidential Elections and (3) Fake News Classification Based on Text Subjectivity. All these tasks involve text documents written in Portuguese.
    </details>
+ **Happy Are Those Who Grade without Seeing: A Multi-Task Learning Approach to Grade Essays Using Gaze Behaviour**
  + Authors: Sandeep Mathias, Rudra Murthy, Diptesh Kanojia, Abhijit Mishra, Pushpak Bhattacharyya
  + Conference: AACL
  + Link: https://aclanthology.org/2020.aacl-main.86/
  + <details>
      <summary>Abstract</summary>
      The gaze behaviour of a reader is helpful in solving several NLP tasks such as automatic essay grading. However, collecting gaze behaviour from readers is costly in terms of time and money. In this paper, we propose a way to improve automatic essay grading using gaze behaviour, which is learnt at run time using a multi-task learning framework. To demonstrate the efficacy of this multi-task learning based approach to automatic essay grading, we collect gaze behaviour for 48 essays across 4 essay sets, and learn gaze behaviour for the rest of the essays, numbering over 7000 essays. Using the learnt gaze behaviour, we can achieve a statistically significant improvement in performance over the state-of-the-art system for the essay sets where we have gaze data. We also achieve a statistically significant improvement for 4 other essay sets, numbering about 6000 essays, where we have no gaze behaviour data available. Our approach establishes that learning gaze behaviour improves automatic essay grading.
    </details>
+ **Language Proficiency Scoring**
  + Authors: Cristina Arhiliuc, Jelena Mitrović, Michael Granitzer
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.690/
  + <details>
      <summary>Abstract</summary>
      The Common European Framework of Reference (CEFR) provides generic guidelines for the evaluation of language proficiency. Nevertheless, for automated proficiency classification systems, different approaches for different languages are proposed. Our paper evaluates and extends the results of an approach to Automatic Essay Scoring proposed as a part of the REPROLANG 2020 challenge. We provide a comparison between our results and the ones from the published paper and we include a new corpus for the English language for further experiments. Our results are lower than the expected ones when using the same approach and the system does not scale well with the added English corpus.
    </details>
+ **Multi-task Learning for Automated Essay Scoring with Sentiment Analysis**
  + Authors: Panitan Muangkammuen, Fumiyo Fukumoto
  + Conference: AACL
  + Link: https://aclanthology.org/2020.aacl-srw.17/
  + <details>
      <summary>Abstract</summary>
      Automated Essay Scoring (AES) is a process that aims to alleviate the workload of graders and improve the feedback cycle in educational systems. Multi-task learning models, one of the deep learning techniques that have recently been applied to many NLP tasks, demonstrate the vast potential for AES. In this work, we present an approach for combining two tasks, sentiment analysis, and AES by utilizing multi-task learning. The model is based on a hierarchical neural network that learns to predict a holistic score at the document-level along with sentiment classes at the word-level and sentence-level. The sentiment features extracted from opinion expressions can enhance a vanilla holistic essay scoring, which mainly focuses on lexicon and text semantics. Our approach demonstrates that sentiment features are beneficial for some essay prompts, and the performance is competitive to other deep learning models on the Automated StudentAssessment Prize (ASAP) benchmark. TheQuadratic Weighted Kappa (QWK) is used to measure the agreement between the human grader’s score and the model’s prediction. Ourmodel produces a QWK of 0.763.
    </details>
+ **REPROLANG 2020: Automatic Proficiency Scoring of Czech, English, German, Italian, and Spanish Learner Essays**
  + Authors: Andrew Caines, Paula Buttery
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.689/
  + <details>
      <summary>Abstract</summary>
      We report on our attempts to reproduce the work described in Vajjala & Rama 2018, ‘Experiments with universal CEFR classification’, as part of REPROLANG 2020: this involves featured-based and neural approaches to essay scoring in Czech, German and Italian. Our results are broadly in line with those from the original paper, with some differences due to the stochastic nature of machine learning and programming language used. We correct an error in the reported metrics, introduce new baselines, apply the experiments to English and Spanish corpora, and generate adversarial data to test classifier robustness. We conclude that feature-based approaches perform better than neural network classifiers for text datasets of this size, though neural network modifications do bring performance closer to the best feature-based models.
    </details>
+ **Reproduction and Replication: A Case Study with Automatic Essay Scoring**
  + Authors: Eva Huber, Çağrı Çöltekin
  + Conference: LREC
  + Link: https://aclanthology.org/2020.lrec-1.688/
  + <details>
      <summary>Abstract</summary>
      As in many experimental sciences, reproducibility of experiments has gained ever more attention in the NLP community. This paper presents our reproduction efforts of an earlier study of automatic essay scoring (AES) for determining the proficiency of second language learners in a multilingual setting. We present three sets of experiments with different objectives. First, as prescribed by the LREC 2020 REPROLANG shared task, we rerun the original AES system using the code published by the original authors on the same dataset. Second, we repeat the same experiments on the same data with a different implementation. And third, we test the original system on a different dataset and a different language. Most of our findings are in line with the findings of the original paper. Nevertheless, there are some discrepancies between our results and the results presented in the original paper. We report and discuss these differences in detail. We further go into some points related to confirmation of research findings through reproduction, including the choice of the dataset, reporting and accounting for variability, use of appropriate evaluation metrics, and making code and data available. We also discuss the varying uses and differences between the terms reproduction and replication, and we argue that reproduction, the confirmation of conclusions through independent experiments in varied settings is more valuable than exact replication of the published values.
    </details>
+ **Scientific Writing Evaluation Using Ensemble Multi-channel Neural Networks**
  + Authors: Yuh-Shyang Wang, Lung-Hao Lee, Bo-Lin Lin, Liang-Chih Yu
  + Conference: ROCLING
  + Link: https://aclanthology.org/2020.rocling-1.32/

## 2019
+ **Automated Essay Scoring with Discourse-Aware Neural Models**
  + Authors: Farah Nadeem, Huy Nguyen, Yang Liu, Mari Ostendorf
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4450/
  + <details>
      <summary>Abstract</summary>
      Automated essay scoring systems typically rely on hand-crafted features to predict essay quality, but such systems are limited by the cost of feature engineering. Neural networks offer an alternative to feature engineering, but they typically require more annotated data. This paper explores network structures, contextualized embeddings and pre-training strategies aimed at capturing discourse characteristics of essays. Experiments on three essay scoring tasks show benefits from all three strategies in different combinations, with simpler architectures being more effective when less training data is available.
    </details>
+ **Discourse Analysis and Its Applications**
  + Authors: Shafiq Joty, Giuseppe Carenini, Raymond Ng, Gabriel Murray
  + Conference: ACL
  + Link: https://aclanthology.org/P19-4003/
  + <details>
      <summary>Abstract</summary>
      Discourse processing is a suite of Natural Language Processing (NLP) tasks to uncover linguistic structures from texts at several levels, which can support many downstream applications. This involves identifying the topic structure, the coherence structure, the coreference structure, and the conversation structure for conversational discourse. Taken together, these structures can inform text summarization, machine translation, essay scoring, sentiment analysis, information extraction, question answering, and thread recovery. The tutorial starts with an overview of basic concepts in discourse analysis – monologue vs. conversation, synchronous vs. asynchronous conversation, and key linguistic structures in discourse analysis. We also give an overview of linguistic structures and corresponding discourse analysis tasks that discourse researchers are generally interested in, as well as key applications on which these discourse structures have an impact.
    </details>
+ **Equity Beyond Bias in Language Technologies for Education**
  + Authors: Elijah Mayfield, Michael Madaio, Shrimai Prabhumoye, David Gerritsen, Brittany McLaughlin, Ezekiel Dixon-Román, Alan W Black
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4446/
  + <details>
      <summary>Abstract</summary>
      There is a long record of research on equity in schools. As machine learning researchers begin to study fairness and bias in earnest, language technologies in education have an unusually strong theoretical and applied foundation to build on. Here, we introduce concepts from culturally relevant pedagogy and other frameworks for teaching and learning, identifying future work on equity in NLP. We present case studies in a range of topics like intelligent tutoring systems, computer-assisted language learning, automated essay scoring, and sentiment analysis in classrooms, and provide an actionable agenda for research.
    </details>
+ **Give Me More Feedback II: Annotating Thesis Strength and Related Attributes in Student Essays**
  + Authors: Zixuan Ke, Hrishikesh Inamdar, Hui Lin, Vincent Ng
  + Conference: ACL
  + Link: https://aclanthology.org/P19-1390/
  + <details>
      <summary>Abstract</summary>
      While the vast majority of existing work on automated essay scoring has focused on holistic scoring, researchers have recently begun work on scoring specific dimensions of essay quality. Nevertheless, progress on dimension-specific essay scoring is limited in part by the lack of annotated corpora. To facilitate advances in this area, we design a scoring rubric for scoring a core, yet unexplored dimension of persuasive essay quality, thesis strength, and annotate a corpus of essays with thesis strength scores. We additionally identify the attributes that could impact thesis strength and annotate the essays with the values of these attributes, which, when predicted by computational models, could provide further feedback to students on why her essay receives a particular thesis strength score.
    </details>
+ **Lexical concreteness in narrative**
  + Authors: Michael Flor, Swapna Somasundaran
  + Conference: ACL
  + Link: https://aclanthology.org/W19-3408/
  + <details>
      <summary>Abstract</summary>
      This study explores the relation between lexical concreteness and narrative text quality. We present a methodology to quantitatively measure lexical concreteness of a text. We apply it to a corpus of student stories, scored according to writing evaluation rubrics. Lexical concreteness is weakly-to-moderately related to story quality, depending on story-type. The relation is mostly borne by adjectives and nouns, but also found for adverbs and verbs.
    </details>
+ **Regression or classification? Automated Essay Scoring for Norwegian**
  + Authors: Stig Johan Berggren, Taraka Rama, Lilja Øvrelid
  + Conference: ACL
  + Link: https://aclanthology.org/W19-4409/
  + <details>
      <summary>Abstract</summary>
      In this paper we present first results for the task of Automated Essay Scoring for Norwegian learner language. We analyze a number of properties of this task experimentally and assess (i) the formulation of the task as either regression or classification, (ii) the use of various non-neural and neural machine learning architectures with various types of input representations, and (iii) applying multi-task learning for joint prediction of essay scoring and native language identification. We find that a GRU-based attention model trained in a single-task setting performs best at the AES task.
    </details>
+ **Unsupervised Learning of Discourse-Aware Text Representation for Essay Scoring**
  + Authors: Farjana Sultana Mim, Naoya Inoue, Paul Reisert, Hiroki Ouchi, Kentaro Inui
  + Conference: ACL
  + Link: https://aclanthology.org/P19-2053/
  + <details>
      <summary>Abstract</summary>
      Existing document embedding approaches mainly focus on capturing sequences of words in documents. However, some document classification and regression tasks such as essay scoring need to consider discourse structure of documents. Although some prior approaches consider this issue and utilize discourse structure of text for document classification, these approaches are dependent on computationally expensive parsers. In this paper, we propose an unsupervised approach to capture discourse structure in terms of coherence and cohesion for document embedding that does not require any expensive parser or annotation. Extrinsic evaluation results show that the document representation obtained from our approach improves the performance of essay Organization scoring and Argument Strength scoring.
    </details>
+ **eRevise: Using Natural Language Processing to Provide Formative Feedback on Text Evidence Usage in Student Writing**
  + Authors: H. Zhang, A. Magooda, D. Litman, R. Correnti, E. Wang, L.C. Matsmura, E. Howe, R. Quintana
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/5025
  + <details>
      <summary>Abstract</summary>
      Writing a good essay typically involves students revising an initial paper draft after receiving feedback. We present eRevise, a web-based writing and revising environment that uses natural language processing features generated for rubricbased essay scoring to trigger formative feedback messages regarding students’ use of evidence in response-to-text writing. By helping students understand the criteria for using text evidence during writing, eRevise empowers students to better revise their paper drafts. In a pilot deployment of eRevise in 7 classrooms spanning grades 5 and 6, the quality of text evidence usage in writing improved after students received formative feedback then engaged in paper revision.
    </details>
## 2018
+ **Augmenting Textual Qualitative Features in Deep Convolution Recurrent Neural Network for Automatic Essay Scoring**
  + Authors: Tirthankar Dasgupta, Abir Naskar, Lipika Dey, Rupsa Saha
  + Conference: ACL
  + Link: https://aclanthology.org/W18-3713/
  + <details>
      <summary>Abstract</summary>
      In this paper we present a qualitatively enhanced deep convolution recurrent neural network for computing the quality of a text in an automatic essay scoring task. The novelty of the work lies in the fact that instead of considering only the word and sentence representation of a text, we try to augment the different complex linguistic, cognitive and psycological features associated within a text document along with a hierarchical convolution recurrent neural network framework. Our preliminary investigation shows that incorporation of such qualitative feature vectors along with standard word/sentence embeddings can give us better understanding about improving the overall evaluation of the input essays.
    </details>
+ **Automated essay scoring with string kernels and word embeddings**
  + Authors: Mădălina Cozma, Andrei Butnaru, Radu Tudor Ionescu
  + Conference: ACL
  + Link: https://aclanthology.org/P18-2080/
  + <details>
      <summary>Abstract</summary>
      In this work, we present an approach based on combining string kernels and word embeddings for automatic essay scoring. String kernels capture the similarity among strings based on counting common character n-grams, which are a low-level yet powerful type of feature, demonstrating state-of-the-art results in various text classification tasks such as Arabic dialect identification or native language identification. To our best knowledge, we are the first to apply string kernels to automatically score essays. We are also the first to combine them with a high-level semantic feature representation, namely the bag-of-super-word-embeddings. We report the best performance on the Automated Student Assessment Prize data set, in both in-domain and cross-domain settings, surpassing recent state-of-the-art deep learning approaches.
    </details>
+ **Give Me More Feedback: Annotating Argument Persuasiveness and Related Attributes in Student Essays**
  + Authors: Winston Carlile, Nishant Gurrapadi, Zixuan Ke, Vincent Ng
  + Conference: ACL
  + Link: https://aclanthology.org/P18-1058/
  + <details>
      <summary>Abstract</summary>
      While argument persuasiveness is one of the most important dimensions of argumentative essay quality, it is relatively little studied in automated essay scoring research. Progress on scoring argument persuasiveness is hindered in part by the scarcity of annotated corpora. We present the first corpus of essays that are simultaneously annotated with argument components, argument persuasiveness scores, and attributes of argument components that impact an argument’s persuasiveness. This corpus could trigger the development of novel computational models concerning argument persuasiveness that provide useful feedback to students on why their arguments are (un)persuasive in addition to how persuasive they are.
    </details>
+ **TDNN: A Two-stage Deep Neural Network for Prompt-independent Automated Essay Scoring**
  + Authors: Cancan Jin, Ben He, Kai Hui, Le Sun
  + Conference: ACL
  + Link: https://aclanthology.org/P18-1100/
  + <details>
      <summary>Abstract</summary>
      Existing automated essay scoring (AES) models rely on rated essays for the target prompt as training data. Despite their successes in prompt-dependent AES, how to effectively predict essay ratings under a prompt-independent setting remains a challenge, where the rated essays for the target prompt are not available. To close this gap, a two-stage deep neural network (TDNN) is proposed. In particular, in the first stage, using the rated essays for non-target prompts as the training data, a shallow model is learned to select essays with an extreme quality for the target prompt, serving as pseudo training data; in the second stage, an end-to-end hybrid deep model is proposed to learn a prompt-dependent rating model consuming the pseudo training data from the first step. Evaluation of the proposed TDNN on the standard ASAP dataset demonstrates a promising improvement for the prompt-independent AES task.
    </details>
+ **A Neural Local Coherence Model for Text Quality Assessment**
  + Authors: Mohsen Mesgar, Michael Strube
  + Conference: EMNLP
  + Link: https://aclanthology.org/D18-1464/
  + <details>
      <summary>Abstract</summary>
      We propose a local coherence model that captures the flow of what semantically connects adjacent sentences in a text. We represent the semantics of a sentence by a vector and capture its state at each word of the sentence. We model what relates two adjacent sentences based on the two most similar semantic states, each of which is in one of the sentences. We encode the perceived coherence of a text by a vector, which represents patterns of changes in salient information that relates adjacent sentences. Our experiments demonstrate that our approach is beneficial for two downstream tasks: Readability assessment, in which our model achieves new state-of-the-art results; and essay scoring, in which the combination of our coherence vectors and other task-dependent features significantly improves the performance of a strong essay scorer.
    </details>
+ **Automatic Essay Scoring Incorporating Rating Schema via Reinforcement Learning**
  + Authors: Yucheng Wang, Zhongyu Wei, Yaqian Zhou, Xuanjing Huang
  + Conference: EMNLP
  + Link: https://aclanthology.org/D18-1090/
  + <details>
      <summary>Abstract</summary>
      Automatic essay scoring (AES) is the task of assigning grades to essays without human interference. Existing systems for AES are typically trained to predict the score of each single essay at a time without considering the rating schema. In order to address this issue, we propose a reinforcement learning framework for essay scoring that incorporates quadratic weighted kappa as guidance to optimize the scoring system. Experiment results on benchmark datasets show the effectiveness of our framework.
    </details>
+ **Atypical Inputs in Educational Applications**
  + Authors: Su-Youn Yoon, Aoife Cahill, Anastassia Loukina, Klaus Zechner, Brian Riordan, Nitin Madnani
  + Conference: NAACL
  + Link: https://aclanthology.org/N18-3008/
  + <details>
      <summary>Abstract</summary>
      In large-scale educational assessments, the use of automated scoring has recently become quite common. While the majority of student responses can be processed and scored without difficulty, there are a small number of responses that have atypical characteristics that make it difficult for an automated scoring system to assign a correct score. We describe a pipeline that detects and processes these kinds of responses at run-time. We present the most frequent kinds of what are called non-scorable responses along with effective filtering models based on various NLP and speech processing technologies. We give an overview of two operational automated scoring systems —one for essay scoring and one for speech scoring— and describe the filtering models they use. Finally, we present an evaluation and analysis of filtering models used for spoken responses in an assessment of language proficiency.
    </details>
+ **Automated Essay Scoring in the Presence of Biased Ratings**
  + Authors: Evelin Amorim, Marcia Cançado, Adriano Veloso
  + Conference: NAACL
  + Link: https://aclanthology.org/N18-1021/
  + <details>
      <summary>Abstract</summary>
      Studies in Social Sciences have revealed that when people evaluate someone else, their evaluations often reflect their biases. As a result, rater bias may introduce highly subjective factors that make their evaluations inaccurate. This may affect automated essay scoring models in many ways, as these models are typically designed to model (potentially biased) essay raters. While there is sizeable literature on rater effects in general settings, it remains unknown how rater bias affects automated essay scoring. To this end, we present a new annotated corpus containing essays and their respective scores. Different from existing corpora, our corpus also contains comments provided by the raters in order to ground their scores. We present features to quantify rater bias based on their comments, and we found that rater bias plays an important role in automated essay scoring. We investigated the extent to which rater bias affects models based on hand-crafted features. Finally, we propose to rectify the training set by removing essays associated with potentially biased scores while learning the scoring model.
    </details>
+ **Co-Attention Based Neural Network for Source-Dependent Essay Scoring**
  + Authors: Haoran Zhang, Diane Litman
  + Conference: NAACL
  + Link: https://aclanthology.org/W18-0549/
  + <details>
      <summary>Abstract</summary>
      This paper presents an investigation of using a co-attention based neural network for source-dependent essay scoring. We use a co-attention mechanism to help the model learn the importance of each part of the essay more accurately. Also, this paper shows that the co-attention based neural network model provides reliable score prediction of source-dependent responses. We evaluate our model on two source-dependent response corpora. Results show that our model outperforms the baseline on both corpora. We also show that the attention of the model is similar to the expert opinions with examples.
    </details>
+ **Neural Automated Essay Scoring and Coherence Modeling for Adversarially Crafted Input**
  + Authors: Youmna Farag, Helen Yannakoudakis, Ted Briscoe
  + Conference: NAACL
  + Link: https://aclanthology.org/N18-1024/
  + <details>
      <summary>Abstract</summary>
      We demonstrate that current state-of-the-art approaches to Automated Essay Scoring (AES) are not well-suited to capturing adversarially crafted input of grammatical but incoherent sequences of sentences. We develop a neural model of local coherence that can effectively learn connectedness features between sentences, and propose a framework for integrating and jointly training the local coherence model with a state-of-the-art AES model. We evaluate our approach against a number of baselines and experimentally demonstrate its effectiveness on both the AES task and the task of flagging adversarial input, further contributing to the development of an approach that strengthens the validity of neural essay scoring models.
    </details>
+ **Argument Mining for Improving the Automated Scoring of Persuasive Essays**
  + Authors: Huy Nguyen, Diane Litman
  + Conference: AAAI
  + Link: https://ojs.aaai.org/index.php/AAAI/article/view/12046
  + <details>
      <summary>Abstract</summary>
      End-to-end argument mining has enabled the development of new automated essay scoring (AES) systems that use argumentative features (e.g., number of claims, number of support relations) in addition to traditional legacy features (e.g., grammar, discourse structure) when scoring persuasive essays. While prior research has proposed different argumentative features as well as empirically demonstrated their utility for AES, these studies have all had important limitations.  In this paper we identify a set of desiderata for evaluating the use of argument mining for AES, introduce an end-to-end argument mining system and associated argumentative feature sets, and present the results of several studies that both satisfy the desiderata and demonstrate the value-added of argument mining for scoring persuasive essays.
    </details>
## 2017
+ **Discourse Mode Identification in Essays**
  + Authors: Wei Song, Dong Wang, Ruiji Fu, Lizhen Liu, Ting Liu, Guoping Hu
  + Conference: ACL
  + Link: https://aclanthology.org/P17-1011/
  + <details>
      <summary>Abstract</summary>
      Discourse modes play an important role in writing composition and evaluation. This paper presents a study on the manual and automatic identification of narration,exposition, description, argument and emotion expressing sentences in narrative essays. We annotate a corpus to study the characteristics of discourse modes and describe a neural sequence labeling model for identification. Evaluation results show that discourse modes can be identified automatically with an average F1-score of 0.7. We further demonstrate that discourse modes can be used as features that improve automatic essay scoring (AES). The impacts of discourse modes for AES are also discussed.
    </details>
+ **A Multi-aspect Analysis of Automatic Essay Scoring for Brazilian Portuguese**
  + Authors: Evelin Amorim, Adriano Veloso
  + Conference: EACL
  + Link: https://aclanthology.org/E17-4010/
  + <details>
      <summary>Abstract</summary>
      Several methods for automatic essay scoring (AES) for English language have been proposed. However, multi-aspect AES systems for other languages are unusual. Therefore, we propose a multi-aspect AES system to apply on a dataset of Brazilian Portuguese essays, which human experts evaluated according to five aspects defined by Brazilian Government to the National Exam to High School Student (ENEM). These aspects are skills that student must master and every skill is assessed apart from each other. Besides the prediction of each aspect, the feature analysis also was performed for each aspect. The AES system proposed employs several features already employed by AES systems for English language. Our results show that predictions for some aspects performed well with the features we employed, while predictions for other aspects performed poorly. Also, it is possible to note the difference between the five aspects in the detailed feature analysis we performed. Besides these contributions, the eight millions of enrollments every year for ENEM raise some challenge issues for future directions in our research.
    </details>
+ **Attention-based Recurrent Convolutional Neural Network for Automatic Essay Scoring**
  + Authors: Fei Dong, Yue Zhang, Jie Yang
  + Conference: CoNLL
  + Link: https://aclanthology.org/K17-1017/
  + <details>
      <summary>Abstract</summary>
      Neural network models have recently been applied to the task of automatic essay scoring, giving promising results. Existing work used recurrent neural networks and convolutional neural networks to model input essays, giving grades based on a single vector representation of the essay. On the other hand, the relative advantages of RNNs and CNNs have not been compared. In addition, different parts of the essay can contribute differently for scoring, which is not captured by existing models. We address these issues by building a hierarchical sentence-document model to represent essays, using the attention mechanism to automatically decide the relative weights of words and sentences. Results show that our model outperforms the previous state-of-the-art methods, demonstrating the effectiveness of the attention mechanism.
    </details>
+ **Detecting Off-topic Responses to Visual Prompts**
  + Authors: Marek Rei
  + Conference: WS
  + Link: https://aclanthology.org/W17-5020/
  + <details>
      <summary>Abstract</summary>
      Automated methods for essay scoring have made great progress in recent years, achieving accuracies very close to human annotators. However, a known weakness of such automated scorers is not taking into account the semantic relevance of the submitted text. While there is existing work on detecting answer relevance given a textual prompt, very little previous research has been done to incorporate visual writing prompts. We propose a neural architecture and several extensions for detecting off-topic responses to visual prompts and evaluate it on a dataset of texts written by language learners.
    </details>
+ **Exploring Relationships Between Writing & Broader Outcomes With Automated Writing Evaluation**
  + Authors: Jill Burstein, Dan McCaffrey, Beata Beigman Klebanov, Guangming Ling
  + Conference: WS
  + Link: https://aclanthology.org/W17-5011/
  + <details>
      <summary>Abstract</summary>
      Writing is a challenge, especially for at-risk students who may lack the prerequisite writing skills required to persist in U.S. 4-year postsecondary (college) institutions. Educators teaching postsecondary courses requiring writing could benefit from a better understanding of writing achievement and its role in postsecondary success. In this paper, novel exploratory work examined how automated writing evaluation (AWE) can inform our understanding of the relationship between postsecondary writing skill and broader success outcomes. An exploratory study was conducted using test-taker essays from a standardized writing assessment of postsecondary student learning outcomes. Findings showed that for the essays, AWE features were found to be predictors of broader outcomes measures: college success and learning outcomes measures. Study findings illustrate AWE’s potential to support educational analytics – i.e., relationships between writing skill and broader outcomes – taking a step toward moving AWE beyond writing assessment and instructional use cases.
    </details>
+ **Fine-grained essay scoring of a complex writing task for native speakers**
  + Authors: Andrea Horbach, Dirk Scholten-Akoun, Yuning Ding, Torsten Zesch
  + Conference: WS
  + Link: https://aclanthology.org/W17-5040/
  + <details>
      <summary>Abstract</summary>
      Automatic essay scoring is nowadays successfully used even in high-stakes tests, but this is mainly limited to holistic scoring of learner essays. We present a new dataset of essays written by highly proficient German native speakers that is scored using a fine-grained rubric with the goal to provide detailed feedback. Our experiments with two state-of-the-art scoring systems (a neural and a SVM-based one) show a large drop in performance compared to existing datasets. This demonstrates the need for such datasets that allow to guide research on more elaborate essay scoring methods.
    </details>
+ **Investigating neural architectures for short answer scoring**
  + Authors: Brian Riordan, Andrea Horbach, Aoife Cahill, Torsten Zesch, Chong Min Lee
  + Conference: WS
  + Link: https://aclanthology.org/W17-5017/
  + <details>
      <summary>Abstract</summary>
      Neural approaches to automated essay scoring have recently shown state-of-the-art performance. The automated essay scoring task typically involves a broad notion of writing quality that encompasses content, grammar, organization, and conventions. This differs from the short answer content scoring task, which focuses on content accuracy. The inputs to neural essay scoring models – ngrams and embeddings – are arguably well-suited to evaluate content in short answer scoring tasks. We investigate how several basic neural approaches similar to those used for automated essay scoring perform on short answer scoring. We show that neural architectures can outperform a strong non-neural baseline, but performance and optimal parameter settings vary across the more diverse types of prompts typical of short answer scoring.
    </details>
## 2016
+ **Constrained Multi-Task Learning for Automated Essay Scoring**
  + Authors: Ronan Cummins, Meng Zhang, Ted Briscoe
  + Conference: ACL
  + Link: https://aclanthology.org/P16-1075/

+ **A Neural Approach to Automated Essay Scoring**
  + Authors: Kaveh Taghipour, Hwee Tou Ng
  + Conference: EMNLP
  + Link: https://aclanthology.org/D16-1193/

+ **Automatic Features for Essay Scoring – An Empirical Study**
  + Authors: Fei Dong, Yue Zhang
  + Conference: EMNLP
  + Link: https://aclanthology.org/D16-1115/

+ **Using Argument Mining to Assess the Argumentation Quality of Essays**
  + Authors: Henning Wachsmuth, Khalid Al-Khatib, Benno Stein
  + Conference: COLING
  + Link: https://aclanthology.org/C16-1158/
  + <details>
      <summary>Abstract</summary>
      Argument mining aims to determine the argumentative structure of texts. Although it is said to be crucial for future applications such as writing support systems, the benefit of its output has rarely been evaluated. This paper puts the analysis of the output into the focus. In particular, we investigate to what extent the mined structure can be leveraged to assess the argumentation quality of persuasive essays. We find insightful statistical patterns in the structure of essays. From these, we derive novel features that we evaluate in four argumentation-related essay scoring tasks. Our results reveal the benefit of argument mining for assessing argumentation quality. Among others, we improve the state of the art in scoring an essay’s organization and its argument strength.
    </details>
+ **Topicality-Based Indices for Essay Scoring**
  + Authors: Beata Beigman Klebanov, Michael Flor, Binod Gyawali
  + Conference: WS
  + Link: https://aclanthology.org/W16-0507/

## 2015
+ **Flexible Domain Adaptation for Automated Essay Scoring Using Correlated Linear Regression**
  + Authors: Peter Phandi, Kian Ming A. Chai, Hwee Tou Ng
  + Conference: EMNLP
  + Link: https://aclanthology.org/D15-1049/

+ **Task-Independent Features for Automated Essay Grading**
  + Authors: Torsten Zesch, Michael Wojatzki, Dirk Scholten-Akoun
  + Conference: WS
  + Link: https://aclanthology.org/W15-0626/
