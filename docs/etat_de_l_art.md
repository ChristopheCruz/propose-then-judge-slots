# État de l'art : vérification des références

Projet : *Laya comme juge de slots*. Document préparé le 30 octobre 2026 pour l'article visé au *Natural Language Processing Journal* (Elsevier).

Les entrées BibTeX correspondantes sont dans [`references.bib`](article/references.bib) (43 entrées). Les clés sont indiquées dans la colonne « Clé ».

## 1. Méthode de vérification

Toutes les consultations ont eu lieu le **30/10/2026**.

| Source | Accès | Usage |
|---|---|---|
| ACL Anthology, export complet `https://aclanthology.org/anthology.bib.gz` | ok (13,5 Mo, 2,56 M lignes) | titres, auteurs, lieux, pages, DOI de toutes les références ACL ; recherche des travaux récents |
| Pages d'articles de l'ACL Anthology (`https://aclanthology.org/<id>/`) | ok | résumés des travaux récents (§ 4) |
| API arXiv (`https://export.arxiv.org/api/query?id_list=…`) | ok | titres, auteurs, dates de dépôt, `journal_ref` |
| API Crossref (`https://api.crossref.org/works`) | ok, avec limitation de débit (429) | DOI hors ACL (AAAI, KDD, JASA, NeurIPS) |
| `https://doi.org/<doi>` | ok | résolution de chaque DOI douteux |
| PMLR (`proceedings.mlr.press`) | ok | Guo et al. 2017 ; mmBERT 2026 |
| NeurIPS (`papers.nips.cc`, fichiers `-Bibtex.bib`) | ok | Geifman et El-Yaniv 2017 ; MT-Bench 2023 |
| API Semantic Scholar | partiel : réponses pour 6 requêtes, puis refus (HTTP 429) | contrôle croisé seulement |
| DBLP (`dblp.org`) | **inaccessible** : page anti-robots (Anubis) | non utilisé |
| Hugging Face (API `huggingface.co/api/models/…` et fiches) | ok | fiche de Laya et des modèles utilisés |
| Recherche web | ok | lieux de publication de D3ST et mmBERT |

Règle suivie : le lieu de publication relu par les pairs prime sur la prépublication arXiv, comme le demande le guide aux auteurs de la revue (« *Where a preprint has subsequently become available as a peer-reviewed publication, the formal publication should be used as the reference* », guide 2023, voir `publication.md`).

## 2. Références du plan : statut

Statuts : **vérifiée** (conforme au plan) ; **corrigée** (le plan se trompait sur l'année, le lieu ou le titre) ; **à préciser** (vérifiée, mais sans lieu relu par les pairs) ; **introuvable**.

| # | Plan | Référence vérifiée (titre exact, lieu) | Identifiant | Statut | Clé |
|---|---|---|---|---|---|
| 1 | JointBERT (Chen et al., 2019) | Chen, Zhuo, Wang. *BERT for Joint Intent Classification and Slot Filling*. arXiv, 2019 | arXiv:1902.10909 | **à préciser** : prépublication seulement. Semantic Scholar donne « arXiv.org » comme lieu, et aucune version ACL n'existe dans l'Anthology | `chen2019jointbert` |
| 2 | MultiATIS++ (Xu et al., 2020) | Xu, Haider, Mansour. *End-to-End Slot Alignment and Recognition for Cross-Lingual NLU*. EMNLP 2020, p. 5052-5063 | 10.18653/v1/2020.emnlp-main.410 | vérifiée. Le nom « MultiATIS++ » ne figure pas dans le titre. Le corpus distribué par le LDC s'appelle *ATIS - Seven Languages* (LDC2021T04) : citer aussi `mansour2021atis7` comme référence de données | `xu2020multiatis` |
| 3 | MTOP (Li et al., 2021) | Li, Arora, Chen, Gupta, Gupta, Mehdad. *MTOP: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark*. EACL 2021 (Main Volume), p. 2950-2962 | 10.18653/v1/2021.eacl-main.257 | vérifiée (prépublication arXiv:2008.09335, août 2020) | `li2021mtop` |
| 4 | MASSIVE (FitzGerald et al., 2023) | FitzGerald et al. (16 auteurs). *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*. ACL 2023 (Long Papers), p. 4277-4302 | 10.18653/v1/2023.acl-long.235 | vérifiée (prépublication arXiv:2204.08582, avril 2022) | `fitzgerald2023massive` |
| 5 | SUMBT (Lee et al., 2019) | Lee, Lee, Kim. *SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking*. ACL 2019, p. 5478-5483 | 10.18653/v1/P19-1546 | vérifiée | `lee2019sumbt` |
| 6 | TRADE (Wu et al., 2019) | Wu, Madotto, Hosseini-Asl, Xiong, Socher, Fung. *Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems*. ACL 2019, p. 808-819 | 10.18653/v1/P19-1078 | vérifiée | `wu2019trade` |
| 7 | DS-DST (Zhang et al., 2020) | Zhang, Hashimoto, Wu, Wang, Yu, Socher, Xiong. *Find or Classify? Dual Strategy for Slot-Value Predictions on Multi-Domain Dialog State Tracking*. \*SEM 2020 (*Ninth Joint Conference on Lexical and Computational Semantics*), p. 154-167 | ACL 2020.starsem-1.17 (pas de DOI : `10.18653/v1/2020.starsem-1.17` renvoie 404) | vérifiée ; lieu précisé (\*SEM) | `zhang2020dsdst` |
| 8 | SGD (Rastogi et al., 2020) | Rastogi, Zang, Sunkara, Gupta, Khaitan. *Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset*. AAAI 2020, vol. 34, n° 05, p. 8689-8696 | 10.1609/aaai.v34i05.6394 | vérifiée (Crossref) | `rastogi2020sgd` |
| 9 | T5DST (Lin et al., 2021) | Lin, Liu, Moon, Crook, Zhou, Wang, Yu, Madotto, Cho, Subba. *Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking*. NAACL-HLT 2021, p. 5640-5648 | 10.18653/v1/2021.naacl-main.448 | vérifiée. Le nom « T5DST » ne figure pas dans le titre | `lin2021t5dst` |
| 10 | D3ST (Zhao et al., 2022) | Zhao, Gupta, Cao, Yu, Wang, Lee, Rastogi, Shafran, Wu. *Description-Driven Task-Oriented Dialog Modeling*. arXiv, 2022 | arXiv:2201.08904 | **à préciser** : aucune version relue par les pairs trouvée (ACL Anthology, page Google Research sans lieu, recherche web). À citer comme prépublication | `zhao2022d3st` |
| 11 | SGD-X (Lee et al., 2022) | Lee, Gupta, Rastogi, Cao, Zhang, Wu. *SGD-X: A Benchmark for Robust Generalization in Schema-Guided Dialogue Systems*. AAAI 2022, vol. 36, n° 10, p. 10938-10946 | 10.1609/aaai.v36i10.21341 | vérifiée (prépublication arXiv:2110.06800, 2021) | `lee2022sgdx` |
| 12 | Multi3WOZ (Hu et al., 2023) | Hu, Zhou, Hergul, Gritta, Zhang, Iacobacci, Vulić, Korhonen. *Multi3WOZ: A Multilingual, Multi-Domain, Multi-Parallel Dataset for Training and Evaluating Culturally Adapted Task-Oriented Dialog Systems*. TACL, vol. 11, 2023, p. 1396-1415 | 10.1162/tacl_a_00609 | vérifiée ; lieu précisé (revue TACL) | `hu2023multi3woz` |
| 13 | X-RiSAWOZ (Moradshahi et al., 2023) | Moradshahi et al. (19 auteurs). *X-RiSAWOZ: High-Quality End-to-End Multilingual Dialogue Datasets and Few-shot Agents*. Findings of ACL 2023, p. 2773-2794 | 10.18653/v1/2023.findings-acl.174 | vérifiée ; lieu précisé (Findings) | `moradshahi2023xrisawoz` |
| 14 | IC-DST (Hu et al., 2022) | Hu, Lee, Xie, Yu, Smith, Ostendorf. *In-Context Learning for Few-Shot Dialogue State Tracking*. Findings of EMNLP 2022, p. 2627-2643 | 10.18653/v1/2022.findings-emnlp.193 | vérifiée ; lieu précisé (Findings) | `hu2022icdst` |
| 15 | ChatGPT zero-shot DST (Heck et al., 2023) | Heck, Lubis, Ruppik, Vukovic, Feng, Geishauser, Lin, van Niekerk, Gašić. *ChatGPT for Zero-shot Dialogue State Tracking: A Solution or an Opportunity?* ACL 2023 (Short Papers), p. 936-950 | 10.18653/v1/2023.acl-short.81 | vérifiée | `heck2023chatgpt` |
| 16 | MT-Bench (Zheng et al., 2023) | Zheng et al. (13 auteurs). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 36 (2023), *Datasets and Benchmarks Track*, p. 46595-46623 | 10.52202/075280-2020 | vérifiée ; lieu précisé (piste D&B) | `zheng2023mtbench` |
| 17 | G-Eval (Liu et al., 2023) | Liu, Iter, Xu, Wang, Xu, Zhu. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. EMNLP 2023, p. 2511-2522 | 10.18653/v1/2023.emnlp-main.153 | vérifiée | `liu2023geval` |
| 18 | Biais de position (Wang et al., 2023) | Wang, Li, Chen, Cai, Zhu, Lin, Cao, Kong, Liu, Liu, Sui. *Large Language Models are not Fair Evaluators*. ACL 2024 (Long Papers), p. 9440-9450 | 10.18653/v1/2024.acl-long.511 | **corrigée** : publié à **ACL 2024**. 2023 est l'année de la prépublication (arXiv:2305.17926, mai 2023). Citer « Wang et al., 2024 » | `wang2024fair` |
| 19 | BLINK (Wu et al., 2020) | Wu, Petroni, Josifoski, Riedel, Zettlemoyer. *Scalable Zero-shot Entity Linking with Dense Entity Retrieval*. EMNLP 2020, p. 6397-6407 | 10.18653/v1/2020.emnlp-main.519 | vérifiée. Le nom « BLINK » ne figure pas dans le titre | `wu2020blink` |
| 20 | GLiNER (Zaratiana et al., 2024) | Zaratiana, Tomeh, Holat, Charnois. *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*. NAACL-HLT 2024 (Long Papers), p. 5364-5376 | 10.18653/v1/2024.naacl-long.300 | vérifiée | `zaratiana2024gliner` |
| 21 | Zero-shot par NLI (Yin et al., 2019) | Yin, Hay, Roth. *Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach*. EMNLP-IJCNLP 2019, p. 3914-3923 | 10.18653/v1/D19-1404 | vérifiée | `yin2019benchmarking` |
| 22 | Température (Guo et al., 2017) | Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks*. ICML 2017, PMLR 70, p. 1321-1330 | proceedings.mlr.press/v70/guo17a | vérifiée | `guo2017calibration` |
| 23 | Isotonique (Zadrozny et Elkan, 2002) | Zadrozny, Elkan. *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD 2002, p. 694-699 | 10.1145/775047.775151 | vérifiée (Crossref). Le lien avec la régression isotonique (algorithme PAV) est confirmé par une présentation du papier (`cs.cornell.edu/courses/cs678/2007sp/ZadroznyElkan.pdf`), pas par le texte original, que nous n'avons pas lu | `zadrozny2002transforming` |
| 24 | Classification sélective (Geifman et El-Yaniv, 2017) | Geifman, El-Yaniv. *Selective Classification for Deep Neural Networks*. NeurIPS 30 (2017) | papers.nips.cc, hash 4a8423d5… | vérifiée (pas de pages dans l'export NeurIPS) | `geifman2017selective` |
| 25 | Règles de score propres (Gneiting et Raftery, 2007) | Gneiting, Raftery. *Strictly Proper Scoring Rules, Prediction, and Estimation*. JASA, vol. 102, n° 477, 2007, p. 359-378 | 10.1198/016214506000001437 | vérifiée. Attention : Crossref liste aussi des rapports techniques de 2004 et 2005 au même titre ; citer l'article de JASA | `gneiting2007scoring` |
| 26 | ModernBERT (Warner et al., 2024) | Warner et al. (13 auteurs). *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*. ACL 2025 (Long Papers), p. 2526-2547 | 10.18653/v1/2025.acl-long.127 | **corrigée** : publié à **ACL 2025**. 2024 est l'année de la prépublication (arXiv:2412.13663, décembre 2024). Citer « Warner et al., 2025 » | `warner2025modernbert` |
| 27 | mmBERT (2025) | Marone, Weller, Fleshman, Yang, Lawrie, Van Durme. *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning*. ICML 2026, PMLR 306, p. 86768-86783 | proceedings.mlr.press/v306/marone26a | **corrigée** : auteurs ajoutés, publié à **ICML 2026**. 2025 est l'année de la prépublication (arXiv:2509.06888, septembre 2025). Citer « Marone et al., 2026 » | `marone2026mmbert` |
| 28 | Laya (fiche HF, rapport technique) | Convai Innovations. Fiche `convaiinnovations/laya` sur Hugging Face, révision `7b928d828b7b0e022f929d9bd2e44165aa270148` (modifiée le 03/10/2026) ; code `github.com/NandhaKishorM/laya` (Apache-2.0, créé le 18/09/2026) | huggingface.co/convaiinnovations/laya | **à préciser** : **aucun rapport technique** n'est lié depuis la fiche, et la fiche ne propose pas de citation. Seul un billet Dev.to est lié. À citer comme logiciel ou modèle non relu par les pairs | `laya2026` |

Le plan cite aussi « Jev (2026, fiches modèles et rapports techniques) ». Cette référence n'était pas dans la liste à vérifier. La fiche de Laya mentionne « TypeSafe Jev » et deux bancs d'essai tiers sur GitHub, sans lien vers un rapport : **non vérifiée**.

### Ce qui a été corrigé, en bref

- **Années** : Wang et al. (2023 → **2024**, ACL), Warner et al. (2024 → **2025**, ACL), mmBERT (2025 → **2026**, ICML ; auteurs : Marone et al.). Dans chaque cas, le plan donnait l'année de la prépublication arXiv.
- **Lieux à préciser** dans l'article : DS-DST (\*SEM 2020), Multi3WOZ (TACL), X-RiSAWOZ et IC-DST (Findings), MT-Bench (piste *Datasets and Benchmarks*).
- **Sans lieu relu par les pairs** : JointBERT, D3ST et Laya. Le guide de la revue demande de les marquer comme prépublications (mot « preprint » ou nom du serveur, avec le DOI de la prépublication) : c'est fait dans `references.bib` (champ `note`).
- **Noms d'usage absents des titres** (MultiATIS++, T5DST, BLINK) : les introduire dans le texte, par exemple « MultiATIS++ (Xu et al., 2020) ».
- **Laya, version** : la fiche Hugging Face décrit le runtime `laya` 0.3.20, alors que le README du dépôt indique 0.3.22, épinglé dans `uv.lock`. Dans l'article, donner la version épinglée et la révision du checkpoint. La fiche précise que le checkpoint anglais repose sur ModernBERT-large (421 M de paramètres en tout) et le checkpoint `multilingual` sur mmBERT-base (322 M) : citer ModernBERT et mmBERT avec Laya. Le README du dépôt indique que C1 utilise le checkpoint `multilingual`.

### Références ajoutées (modèles et données utilisés)

Ces entrées ne figuraient pas dans le plan, mais l'article devra les citer, puisqu'il utilise les modèles et les données correspondants :

| Référence | Source de vérification | Clé |
|---|---|---|
| Mansour, Haider. *ATIS - Seven Languages*, LDC2021T04, 2021 [dataset] | catalogue LDC, DOI 10.35111/g9h5-0p74 (résout vers `catalog.ldc.upenn.edu/LDC2021T04`) ; c'est la citation suggérée par le LDC | `mansour2021atis7` |
| Reimers, Gurevych. *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP 2020, p. 4512-4525 | ACL Anthology ; méthode d'entraînement des modèles `paraphrase-multilingual-*` | `reimers2020multilingual` |
| Gemma Team. *Gemma 4 Technical Report*. arXiv:2607.02770, 2026 | API arXiv ; lien « Technical Report » de la fiche `google/gemma-4-31B-it` | `gemma4report` |
| Qwen Team. *Qwen3.8-Max: A New Bar for Coding and Cowork*, billet, août 2026 | bloc BibTeX de la fiche `Qwen/Qwen3.8-27B` | `qwen38` |
| Qwen Team. *Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency*, billet, août 2026 | bloc BibTeX de la fiche `Qwen/Qwen3.8-Flash-Next-FP8`. La fiche renvoie aussi à un rapport technique (`github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf`), que nous n'avons pas ouvert | `qwen38flashnext` |

À ajouter plus tard, non vérifiées ici : mDeBERTa-v3 (encodeur prévu pour C0), vLLM, et Sentence-BERT (Reimers et Gurevych, EMNLP-IJCNLP 2019, D19-1410), citée par la fiche du modèle d'embeddings. Sentence-BERT a été repérée dans l'Anthology, mais n'est pas incluse dans `references.bib`.

## 3. Positionnement par axe (repris du plan)

Le tableau reprend les axes du plan, avec les clés vérifiées. La colonne « Ce qu'on en retient » est celle du plan, c'est-à-dire la lecture de l'auteur : elle reste à confronter aux articles lors de la rédaction.

| Axe | Clés | Ce qu'on en retient (plan) |
|---|---|---|
| NLU joint intention + slots, multilingue | `chen2019jointbert`, `xu2020multiatis`, `li2021mtop`, `fitzgerald2023massive` | baselines d'extraction, protocoles interlingues |
| Suivi d'état par liste ou extraction | `lee2019sumbt`, `wu2019trade`, `zhang2020dsdst` | l'opposition « choisir ou extraire », ancêtre direct de notre approche |
| Schémas décrits, zero-shot | `rastogi2020sgd`, `lin2021t5dst`, `zhao2022d3st`, `lee2022sgdx` | descriptions en langage naturel, domaines non vus, robustesse aux reformulations |
| Dialogue multilingue | `hu2023multi3woz`, `moradshahi2023xrisawoz` | données françaises multi-tours, adaptation culturelle |
| LLM pour l'état de dialogue | `hu2022icdst`, `heck2023chatgpt` (+ `li2024fnctod`, `lee2024orchestrallm`, `lin2026reactod`, § 4) | références de C4 et C5, et leur coût |
| LLM-as-judge et biais | `zheng2023mtbench`, `liu2023geval`, `wang2024fair` (+ `li2025calibraeval`, `radharapu2026calibrating`, § 4) | notre juge travaille en ligne, pas en évaluation ; tests de permutation |
| Candidats puis reclassement | `wu2020blink`, `zaratiana2024gliner`, `yin2019benchmarking` | génération puis sélection, comme en liaison d'entités |
| Calibration et abstention | `guo2017calibration`, `zadrozny2002transforming`, `geifman2017selective`, `gneiting2007scoring` | seuils à trois zones, courbes risque-couverture |
| Encodeurs de décision | `warner2025modernbert`, `marone2026mmbert`, `laya2026` | le modèle étudié ; Laya et Jev ne sont pas relus par les pairs, à citer comme tels |
| Petits modèles contre LLM | `pecher2025comparing`, `tikhomirov2026selection`, `aguirre2025finetuning`, `li2025synergistic`, `rana2025zeroshot` (§ 4) | à ajouter au plan : c'est le cadre de QR1 |

## 4. Travaux récents à surveiller (2024-2026)

Méthode : recherche par mots-clés dans les titres de l'export de l'ACL Anthology (années 2024 à 2026) : *dialogue state tracking*, *slot*, *judge* combiné à *calibration*, *bias*, *confidence* ou *uncertainty*, et *small* ou *encoder* combiné à *LLM*. Puis lecture des résumés sur les pages de l'Anthology (30/10/2026). Dix travaux retenus. Chaque phrase s'appuie sur le résumé seul : les articles n'ont pas été lus en entier. Les chiffres sont ceux des résumés.

1. **Li et al. (2024), *Large Language Models as Zero-shot Dialogue State Tracker through Function Calling* (FnCTOD), ACL 2024** (`li2024fnctod`). Ils formulent le DST comme un appel de fonctions. Des LLM ouverts de 7 à 13 B dépassent ChatGPT en zero-shot, et un LLaMA2-13B fine-tuné l'égale. C'est la référence « LLM génératif » la plus proche de C5 : notre argument est que le juge ne génère pas de valeur.
2. **Lee, Cheng et Ostendorf (2024), *OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking*, NAACL 2024** (`lee2024orchestrallm`). Un routage entre petits et grands modèles améliore le DST par rapport au LLM seul, et réduit le coût de calcul de plus de 50 %. C'est le concurrent direct de la cascade « Laya, puis LLM sur les cas incertains » envisagée dans le plan.
3. **Lin et al. (2026), *ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking*, TrustNLP 2026** (`lin2026reactod`). Les mises à jour d'état passent par des appels d'outils validés par un vérificateur symbolique, contre les hallucinations et les erreurs de format des LLM de taille moyenne (JGA zero-shot de 52,71 % sur MultiWOZ 2.1). Cet article partage notre motivation (prévisibilité, IA hybride) avec une solution générative : utile pour le positionnement « digne de confiance » attendu par la revue.
4. **Li et al. (2025), *Synergistic Augmentation: Enhancing Cross-Domain Zero-Shot Slot Filling with Small Model-Assisted Large Language Models*, Findings of ACL 2025** (`li2025synergistic`). Selon eux, les LLM font moins bien que les petits modèles sur les slots vus, et ils injectent les prédictions et la confiance d'un petit modèle dans le prompt du LLM. C'est la combinaison inverse de la nôtre : le petit modèle aide le LLM, alors que chez nous il tranche.
5. **Rana et al. (2025), *Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems*, COLING 2025 (Industry)** (`rana2025zeroshot`). Une distillation en boîte noire d'un LLM enseignant vers un petit modèle dépasse les LLM seuls sur des données internes de centre d'appels (+26 points de F1), à faible latence. C'est un argument industriel pour les petits modèles en slot filling, mais sur des données non publiques.
6. **Aguirre et al. (2025), *Fine-Tuning Medium-Scale LLMs for Joint Intent Classification and Slot Filling: A Data-Efficient and Cost-Effective Solution for SMEs*, COLING 2025 (Industry)** (`aguirre2025finetuning`). Sur MASSIVE en espagnol et en anglais, Llama-3-8B fine-tuné sur 10 % des données dépasse JointBERT et GPT-4o en zero-shot, en monolingue comme en interlingue. C'est la comparaison « fine-tuné contre LLM » la plus proche de QR1 et QR3, avec un LLM de 8 B, là où notre juge a environ 0,4 B de paramètres.
7. **Li et al. (2025), *CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges*, ACL 2025** (`li2025calibraeval`). Les juges LLM changent d'avis quand on permute la position ou l'identifiant des options, et les auteurs corrigent ce biais sans étiquettes. Ce travail concerne directement notre LLM-juge à lettres d'options et nos tests de permutation (robustesse).
8. **Radharapu et al. (2026), *Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation*, ACL 2026 (Industry)** (`radharapu2026calibrating`). Des sondes linéaires entraînées avec une perte de Brier sur les états cachés donnent des juges mieux calibrés, à moindre coût que la confiance verbalisée ou la génération multiple. C'est une alternative à nos probabilités lues sur les logprobs de C4 : à discuter pour QR4.
9. **Pecher et al. (2025), *Comparing Specialised Small and General Large Language Models on Text Classification: 100 Labelled Samples to Achieve Break-Even Performance*, EMNLP 2025** (`pecher2025comparing`). Sur 8 tâches, il suffit en moyenne d'environ 100 exemples étiquetés pour qu'un petit modèle spécialisé égale un LLM général, mais ce nombre augmente de 100 à 200 % si l'on tient compte de la variance. Cela justifie nos courbes d'apprentissage et nos graines multiples (QR1, protocole statistique).
10. **Tikhomirov et Savchenko (2026), *The Selection Effect: Why Compact LLM Verifiers for Emotion Classification Fail Behind Encoder Confidence Gates*, EMNLP 2026** (`tikhomirov2026selection`). Quand un encodeur ne transmet que ses cas incertains, le vérificateur LLM compact n'améliore pas la frontière coût-qualité (effet de sélection). Le seuil sur la confiance de l'encodeur et l'acceptation conforme sont préférables. Ce résultat touche directement la cascade et la décision à trois zones (QR4) : à tester sur nos données avant de défendre une cascade.

Remarques :
- Le DOI des articles d'EMNLP 2026 n'a pas de segment `v1` (`10.18653/2026.emnlp-main.1049`). Il résout bien vers l'Anthology (vérifié). Les articles de COLING 2025 Industry n'ont pas de DOI dans l'Anthology.
- Aucun travail récent retenu ne porte à la fois sur le **français** et le **multi-tours** avec une analyse coût-précision-abstention. Cela va dans le sens du plan (« peu de travaux couvrent le français multi-tours… »), mais **notre recherche s'est limitée aux titres de l'ACL Anthology** : elle ne remplace pas une recherche sur arXiv, Semantic Scholar (refusée ici après quelques requêtes), TALN ou SIGDIAL hors Anthology. À refaire avant la soumission.
- D'autres titres de 2024-2026 repérés mais non retenus (résumés non lus) : Finch et Choi 2024 (données synthétiques pour le DST zero-shot), Xu et al. 2025 (DST zero-shot à partir du schéma seul), Wu et al. 2025 (INT, ID et SF multilingues sur MASSIVE), Shi et al. 2025 (biais de position des juges LLM), Zhang et al. 2026 (routage calibré entre juge LLM et experts humains, EMNLP 2026 Industry : résumé absent de la page).
