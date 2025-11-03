#!/home/user/LLM/.venv/bin/python3

from langchain_community.llms import Ollama

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:23:18 2024

@author: erikz
"""

import pandas as pd

litsearch_df = pd.read_csv("/home/user/LLM/abstracts_data/all_labeled2include_trainingset_plus_included_and_maybe_by_rayaan.csv")
test_litsearch_df_abstracts = litsearch_df[~litsearch_df.abstract.isna()][['id', 'abstract']].drop_duplicates().dropna()
test_litsearch_df_abstracts = test_litsearch_df_abstracts.head(1)

from langchain_community.llms import Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="deepseek-r1"
)

ai_generation_class = {  
    "Classic Machine Learning": {  
        "Definition": "Traditional statistical and algorithmic methods without deep neural networks.",  
        "Subcategories": {  
            "Supervised Learning": [  
                "Linear/logistic regression",  
                "SVMs",  
                "decision trees",  
                "Random Forests"  
            ],  
            "Unsupervised Learning": [  
                "Clustering (k-means)",  
                "dimensionality reduction (PCA)"  
            ],  
            "Reinforcement Learning": [  
                "Q-learning",  
                "policy gradients (pre-deep RL)"  
            ],  
            "Ensemble Methods": [  
                "Bagging",  
                "Boosting (XGBoost)"  
            ]  
        }  
    },  
    "Classic AI & Neural Network Architectures": {  
        "Definition": "Early AI systems and pre-foundation-model neural networks.",  
        "Subcategories": {  
            "Symbolic AI": [  
                "Rule-based systems",  
                "expert systems"  
            ],  
            "Basic Neural Networks": [  
                "ANNs",  
                "CNNs",  
                "RNNs",  
                "LSTMs",  
                "GNNs"  
            ],  
            "Early Deep Learning": [  
                "AlexNet",  
                "ResNet",  
                "VGG (pre-transformer architectures)"  
            ],  
            "Traditional NLP": [  
                "Word2Vec",  
                "TF-IDF",  
                "early RNN-based language models"  
            ]  
        }  
    },  
    "New Generation AI": {  
        "Definition": "Post-2020 paradigms focused on generative tasks and scalability.",  
        "Subcategories": {  
            "Generative AI": [  
                "Diffusion models",  
                "GANs",  
                "VAEs"  
            ],  
            "Foundation Models": [  
                "LLMs (GPT-4, ChatGPT, LLaMA)",  
                "multimodal models (CLIP)"  
            ],  
            "Modern Architectures": [  
                "Transformers",  
                "Vision Transformers (ViTs)"  
            ],  
            "Applications": [  
                "NLP (ChatGPT, BERT)",  
                "geospatial AI (Prithvi)",  
                "code generation (Codex)"  
            ]  
        }  
    }  
}


ai_generation_class = {  
    # 💡 Added temporal boundaries and technical specificity  
    "Classic Machine Learning": {  
        "Definition": "Pre-deep learning statistical methods (1990s-2010s) using explicit feature engineering",  
        "Subcategories": {  
            # 💡 Expanded coverage with missing key methods  
            "Supervised Learning": [  
                "Linear regression", "Logistic regression",   
                "Support Vector Machines (SVMs)",   
                "Decision Trees", "Random Forests",  
                "Naive Bayes", "k-Nearest Neighbors (kNN)"  
            ],  
            "Unsupervised Learning": [  
                "k-means Clustering",   
                "Hierarchical Clustering",  
                "Principal Component Analysis (PCA)",  
                "t-SNE", "DBSCAN"  
            ],  
            "Reinforcement Learning": [  
                "Q-learning", "SARSA",  
                "Policy Iteration",   
                "Value Iteration",  
                "Monte Carlo Methods"  
            ],  
            # 💡 Unified optimization approaches  
            "Optimization Methods": [  
                "Gradient Descent",  
                "Genetic Algorithms",  
                "Simulated Annealing"  
            ]  
        }  
    },  
    
    # 💡 Renamed for historical accuracy  
    "Classic Deep Learning": {  
        "Definition": "First-wave neural networks enabled by GPU computing and big data (pre-transformer era)",  
        "Subcategories": {  
            # 💡 Added architectural evolution  
            "Neural Architectures": [  
                "Multilayer Perceptrons (MLPs)",  
                "Convolutional Neural Networks (CNNs)",  
                "Recurrent Neural Networks (RNNs)",  
                "Long Short-Term Memory (LSTMs)",  
                "Autoencoders", "Siamese Networks"  
            ],  
            "Training Paradigms": [  
                "Backpropagation",  
                "Dropout Regularization",  
                "Batch Normalization",  
                "Transfer Learning"  
            ],  
            # 💡 Expanded NLP coverage  
            "Traditional NLP": [  
                "Word2Vec", "GloVe",  
                "TF-IDF", "n-gram Models",  
                "Conditional Random Fields (CRFs)",  
                "Hidden Markov Models (HMMs)"  
            ],  
            # 💡 Added hardware/software context  
            "Enabling Technologies": [  
                "CUDA-enabled GPUs",  
                "Distributed Training",  
                "FP16 Precision Training"  
            ]  
        }  
    },  

    # 💡 Restructured for modern paradigm shifts  
    "New Generation of AI": {  
        "Definition": "Post-2020 paradigms focused on generative tasks and scalability",  
        "Subcategories": {  
            "Generative AI": [  
                "Diffusion models",  
                "GANs",  
                "VAEs"  
            ],  
            "Foundation Models": [  
                "LLMs (GPT-4, ChatGPT, LLaMA)",  
                "multimodal models (CLIP)"  
            ],  
            "Modern Architectures": [  
                "Transformers",  
                "Vision Transformers (ViTs)"  
            ],  
            "Applications": [  
                "NLP (ChatGPT, BERT)",  
                "geospatial AI (Prithvi)",  
                "code generation (Codex)"  
            ]  
        }  
    }  
}  


ai_functional_use_cases = {  
    "Data-Centric AI": {  
        "Label": "AI for Data Enhancement & Monitoring",  
        "Definition": "Tools that improve data quality, automate collection/processing, or enable real-time monitoring.",  
        "Subcategories": {  
            "Data Generation": [  
                "Synthetic data creation (XXX)",  
                "Data augmentation (e.g., image rotation for training)"  
            ],  
            "Data Cleaning": [  
                "Anomaly detection (autoencoders)",  
                "missing value imputation"  
            ],  
            "Monitoring Systems": [  
                "IoT sensor analytics",  
                "network intrusion detection"  
            ]  
        },  
        "Examples": [  
            "Satellite imagery preprocessing with CNNs",  
            "XXX"  
        ]  
    },  
    "Predictive AI": {  
        "Label": "AI for Forecasting & Pattern Recognition",  
        "Definition": "Models that predict future states or classify patterns in structured/unstructured data.",  
        "Subcategories": {  
            "Time-Series Forecasting": [  
                "Weather prediction (LSTMs)",  
                "XXX"  
            ],  
            "Classification": [  
                "CNNs for animal identification",  
                "XXX"  
            ],  
            "Risk Modeling": [  
                "XXX",  
                "disease outbreak prediction"  
            ]  
        },  
        "Examples": [  
            "CNN for species identification in forest images",  
            "XXX"  
        ]  
    },  
    "Decision-Support AI": {  
        "Label": "AI for Actionable Insights & Automation in sustainability science",  
        "Definition": "Systems that recommend or automate decisions based on data analysis.",  
        "Subcategories": {  
            "AI for biodiversity managers": [  
                "Personalized content (Netflix, Spotify)"  
            ],  
            "AI for policy advising": [  
                "Supply chain optimization",  
                "treatment plans"  
            ],  
            "Autonomous Systems": [  
                "XXX in urban planning",  
                "XXX in biodiversity monitoring"  
            ]  
        },  
        "Examples": [  
            "LLM-powered XXX",  
            "Reinforcement learning for XXX"  
        ]  
    }  
}  


ai_functional_use_cases = {  
    "Data-Centric AI": {  
        "Label": "AI for Data Enhancement & Monitoring",  
        "Definition": "Tools that improve data quality, automate collection/processing, or enable real-time monitoring.",  
        "Subcategories": {  
            "Data Generation": [  
                "Synthetic data creation (GANs, diffusion models)",  
                "Data augmentation (neural style transfer)"  
            ],  
            "Data Cleaning": [  
                "Anomaly detection (autoencoder-based)",  
                "Missing value imputation on precipitation data (kNN imputation)"  
            ],  
            "Monitoring Systems": [  
                "IoT sensor analytics (LSTM-based)",  
                "Network intrusion detection (graph neural networks)"  
            ]  
        },  
        "Examples": [  
            "Satellite imagery preprocessing with CNNs",  
            "Diffusion-based satellite data synthesis"  
        ]  
    },  
    "Predictive AI": {  
        "Label": "AI for Forecasting & Pattern Recognition",  
        "Definition": "Models that predict future states or classify patterns in structured/unstructured data.",  
        "Subcategories": {  
            "Time-Series Forecasting": [  
                "Weather prediction (Transformer-based)",  
                "Tipping points identification"  
            ],  
            "Classification": [  
                "CNNs for animal identification",  
                "Acoustic classification of bird species"  
            ],  
            "Risk Modeling": [  
                "Flood risk scoring (XGBoost)",  
                "Disease outbreak prediction (graph-based SEIR models)"  
            ]  
        },  
        "Examples": [  
            "CNN for species identification in forest images",  
            "Transformer-XL for electricity demand forecasting"  
        ]  
    },  
    "Decision-Support AI": {  
        "Label": "AI for Actionable Insights & Automation in sustainability science",  
        "Definition": "Systems that recommend or automate decisions based on data analysis.",  
        "Subcategories": {  
            "AI for biodiversity managers": [  
                "Habitat restoration planning",  
                "Invasive species management"  
            ],  
            "AI for policy advising": [  
                "Carbon credit allocation systems",  
                "Urban heat island mitigation planning"  
            ],  
            "Autonomous Systems": [  
                "Drone-based traffic optimization in urban planning",  
                "Autonomous camera traps for biodiversity monitoring"  
            ]  
        },  
        "Examples": [  
            "LLM-powered environmental policy analysis",  
            "Reinforcement learning for reforestation planning"  
        ]  
    }  
}  

abstracts = ["Toponym identification, or place name recognition, within epidemiology articles is a crucial task for phylogeographers, as it allows them to analyze the development, spread, and migration of viruses. Although, public databases, such as GenBank (Benson et al., November 2012), contain the geographical information, this information is typically restricted to country and state levels. In order to identify more fine-grained localization information, epidemiologists need to read relevant scientific articles and manually extract place name mentions.\r\nIn this thesis, we investigate the use of various neural network architectures and language representations to automatically segment and label toponyms within biomedical texts. We demonstrate how our language model based toponym recognizer relying on transformer architecture can achieve state-of-the-art performance. This model uses pre-trained BERT as the backbone and fine tunes on two domains of datasets (general articles and medical articles) in order to measure the generalizability of the approach and cross-domain transfer learning.\r\nUsing BERT as the backbone of the model, resulted in a large highly parameterized model (340M parameters). In order to obtain a light model architecture we experimented with parameter pruning techniques, specifically we experimented with Lottery Ticket Hypothesis (Frankle and Carbin, May 2019) (LTH), however as indicated by Frankle and Carbin (May 2019), their pruning technique does not scale well to highly parametrized models and loses stability. We proposed a novel technique to augment LTH in order to increase the scalability and stability of this technique to highly parametrized models such as BERT and tested our technique on toponym identification task. \r\nThe evaluation of the model was performed using a collection of 105 epidemiology articles from PubMed Central (Weissenbacher et al., June 2015). Our proposed model significantly improves the state-of-the-art model by achieving an F-measure of 90.85% compared to 89.13%",
          "Flowers are admired and used by people all around the world for their fragrance, religious significance, and medicinal capabilities. The accurate taxonomy of these flower species is critical for biodiversity conservation and research. Non-experts typically need to spend a lot of time examining botanical guides in order to accurately identify a flower, which can be challenging and time-consuming. In this study, an innovative mobile application named FloralCam has been developed for the identification of flower species that are commonly found in Mauritius. Our dataset, named FlowerNet, was collected using a smartphone in a natural environment setting and consists of 11660 images, with 110 images for each of the 106 flower species. Seventy percent of the data was used for training, twenty percent for validation and the remaining ten percent for testing. Using the approach of transfer learning, pre-trained convolutional neural networks (CNNs) such as the InceptionV3, MobileNetV2 and ResNet50V2 were fine tuned on the custom dataset created. The best performance was achieved with the fine tuned MobileNetV2 model with accuracy 99.74% and prediction time 0.09 seconds. The best model was then converted to TensorFlow Lite format and integrated in a mobile application which was built using Flutter. Furthermore, the models were also tested on the benchmark Oxford 102 dataset and MobileNetV2 obtained the highest classification accuracy of 95.90%. The mobile application, the dataset and the deep learning models developed can be used to support future research in the field of flower recognition",
          "For environmental management and conservation, forecasting the forest's tree cover is essential. Accurate prediction models are needed to evaluate the possible effects and pinpoint places that need rapid attention as deforestation and forest degradation become a more widespread concern. In this study, we employ machine learning to predict forest tree cover using dataset of environmental variables and historical tree cover data. The foundation of our approach is the Random Forests ensemble learning technique, which combines different decision trees to improve prediction accuracy. Our goal is to create a model that, using environmental variables like elevation, rainfall, temperature, and soil type, can precisely forecast the number of trees in each location. To train and evaluate our model, we employ a dataset of more than 500,000 forested regions located throughout the United States. With predictions of forest tree cover being upto 97% accurate, the results demonstrate that the Random Forest model outperforms conventional machine learning techniques. Machine learning may be utilised as a useful tool in environmental management and conservation efforts by estimating the amount of forest tree cover. Accurate tree cover forecast can help in determining which regions need forest preservation, reforestation, or both.",
          "There is an increasing need for skillful runoff season (i.e., spring) streamflow forecasts that extend beyond a 12-month lead time for water resources management, especially under multiyear droughts and particularly in basins with highly variable streamflow, large storage capacity, proclivity to droughts, and many competing water users such as in the Colorado River Basin (CRB). Ensemble streamflow prediction (ESP) is a probabilistic prediction method widely used in hydrology, including at the National Oceanic and Atmospheric Administration (NOAA) Colorado Basin River Forecasting Center (CBRFC) to forecast flows that the Bureau of Reclamation uses in their water resources operational decision models. However, it tends toward climatology at 5-month and longer lead times, causing decreased skill, particularly in forecasts critical for management decisions. We developed a modeling approach for seasonal streamflow forecasts using a machine learning technique, random forest (RF), for runoff season flows (April 1–July 31 total) at the important gauge of Lees Ferry, Arizona, on the CRB. The model predictors include antecedent basin conditions, large-scale climate teleconnections, climate model projections of temperature and precipitation, and the mean ESP forecast from CBRFC. The RF model is fitted and validated separately for lead times spanning 0 to 18 months over the period 1983–2017. The performance of the RF model forecasts and CBRFC ESP forecasts are separately assessed against observed streamflows in a cross validation mode. Forecast performance was evaluated using metrics including relative bias, root mean square error, ranked probability skill score, and reliability. Measured by ranked probability skill score, RF outperforms a climatological benchmark at all lead times and outperforms CBRFC's ESP hindcasts for lead times spanning 6 to 18 months. For the 6- to 18-month lead times, the RF ensemble median had a root mean square error that was between ∼410- and ∼620-thousand acre-feet lower than that of the ESP ensemble median (i.e., RF reduced ensemble median RMSE by −9% to −12% relative to ESP). Reliability was comparable between RF and ESP. More skillful long-lead cross-validated forecasts using machine learning methods show promise for their use in real time forecasts and better informed and efficient water resources management; however, further testing in various decision models is needed to examine RF forecasts' downstream impacts on key water resources metrics like robustness, reliability, and vulnerability",
          ]
          
for index, row in test_litsearch_df_abstracts.iterrows():  
    abstract_id = row['id']
    abstract = row['abstract']
          
    #call LLM
    print(abstract_id)
    #prompt1 = "Acting as an AI sustainability researcher, please analyze the following abstract to identify the AI methods used and label these methods according to the guiding instructions regarding the three types of AI methods. You will also see examples of subtypes and the actual AI methods in the the guiding instructions provided. Please return your response in the form of a Python dictionary, where the key is one of the three types of AI methods, and the value is the list of keywords you identified in the abstract as relevant for making these decisions. Do not use any other labels or words in your response. Don't provide any subtypes, only one of the main classes as a key and the list of keywords as value. You must make a definitive decision. The abstract is here - {}. The guiding instructions about AI generation classes are available here - {} in the form of a dictionary".format(abstract,ai_generation_class)

    prompt1 = "Acting as an AI researcher with 10+ years experience in technical taxonomy, analyze the following abstract to identify AI methods and classify them using EXACTLY ONE main class from the provided AI generation classes: 1) Classic Machine Learning 2) Classic AI & Neural Network Architectures 3) New Generation of AI. Return a Python dictionary where: - Key: One of the three main classes from the abstract available here {} - Value: List of EXACT technical terms appearing in the guiding instructions about AI generation classes available here {} that MATCH the class's official subtypes. STRICT RULES: 1. Use only terms verbatim from the abstract. 2. Never infer or invent terms. 3. If multiple classes apply, choose the one with most matches. 4. If none class apply, return 'Other' as class category (also use 'Other' as Key) with all technical terms indicative of method applied (value). 5. Return empty list if no matches. Example 1: Abstract: 'Using CNNs and Transformers for image analysis' Answer: ```python {{'Classic AI & Neural Network Architectures': ['CNNs', 'Transformers']}} ```. Example 2: Abstract: 'Implemented Q-learning with statistical methods' Answer: ```python {{'Classic Machine Learning': ['Q-learning', 'statistical methods']}} ```. Example 3: Abstract: 'Customer churn prediction using Foundation Models' Answer: ```python {{'New Generation of AI': ['Foundation Models']}} ```.".format(abstract, ai_generation_class)  
    print(ollama.invoke(prompt1))
	
    #prompt2 = "Acting as an AI researcher, please analyze the following abstract to identify the AI methods used and label these methods according to the typology of the AI functional use cases. You will also see examples of subtypes and the actual AI methods in the AI functional use cases typology provided. Please return your response in the form of a Python dictionary, where the key is one of the three types of AI functional use cases, and the value is the list of keywords you identified in the abstract as relevant for making these decisions. Do not use any other labels or words in your response. Don't provide any subtypes, only one of the main classes as a key and the list of keywords as value. You must make a definitive decision. The abstract is here - {}. The typology of the AI functional use cases is available here - {} in the form of a dictionary".format(abstract,ai_functional_use_cases)

    #prompt2 = "Acting as an AI sustainability researcher, analyze this abstract to classify methods into EXACTLY ONE category: Data-Centric AI, Predictive AI, or Decision-Support AI. Return a Python dictionary where: - Key: One main class - Value: List of EXACT technical terms from abstract matching official Subcategories. STRICT RULES: 1. Use only verbatim terms 2. No term invention 3. Choose class with most matches 4. 'Other' category for no matches 5. Empty list if no terms. Example 1: Abstract: 'GAN-based sensor analytics' Answer: ```python {{'Data-Centric AI': ['GANs', 'sensor analytics']}} ```. Example 2: Abstract: 'LSTM species identification' Answer: ```python {{'Predictive AI': ['LSTMs', 'species identification']}} ``` Example 3: Abstract: 'Reinforcement learning for urban planning' Answer: ```python {{'Decision-Support AI': ['reinforcement learning', 'urban planning']}} ```. Abstract: '{}' AI Typology: {}".format(abstract,ai_functional_use_cases)
    #print(ollama.invoke(prompt2))
    print('NEXT!')
