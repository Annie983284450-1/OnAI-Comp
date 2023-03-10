# OnAI-Comp: An Online AI Experts Competing Framework for Early Sepsis Detection

 
Sepsis is a major public concern due to its high mortality, morbidity, and financial cost. There are many existing works of early sepsis prediction using different machine learning models to mitigate the outcomes brought by sepsis. In the practical scenario, the dataset grows dynamically as new patients visit the hospital. Most existing models, being ``offline'' models and having used retrospective observational data, cannot be updated and improved dynamically using the new observational data. Incorporating the new data to improve the offline models requires retraining the model, which is very computationally expensive. To solve the challenge mentioned above, we propose an Online Artificial Intelligence Experts Competing Framework (OnAI-Comp) for early sepsis detection. We selected several machine learning models as the artificial intelligence experts and used average regret to evaluate the performance of our model. The experimental analysis demonstrated that our model would converge to the optimal strategy in the long run. Meanwhile, our model can provide clinically interpretable predictions using existing local interpretable model-agnostic explanation technologies, which can aid clinicians in making decisions and might improve the probability of survival.

Paper: https://pubmed.ncbi.nlm.nih.gov/34699366/
