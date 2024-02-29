from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# List of blackmailing sentences for training
blackmailing_sentences = [
    "If you go out with your friends tonight I won't be here when you get back.",
    "If you don't do what I say, I'll tell everyone your secret.",
    "Pay up, or I'll make sure you regret it.",
    "If you don't comply, I'll ruin your reputation.",
    "Do as I say, or I'll harm your loved ones.",
    "Hand over the money, or else there will be consequences.",
    "If you don't meet my demands, I'll expose your past.",
    "Give me what I want, or I'll destroy your career.",
    "If you don't cooperate, I'll make sure you suffer.",
    "Pay me, or I'll make sure everyone knows about your mistakes.",
    "Comply with my wishes, or I'll make your life miserable.",
    "If you don't follow through, I'll make sure you regret it.",
    "Do what I say, or I'll ruin your chances for success.",
    "Hand over the cash, or I'll reveal your darkest secrets.",
    "If you don't listen to me, I'll make sure you pay the price.",
    "Give me what I want, or I'll make your life a living hell.",
    "If you don't obey, I'll make sure you suffer the consequences.",
    "Pay me now, or I'll make sure you never work in this town again.",
    "Comply with my demands, or I'll expose your lies to the world.",
    "Do as I say, or I'll make sure your family pays for your mistakes.",
    "Hand over the funds, or I'll ensure you regret crossing me.",
    "If you don't meet my expectations, I'll make sure you regret it.",
    "Give me what I want, or I'll ensure you never find peace again.",
    "If you don't cooperate, I'll make sure you suffer the consequences.",
    "Pay me what I'm owed, or I'll make sure you regret it for life.",
    "Comply with my demands, or I'll ensure your downfall is swift.",
    "Do as I say, or I'll make sure you're haunted by your decisions.",
    "Hand over the money, or I'll ensure your reputation is destroyed.",
    "If you don't listen, I'll make sure you regret crossing me forever.",
    "Give me what I want, or I'll ensure your life is filled with misery.",
    "If you don't comply, I'll make sure your secrets are exposed to all.",
    "Pay me now, or I'll ensure you suffer the consequences of your actions.",
    "Comply with my wishes, or I'll make sure you regret ever crossing me.",
    "Do as I say, or I'll make sure you're punished for your disobedience.",
    "Hand over the cash, or I'll ensure your life becomes a living nightmare.",
    "If you don't meet my demands, I'll make sure your life is ruined forever.",
    "Give me what I want, or I'll ensure your reputation is tarnished beyond repair.",
    "If you don't cooperate, I'll make sure you regret the day you crossed me.",
    "Pay me what you owe, or I'll make sure you pay the price for your deceit.",
    "Comply with my demands, or I'll ensure your downfall is both swift and severe.",
    "Do as I say, or I'll make sure you suffer the consequences of your actions.",
    "Hand over the money, or I'll ensure your secrets are exposed to the world.",
    "If you don't listen, I'll make sure you regret the day you ever crossed me.",
    "Give me what I want, or I'll ensure your life becomes a living nightmare.",
    "If you don't comply, I'll make sure your reputation is tarnished forever.",
    "Pay me now, or I'll make sure you suffer the consequences of your betrayal.",
    "Comply with my wishes, or I'll ensure you face the full force of my wrath.",
    "Do as I say, or I'll make sure your life is filled with pain and regret.",
    "Hand over the cash, or I'll ensure your life becomes a never-ending nightmare.",
    "If you don't meet my demands, I'll make sure you regret the day you crossed me.",
    "Give me what I want, or I'll ensure your reputation is ruined beyond repair.",
    "If you don't cooperate, I'll make sure you pay dearly for your disobedience.",
    "Pay me what you owe, or I'll make sure you face the consequences of your actions.",
    "Comply with my demands, or I'll ensure you suffer the full extent of my vengeance.",
    "Do as I say, or I'll make sure you regret the day you ever dared to defy me.",
    "Hand over the money, or I'll ensure your secrets are revealed to the world.",
    "If you don't listen, I'll make sure you suffer the consequences of your betrayal.",
    "Give me what I want, or I'll ensure your life becomes a living hell on earth.",
    "If you don't comply, I'll make sure your reputation is destroyed beyond repair.",
    "Pay me now, or I'll make sure you suffer the consequences of your treachery.",
    "Comply with my wishes, or I'll ensure you face the full wrath of my fury.",
    "Do as I say, or I'll make sure your life becomes a never-ending nightmare.",
    "Hand over the cash, or I'll ensure you regret the day you ever crossed me.",
    "If you don't meet my demands, I'll make sure you suffer the consequences.",
    "Give me what I want, or I'll ensure your reputation is tarnished forever.",
    "If you don't cooperate, I'll make sure you regret the day you defied me.",
    "Pay me what you owe, or I'll make sure you pay dearly for your deception.",
    "Comply with my demands, or I'll ensure your downfall is both swift and severe.",
    "Do as I say, or I'll make sure you suffer the consequences of your actions.",
    "Hand over the money, or I'll ensure your life is filled with pain and regret.",
    "If you don't listen, I'll make sure your reputation is tarnished forever.",
    "Give me what I want, or I'll ensure your life becomes a living nightmare.",
    "If you don't comply, I'll make sure you pay the price for your disobedience.",
    "Pay me now, or I'll make sure your secrets are revealed to the world.",
    "Comply with my wishes, or I'll ensure you suffer the consequences.",
    "Do as I say, or I'll make sure your reputation is ruined beyond repair.",
    "Hand over the cash, or I'll ensure your life becomes a never-ending nightmare.",
    "If you don't meet my demands, I'll make sure you regret the day you crossed me.",
    "Give me what I want, or I'll ensure your secrets are exposed to all.",
    "If you don't cooperate, I'll make sure you pay the price for your defiance.",
    "Pay me what you owe, or I'll make sure you face the full force of my wrath.",
    "Comply with my demands, or I'll ensure your life is filled with regret.",
    "Do as I say, or I'll make sure your reputation is tarnished forever.",
    "Hand over the money, or I'll make sure you suffer the consequences.",
    "If you don't listen, I'll make sure your life becomes a living hell.",
    "Give me what I want, or I'll ensure your downfall is swift and severe.",
    "If you don't comply, I'll make sure you suffer the consequences.",
    "Pay me now, or I'll make sure you regret the day you ever crossed me.",
    "Comply with my wishes, or I'll ensure your life is filled with pain and suffering.",
    "Do as I say, or I'll make sure you pay the price for your disobedience.",
    "Hand over the cash, or I'll make sure you regret the day you ever crossed me.",
    "If you don't meet my demands, I'll make sure you suffer the consequences.",
    "Give me what I want, or I'll make sure your secrets are revealed to all.",
    "If you don't cooperate, I'll make sure you suffer the full extent of my wrath.",
    "Pay me what you owe, or I'll make sure you face the consequences of your actions.",
    "Comply with my demands, or I'll make sure your life is filled with misery.",
    "Do as I say, or I'll make sure your reputation is tarnished beyond repair.",
    "Hand over the money, or I'll make sure you pay the price for your betrayal.",
    "If you don't listen, I'll make sure you regret the day you ever crossed me.",
    "Give me what I want, or I'll make sure you suffer the consequences of your actions.",
    "If you don't comply, I'll make sure you pay the price for your disobedience."
]


# Labels for blackmailing sentences (1 for blackmailing, 0 for not blackmailing)
labels = np.ones(len(blackmailing_sentences))

# List of non-blackmailing sentences for training
non_blackmailing_sentences = [
    "I'm going to visit the farmer's market to buy fresh fruits and vegetables.",
    "I'm going to take a painting class to learn how to paint landscapes.",
    "I'm going to go on a guided tour of the city.",
    "I'm going to spend the day exploring the local hiking trails.",
    "I'm going to visit the zoo to see the new baby animals.",
    "I'm going to take a cooking class to learn how to cook Thai food.",
    "I'm going to go wine tasting at the vineyard.",
    "I'm going to spend the day at the beach building sandcastles.",
    "I'm going to take a pottery class to learn how to make ceramics.",
    "I'm going to visit the aquarium to see the dolphins.",
    "I'm going to go on a hot air balloon ride at sunset.",
    "I'm going to spend the day at the spa getting massages.",
    "I'm going to take a knitting class to learn how to knit sweaters.",
    "I'm going to go on a road trip to visit national parks.",
    "I'm going to spend the weekend at a cabin in the mountains.",
    "I'm going to visit the art gallery to see the new paintings.",
    "I'm going to take a photography course to improve my photography skills.",
    "I'm going to go horseback riding at the stables.",
    "I'm going to spend the day at the amusement park riding the rides.",
    "I'm going to take a dance class to learn how to salsa dance.",
    "I'm going to go kayaking on the river.",
    "I'm going to visit the science museum to see the new exhibits.",
    "I'm going to go rock climbing at the indoor climbing gym.",
    "I'm going to spend the weekend at a bed and breakfast in the countryside.",
    "I'm going to go fishing at the lake.",
    "I'm going to visit the farmer's market to buy fresh produce.",
    "I'm going to take a painting class to learn how to paint landscapes.",
    "I'm going to go on a guided tour of the city.",
    "I'm going to spend the day exploring the local hiking trails.",
    "I'm going to visit the zoo to see the animals.",
    "I'm going to take a cooking class to learn how to make pasta.",
    "I'm going to go wine tasting at the winery.",
    "I'm going to spend the day at the beach building sandcastles.",
    "I'm going to take a pottery class to learn how to make pottery.",
    "I'm going to visit the aquarium to see the fish.",
    "I'm going to go on a hot air balloon ride at sunrise.",
    "I'm going to spend the day at the spa getting pampered.",
    "I'm going to take a knitting class to learn how to knit.",
    "I'm going to go on a road trip to visit national parks.",
    "I'm going to spend the weekend at a cabin in the woods.",
    "I'm going to visit the art gallery to see the art.",
    "I'm going to take a photography course to improve my photography.",
    "I'm going to go horseback riding at the ranch.",
    "I'm going to spend the day at the amusement park riding roller coasters.",
    "I'm going to take a dance class to learn how to dance.",
    "I'm going to go kayaking on the river.",
    "I'm going to visit the science museum to see the exhibits.",
    "I'm going to go rock climbing at the indoor climbing gym.",
    "I'm going to spend the weekend at a bed and breakfast in the countryside.",
    "I'm going to go fishing at the lake.",
    "I'm going to visit the farmer's market to buy fresh fruits.",
    "I'm going to take a painting class to learn how to paint.",
    "I'm going to go on a guided tour of the city.",
    "I'm going to spend the day exploring the local hiking trails.",
    "I'm going to visit the zoo to see the animals.",
    "I'm going to take a cooking class to learn how to cook.",
    "I'm going to go wine tasting at the winery.",
    "I'm going to spend the day at the beach building sandcastles.",
    "I'm going to take a pottery class to learn how to make pottery.",
    "I'm going to visit the aquarium to see the fish.",
    "I'm going to go on a hot air balloon ride at sunrise.",
    "I'm going to spend the day at the spa getting pampered.",
    "I'm going to take a knitting class to learn how to knit.",
    "I'm going to go on a road trip to visit national parks.",
    "I'm going to spend the weekend at a cabin in the woods.",
    "I'm going to visit the art gallery to see the art.",
    "I'm going to take a photography course to improve my photography.",
    "I'm going to go horseback riding at the ranch.",
    "I'm going to spend the day at the amusement park riding roller coasters.",
    "I'm going to take a dance class to learn how to dance.",
    "I'm going to go kayaking on the river.",
    "I'm going to visit the science museum to see the exhibits.",
    "I'm going to go rock climbing at the indoor climbing gym.",
    "I'm going to spend the weekend at a bed and breakfast in the countryside.",
    "I'm going to go fishing at the lake.",
    "I'm going to visit the farmer's market to buy fresh fruits.",
    "I'm going to take a painting class to learn how to paint.",
    "I'm going to go on a guided tour of the city.",
    "I'm going to spend the day exploring the local hiking trails.",
    "I'm going to visit the zoo to see the animals.",
    "I'm going to take a cooking class to learn how to cook.",
    "I'm going to go wine tasting at the winery.",
    "I'm going to spend the day at the beach building sandcastles.",
    "I'm going to take a pottery class to learn how to make pottery.",
    "I'm going to visit the aquarium to see the fish.",
    "I'm going to go on a hot air balloon ride at sunrise.",
    "I'm going to spend the day at the spa getting pampered.",
    "I'm going to take a knitting class to learn how to knit.",
    "I'm going to go on a road trip to visit national parks.",
    "I'm going to spend the weekend at a cabin in the woods.",
    "I'm going to visit the art gallery to see the art.",
    "I'm going to take a photography course to improve my photography.",
    "I'm going to go horseback riding at the ranch.",
    "I'm going to spend the day at the amusement park riding roller coasters.",
    "I'm going to take a dance class to learn how to dance.",
    "I'm going to go kayaking on the river.",
    "I'm going to visit the science museum to see the exhibits.",
    "I'm going to go rock climbing at the indoor climbing gym."
]


# Labels for non-blackmailing sentences (0 for not blackmailing)
labels_non_blackmailing = np.zeros(len(non_blackmailing_sentences))

# Combine blackmailing and non-blackmailing sentences and labels
all_sentences = blackmailing_sentences + non_blackmailing_sentences
all_labels = np.concatenate([labels, labels_non_blackmailing])

# Define pipeline for the machine learning model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
model.fit(all_sentences, all_labels)

# Save the model
import joblib
joblib.dump(model, "blackmail_detection_model.joblib")