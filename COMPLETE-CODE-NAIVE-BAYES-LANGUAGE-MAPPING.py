import nltk
nltk.download('stopwords')
nltk.download('punkt')  # For tokenization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string

pip install datasets

from datasets import load_dataset

data = load_dataset("MartinThoma/wili_2018")
 
df = pd.DataFrame(data['train'])   
 
print(df.head())


from datasets import load_dataset
import pandas as pd
 
data = load_dataset("MartinThoma/wili_2018")
df = pd.DataFrame(data['train'])  
 
print("Available columns:", df.columns.tolist())
 
print(df.head())


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
 
nltk.download('stopwords')
 
stemmer = PorterStemmer()
 
def preprocess_text(text):
 
    text = text.lower()
 
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
   
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)
 
df['cleaned_text'] = df['sentence'].apply(preprocess_text)
 
print(df[['sentence', 'cleaned_text', 'label']].head())



from sklearn.model_selection import train_test_split

 
X = df['cleaned_text']
y = df['label']

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




from sklearn.feature_extraction.text import CountVectorizer

 
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
 
print(f"Training data shape: {X_train_vectorized.shape}")
print(f"Testing data shape: {X_test_vectorized.shape}")



from sklearn.naive_bayes import MultinomialNB
 
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_vectorized)
 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


def predict_language(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

new_texts = [
    "Hello, how are you?", 
    "Bonjour, comment ça va?",  # French (this may not be correctly identified without more training data)
    "Hola, ¿cómo estás?",  # Spanish (same as above)
    "ब्रेकिंग न्यूज़, वीडियो, ऑडियो और फ़ीचर.?"
]

label_mapping = {
    0: 'Chavacano',
    1: 'Gola',
    2: 'Jamaican Patois',
    3: 'Luganda',
    4: 'Sanskrit',
    5: 'Rusyn',
    6: 'Wolof',
    7: 'Newar',
    8: 'Mirandese',
    9: 'Breton',
    10: 'Arabic',
    11: 'Armenian',
    12: 'Mingrelian',
    13: 'Extremaduran',
    14: 'Cornish',
    15: 'Yoruba',
    16: 'Divehi (Maldivian)',
    17: 'Assamese',
    18: 'Latin',
    19: 'Welsh',
    20: 'Fijian Hindi',
    21: 'Achinese',
    22: 'Kabardian',
    23: 'Tajik',
    24: 'Russian',
    25: 'Northern Sotho (Sesotho sa Leboa)',
    26: 'Burmese',
    27: 'Malay (Bahasa Melayu)',
    28: 'Avar',
    29: 'Chavacano de Zamboanga (Zamboangueño)',
    30: 'Urdu',
    31: 'German',
    32: 'Swahili',
    33: 'Pashto',
    34: 'Buryat',
    35: 'Udmurt',
    36: 'Kashubian',
    37: 'Yiddish',
    38: 'Vepsian (Veps)',
    39: 'Portuguese',
    40: 'Pennsylvania German (Pennsilfaanisch Deitsch)',
    41: 'English',
    42: 'Thai',
    43: 'Haitian Creole (Kreyòl Ayisyen)',
    44: 'Lombard (Lombardo)',
    45: 'Pangasinan',
    46: 'Javanese (Basa Jawa)',
    47: 'Chuvash (Чӑваш)',
    48: "Nanai",
    49: "Scottish Gaelic",
    50: "Georgian",
    51: "Bhojpuri",
    52: "Bosnian",
    53: "Konkani",
    54: "Ossetic",
    55: "Māori",
    56: "Frisian",
    57: "Catalan",
    58: "Azerbaijani (North Azerbaijani)",
    59: "Kinyarwanda",
    60: "Hindi",
    61: "Shona",
    62: "Danish",
    63: "Emilian",
    64: "Macedonian",
    65: "Romanian",
    66: "Bulgarian",
    67: "Croatian",
    68: "Somali",
    69: "Kapampangan",
    70: "Navajo",
    71: "Colognian",
    72: "Classical Nahuatl",
    73: "Khmer",
    74: "Samogitian",
    75: "Sranan Tongo",
    76: "Bavarian",
    77: "Corsican",
    78: "Sorani Kurdish",
    79: "Palatine German",
    80: "Egyptian Arabic",
    81: "Tarantino",
    82: "French",
    83: "Maithili",
    84: "Cantonese",
    85: "Gujarati",
    86: "Finnish",
    87: "Kyrgyz",
    88: "Volapük",
    89: "Hausa",
    90: "Afrikaans",
    91: "Uyghur",
    92: "Lao",
    93: "Swedish",
    94: "Slovenian",
    95: "Korean",
    96: "Silesian",
    97: "Serbian",
    98: "Doteli",
    99: "Norman",
    100: "Lower Sorbian",
    101: "Indonesian",
    102: "Walloon",
    103: "Western Punjabi",
    104: "Ukrainian",
    105: "Bishnupriya Manipuri",
    106: "Vietnamese",
    107: "Turkish",
    108: "Aymara",
    109: "Lithuanian",
    110: "Zealandic",
    111: "Polish",
    112: "Estonian",
    113: "Sicilian",
    114: "West Flemish",
    115: "Saterland Frisian",
    116: "Gagauz",
    117: "Guarani",
    118: "Kazakh",
    119: "Bengali",
    120: "Picard",
    121: "Banjar",
    122: "Karachay-Balkar",
    123: "Amharic",
    124: "Zazaki",
    125: "Luxembourgish",
    126: "Italian",
    127: "Kabyle",
    128: "Belarusian",
    129: "Old English",
    130: "Eastern Mari",
    131: "Chechen",
    132: "Komi-Permyak",
    133: "Manx",
    134: "Ido",
    135: "Faroese",
    136: "Bashkir",
    137: "Icelandic",
    138: "Central Bicolano",
    139: "Tetum",
    140: "Japanese",
    141: "Kurdish",
    142: "Basa Banyumasan",
    143: "Tuvan",
    144: "Livvi-Karelian",
    145: "Aragonese",
    146: "Oriya",
    147: "Limburgish",
    148: "Telugu",
    149: "Lingala",
    150: "Romansh",
    151: "Albanian",
    152: "Xhosa",
    153: "Malagasy",
    154: "Persian",
    155: "Serbo-Croatian",
    156: "Tamil",
    157: "Azerbaijani",
    158: "Ladino",
    159: "Norwegian Bokmål",
    160: "Sinhala",
    161: "Scottish Gaelic",
    162: "Neapolitan",
    163: "Sindhi",
    164: "Asturian",
    165: "Malayalam",
    166: "Moksha",
    167: "Tswana",
    168: "Low German",
    169: "Tagalog",
    170: "Norwegian Nynorsk",
    171: "Sundanese",
    172: "Classical Chinese",
    173: "Lojban",
    174: "Crimean Tatar",
    175: "Papiamento",
    176: "Occitan",
    177: "Hakka Chinese",
    178: "Uzbek",
    179: "Chinese",
    180: "Upper Sorbian",
    181: "Northern Sami",
    182: "Maltese",
    183: "Veps",
    184: "Lezgian",
    185: "Dutch",
    186: "Dutch Low Saxon",
    187: "Hill Mari",
    188: "Spanish",
    189: "Cebuano",
    190: "Interlingua",
    191: "Hebrew",
    192: "Hungarian",
    193: "Quechua",
    194: "Karakalpak",
    195: "Marathi",
    196: "Venetian",
    197: "Arpitan",
    198: "Greek",
    199: "Yakut",
    200: "Basque",
    201: "Czech",
    202: "Slovak",
    203: "Cherokee",
    204: "Ligurian",
    205: "Nepali",
    206: "Sardinian",
    207: "Ilocano",
    208: "Taraškievica Belarusian",
    209: "Tibetan",
    210: "Oromo",
    211: "Waray",
    212: "Galician",
    213: "Mongolian",
    214: "Irish",
    215: "Minangkabau",
    216: "Igbo",
    217: "Occidental",
    218: "Esperanto",
    219: "Latvian",
    220: "Northern Luri",
    221: "Alsatian",
    222: "Mazanderani",
    223: "Aromanian",
    224: "Friulian",
    225: "Tatar",
    226: "Erzya",
    227: "Punjabi",
    228: "Tongan",
    229: "Komi",
    230: "Wu Chinese",
    231: "Tulu",
    232: "Turkmen",
    233: "Kannada",
    234: "Latgalian",


}

for text in new_texts:
    predicted_class_id = predict_language(text)
    lang_prediction = label_mapping[predicted_class_id]
    print(f'Text: "{text}" is predicted to be in language: {lang_prediction}')